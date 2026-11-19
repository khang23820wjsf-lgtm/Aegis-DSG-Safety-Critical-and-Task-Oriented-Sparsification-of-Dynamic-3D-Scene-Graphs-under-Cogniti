"""Record REAL MuJoCo 4.0.1 3D Rigid-Body Physics episodes on REAL 3DSSG rooms
and export an interactive 360°-rotatable 3D Web Viewer (results/mujoco_3dssg_viewer.html).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set
import mujoco
import networkx as nx
import numpy as np

from src.baselines import DistanceSparsifier
from src.dsg_generator import AgentState, NOISE_CONFIGS
from src.phase_b.aegis_v2 import AegisV2Sparsifier
from src.phase_b.coarsening import CompressedEntity
from src.physics.mujoco_env import _is_collidable_entity, _plan_dwa_velocity, build_mjcf_from_3dssg
from src.real_data.loader_3dssg import load_3dssg_scan


def record_mujoco_episode(
    scan_index: int,
    sparsifier,
    k: int = 5,
    max_steps: int = 120,
) -> Dict[str, Any]:
    """Run real MuJoCo 4.0 rigid-body physics and record every frame for 3D web playback."""
    room_size = (8.0, 10.0)
    w_x, l_y = room_size
    scan_id, g_room = load_3dssg_scan(scan_index, room_size=room_size, seed=42 + scan_index)
    mjcf, obj_meta, dyn_peds = build_mjcf_from_3dssg(g_room, room_size)

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42 + scan_index)
    noise_cfg = NOISE_CONFIGS["medium"]
    sparsifier.reset()

    agent_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mocap_ids = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["id"])] for p in dyn_peds]

    start_xy = np.array([w_x / 2.0, 0.75], dtype=float)
    goal_xy = np.array([w_x / 2.0, l_y - 0.65], dtype=float)
    prev_xy = start_xy.copy()
    hit_geoms: Set[int] = set()
    stall_steps = 0
    reached = False

    static_objs = [
        {
            "id": nid,
            "label": m["label"],
            "pos": [round(float(x), 3) for x in m["pos"]],
            "size": [round(float(x), 3) for x in m["size"]],
            "collidable": _is_collidable_entity(m["pos"][2], m["size"][2], m["hazard_severity"]),
            "haz": round(float(m["hazard_severity"]), 2),
        }
        for nid, m in obj_meta.items()
    ]

    frames: List[Dict[str, Any]] = []
    recalls: List[float] = []

    for step in range(max_steps):
        ax, ay = float(data.qpos[0] + start_xy[0]), float(data.qpos[1] + start_xy[1])
        avx, avy = float(data.qvel[0]), float(data.qvel[1])
        agent = AgentState(pos=np.array([ax, ay, 0.85]), vel=np.array([avx, avy, 0.0]), radius=0.30)

        g_obs = nx.DiGraph()
        spd = float(np.linalg.norm(agent.vel[:2]))
        heading = agent.vel[:2] / spd if spd > 0.1 else np.array([0.0, 1.0])

        all_entities = dict(obj_meta)
        for p in dyn_peds:
            all_entities[p["id"]] = {
                "label": "pedestrian",
                "pos": np.array([p["x"], p["y"], 0.85]),
                "size": np.array([0.48, 0.48, 1.70]),
                "vel": np.array([p["vx"], p["vy"], 0.0]),
                "structural_importance": 0.10,
                "hazard_severity": 0.95,
                "is_dynamic": True,
            }

        obs_hazards: Set[str] = set()
        for nid, m in all_entities.items():
            delta = m["pos"][:2] - agent.pos[:2]
            dist = float(np.linalg.norm(delta))
            if dist > agent.max_range or dist < 0.05:
                continue
            ang = math.acos(float(np.clip(np.dot(delta / dist, heading), -1.0, 1.0)))
            if ang > agent.fov_half_angle_rad:
                continue
            fov_ratio = ang / agent.fov_half_angle_rad
            if rng.random() < noise_cfg["dropout_base"] * (0.5 + 1.5 * fov_ratio ** 2):
                continue

            is_haz = False
            if _is_collidable_entity(m["pos"][2], m["size"][2], m["hazard_severity"]):
                rv = m["vel"][:2] - agent.vel[:2]
                v2 = float(np.dot(rv, rv))
                ts = 0.0 if v2 < 1e-6 else float(np.clip(-np.dot(delta, rv) / v2, 0.0, 3.0))
                if np.all(np.abs(delta + rv * ts) <= m["size"][:2] / 2.0 + agent.radius + 0.25):
                    is_haz = True
                    obs_hazards.add(nid)

            conf = float(np.clip(0.98 - 0.35 * fov_ratio ** 2 - 0.12 * (dist / agent.max_range) + rng.normal(0, noise_cfg["conf_noise"]), 0.20, 0.99))
            g_obs.add_node(
                nid, layer="object", label=m["label"],
                pos=m["pos"] + rng.normal(0, noise_cfg["pos_std"], size=3),
                size=m["size"].copy(), vel=m["vel"].copy(), confidence=conf,
                structural_importance=m["structural_importance"],
                hazard_severity=m["hazard_severity"], is_dynamic=m["is_dynamic"], is_hazard_gt=is_haz,
            )

        if hasattr(sparsifier, "select_entities"):
            entities = sparsifier.select_entities(g_obs, agent, k)
            covered = set().union(*(e.member_ids for e in entities)) if entities else set()
        else:
            sel = sparsifier.select(g_obs, agent, k)
            covered = sel
            entities = [
                CompressedEntity(
                    entity_id=nid, label=str(g_obs.nodes[nid]["label"]),
                    pos=g_obs.nodes[nid]["pos"].copy(), size=g_obs.nodes[nid]["size"].copy(),
                    vel=g_obs.nodes[nid]["vel"].copy(), score=1.0, member_ids={nid},
                )
                for nid in sel if nid in g_obs
            ]

        recalls.append(1.0 if not obs_hazards else len(covered & obs_hazards) / len(obs_hazards))
        vx_cmd, vy_cmd = _plan_dwa_velocity(agent, entities, goal_xy, room_size, stall_steps)
        data.ctrl[0], data.ctrl[1] = vx_cmd, vy_cmd

        step_contact = False
        for _ in range(20):
            for p, mid in zip(dyn_peds, mocap_ids):
                p["x"] += p["vx"] * 0.005
                if p["x"] < 0.7 or p["x"] > w_x - 0.7:
                    p["vx"] *= -1.0
                data.mocap_pos[mid] = [p["x"], p["y"], 0.85]
            mujoco.mj_step(model, data)
            for ci in range(data.ncon):
                con = data.contact[ci]
                g1, g2 = int(con.geom1), int(con.geom2)
                if agent_geom_id in (g1, g2):
                    other = g2 if g1 == agent_geom_id else g1
                    if other != floor_geom_id:
                        hit_geoms.add(other)
                        step_contact = True

        curr_xy = np.array([data.qpos[0] + start_xy[0], data.qpos[1] + start_xy[1]])
        step_dist = float(np.linalg.norm(curr_xy - prev_xy))
        stall_steps = (stall_steps + 1) if step_dist < 0.03 else 0
        prev_xy = curr_xy

        frames.append({
            "t": step,
            "agent": [round(float(curr_xy[0]), 3), round(float(curr_xy[1]), 3), 0.85],
            "vel": [round(float(data.qvel[0]), 2), round(float(data.qvel[1]), 2)],
            "peds": [[round(p["x"], 3), round(p["y"], 3), 0.85] for p in dyn_peds],
            "covered": sorted(covered),
            "supers": [
                {"pos": [round(float(x), 2) for x in e.pos], "size": [round(float(x), 2) for x in e.size], "label": e.label}
                for e in entities if e.is_super_node
            ],
            "contact": step_contact,
            "total_hits": len(hit_geoms),
            "avg_cov": round(float(np.mean(recalls)) * 100, 1),
        })

        if float(np.linalg.norm(curr_xy - goal_xy)) <= 0.85:
            reached = True
            break

    return {
        "scan_id": scan_id[:12],
        "room_size": [w_x, l_y],
        "goal": [round(float(goal_xy[0]), 2), round(float(goal_xy[1]), 2)],
        "objects": static_objs,
        "frames": frames,
        "reached": reached,
        "total_hits": len(hit_geoms),
        "avg_cov": round(float(np.mean(recalls)) * 100, 1),
    }


HTML_3D = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>MuJoCo 4.0 x Real 3DSSG — Trình xem 3D Tương tác 360°</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { padding: 10px 16px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  header h1 { margin: 0; font-size: 14.5px; color: #38bdf8; }
  .pill { background: rgba(249,115,22,0.22); color: #fdba74; font-size: 10.5px; padding: 2px 8px; border-radius: 10px; font-weight: 700; margin-left: 6px; }
  .controls { display: flex; gap: 8px; align-items: center; font-size: 12px; flex-wrap: wrap; }
  button { background: #0284c7; color: white; border: none; padding: 5px 11px; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button:hover { background: #0369a1; }
  select { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 4px 7px; font-size: 12px; }
  input[type=range] { width: 170px; accent-color: #38bdf8; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 310px; gap: 10px; padding: 10px; min-height: 0; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  canvas { flex: 1; width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .sidebar { padding: 12px; overflow-y: auto; font-size: 12.5px; display: flex; flex-direction: column; gap: 10px; }
  .card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; }
  .card h3 { margin: 0 0 6px 0; font-size: 13px; color: #f8fafc; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1e293b; }
  .ok { color: #4ade80; font-weight: 700; }
  .bad { color: #f87171; font-weight: 700; }
  .hint { background: rgba(56,189,248,0.12); border-left: 3px solid #38bdf8; padding: 7px 9px; border-radius: 4px; font-size: 11.5px; color: #bae6fd; line-height: 1.45; }
</style>
</head>
<body>
<header>
  <h1>MuJoCo 4.0 × Real 3DSSG Rooms <span class="pill">VẬT LÝ CỨNG 3D THẬT · XOAY 360°</span></h1>
  <div class="controls">
    <select id="roomSelect"></select>
    <select id="algSelect">
      <option value="aegis" selected>Thuật toán: Aegis-v2 (Phase B)</option>
      <option value="distance">Thuật toán: Distance Baseline</option>
    </select>
    <button id="playBtn">Tạm dừng</button>
    <input type="range" id="slider" min="0" max="0" value="0" />
    <span id="frameLbl">Frame 0</span>
  </div>
</header>

<main>
  <div class="panel">
    <canvas id="view3d"></canvas>
  </div>
  <div class="panel sidebar">
    <div class="hint">
      <b>Điều khiển 3D:</b><br/>
      • <b>Kéo chuột trái</b> để xoay góc nhìn 360° quanh căn phòng thật 3DSSG.<br/>
      • <b>Cuộn chuột</b> để phóng to / thu nhỏ.<br/>
      • Quỹ đạo này được ghi trực tiếp từ bộ giải vật lý cứng <b>MuJoCo 4.0 (200Hz)</b> — đâm vào bàn ghế là bị chặn cứng thật!
    </div>
    <div class="card">
      <h3>Thông số MuJoCo thời gian thực</h3>
      <div id="statsBox"></div>
    </div>
    <div class="card">
      <h3>Chú thích 3D</h3>
      <div style="font-size:12px; line-height:1.7;">
        🟠 <b>Trụ màu cam:</b> Agent (người dùng)<br/>
        🔴 <b>Trụ đỏ di chuyển:</b> Người đi bộ<br/>
        🟫 <b>Khối hộp nâu/đỏ:</b> Đồ nội thất thật 3DSSG (cứng)<br/>
        🟦 <b>Viền xanh dương:</b> Đang nằm trong Top-k<br/>
        🟧 <b>Khung cam lớn:</b> Siêu nút Gom cụm (Super-Node)<br/>
        🟢 <b>Đĩa xanh lá cuối phòng:</b> Đích đến
      </div>
    </div>
  </div>
</main>

<script>
const DB = __DATA_JSON__;
const roomSelect = document.getElementById("roomSelect");
const algSelect = document.getElementById("algSelect");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("playBtn");
const frameLbl = document.getElementById("frameLbl");
const statsBox = document.getElementById("statsBox");

Object.keys(DB).forEach(k => {
  const opt = document.createElement("option");
  opt.value = k;
  opt.textContent = `Phòng thật 3DSSG #${k} (${DB[k].aegis.scan_id})`;
  roomSelect.appendChild(opt);
});

let curRoom = Object.keys(DB)[0];
let curAlg = "aegis";
let idx = 0, playing = true, lastTick = performance.now();

// Interactive 3D camera spherical coordinates
let yaw = -0.65, pitch = 0.62, camDist = 14.5;
let dragging = false, lastX = 0, lastY = 0;

const canvas = document.getElementById("view3d");
canvas.addEventListener("mousedown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.008;
  pitch = Math.max(0.15, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
  lastX = e.clientX; lastY = e.clientY;
  render();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  camDist = Math.max(6, Math.min(26, camDist + e.deltaY * 0.01));
  render();
}, { passive: false });

function getEp() { return DB[curRoom][curAlg]; }
function resetEp() { idx = 0; slider.max = getEp().frames.length - 1; render(); }

roomSelect.onchange = e => { curRoom = e.target.value; resetEp(); };
algSelect.onchange = e => { curAlg = e.target.value; resetEp(); };
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Tạm dừng" : "Chạy tiếp"; };
slider.oninput = e => { idx = parseInt(e.target.value, 10); render(); };

function project3D(x, y, z, w, h, cx, cy) {
  const rx = x - cx, ry = y - cy, rz = z - 0.8;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const x1 = rx * cosY - ry * sinY;
  const y1 = rx * sinY + ry * cosY;
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const y2 = y1 * cosP - rz * sinP;
  const z2 = y1 * sinP + rz * cosP;
  const depth = y2 + camDist;
  const fov = Math.min(w, h) * 0.95;
  const scale = fov / Math.max(1.0, depth);
  return { sx: w * 0.5 + x1 * scale, sy: h * 0.56 - z2 * scale, depth };
}

function drawBox3D(ctx, pos, size, fill, stroke, lw, w, h, cx, cy, label) {
  const [px, py, pz] = pos;
  const [sx, sy, sz] = size;
  const hx = sx / 2, hy = sy / 2, hz = sz / 2;
  const corners = [
    [px-hx, py-hy, pz-hz], [px+hx, py-hy, pz-hz], [px+hx, py+hy, pz-hz], [px-hx, py+hy, pz-hz],
    [px-hx, py-hy, pz+hz], [px+hx, py-hy, pz+hz], [px+hx, py+hy, pz+hz], [px-hx, py+hy, pz+hz],
  ].map(c => project3D(c[0], c[1], c[2], w, h, cx, cy));

  const faces = [
    [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]
  ];
  ctx.lineWidth = lw;
  for (const f of faces) {
    ctx.beginPath();
    f.forEach((ci, i) => i === 0 ? ctx.moveTo(corners[ci].sx, corners[ci].sy) : ctx.lineTo(corners[ci].sx, corners[ci].sy));
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  if (label) {
    const top = project3D(px, py, pz + hz + 0.2, w, h, cx, cy);
    ctx.fillStyle = "#f8fafc"; ctx.font = "10px system-ui";
    ctx.fillText(label, top.sx - 14, top.sy);
  }
}

function render() {
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== r.width || canvas.height !== r.height) { canvas.width = r.width; canvas.height = r.height; }
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const ep = getEp();
  const fr = ep.frames[idx];
  const [rw, rl] = ep.room_size;
  const cx = rw / 2, cy = rl / 2;

  if (fr.contact) {
    ctx.fillStyle = "rgba(239,68,68,0.22)";
    ctx.fillRect(0, 0, w, h);
  }

  // Floor grid
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  for (let gx = 0; gx <= rw; gx += 1) {
    const a = project3D(gx, 0, 0, w, h, cx, cy), b = project3D(gx, rl, 0, w, h, cx, cy);
    ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
  }
  for (let gy = 0; gy <= rl; gy += 1) {
    const a = project3D(0, gy, 0, w, h, cx, cy), b = project3D(rw, gy, 0, w, h, cx, cy);
    ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
  }

  // Room outer wireframe boundary
  const bpts = [[0,0],[rw,0],[rw,rl],[0,rl]].map(p => project3D(p[0], p[1], 0, w, h, cx, cy));
  ctx.strokeStyle = "#64748b"; ctx.lineWidth = 2;
  ctx.beginPath(); bpts.forEach((p,i)=> i===0?ctx.moveTo(p.sx,p.sy):ctx.lineTo(p.sx,p.sy)); ctx.closePath(); ctx.stroke();

  // Goal beacon
  drawBox3D(ctx, [ep.goal[0], ep.goal[1], 0.08], [1.1, 1.1, 0.12], "rgba(34,197,94,0.45)", "#4ade80", 1.5, w, h, cx, cy, "ĐÍCH");

  // Agent walked trajectory on floor
  ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2.5; ctx.beginPath();
  for (let i = 0; i <= idx; i++) {
    const ap = ep.frames[i].agent;
    const p = project3D(ap[0], ap[1], 0.05, w, h, cx, cy);
    i === 0 ? ctx.moveTo(p.sx, p.sy) : ctx.lineTo(p.sx, p.sy);
  }
  ctx.stroke();

  // Collect all 3D boxes and painter-sort back-to-front by camera depth
  const covSet = new Set(fr.covered);
  const drawables = [];

  for (const o of ep.objects) {
    const p = project3D(o.pos[0], o.pos[1], o.pos[2], w, h, cx, cy);
    const isSel = covSet.has(o.id);
    const fill = !o.collidable ? "rgba(34,197,94,0.22)" : (o.haz >= 0.8 ? "rgba(220,75,60,0.68)" : "rgba(180,130,85,0.68)");
    const stroke = isSel ? "#38bdf8" : "rgba(15,23,42,0.7)";
    drawables.push({ depth: p.depth, fn: () => drawBox3D(ctx, o.pos, o.size, fill, stroke, isSel ? 2.4 : 0.9, w, h, cx, cy, isSel ? o.label : "") });
  }

  fr.peds.forEach((pp, pi) => {
    const p = project3D(pp[0], pp[1], pp[2], w, h, cx, cy);
    const isSel = covSet.has(`dyn_ped_${pi}`);
    drawables.push({ depth: p.depth, fn: () => drawBox3D(ctx, pp, [0.46, 0.46, 1.65], "rgba(239,68,68,0.88)", isSel ? "#38bdf8" : "#fecaca", isSel ? 2.6 : 1.2, w, h, cx, cy, "Người") });
  });

  const agP = project3D(fr.agent[0], fr.agent[1], fr.agent[2], w, h, cx, cy);
  drawables.push({ depth: agP.depth, fn: () => drawBox3D(ctx, fr.agent, [0.54, 0.54, 1.68], fr.contact ? "rgba(239,68,68,0.95)" : "rgba(245,158,11,0.92)", "#fef3c7", 2.0, w, h, cx, cy, fr.contact ? "💥 ĐỤNG!" : "AGENT") });

  drawables.sort((a, b) => b.depth - a.depth);
  drawables.forEach(d => d.fn());

  // Super-Nodes wireframe on top
  for (const s of fr.supers) {
    ctx.setLineDash([5, 4]);
    drawBox3D(ctx, s.pos, s.size, "rgba(249,115,22,0.16)", "#f97316", 2.5, w, h, cx, cy, s.label);
    ctx.setLineDash([]);
  }

  slider.value = idx;
  frameLbl.textContent = `Frame ${idx}/${ep.frames.length - 1}`;
  statsBox.innerHTML = `
    <div class="stat-row"><span>Phòng thật 3DSSG</span><b>${ep.scan_id}</b></div>
    <div class="stat-row"><span>Đồ nội thất cứng</span><b>${ep.objects.length} vật</b></div>
    <div class="stat-row"><span>Va chạm cứng MuJoCo</span><span class="${fr.total_hits === 0 ? 'ok' : 'bad'}">${fr.total_hits} lần</span></div>
    <div class="stat-row"><span>Hazard Coverage TB</span><span class="ok">${fr.avg_cov}%</span></div>
    <div class="stat-row"><span>Tới đích cuối tập</span><span class="${ep.reached ? 'ok' : 'bad'}">${ep.reached ? '✓ TỚI ĐÍCH' : '✕ Bị kẹt chữ U'}</span></div>
  `;
}

function loop(now) {
  const ep = getEp();
  if (playing && now - lastTick >= 85) {
    idx = (idx + 1) % ep.frames.length;
    render();
    lastTick = now;
  }
  requestAnimationFrame(loop);
}

resetEp();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def main() -> None:
    db: Dict[str, Any] = {}
    for scan_idx in (0, 1, 2, 3, 4):
        print(f"Recording REAL MuJoCo 4.0 physics on 3DSSG Scan #{scan_idx}...")
        db[str(scan_idx)] = {
            "aegis": record_mujoco_episode(scan_idx, AegisV2Sparsifier(), k=5),
            "distance": record_mujoco_episode(scan_idx, DistanceSparsifier(), k=5),
        }

    out_path = Path("results/mujoco_3dssg_viewer.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_3D.replace("__DATA_JSON__", json.dumps(db, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")
    print(f"3D Interactive MuJoCo x 3DSSG Web Viewer saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
