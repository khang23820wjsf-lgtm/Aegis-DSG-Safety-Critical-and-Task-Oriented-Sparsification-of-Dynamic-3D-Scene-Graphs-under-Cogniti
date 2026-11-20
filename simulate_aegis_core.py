"""Interactive Web Simulation driven 100% by the REAL Python AegisDSG Core (src/aegis_core.py).

Records exact frame-by-frame outputs of all 4 AegisDSG layers:
  - Layer 1: Temporal Belief p_t(v) & Object Permanence (occluded memory nodes)
  - Layer 2: Unified 3D Risk Field R_v & Asymmetric Weight w_v
  - Layer 3: Free-Space-Preserving Super-Nodes (gap < 0.70m)
  - Layer 4: Dynamic-Budget Submodular Selection (|V'_t| <= k, stopping < tau_min)

Side-by-side comparison against Distance, D-Lite(True Dijkstra), and Clio(True aIB)
across all 4 unseen 3D environments (seed=100, k=5, medium noise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.aegis_core import AegisDSG
from src.baselines import DistanceSparsifier
from src.dsg_generator import DSGFrame, ScenarioType, generate_dsg_sequence
from src.metrics import compute_hazard_recall, compute_jaccard_flicker
from src.phase_b.true_baselines import ClioTrueIBSparsifier, DLiteTrueSparsifier


VI_LABELS: Dict[str, str] = {
    "chair": "Ghế",
    "table": "Bàn",
    "low_barrier": "Rào chắn",
    "stair_edge": "Mép cầu thang",
    "overhead_sign": "Biển ngang đầu",
    "door": "Cửa ra vào",
    "wall_pillar": "Cột tường",
    "pedestrian": "Người đi bộ",
    "wall_painting": "Tranh cao",
    "ceiling_light": "Đèn trần",
    "floor_mat": "Thảm sàn",
    "shelf": "Kệ tủ",
    "unknown_ghost": "Bóng ma FP",
}

SCENARIO_META: Dict[ScenarioType, Dict[str, Any]] = {
    "corridor": {"title": "1. Hành lang thẳng", "width": 4.0, "length": 24.0},
    "l_turn_corner": {"title": "2. Ngã rẽ góc khuất 90°", "width": 16.0, "length": 13.0},
    "crowded_lobby": {"title": "3. Sảnh đông cắt ngang", "width": 10.0, "length": 18.0},
    "vertical_hazards": {"title": "4. Phân tầng độ cao 3D", "width": 4.0, "length": 24.0},
}


def _record_scenario(scenario: ScenarioType, k: int = 5, seed: int = 100) -> Dict[str, Any]:
    seq: List[DSGFrame] = generate_dsg_sequence(
        num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=seed, scenario=scenario
    )

    aegis = AegisDSG()
    baselines = {
        "distance": DistanceSparsifier(),
        "dlite": DLiteTrueSparsifier(),
        "clio": ClioTrueIBSparsifier(),
    }
    aegis.reset()
    for b in baselines.values():
        b.reset()

    prev_aegis: Set[str] = set()
    prev_base: Dict[str, Set[str]] = {name: set() for name in baselines}
    sum_rec_a = sum_fl_a = sum_used_a = 0.0
    sum_rec_b = {name: 0.0 for name in baselines}
    sum_fl_b = {name: 0.0 for name in baselines}

    frames_out: List[Dict[str, Any]] = []

    for idx, f in enumerate(seq):
        obs_ids = {n for n, d in f.graph_obs.nodes(data=True) if d.get("layer") == "object"}
        all_ids = {n for n, d in f.graph_gt.nodes(data=True) if d.get("layer") == "object"} | obs_ids
        obs_haz = {n for n in obs_ids if f.graph_obs.nodes[n].get("is_hazard_gt", False)}

        # Run REAL Python AegisDSG core!
        ents = aegis.compress(f.graph_obs, f.agent, k)
        cov_a = set().union(*(e.member_ids for e in ents)) if ents else set()
        miss_a = obs_haz - cov_a
        rec_a = compute_hazard_recall(cov_a, obs_haz)
        fl_a = compute_jaccard_flicker(prev_aegis, cov_a) if idx > 0 else 0.0
        sum_rec_a += rec_a
        sum_fl_a += fl_a
        sum_used_a += len(ents)
        prev_aegis = cov_a

        # Run REAL Python baselines
        base_step: Dict[str, Any] = {}
        for bname, balg in baselines.items():
            sel_b = balg.select(f.graph_obs, f.agent, k)
            miss_b = obs_haz - sel_b
            rec_b = compute_hazard_recall(sel_b, obs_haz)
            fl_b = compute_jaccard_flicker(prev_base[bname], sel_b) if idx > 0 else 0.0
            sum_rec_b[bname] += rec_b
            sum_fl_b[bname] += fl_b
            prev_base[bname] = sel_b
            base_step[bname] = {
                "sel": sorted(sel_b),
                "missed": sorted(miss_b),
                "avg_rec": round((sum_rec_b[bname] / (idx + 1)) * 100, 1),
                "avg_fl": round(sum_fl_b[bname] / max(1, idx), 3),
            }

        objs_list = []
        for nid in sorted(all_ids):
            in_obs = nid in obs_ids
            d = f.graph_obs.nodes[nid] if in_obs else f.graph_gt.nodes[nid]
            objs_list.append({
                "id": nid,
                "label": VI_LABELS.get(d["label"], d["label"]),
                "pos": [round(float(x), 3) for x in d["pos"]],
                "size": [round(float(x), 3) for x in d["size"]],
                "vel": [round(float(x), 3) for x in d["vel"]],
                "conf": round(float(d.get("confidence", 1.0)), 2),
                "is_haz": bool(d.get("is_hazard_gt", False)),
                "is_dyn": bool(d.get("is_dynamic", False)),
                "in_obs": in_obs,
            })

        aegis_ents_json = [
            {
                "id": e.entity_id,
                "label": e.label if e.is_super_node else VI_LABELS.get(e.label, e.label),
                "pos": [round(float(x), 3) for x in e.pos],
                "size": [round(float(x), 3) for x in e.size],
                "risk": round(e.risk, 2),
                "weight": round(e.weight, 2),
                "belief": round(e.belief, 2),
                "is_super": e.is_super_node,
                "is_occ": e.is_occluded_memory,
                "members": sorted(e.member_ids),
            }
            for e in ents
        ]

        frames_out.append({
            "t": f.t,
            "time": f.timestamp,
            "agent": {
                "pos": [round(float(x), 3) for x in f.agent.pos],
                "vel": [round(float(x), 3) for x in f.agent.vel],
                "fov": round(f.agent.fov_half_angle_rad, 4),
                "max_range": f.agent.max_range,
            },
            "objects": objs_list,
            "aegis": {
                "entities": aegis_ents_json,
                "covered": sorted(cov_a),
                "missed": sorted(miss_a),
                "avg_rec": round((sum_rec_a / (idx + 1)) * 100, 1),
                "avg_fl": round(sum_fl_a / max(1, idx), 3),
                "avg_used": round(sum_used_a / (idx + 1), 2),
                "used": len(ents),
            },
            "baselines": base_step,
        })

    return {"meta": SCENARIO_META[scenario], "k": k, "frames": frames_out}


HTML_CORE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Aegis-Core v1.0 — Mô phỏng 4 Tầng Nguyên Khối (Python Backend)</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { padding: 9px 16px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  header h1 { margin: 0; font-size: 14.5px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
  .pill { background: rgba(34,197,94,0.2); color: #4ade80; font-size: 10.5px; padding: 2px 8px; border-radius: 10px; font-weight: 700; }
  .controls { display: flex; gap: 8px; align-items: center; font-size: 12px; flex-wrap: wrap; }
  button { background: #0284c7; color: white; border: none; padding: 5px 11px; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button:hover { background: #0369a1; }
  select { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 4px 7px; font-size: 12px; }
  input[type=range] { width: 165px; accent-color: #38bdf8; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr 355px; gap: 10px; padding: 10px; min-height: 0; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { padding: 7px 11px; font-size: 12.5px; font-weight: 700; background: #0f172a; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .b-green { background: rgba(34,197,94,0.2); color: #4ade80; }
  .b-red { background: rgba(239,68,68,0.22); color: #f87171; }
  canvas { flex: 1; width: 100%; height: 100%; display: block; }
  .sidebar { padding: 10px; overflow-y: auto; font-size: 12px; display: flex; flex-direction: column; gap: 8px; }
  .card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 9px; }
  .card h3 { margin: 0 0 6px 0; font-size: 12.5px; color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; font-size: 11.8px; }
  th, td { padding: 4px 5px; border-bottom: 1px solid #1e293b; text-align: center; }
  th { color: #94a3b8; }
  .win { color: #4ade80; font-weight: 700; }
  .lose { color: #f87171; font-weight: 700; }
  .ent-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; }
  .ent-list li { padding: 5px 7px; border-radius: 5px; font-size: 11.2px; display: flex; justify-content: space-between; align-items: center; }
  .li-haz { background: rgba(239,68,68,0.16); border-left: 3px solid #ef4444; }
  .li-super { background: rgba(249,115,22,0.22); border-left: 3px solid #f97316; color: #fed7aa; font-weight: 600; }
  .li-occ { background: rgba(168,85,247,0.22); border-left: 3px solid #a855f7; color: #e9d5ff; }
  .li-safe { background: rgba(56,189,248,0.12); border-left: 3px solid #38bdf8; }
  .li-miss { background: rgba(239,68,68,0.3); border-left: 3px solid #ff2e2e; color: #fecaca; font-weight: 700; }
  .legend { font-size: 11px; color: #cbd5e1; line-height: 1.6; }
</style>
</head>
<body>
<header>
  <h1>Aegis-Core v1.0 Simulator <span class="pill">100% REAL PYTHON CORE (src/aegis_core.py)</span></h1>
  <div class="controls">
    <select id="scSelect">
      <option value="corridor">1. Hành lang thẳng</option>
      <option value="l_turn_corner">2. Ngã rẽ góc khuất 90°</option>
      <option value="crowded_lobby" selected>3. Sảnh đông cắt ngang</option>
      <option value="vertical_hazards">4. Phân tầng độ cao 3D</option>
    </select>
    <select id="baseSelect">
      <option value="clio" selected>So với: Clio (True aIB)</option>
      <option value="dlite">So với: D-Lite (True Dijkstra)</option>
      <option value="distance">So với: Distance</option>
    </select>
    <button id="playBtn">Tạm dừng</button>
    <input type="range" id="slider" min="0" max="0" value="0" />
    <span id="timeLbl">Frame 0</span>
  </div>
</header>

<main>
  <div class="panel">
    <div class="panel-title">
      <span id="leftTitle">A. Baseline: Clio (True aIB)</span>
      <span id="leftBadge" class="badge b-red">Sót: 0</span>
    </div>
    <canvas id="cLeft"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">
      <span>B. Aegis-Core v1.0 (4 Tầng: Belief + Risk + Coarsening + Submodular)</span>
      <span id="rightBadge" class="badge b-green">|V'| = 0/5</span>
    </div>
    <canvas id="cRight"></canvas>
  </div>
  <div class="panel sidebar">
    <div class="card">
      <h3>Đối chiếu tích lũy (Seed 100 — Unseen, k=5)</h3>
      <table>
        <thead><tr><th>Chỉ số</th><th id="thBase">Clio (aIB)</th><th>Aegis-Core</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
    <div class="card">
      <h3>Đầu ra 4 tầng của Aegis-Core ngay frame này</h3>
      <ul id="entList" class="ent-list"></ul>
    </div>
    <div class="card">
      <h3>Chú thích 4 Tầng Aegis-Core</h3>
      <div class="legend">
        🟣 <b>Tầng 1 (Object Permanence):</b> Khung tím = Vật bị che khuất vẫn được nhớ vị trí!<br/>
        🟢 <b>Tầng 2 (Risk Field):</b> Vòng xanh lá = Bảo vệ hai vai + Phễu xanh dương TTC.<br/>
        🟧 <b>Tầng 3 (Coarsening):</b> Khung cam nét đứt = Siêu nút gộp vật sát nhau (&lt;0.7m).<br/>
        🔵 <b>Tầng 4 (Dynamic Budget):</b> Tự dừng sớm nếu ít nguy hiểm (|V'| ≤ k).<br/>
        ❌ <b>Chữ X đỏ:</b> Vật cản nguy hiểm bị bỏ sót!
      </div>
    </div>
  </div>
</main>

<script>
const DB = __DATA_JSON__;
let scKey = "crowded_lobby", baseKey = "clio", idx = 0, playing = true, lastTick = performance.now();

const scSel = document.getElementById("scSelect");
const baseSel = document.getElementById("baseSelect");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("playBtn");

function getSc() { return DB[scKey]; }
function resetSc() { idx = 0; slider.max = getSc().frames.length - 1; render(); }

scSel.onchange = e => { scKey = e.target.value; resetSc(); };
baseSel.onchange = e => { baseKey = e.target.value; render(); };
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Tạm dừng" : "Chạy tiếp"; };
slider.oninput = e => { idx = parseInt(e.target.value, 10); render(); };

function resize(c) {
  const r = c.getBoundingClientRect();
  if (c.width !== r.width || c.height !== r.height) { c.width = r.width; c.height = r.height; }
}

function drawScene(cid, frame, isCore) {
  const c = document.getElementById(cid);
  resize(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const meta = getSc().meta;
  const ax = frame.agent.pos[0], ay = frame.agent.pos[1];
  const scale = Math.min(c.width / 15.0, c.height / 15.0);
  const toScreen = (x, y) => [c.width * 0.5 + (x - ax) * scale, c.height * 0.72 - (y - ay) * scale];

  // Walls
  ctx.strokeStyle = "#475569"; ctx.lineWidth = 3; ctx.beginPath();
  if (scKey === "l_turn_corner") {
    [[0,0],[0,13],[16,13]].forEach((p,i)=>{ const [sx,sy]=toScreen(p[0],p[1]); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); });
    [[4,0],[4,9],[16,9]].forEach((p,i)=>{ const [sx,sy]=toScreen(p[0],p[1]); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); });
  } else {
    const [x1,y1]=toScreen(0,0), [x2,y2]=toScreen(0,meta.length);
    const [x3,y3]=toScreen(meta.width,0), [x4,y4]=toScreen(meta.width,meta.length);
    ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.moveTo(x3,y3); ctx.lineTo(x4,y4);
  }
  ctx.stroke();

  // FOV + Aegis Risk Field
  const [sx, sy] = toScreen(ax, ay);
  const heading = Math.atan2(frame.agent.vel[1], frame.agent.vel[0]);
  ctx.fillStyle = "rgba(250,204,21,0.07)"; ctx.strokeStyle = "rgba(250,204,21,0.3)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(sx, sy); ctx.arc(sx, sy, frame.agent.max_range * scale, -heading - frame.agent.fov, -heading + frame.agent.fov); ctx.closePath(); ctx.fill(); ctx.stroke();

  if (isCore) {
    const ch = 38 * Math.PI / 180;
    ctx.fillStyle = "rgba(56,189,248,0.13)"; ctx.strokeStyle = "rgba(56,189,248,0.7)"; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.arc(sx, sy, 6.5 * scale, -heading - ch, -heading + ch); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = "rgba(74,222,128,0.55)"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(sx, sy, 1.8 * scale, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }

  const covSet = new Set(isCore ? frame.aegis.covered : frame.baselines[baseKey].sel);
  const missSet = new Set(isCore ? frame.aegis.missed : frame.baselines[baseKey].missed);

  for (const o of frame.objects) {
    const [ox, oy] = toScreen(o.pos[0], o.pos[1]);
    if (ox < -40 || ox > c.width + 40 || oy < -40 || oy > c.height + 40) continue;
    const isCov = covSet.has(o.id), isMiss = missSet.has(o.id);
    const w = Math.max(8, o.size[0] * scale), h = Math.max(8, o.size[1] * scale);
    ctx.save();
    ctx.globalAlpha = isCov || isMiss ? 1.0 : (o.in_obs ? 0.32 : 0.12);
    ctx.fillStyle = o.in_obs ? (o.is_haz ? "#ef4444" : "#22c55e") : "#64748b";
    ctx.fillRect(ox - w/2, oy - h/2, w, h);

    if (isCov) {
      ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.3;
      ctx.beginPath(); ctx.arc(ox, oy, Math.max(w, h) * 0.65 + 4, 0, Math.PI * 2); ctx.stroke();
    }
    if (isMiss) {
      ctx.strokeStyle = "#ff2e2e"; ctx.lineWidth = 3.2;
      ctx.beginPath(); ctx.moveTo(ox-8,oy-8); ctx.lineTo(ox+8,oy+8); ctx.moveTo(ox+8,oy-8); ctx.lineTo(ox-8,oy+8); ctx.stroke();
    }
    if (o.is_dyn && o.in_obs) {
      ctx.strokeStyle = "#fca5a5"; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + o.vel[0]*scale*0.7, oy - o.vel[1]*scale*0.7); ctx.stroke();
    }
    if (isCov || isMiss) {
      ctx.fillStyle = isMiss ? "#fca5a5" : "#f8fafc";
      ctx.font = isMiss ? "bold 10.5px system-ui" : "10px system-ui";
      ctx.fillText(isMiss ? `SÓT: ${o.label}` : o.label, ox + w/2 + 5, oy + 3);
    }
    ctx.restore();
  }

  // Draw Layer 1 Occluded Memory (purple) & Layer 3 Super-Nodes (orange) on Core panel
  if (isCore) {
    for (const ent of frame.aegis.entities) {
      const [ex, ey] = toScreen(ent.pos[0], ent.pos[1]);
      const ew = Math.max(14, ent.size[0] * scale + 8), eh = Math.max(14, ent.size[1] * scale + 8);
      if (ent.is_super) {
        ctx.save();
        ctx.fillStyle = "rgba(249,115,22,0.18)"; ctx.strokeStyle = "#f97316"; ctx.lineWidth = 2.5; ctx.setLineDash([6,4]);
        ctx.fillRect(ex - ew/2, ey - eh/2, ew, eh); ctx.strokeRect(ex - ew/2, ey - eh/2, ew, eh);
        ctx.setLineDash([]); ctx.fillStyle = "#fdba74"; ctx.font = "bold 10.5px system-ui";
        ctx.fillText(`🔶 ${ent.label}`, ex - ew/2, ey - eh/2 - 4);
        ctx.restore();
      } else if (ent.is_occ) {
        ctx.save();
        ctx.strokeStyle = "#a855f7"; ctx.lineWidth = 2.5; ctx.setLineDash([4,3]);
        ctx.strokeRect(ex - ew/2, ey - eh/2, ew, eh);
        ctx.setLineDash([]); ctx.fillStyle = "#e9d5ff"; ctx.font = "bold 10px system-ui";
        ctx.fillText(`🟣 NHỚ: ${ent.label}`, ex - ew/2, ey - eh/2 - 4);
        ctx.restore();
      }
    }
  }

  // Agent
  ctx.fillStyle = "#f59e0b"; ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#fef3c7"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + frame.agent.vel[0]*scale*0.9, sy - frame.agent.vel[1]*scale*0.9); ctx.stroke();
}

function render() {
  const sc = getSc();
  const fr = sc.frames[idx];
  const baseName = { clio: "Clio (True aIB)", dlite: "D-Lite (Dijkstra)", distance: "Distance" }[baseKey];
  const b = fr.baselines[baseKey], a = fr.aegis;

  slider.value = idx;
  document.getElementById("timeLbl").textContent = `Frame ${fr.t}/${sc.frames.length - 1} (${fr.time.toFixed(1)}s)`;
  document.getElementById("leftTitle").textContent = `A. Baseline: ${baseName}`;
  document.getElementById("thBase").textContent = baseName;

  const lb = document.getElementById("leftBadge");
  lb.textContent = b.missed.length === 0 ? "✓ Không sót" : `✕ Sót ${b.missed.length} vật!`;
  lb.className = "badge " + (b.missed.length === 0 ? "b-green" : "b-red");

  const rb = document.getElementById("rightBadge");
  rb.textContent = a.missed.length === 0 ? `✓ |V'|=${a.used}/${sc.k}` : `✕ Sót ${a.missed.length} | |V'|=${a.used}/${sc.k}`;
  rb.className = "badge " + (a.missed.length === 0 ? "b-green" : "b-red");

  const winR = a.avg_rec >= b.avg_rec, winF = a.avg_fl <= b.avg_fl;
  document.getElementById("metricsBody").innerHTML = `
    <tr><td>Hazard Coverage (↑)</td><td class="${!winR?'win':'lose'}">${b.avg_rec}%</td><td class="${winR?'win':'lose'}">${a.avg_rec}%</td></tr>
    <tr><td>Flicker F_t (↓)</td><td class="${!winF?'win':'lose'}">${b.avg_fl}</td><td class="${winF?'win':'lose'}">${a.avg_fl}</td></tr>
    <tr><td>Avg Ngân sách dùng</td><td>${sc.k}.00/${sc.k}</td><td class="win">${a.avg_used}/${sc.k}</td></tr>
  `;

  const byId = Object.fromEntries(fr.objects.map(o => [o.id, o]));
  const items = [];
  for (const id of a.missed) items.push(`<li class="li-miss"><span>✕ BỎ SÓT: ${byId[id]?.label || id}</span><span>NGUY HIỂM</span></li>`);
  for (const ent of a.entities) {
    let cls = ent.risk > 0.28 ? "li-haz" : "li-safe";
    let tag = `R=${ent.risk} · p=${ent.belief}`;
    if (ent.is_super) { cls = "li-super"; tag = `SIÊU NÚT · R=${ent.risk}`; }
    else if (ent.is_occ) { cls = "li-occ"; tag = `NHỚ KHUẤT · p=${ent.belief}`; }
    items.push(`<li class="${cls}"><span>${ent.label}</span><span>${tag}</span></li>`);
  }
  document.getElementById("entList").innerHTML = items.join("");

  drawScene("cLeft", fr, false);
  drawScene("cRight", fr, true);
}

function loop(now) {
  if (playing && now - lastTick >= 95) {
    idx = (idx + 1) % getSc().frames.length;
    render();
    lastTick = now;
  }
  requestAnimationFrame(loop);
}

resetSc();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def main() -> None:
    db = {sc: _record_scenario(sc, k=5, seed=100) for sc in SCENARIO_META}
    out = Path("results/aegis_core_sim.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML_CORE.replace("__DATA_JSON__", json.dumps(db, ensure_ascii=False)), encoding="utf-8")
    print(f"Aegis-Core v1.0 Simulator generated at: {out.resolve()}")


if __name__ == "__main__":
    main()
