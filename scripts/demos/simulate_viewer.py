"""Interactive HTML5 Side-by-Side Simulation + Live Metrics Comparison Dashboard.

Compares Distance Baseline vs. Proposed (Aegis-DSG) frame-by-frame across
all 4 3D environments and cognitive budgets k in {3, 5, 7}.
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path; _ROOT = str(_Path(__file__).resolve().parents[2]); (_sys.path.insert(0, _ROOT) if _ROOT not in _sys.path else None)

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.baselines import DistanceSparsifier
from src.dsg_generator import DSGFrame, ScenarioType, generate_dsg_sequence
from src.metrics import compute_hazard_recall, compute_jaccard_flicker
from src.proposed import ProposedSparsifier


VI_LABELS: Dict[str, str] = {
    "chair": "Ghế",
    "table": "Bàn",
    "low_barrier": "Rào chắn thấp",
    "stair_edge": "Mép cầu thang",
    "overhead_sign": "Biển treo ngang đầu",
    "door": "Cửa ra vào",
    "wall_pillar": "Cột tường",
    "pedestrian": "Người đi bộ",
    "wall_painting": "Tranh treo cao",
    "ceiling_light": "Đèn trần",
    "floor_mat": "Thảm sàn",
    "shelf": "Kệ tủ",
    "unknown_ghost": "Nhiễu ảo (FP)",
}

SCENARIO_META: Dict[ScenarioType, Dict[str, Any]] = {
    "corridor": {"title": "1. Hành lang thẳng", "width": 4.0, "length": 24.0},
    "l_turn_corner": {"title": "2. Ngã rẽ góc khuất 90°", "width": 16.0, "length": 13.0},
    "crowded_lobby": {"title": "3. Sảnh đông người cắt ngang", "width": 10.0, "length": 18.0},
    "vertical_hazards": {"title": "4. Vật cản phân tầng độ cao 3D", "width": 4.0, "length": 24.0},
}

K_OPTIONS = [3, 5, 7]


def _run_trace(seq: List[DSGFrame], k: int) -> Dict[str, List[Dict[str, Any]]]:
    """Run Distance and Proposed over a sequence and record per-frame selections and metrics."""
    dist_alg = DistanceSparsifier()
    prop_alg = ProposedSparsifier()
    dist_alg.reset()
    prop_alg.reset()

    out: Dict[str, List[Dict[str, Any]]] = {"distance": [], "proposed": []}
    prev_d: Set[str] = set()
    prev_p: Set[str] = set()
    sum_rec_d = sum_rec_p = 0.0
    sum_fl_d = sum_fl_p = 0.0

    for idx, f in enumerate(seq):
        obs_haz = {
            nid for nid, d in f.graph_obs.nodes(data=True)
            if d.get("layer") == "object" and d.get("is_hazard_gt", False)
        }
        sel_d = dist_alg.select(f.graph_obs, f.agent, k)
        sel_p = prop_alg.select(f.graph_obs, f.agent, k)

        rec_d = compute_hazard_recall(sel_d, obs_haz)
        rec_p = compute_hazard_recall(sel_p, obs_haz)
        fl_d = compute_jaccard_flicker(prev_d, sel_d) if idx > 0 else 0.0
        fl_p = compute_jaccard_flicker(prev_p, sel_p) if idx > 0 else 0.0

        sum_rec_d += rec_d
        sum_rec_p += rec_p
        sum_fl_d += fl_d
        sum_fl_p += fl_p
        n_fl = max(1, idx)

        out["distance"].append({
            "sel": sorted(sel_d),
            "missed": sorted(obs_haz - sel_d),
            "recall": round(rec_d * 100, 1),
            "avg_recall": round((sum_rec_d / (idx + 1)) * 100, 1),
            "avg_flicker": round(sum_fl_d / n_fl, 3),
        })
        out["proposed"].append({
            "sel": sorted(sel_p),
            "missed": sorted(obs_haz - sel_p),
            "recall": round(rec_p * 100, 1),
            "avg_recall": round((sum_rec_p / (idx + 1)) * 100, 1),
            "avg_flicker": round(sum_fl_p / n_fl, 3),
        })
        prev_d, prev_p = sel_d, sel_p

    return out


def serialize_scenario(seq: List[DSGFrame]) -> Dict[str, Any]:
    """Serialize frames + algorithm traces for k in {3, 5, 7}."""
    frames_out: List[Dict[str, Any]] = []
    for f in seq:
        obs_ids = {nid for nid, d in f.graph_obs.nodes(data=True) if d.get("layer") == "object"}
        all_ids = {nid for nid, d in f.graph_gt.nodes(data=True) if d.get("layer") == "object"} | obs_ids

        objects_list: List[Dict[str, Any]] = []
        for nid in sorted(all_ids):
            in_obs = nid in obs_ids
            data = f.graph_obs.nodes[nid] if in_obs else f.graph_gt.nodes[nid]
            objects_list.append({
                "id": nid,
                "label": VI_LABELS.get(data["label"], data["label"]),
                "pos": [round(float(x), 3) for x in data["pos"]],
                "size": [round(float(x), 3) for x in data["size"]],
                "vel": [round(float(x), 3) for x in data["vel"]],
                "conf": round(float(data.get("confidence", 1.0)), 2),
                "is_hazard": bool(data.get("is_hazard_gt", False)),
                "is_dynamic": bool(data.get("is_dynamic", False)),
                "in_obs": in_obs,
            })

        frames_out.append({
            "t": f.t,
            "time": f.timestamp,
            "agent": {
                "pos": [round(float(x), 3) for x in f.agent.pos],
                "vel": [round(float(x), 3) for x in f.agent.vel],
                "fov_rad": round(f.agent.fov_half_angle_rad, 4),
                "max_range": f.agent.max_range,
            },
            "objects": objects_list,
        })

    traces = {str(k): _run_trace(seq, k) for k in K_OPTIONS}
    return {"frames": frames_out, "traces": traces}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Aegis-DSG: Mô phỏng & Đối chiếu Số liệu Trực tiếp</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0f172a; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 18px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
  header h1 { margin: 0; font-size: 15px; color: #38bdf8; }
  .controls { display: flex; gap: 10px; align-items: center; font-size: 13px; flex-wrap: wrap; }
  button { background: #0284c7; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button:hover { background: #0369a1; }
  input[type=range] { width: 170px; accent-color: #38bdf8; }
  select { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 5px 8px; font-size: 13px; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr 340px; gap: 10px; padding: 10px; min-height: 0; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { padding: 8px 12px; font-size: 13px; font-weight: 700; background: #0f172a; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; }
  .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
  .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
  canvas { flex: 1; width: 100%; height: 100%; display: block; }
  .sidebar { padding: 12px; overflow-y: auto; font-size: 12.5px; line-height: 1.45; }
  .card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
  .card h3 { margin: 0 0 8px 0; font-size: 13px; color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 5px 6px; border-bottom: 1px solid #1e293b; text-align: center; }
  th { color: #94a3b8; font-weight: 600; }
  .win { color: #4ade80; font-weight: 700; }
  .lose { color: #f87171; font-weight: 700; }
  .item-list { list-style: none; padding: 0; margin: 4px 0 0 0; }
  .item-list li { padding: 3px 7px; margin-bottom: 3px; border-radius: 4px; font-size: 11.5px; display: flex; justify-content: space-between; }
  .li-kept-haz { background: rgba(34, 197, 94, 0.16); border-left: 3px solid #22c55e; }
  .li-kept-safe { background: rgba(148, 163, 184, 0.12); border-left: 3px solid #64748b; }
  .li-missed { background: rgba(239, 68, 68, 0.22); border-left: 3px solid #ef4444; color: #fecaca; font-weight: 600; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; font-size: 11px; color: #cbd5e1; margin-top: 4px; }
  .legend span { display: inline-flex; align-items: center; gap: 4px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
</style>
</head>
<body>
<header>
  <h1>Aegis-DSG — Mô phỏng Trực tiếp & Đối chiếu Số liệu (Distance vs. Proposed)</h1>
  <div class="controls">
    <label>Môi trường:
      <select id="scenarioSelect">
        <option value="corridor">1. Hành lang thẳng</option>
        <option value="l_turn_corner">2. Ngã rẽ góc khuất 90°</option>
        <option value="crowded_lobby">3. Sảnh đông người cắt ngang</option>
        <option value="vertical_hazards">4. Vật cản phân tầng độ cao 3D</option>
      </select>
    </label>
    <label>Ngân sách k:
      <select id="kSelect">
        <option value="3">k = 3 vật</option>
        <option value="5" selected>k = 5 vật</option>
        <option value="7">k = 7 vật</option>
      </select>
    </label>
    <button id="playBtn">Tạm dừng</button>
    <input type="range" id="frameSlider" min="0" max="0" value="0" />
    <span id="timeLabel">Frame 0 | 0.0s</span>
    <select id="speedSelect">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
    </select>
  </div>
</header>

<main>
  <div class="panel">
    <div class="panel-title">
      <span>A. Distance Baseline (Chọn k vật gần nhất)</span>
      <span id="distBadge" class="badge badge-blue">Recall: 0%</span>
    </div>
    <canvas id="canvasDist"></canvas>
  </div>

  <div class="panel">
    <div class="panel-title">
      <span>B. Proposed — Aegis-DSG (Phễu động + Bất đối xứng)</span>
      <span id="propBadge" class="badge badge-green">Recall: 0%</span>
    </div>
    <canvas id="canvasProp"></canvas>
  </div>

  <div class="panel sidebar">
    <div class="card">
      <h3>Đối chiếu số liệu tích lũy (đến frame hiện tại)</h3>
      <table>
        <thead>
          <tr><th>Chỉ số</th><th>Distance</th><th>Proposed</th></tr>
        </thead>
        <tbody id="liveMetricsBody"></tbody>
      </table>
      <div class="legend" style="margin-top:8px;">
        <span><i class="dot" style="background:#f59e0b;"></i>Người dùng</span>
        <span><i class="dot" style="border:2px solid #38bdf8; background:transparent;"></i>Được chọn (Top-k)</span>
        <span><i class="dot" style="background:#ef4444;"></i>Vật nguy hiểm</span>
        <span><b style="color:#ef4444;">✕</b> Bị bỏ sót!</span>
      </div>
    </div>

    <div class="card">
      <h3>Proposed giữ lại (Top-k) & Bỏ sót ở frame này</h3>
      <ul id="propList" class="item-list"></ul>
    </div>

    <div class="card">
      <h3>Distance giữ lại (Top-k) & Bỏ sót ở frame này</h3>
      <ul id="distList" class="item-list"></ul>
    </div>
  </div>
</main>

<script>
const SCENARIOS = __DATA_JSON__;
let currentKey = "corridor";
let currentK = "5";
let idx = 0;
let playing = true;
let speed = 1.0;
let lastTick = performance.now();

const scenarioSelect = document.getElementById("scenarioSelect");
const kSelect = document.getElementById("kSelect");
const slider = document.getElementById("frameSlider");
const playBtn = document.getElementById("playBtn");
const timeLabel = document.getElementById("timeLabel");
const speedSelect = document.getElementById("speedSelect");

function getFrames() { return SCENARIOS[currentKey].frames; }
function getMeta() { return SCENARIOS[currentKey].meta; }
function getTrace(alg) { return SCENARIOS[currentKey].traces[currentK][alg]; }

function resetView() {
  idx = 0;
  slider.max = getFrames().length - 1;
  render();
}

scenarioSelect.onchange = (e) => { currentKey = e.target.value; resetView(); };
kSelect.onchange = (e) => { currentK = e.target.value; render(); };
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Tạm dừng" : "Chạy tiếp"; };
slider.oninput = (e) => { idx = parseInt(e.target.value, 10); render(); };
speedSelect.onchange = (e) => { speed = parseFloat(e.target.value); };

function resizeCanvas(c) {
  const rect = c.getBoundingClientRect();
  if (c.width !== rect.width || c.height !== rect.height) {
    c.width = rect.width;
    c.height = rect.height;
  }
}

function drawMap(canvasId, frame, traceStep, showCollisionCone) {
  const c = document.getElementById(canvasId);
  resizeCanvas(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const ax = frame.agent.pos[0], ay = frame.agent.pos[1];
  const scale = Math.min(c.width / 15.0, c.height / 15.0);
  const toScreen = (x, y) => [
    c.width * 0.5 + (x - ax) * scale,
    c.height * 0.68 - (y - ay) * scale
  ];

  // Walls
  const meta = getMeta();
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 3;
  ctx.beginPath();
  if (currentKey === "l_turn_corner") {
    [[0,0],[0,13],[16,13]].forEach((p,i)=>{ const [sx,sy]=toScreen(p[0],p[1]); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); });
    [[4,0],[4,9],[16,9]].forEach((p,i)=>{ const [sx,sy]=toScreen(p[0],p[1]); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); });
  } else {
    const [x1,y1]=toScreen(0,0), [x2,y2]=toScreen(0,meta.length);
    const [x3,y3]=toScreen(meta.width,0), [x4,y4]=toScreen(meta.width,meta.length);
    ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
    ctx.moveTo(x3,y3); ctx.lineTo(x4,y4);
  }
  ctx.stroke();

  // FOV wedge
  const [sx, sy] = toScreen(ax, ay);
  const heading = Math.atan2(frame.agent.vel[1], frame.agent.vel[0]);
  const fov = frame.agent.fov_rad;
  ctx.fillStyle = "rgba(250, 204, 21, 0.08)";
  ctx.strokeStyle = "rgba(250, 204, 21, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.arc(sx, sy, frame.agent.max_range * scale, -heading - fov, -heading + fov);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Dynamic Collision Cone (only on Proposed panel)
  if (showCollisionCone) {
    const coneHalf = 35 * Math.PI / 180;
    const coneLen = Math.hypot(frame.agent.vel[0], frame.agent.vel[1]) * 3.2 + 2.5;
    ctx.fillStyle = "rgba(56, 189, 248, 0.14)";
    ctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.arc(sx, sy, coneLen * scale, -heading - coneHalf, -heading + coneHalf);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const selSet = new Set(traceStep.sel);
  const missSet = new Set(traceStep.missed);

  for (const obj of frame.objects) {
    const [ox, oy] = toScreen(obj.pos[0], obj.pos[1]);
    if (ox < -40 || ox > c.width + 40 || oy < -40 || oy > c.height + 40) continue;

    const isSel = selSet.has(obj.id);
    const isMiss = missSet.has(obj.id);
    const w = Math.max(7, obj.size[0] * scale);
    const h = Math.max(7, obj.size[1] * scale);

    let color = "#475569";
    if (obj.in_obs) color = obj.is_hazard ? "#ef4444" : "#22c55e";

    ctx.globalAlpha = isSel || isMiss ? 1.0 : (obj.in_obs ? 0.35 : 0.14);
    ctx.fillStyle = color;
    ctx.fillRect(ox - w / 2, oy - h / 2, w, h);

    // Highlight ring for selected top-k items
    if (isSel) {
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(ox, oy, Math.max(w, h) * 0.75 + 4, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Big red X for MISSED hazards
    if (isMiss) {
      ctx.strokeStyle = "#ff2e2e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(ox - 8, oy - 8); ctx.lineTo(ox + 8, oy + 8);
      ctx.moveTo(ox + 8, oy - 8); ctx.lineTo(ox - 8, oy + 8);
      ctx.stroke();
    }

    if (obj.is_dynamic && obj.in_obs) {
      ctx.strokeStyle = "#fca5a5";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + obj.vel[0] * scale * 0.75, oy - obj.vel[1] * scale * 0.75);
      ctx.stroke();
    }

    if (isSel || isMiss) {
      ctx.fillStyle = isMiss ? "#fca5a5" : "#f8fafc";
      ctx.font = isMiss ? "bold 11px system-ui" : "10.5px system-ui";
      ctx.fillText(isMiss ? `BỎ SÓT: ${obj.label}` : obj.label, ox + w / 2 + 6, oy + 3);
    }
    ctx.globalAlpha = 1.0;
  }

  // Agent
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(sx, sy, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#fef3c7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(sx + frame.agent.vel[0] * scale * 0.9, sy - frame.agent.vel[1] * scale * 0.9);
  ctx.stroke();
}

function renderList(elId, frame, traceStep) {
  const byId = Object.fromEntries(frame.objects.map(o => [o.id, o]));
  const items = [];
  for (const id of traceStep.missed) {
    const o = byId[id];
    items.push(`<li class="li-missed"><span>✕ BỎ SÓT: ${o ? o.label : id}</span><span>NGUY HIỂM</span></li>`);
  }
  for (const id of traceStep.sel) {
    const o = byId[id];
    const cls = o && o.is_hazard ? "li-kept-haz" : "li-kept-safe";
    const tag = o && o.is_hazard ? "Cứu được" : "Định hướng/Khác";
    items.push(`<li class="${cls}"><span>✓ ${o ? o.label : id}</span><span>${tag}</span></li>`);
  }
  document.getElementById(elId).innerHTML = items.join("");
}

function render() {
  const frames = getFrames();
  const frame = frames[idx];
  const dStep = getTrace("distance")[idx];
  const pStep = getTrace("proposed")[idx];

  slider.value = idx;
  timeLabel.textContent = `Frame ${frame.t}/${frames.length - 1} | ${frame.time.toFixed(1)}s`;

  const distBadge = document.getElementById("distBadge");
  const propBadge = document.getElementById("propBadge");
  distBadge.textContent = `Frame Recall: ${dStep.recall}%`;
  distBadge.className = "badge " + (dStep.recall >= 99 ? "badge-green" : "badge-red");
  propBadge.textContent = `Frame Recall: ${pStep.recall}%`;
  propBadge.className = "badge " + (pStep.recall >= 99 ? "badge-green" : "badge-red");

  const recWinP = pStep.avg_recall >= dStep.avg_recall;
  const flWinP = pStep.avg_flicker <= dStep.avg_flicker;
  document.getElementById("liveMetricsBody").innerHTML = `
    <tr>
      <td>Recall TB (↑)</td>
      <td class="${!recWinP ? 'win' : 'lose'}">${dStep.avg_recall}%</td>
      <td class="${recWinP ? 'win' : 'lose'}">${pStep.avg_recall}%</td>
    </tr>
    <tr>
      <td>Flicker F_t (↓)</td>
      <td class="${!flWinP ? 'win' : 'lose'}">${dStep.avg_flicker}</td>
      <td class="${flWinP ? 'win' : 'lose'}">${pStep.avg_flicker}</td>
    </tr>
    <tr>
      <td>Sót ở frame này</td>
      <td class="${dStep.missed.length === 0 ? 'win' : 'lose'}">${dStep.missed.length} vật</td>
      <td class="${pStep.missed.length === 0 ? 'win' : 'lose'}">${pStep.missed.length} vật</td>
    </tr>
  `;

  renderList("propList", frame, pStep);
  renderList("distList", frame, dStep);

  drawMap("canvasDist", frame, dStep, false);
  drawMap("canvasProp", frame, pStep, true);
}

function loop(now) {
  if (playing && now - lastTick >= 110 / speed) {
    idx = (idx + 1) % getFrames().length;
    render();
    lastTick = now;
  }
  requestAnimationFrame(loop);
}

resetView();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def build_html_simulation(out_path: Path) -> None:
    """Generate all 4 scenarios with live Distance vs Proposed traces into simulation.html."""
    payload: Dict[str, Any] = {}
    for sc, meta in SCENARIO_META.items():
        seq = generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=42, scenario=sc)
        sc_data = serialize_scenario(seq)
        sc_data["meta"] = meta
        payload[sc] = sc_data

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Side-by-side comparison dashboard generated at: {out_path.resolve()}")


if __name__ == "__main__":
    build_html_simulation(Path("results/simulation.html"))
