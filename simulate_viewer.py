"""Interactive HTML5 3D + Top-Down Simulation Viewer for the DSG sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.dsg_generator import DSGFrame, generate_dsg_sequence


VI_LABELS: Dict[str, str] = {
    "chair": "Ghế",
    "table": "Bàn",
    "low_barrier": "Rào chắn thấp",
    "stair_edge": "Mép cầu thang",
    "door": "Cửa ra vào",
    "wall_pillar": "Cột tường",
    "pedestrian": "Người đi bộ",
    "wall_painting": "Tranh treo tường",
    "ceiling_light": "Đèn trần",
    "shelf": "Kệ tủ",
    "unknown_ghost": "Nhiễu ảo (báo giả)",
}


def serialize_sequence(seq: List[DSGFrame]) -> List[Dict[str, Any]]:
    """Convert DSGFrame sequence into JSON-friendly dicts for the HTML5 viewer."""
    frames_out: List[Dict[str, Any]] = []
    for f in seq:
        obs_ids = {
            nid for nid, d in f.graph_obs.nodes(data=True) if d.get("layer") == "object"
        }
        objects_list: List[Dict[str, Any]] = []
        # Include all GT objects + any false-positive ghosts in obs
        all_ids = set(
            nid for nid, d in f.graph_gt.nodes(data=True) if d.get("layer") == "object"
        ) | obs_ids

        for nid in sorted(all_ids):
            in_obs = nid in obs_ids
            data = f.graph_obs.nodes[nid] if in_obs else f.graph_gt.nodes[nid]
            objects_list.append(
                {
                    "id": nid,
                    "label": VI_LABELS.get(data["label"], data["label"]),
                    "raw_label": data["label"],
                    "pos": [round(float(x), 3) for x in data["pos"]],
                    "size": [round(float(x), 3) for x in data["size"]],
                    "vel": [round(float(x), 3) for x in data["vel"]],
                    "conf": round(float(data.get("confidence", 1.0)), 2),
                    "is_hazard": bool(data.get("is_hazard_gt", False)),
                    "is_dynamic": bool(data.get("is_dynamic", False)),
                    "in_obs": in_obs,
                }
            )

        frames_out.append(
            {
                "t": f.t,
                "time": f.timestamp,
                "agent": {
                    "pos": [round(float(x), 3) for x in f.agent.pos],
                    "vel": [round(float(x), 3) for x in f.agent.vel],
                    "fov_rad": round(f.agent.fov_half_angle_rad, 4),
                    "max_range": f.agent.max_range,
                },
                "objects": objects_list,
            }
        )
    return frames_out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Mô phỏng Không gian 3D - Dynamic Scene Graph</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0f172a; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 12px 20px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 16px; color: #38bdf8; }
  .controls { display: flex; gap: 12px; align-items: center; font-size: 13px; }
  button { background: #0284c7; color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button:hover { background: #0369a1; }
  input[type=range] { width: 220px; accent-color: #38bdf8; }
  select { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 4px 8px; }
  main { flex: 1; display: grid; grid-template-columns: 1.1fr 1fr 300px; gap: 10px; padding: 10px; min-height: 0; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { padding: 8px 12px; font-size: 13px; font-weight: 600; background: #0f172a; border-bottom: 1px solid #334155; color: #cbd5e1; }
  canvas { flex: 1; width: 100%; height: 100%; display: block; }
  .sidebar { padding: 12px; overflow-y: auto; font-size: 13px; line-height: 1.5; }
  .stat-box { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
  .legend-item { display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 12px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .hazard-list { list-style: none; padding: 0; margin: 6px 0 0 0; }
  .hazard-list li { background: rgba(239, 68, 68, 0.15); border-left: 3px solid #ef4444; padding: 4px 8px; margin-bottom: 4px; border-radius: 3px; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Mô phỏng Chuỗi Không gian 3D (Dynamic 3D Scene Graph)</h1>
  <div class="controls">
    <button id="playBtn">Tạm dừng</button>
    <input type="range" id="frameSlider" min="0" max="0" value="0" />
    <span id="timeLabel">Frame 0 | 0.0s</span>
    <label>Tốc độ:
      <select id="speedSelect">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
      </select>
    </label>
  </div>
</header>

<main>
  <div class="panel">
    <div class="panel-title">1. Góc nhìn 3D (Thấy độ cao: đèn trần, tranh, bàn ghế, người)</div>
    <canvas id="canvas3d"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">2. Bản đồ Radar từ trên xuống (Tầm nhìn của kính & Vật cản)</div>
    <canvas id="canvas2d"></canvas>
  </div>
  <div class="panel sidebar">
    <div class="stat-box">
      <strong>Trạng thái hiện tại</strong>
      <div id="statsText" style="margin-top:6px; color:#94a3b8;"></div>
    </div>
    <div class="stat-box">
      <strong>Chú thích màu sắc</strong>
      <div class="legend-item"><span class="dot" style="background:#f59e0b;"></span> Người dùng (Đang đi tới)</div>
      <div class="legend-item"><span class="dot" style="background:#ef4444;"></span> Vật nguy hiểm (Sắp đâm vào)</div>
      <div class="legend-item"><span class="dot" style="background:#22c55e;"></span> Đồ vô hại (Tranh, đèn, vật ở xa)</div>
      <div class="legend-item"><span class="dot" style="background:#64748b;"></span> Ngoài tầm nhìn / Bị che khuất</div>
      <div class="legend-item"><span class="dot" style="background:rgba(250,204,21,0.35); border:1px dashed #eab308;"></span> Vùng quét của kính (FOV)</div>
    </div>
    <div class="stat-box">
      <strong>Cảnh báo nguy hiểm trước mặt:</strong>
      <ul id="hazardList" class="hazard-list"></ul>
    </div>
  </div>
</main>

<script>
const DATA = __DATA_JSON__;
let idx = 0;
let playing = true;
let speed = 1.0;
let lastTick = performance.now();

const slider = document.getElementById("frameSlider");
const playBtn = document.getElementById("playBtn");
const timeLabel = document.getElementById("timeLabel");
const speedSelect = document.getElementById("speedSelect");
const statsText = document.getElementById("statsText");
const hazardList = document.getElementById("hazardList");

slider.max = DATA.length - 1;
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

function draw2D(frame) {
  const c = document.getElementById("canvas2d");
  resizeCanvas(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const scale = Math.min(c.width / 6.0, c.height / 15.0);
  const ax = frame.agent.pos[0], ay = frame.agent.pos[1];
  const toScreen = (x, y) => [
    c.width / 2 + (x - 2.0) * scale,
    c.height * 0.82 - (y - ay) * scale
  ];

  // Corridor walls
  const [wl] = toScreen(0, ay);
  const [wr] = toScreen(4, ay);
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(wl, 0); ctx.lineTo(wl, c.height);
  ctx.moveTo(wr, 0); ctx.lineTo(wr, c.height);
  ctx.stroke();

  // FOV wedge
  const [sx, sy] = toScreen(ax, ay);
  const heading = Math.atan2(frame.agent.vel[1], frame.agent.vel[0]);
  const fov = frame.agent.fov_rad;
  ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
  ctx.strokeStyle = "rgba(250, 204, 21, 0.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.arc(sx, sy, frame.agent.max_range * scale, -heading - fov, -heading + fov);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Objects
  for (const obj of frame.objects) {
    const [ox, oy] = toScreen(obj.pos[0], obj.pos[1]);
    if (oy < -40 || oy > c.height + 40) continue;
    const w = Math.max(8, obj.size[0] * scale);
    const h = Math.max(8, obj.size[1] * scale);

    let color = "#64748b";
    if (obj.in_obs) color = obj.is_hazard ? "#ef4444" : "#22c55e";

    ctx.globalAlpha = obj.in_obs ? 0.9 : 0.35;
    ctx.fillStyle = color;
    ctx.fillRect(ox - w / 2, oy - h / 2, w, h);

    if (obj.is_dynamic && obj.in_obs) {
      ctx.strokeStyle = "#fca5a5";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + obj.vel[0] * scale * 0.8, oy - obj.vel[1] * scale * 0.8);
      ctx.stroke();
    }

    if (obj.in_obs) {
      ctx.fillStyle = "#f8fafc";
      ctx.font = "11px system-ui";
      ctx.fillText(`${obj.label} (${Math.round(obj.conf * 100)}%)`, ox + w / 2 + 4, oy + 3);
    }
    ctx.globalAlpha = 1.0;
  }

  // Agent
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(sx, sy, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#fef3c7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(sx + frame.agent.vel[0] * scale * 0.9, sy - frame.agent.vel[1] * scale * 0.9);
  ctx.stroke();
}

function draw3D(frame) {
  const c = document.getElementById("canvas3d");
  resizeCanvas(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const ax = frame.agent.pos[0], ay = frame.agent.pos[1];
  // Simple isometric projection relative to agent Y
  const proj = (x, y, z) => {
    const rx = x - 2.0;
    const ry = y - ay;
    const rz = z;
    const sx = c.width * 0.48 + (rx * 48) + (ry * 22);
    const sy = c.height * 0.78 - (ry * 20) - (rz * 52);
    return [sx, sy];
  };

  // Floor grid lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  for (let dy = -2; dy <= 12; dy += 2) {
    const y = Math.floor(ay / 2) * 2 + dy;
    const [x1, y1] = proj(0, y, 0);
    const [x2, y2] = proj(4, y, 0);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  const [fl1, fy1] = proj(0, ay - 2, 0), [fl2, fy2] = proj(0, ay + 12, 0);
  const [fr1, ry1] = proj(4, ay - 2, 0), [fr2, ry2] = proj(4, ay + 12, 0);
  ctx.strokeStyle = "#64748b";
  ctx.beginPath(); ctx.moveTo(fl1, fy1); ctx.lineTo(fl2, fy2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(fr1, ry1); ctx.lineTo(fr2, ry2); ctx.stroke();

  // Sort back-to-front by Y
  const sorted = [...frame.objects].sort((a, b) => b.pos[1] - a.pos[1]);
  for (const obj of sorted) {
    const ry = obj.pos[1] - ay;
    if (ry < -2 || ry > 13) continue;
    const [bx, by] = proj(obj.pos[0], obj.pos[1], obj.pos[2] - obj.size[2] / 2);
    const [tx, ty] = proj(obj.pos[0], obj.pos[1], obj.pos[2] + obj.size[2] / 2);
    const [gx, gy] = proj(obj.pos[0], obj.pos[1], 0);

    let color = "#64748b";
    if (obj.in_obs) color = obj.is_hazard ? "#ef4444" : "#22c55e";

    ctx.globalAlpha = obj.in_obs ? 0.9 : 0.25;
    // Drop shadow to floor
    ctx.strokeStyle = "#475569";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);

    // Vertical pillar representing 3D height
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(6, obj.size[0] * 18);
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();

    if (obj.in_obs) {
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px system-ui";
      ctx.fillText(obj.label, tx + 8, ty);
    }
    ctx.globalAlpha = 1.0;
  }

  // Agent 3D
  const [abx, aby] = proj(ax, ay, 0);
  const [atx, aty] = proj(ax, ay, 1.7);
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 10;
  ctx.beginPath(); ctx.moveTo(abx, aby); ctx.lineTo(atx, aty); ctx.stroke();
  ctx.fillStyle = "#fbbf24";
  ctx.font = "bold 12px system-ui";
  ctx.fillText("NGƯỜI DÙNG", atx - 32, aty - 8);
}

function render() {
  const frame = DATA[idx];
  slider.value = idx;
  timeLabel.textContent = `Frame ${frame.t}/${DATA.length - 1} | ${frame.time.toFixed(1)}s`;

  const obsObjs = frame.objects.filter(o => o.in_obs);
  const hazards = obsObjs.filter(o => o.is_hazard);
  statsText.innerHTML = `
    • Tổng đồ vật quanh hành lang: <b>${frame.objects.length}</b><br/>
    • Kính đang nhìn thấy: <b>${obsObjs.length}</b> thứ<br/>
    • Số vật nguy hiểm: <b style="color:#ef4444;">${hazards.length}</b> thứ
  `;

  hazardList.innerHTML = hazards.length === 0
    ? "<li style='border-color:#22c55e; background:rgba(34,197,94,0.12);'>Đường đang thoáng, không có vật cản nguy hiểm!</li>"
    : hazards.map(h => {
        const dist = Math.hypot(h.pos[0] - frame.agent.pos[0], h.pos[1] - frame.agent.pos[1]).toFixed(1);
        return `<li><b>${h.label}</b> — cách ${dist}m (độ rõ ${Math.round(h.conf * 100)}%)</li>`;
      }).join("");

  draw3D(frame);
  draw2D(frame);
}

function loop(now) {
  if (playing && now - lastTick >= 100 / speed) {
    idx = (idx + 1) % DATA.length;
    render();
    lastTick = now;
  }
  requestAnimationFrame(loop);
}

render();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def build_html_simulation(out_path: Path) -> None:
    """Generate a DSG sequence and write a self-contained interactive HTML5 viewer."""
    seq = generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=42)
    payload = json.dumps(serialize_sequence(seq), ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Interactive simulation generated at: {out_path.resolve()}")


if __name__ == "__main__":
    build_html_simulation(Path("results/simulation.html"))
