import sys as _sys; from pathlib import Path as _Path; _ROOT = str(_Path(__file__).resolve().parents[2]); (_sys.path.insert(0, _ROOT) if _ROOT not in _sys.path else None)
"""Generate a standalone, 60 FPS Closed-Loop Infinite Procedural World HTML5 Demo for Aegis-DSG.

Features:
- True Closed-Loop Reactive Navigation (mini-DWA) for BOTH agents independently:
  * Agent A (Distance Baseline) steers/brakes using ONLY Distance's top-k output.
  * Agent B (Proposed Aegis-DSG) steers/brakes using ONLY Proposed's top-k output.
- Handles multi-sided encirclement (too many hazards on all sides):
  * Evaluates forward steering arcs; if all forward arcs are blocked -> Emergency Brake / Hold
    (waits for moving crowds to pass) + repulsive escape steering if boxed in by static obstacles.
- Counts REAL physical collisions against Ground-Truth 3D objects (showing why missed hazards hurt!).
- 60 FPS smooth physics & rendering, 10 Hz sensor/algorithm updates, click-to-spawn oncoming pedestrians.
"""

from pathlib import Path


HTML_INFINITE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Aegis-DSG: Mô phỏng Đóng vòng (Closed-Loop Né Vật Cản + Bản đồ Vô tận)</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { padding: 9px 16px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  header h1 { margin: 0; font-size: 14.5px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
  .pill { background: rgba(34,197,94,0.2); color: #4ade80; font-size: 10.5px; padding: 2px 8px; border-radius: 10px; font-weight: 700; }
  .controls { display: flex; gap: 7px; align-items: center; font-size: 12px; flex-wrap: wrap; }
  button { background: #0284c7; color: white; border: none; padding: 5px 10px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; }
  button:hover { background: #0369a1; }
  button.sec { background: #334155; }
  button.sec:hover { background: #475569; }
  select { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 4px 7px; font-size: 12px; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr 335px; gap: 10px; padding: 10px; min-height: 0; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { padding: 7px 11px; font-size: 12.5px; font-weight: 700; background: #0f172a; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .b-green { background: rgba(34,197,94,0.2); color: #4ade80; }
  .b-red { background: rgba(239,68,68,0.25); color: #f87171; }
  .b-amber { background: rgba(245,158,11,0.25); color: #fbbf24; }
  canvas { flex: 1; width: 100%; height: 100%; display: block; cursor: crosshair; }
  .sidebar { padding: 10px; overflow-y: auto; font-size: 12px; display: flex; flex-direction: column; gap: 8px; }
  .card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 9px; }
  .card h3 { margin: 0 0 6px 0; font-size: 12.5px; color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; font-size: 11.8px; }
  th, td { padding: 4px 4px; border-bottom: 1px solid #1e293b; text-align: center; }
  th { color: #94a3b8; }
  .win { color: #4ade80; font-weight: 700; }
  .lose { color: #f87171; font-weight: 700; }
  .hint { background: rgba(56,189,248,0.12); border-left: 3px solid #38bdf8; padding: 6px 8px; border-radius: 4px; font-size: 11.3px; color: #bae6fd; line-height: 1.4; }
  #sparkCanvas { width: 100%; height: 96px; display: block; background: #0b1120; border-radius: 4px; cursor: default; }
  .item-list { list-style: none; padding: 0; margin: 0; max-height: 135px; overflow-y: auto; }
  .item-list li { padding: 3px 6px; margin-bottom: 3px; border-radius: 4px; font-size: 11px; display: flex; justify-content: space-between; }
  .li-ok { background: rgba(34,197,94,0.15); border-left: 3px solid #22c55e; }
  .li-safe { background: rgba(148,163,184,0.12); border-left: 3px solid #64748b; }
  .li-miss { background: rgba(239,68,68,0.22); border-left: 3px solid #ef4444; color: #fecaca; font-weight: 700; }
</style>
</head>
<body>
<header>
  <h1>Aegis-DSG Closed-Loop Infinite <span class="pill">TỰ NÉ VẬT CẢN · 60 FPS</span></h1>
  <div class="controls">
    <select id="modeSelect">
      <option value="corridor">Hành lang vô tận (4m)</option>
      <option value="lobby">Sảnh đông cắt ngang (9m)</option>
      <option value="vertical">Phân tầng 3D độ cao (4m)</option>
    </select>
    <select id="kSelect">
      <option value="3">k = 3 vật</option>
      <option value="5" selected>k = 5 vật</option>
      <option value="7">k = 7 vật</option>
    </select>
    <select id="noiseSelect">
      <option value="clean">Nhiễu: Sạch</option>
      <option value="medium" selected>Nhiễu: Trung bình</option>
      <option value="heavy">Nhiễu: Nặng</option>
    </select>
    <select id="speedSelect">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="1.5">1.5x</option>
    </select>
    <button id="playBtn">Tạm dừng</button>
    <button id="resetBtn" class="sec">Bản đồ mới</button>
  </div>
</header>

<main>
  <div class="panel">
    <div class="panel-title">
      <span>A. Agent dùng Distance (Chỉ né k vật gần nhất)</span>
      <span id="distStatus" class="badge b-green">ĐI THẲNG</span>
    </div>
    <canvas id="canvasDist"></canvas>
  </div>

  <div class="panel">
    <div class="panel-title">
      <span>B. Agent dùng Proposed Aegis-DSG (Né theo phễu 3D)</span>
      <span id="propStatus" class="badge b-green">ĐI THẲNG</span>
    </div>
    <canvas id="canvasProp"></canvas>
  </div>

  <div class="panel sidebar">
    <div class="hint">
      <b>Cơ chế né & bao vây (Mini-DWA):</b><br/>
      • Mỗi Agent <b>chỉ thấy đúng k vật</b> thuật toán cấp cho.<br/>
      • Gặp vật cản: tự bẻ lái lách trái/phải.<br/>
      • <b>Bị bao vây nhiều bên:</b> tự <b>PHANH GẤP</b> đợi đám đông qua, hoặc lùi nhẹ tìm khe thoát.<br/>
      • <b>Click lên bản đồ</b> để thả thêm người lao tới test!
    </div>

    <div class="card">
      <h3>Đối chiếu Đóng vòng (<span id="frameCount">0</span> frames)</h3>
      <table>
        <thead><tr><th>Chỉ số</th><th>Distance</th><th>Proposed</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>

    <div class="card">
      <h3>Biểu đồ Hazard Recall (%) hội tụ theo thời gian</h3>
      <canvas id="sparkCanvas"></canvas>
      <div style="display:flex; justify-content:space-between; font-size:10.5px; margin-top:4px;">
        <span style="color:#4ade80;">━ Proposed (Aegis-DSG)</span>
        <span style="color:#38bdf8;">━ Distance Baseline</span>
      </div>
    </div>

    <div class="card">
      <h3>Proposed đang nhìn thấy (Top-k) & Bỏ sót</h3>
      <ul id="propList" class="item-list"></ul>
    </div>
  </div>
</main>

<script>
const NOISE_CFG = {
  clean:  { posStd: 0.02, dropout: 0.01, fp: 0.00, confNoise: 0.03 },
  medium: { posStd: 0.08, dropout: 0.08, fp: 0.04, confNoise: 0.12 },
  heavy:  { posStd: 0.18, dropout: 0.18, fp: 0.10, confNoise: 0.22 },
};

const SPECS = {
  chair:         { name: "Ghế",            size: [0.55, 0.55, 0.85], z: 0.425, cv: 0.15, haz: 0.75, dyn: false },
  table:         { name: "Bàn",            size: [1.05, 0.65, 0.75], z: 0.375, cv: 0.30, haz: 0.80, dyn: false },
  low_barrier:   { name: "Rào chắn",       size: [0.90, 0.22, 1.15], z: 0.575, cv: 0.20, haz: 0.95, dyn: false },
  stair_edge:    { name: "Mép cầu thang",  size: [1.50, 0.38, 0.25], z: 0.125, cv: 0.90, haz: 1.00, dyn: false },
  overhead_sign: { name: "Biển ngang đầu", size: [1.10, 0.16, 0.40], z: 1.550, cv: 0.35, haz: 0.95, dyn: false },
  door:          { name: "Cửa ra vào",     size: [0.22, 1.00, 2.10], z: 1.050, cv: 0.95, haz: 0.30, dyn: false },
  pedestrian:    { name: "Người đi bộ",    size: [0.48, 0.44, 1.72], z: 0.860, cv: 0.10, haz: 0.95, dyn: true  },
  wall_painting: { name: "Tranh (z=2.1m)", size: [0.08, 0.90, 0.65], z: 2.150, cv: 0.05, haz: 0.00, dyn: false },
  ceiling_light: { name: "Đèn trần",       size: [0.50, 0.50, 0.12], z: 2.750, cv: 0.05, haz: 0.00, dyn: false },
  floor_mat:     { name: "Thảm sàn",       size: [1.20, 0.80, 0.02], z: 0.010, cv: 0.05, haz: 0.00, dyn: false },
};

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function pickWeighted(items, probs) {
  let r = Math.random(), acc = 0;
  for (let i = 0; i < items.length; i++) { acc += probs[i]; if (r <= acc) return items[i]; }
  return items[items.length - 1];
}

function makeAgent(width) {
  return {
    x: width / 2, y: 0.5, z: 0.85,
    vx: 0, vy: 1.35, targetVx: 0, targetVy: 1.35,
    radius: 0.32, height: 1.70, fov: 72 * Math.PI / 180, maxRange: 12.0,
    status: "ĐI THẲNG", statusKind: "b-green",
    brakeTime: 0, escapeTimer: 0, escapeTargetX: width / 2, hitCooldown: 0, hitFlash: 0,
    collisions: 0, hitSet: new Set(),
    prevSel: new Set(), persistence: {},
    obs: [], sel: new Set(), miss: new Set(),
    sumRec: 0, sumFl: 0, frames: 0,
    trail: [[width / 2, 0.5]],
  };
}

let state = {};

function initWorld() {
  const mode = document.getElementById("modeSelect").value;
  const width = mode === "lobby" ? 9.0 : 4.2;
  state = {
    mode, width, t: 0,
    agentD: makeAgent(width),
    agentP: makeAgent(width),
    objects: [],
    nextId: 1,
    generatedUpToY: 3.0,
    historyD: [],
    historyP: [],
  };
  ensureChunksAhead();
}

function spawnChunk(yStart, yEnd) {
  const { mode, width } = state;
  const count = mode === "lobby" ? 16 : 10;
  let pool, probs;
  if (mode === "lobby") {
    pool = ["pedestrian", "chair", "table", "low_barrier", "ceiling_light", "floor_mat"];
    probs = [0.36, 0.22, 0.14, 0.08, 0.10, 0.10];
  } else if (mode === "vertical") {
    pool = ["overhead_sign", "stair_edge", "low_barrier", "pedestrian", "chair", "ceiling_light", "wall_painting", "floor_mat"];
    probs = [0.15, 0.08, 0.08, 0.12, 0.09, 0.18, 0.15, 0.15];
  } else {
    pool = ["chair", "table", "low_barrier", "stair_edge", "pedestrian", "door", "wall_painting", "ceiling_light"];
    probs = [0.18, 0.12, 0.10, 0.06, 0.18, 0.08, 0.16, 0.12];
  }
  for (let i = 0; i < count; i++) {
    const type = pickWeighted(pool, probs);
    const sp = SPECS[type];
    const y = yStart + Math.random() * (yEnd - yStart);
    const x = (type === "wall_painting" || type === "door")
      ? (Math.random() < 0.5 ? 0.15 : width - 0.15)
      : (0.55 + Math.random() * (width - 1.1));
    let vx = 0, vy = 0;
    if (sp.dyn) {
      if (mode === "lobby" && Math.random() < 0.65) {
        vx = (Math.random() < 0.5 ? 1 : -1) * (0.75 + Math.random() * 0.65);
        vy = -0.3 + Math.random() * 0.35;
      } else {
        vx = (Math.random() - 0.5) * 0.35;
        vy = -(0.75 + Math.random() * 0.7);
      }
    }
    state.objects.push({
      id: `${type}_${state.nextId++}`, type, name: sp.name,
      x, y, z: sp.z, sx: sp.size[0], sy: sp.size[1], sz: sp.size[2],
      vx, vy, cv: sp.cv, haz: sp.haz, dyn: sp.dyn, trail: [[x, y]],
    });
  }
}

function ensureChunksAhead() {
  const maxY = Math.max(state.agentD.y, state.agentP.y);
  const minY = Math.min(state.agentD.y, state.agentP.y);
  while (state.generatedUpToY < maxY + 22.0) {
    spawnChunk(state.generatedUpToY, state.generatedUpToY + 8.0);
    state.generatedUpToY += 8.0;
  }
  state.objects = state.objects.filter(o => o.y > minY - 6.0);
}

function isHeightCollidable(agent, o) {
  if (o.haz <= 0) return false;
  const zMin = o.z - o.sz / 2, zMax = o.z + o.sz / 2;
  return !(zMin > agent.height + 0.15 || zMax < 0.05);
}

function isFutureHazard(agent, o) {
  if (!isHeightCollidable(agent, o)) return false;
  const rx = o.x - agent.x, ry = o.y - agent.y;
  const rvx = o.vx - agent.vx, rvy = o.vy - agent.vy;
  const v2 = rvx * rvx + rvy * rvy;
  const tStar = v2 < 1e-6 ? 0 : Math.max(0, Math.min(3.0, -(rx * rvx + ry * rvy) / v2));
  const cx = Math.abs(rx + rvx * tStar), cy = Math.abs(ry + rvy * tStar);
  return cx <= (o.sx / 2 + agent.radius + 0.25) && cy <= (o.sy / 2 + agent.radius + 0.25);
}

function computeRisk(agent, o) {
  if (!isHeightCollidable(agent, o)) return 0.0;
  const rx = o.x - agent.x, ry = o.y - agent.y;
  const dist = Math.hypot(rx, ry);
  const rvx = o.vx - agent.vx, rvy = o.vy - agent.vy;
  const v2 = rvx * rvx + rvy * rvy;
  const horizon = 3.2;
  const ttc = v2 < 1e-6 ? 0 : Math.max(0, Math.min(horizon, -(rx * rvx + ry * rvy) / v2));
  const ex = Math.max(Math.abs(rx + rvx * ttc) - (o.sx / 2 + agent.radius + 0.22), 0);
  const ey = Math.max(Math.abs(ry + rvy * ttc) - (o.sy / 2 + agent.radius + 0.22), 0);
  const dMin = Math.hypot(ex, ey);

  const aSpd = Math.hypot(agent.vx, agent.vy);
  let ang = 0;
  if (aSpd > 0.1 && dist > 1e-4) {
    ang = Math.acos(Math.max(-1, Math.min(1, (rx * agent.vx + ry * agent.vy) / (dist * aSpd))));
  } else if (dist > 1e-4) {
    ang = Math.acos(Math.max(-1, Math.min(1, ry / dist))); // Default forward +Y when stopped
  }
  const coneRad = 38 * Math.PI / 180;
  const inSafetyBubble = dist <= 2.2;
  let coneFactor = (inSafetyBubble || ang <= coneRad) ? 1.0 : Math.exp(-Math.pow((ang - coneRad) / 0.35, 2));
  const closing = -((rx * rvx + ry * rvy) / (dist + 1e-6));
  if (closing > 0.3) coneFactor = Math.max(coneFactor, 0.78);

  const exNow = Math.max(Math.abs(rx) - (o.sx / 2 + agent.radius + 0.22), 0);
  const eyNow = Math.max(Math.abs(ry) - (o.sy / 2 + agent.radius + 0.22), 0);
  const effDist = inSafetyBubble ? Math.min(dMin, Math.hypot(exNow, eyNow)) : dMin;

  const spatial = Math.exp(-(effDist * effDist) / (2 * 0.65 * 0.65));
  const urgency = inSafetyBubble ? 1.0 : (1.0 - 0.5 * (ttc / horizon));
  return Math.max(0, Math.min(1, spatial * urgency * coneFactor * Math.max(o.haz, 0.2)));
}

// --- Mini-DWA Reactive Local Planner + Gap-Seeking Anti-Deadlock ---
function findWidestGapX(agent, activeObs, width) {
  let bestX = width / 2;
  let bestClear = -1;
  const probeY = agent.y + 1.1;
  for (let px = 0.5; px <= width - 0.5; px += 0.2) {
    let minClr = Math.min(px - 0.35, (width - 0.35) - px);
    for (const o of activeObs) {
      if (o.y < agent.y - 0.3 || o.y > agent.y + 2.8) continue;
      const dx = Math.max(Math.abs(px - o.x) - (o.sx / 2 + agent.radius), 0);
      const dy = Math.max(Math.abs(probeY - o.y) - (o.sy / 2 + agent.radius), 0);
      minClr = Math.min(minClr, Math.hypot(dx, dy));
    }
    if (minClr > bestClear) { bestClear = minClr; bestX = px; }
  }
  return bestX;
}

function planReactiveVelocity(agent, perceivedObstacles, width) {
  const maxSpd = 1.35;
  const angles = [0, -15, 15, -30, 30, -48, 48, -68, 68, -86, 86];
  const speeds = [maxSpd, 0.90, 0.50];
  const horizon = 1.65;
  const steps = 6;
  const margin = agent.radius + 0.12;

  const activeObs = perceivedObstacles.filter(o => isHeightCollidable(agent, o));

  // If currently committed to a gap-escape sidestep, finish moving toward the open gap
  if (agent.escapeTimer > 0) {
    agent.escapeTimer -= 0.1;
    const dx = agent.escapeTargetX - agent.x;
    if (Math.abs(dx) > 0.25) {
      agent.targetVx = Math.sign(dx) * 1.15;
      agent.targetVy = -0.10;
      agent.status = dx < 0 ? "↩ LÁCH KHE TRÁI" : "↩ LÁCH KHE PHẢI";
      agent.statusKind = "b-amber";
      return;
    }
    agent.escapeTimer = 0;
  }

  let bestScore = -1e9;
  let bestVx = 0, bestVy = 0;
  let foundSafeForward = false;

  for (const spd of speeds) {
    for (const deg of angles) {
      const rad = deg * Math.PI / 180;
      const vx = spd * Math.sin(rad);
      const vy = spd * Math.cos(rad);

      let minClearance = 9.0;
      let collided = false;

      for (let s = 1; s <= steps; s++) {
        const dt = (horizon * s) / steps;
        const px = agent.x + vx * dt;
        const py = agent.y + vy * dt;

        const wallClear = Math.min(px - 0.36, (width - 0.36) - px);
        if (wallClear < 0) { collided = true; break; }
        minClearance = Math.min(minClearance, wallClear);

        for (const o of activeObs) {
          const ox = o.x + o.vx * dt;
          const oy = o.y + o.vy * dt;
          const dx = Math.max(Math.abs(px - ox) - (o.sx / 2 + margin), 0);
          const dy = Math.max(Math.abs(py - oy) - (o.sy / 2 + margin), 0);
          const clr = Math.hypot(dx, dy);
          if (clr <= 0.02) { collided = true; break; }
          minClearance = Math.min(minClearance, clr);
        }
        if (collided) break;
      }

      if (!collided) {
        foundSafeForward = true;
        const centerBias = -0.06 * Math.abs((agent.x + vx * 0.8) - width / 2);
        const steerPenalty = -0.005 * Math.abs(deg);
        const score = 1.45 * vy + 1.05 * Math.min(minClearance, 1.5) + centerBias + steerPenalty;
        if (score > bestScore) {
          bestScore = score;
          bestVx = vx;
          bestVy = vy;
        }
      }
    }
  }

  if (foundSafeForward) {
    agent.brakeTime = 0;
    agent.targetVx = bestVx;
    agent.targetVy = bestVy;
    if (Math.abs(bestVx) > 0.35) {
      agent.status = bestVx < 0 ? "↪ NÉ TRÁI" : "↪ NÉ PHẢI";
      agent.statusKind = "b-amber";
    } else {
      agent.status = "✓ ĐI THẲNG";
      agent.statusKind = "b-green";
    }
  } else {
    // SURROUNDED / BLOCKED BY WIDE OBSTACLES!
    agent.brakeTime += 0.1;
    if (agent.brakeTime < 0.7) {
      // Phase 1: Brief Emergency Brake to let moving pedestrians clear
      agent.targetVx = 0;
      agent.targetVy = 0;
      agent.status = "🛑 PHANH GẤP";
      agent.statusKind = "b-red";
    } else {
      // Phase 2: Find the widest open horizontal gap across the corridor and commit to sidestepping!
      agent.escapeTargetX = findWidestGapX(agent, activeObs, width);
      agent.escapeTimer = 0.9;
      agent.brakeTime = 0;
      const dx = agent.escapeTargetX - agent.x;
      agent.targetVx = (Math.abs(dx) > 0.1 ? Math.sign(dx) : 1) * 1.15;
      agent.targetVy = -0.15;
      agent.status = dx < 0 ? "↩ LÁCH KHE TRÁI" : "↩ LÁCH KHE PHẢI";
      agent.statusKind = "b-amber";
    }
  }
}

function jaccardFlicker(a, b) {
  const union = new Set([...a, ...b]);
  if (union.size === 0) return 0;
  let diff = 0;
  for (const x of union) if (a.has(x) !== b.has(x)) diff++;
  return diff / union.size;
}

function updateAgentAlgorithm(agent, isProposed, k, noise) {
  const obs = [];
  const aSpd = Math.hypot(agent.vx, agent.vy);
  const hx = aSpd > 0.1 ? agent.vx / aSpd : 0;
  const hy = aSpd > 0.1 ? agent.vy / aSpd : 1;

  for (const o of state.objects) {
    const isHaz = isFutureHazard(agent, o);
    const dx = o.x - agent.x, dy = o.y - agent.y;
    const dist = Math.hypot(dx, dy);
    if (dist > agent.maxRange || dist < 0.05) continue;
    const ang = Math.acos(Math.max(-1, Math.min(1, (dx * hx + dy * hy) / dist)));
    if (ang > agent.fov) continue;

    const fovRatio = ang / agent.fov;
    if (Math.random() < noise.dropout * (0.5 + 1.5 * fovRatio * fovRatio)) continue;

    const conf = Math.max(0.2, Math.min(0.99, 0.98 - 0.35 * fovRatio * fovRatio - 0.15 * (dist / agent.maxRange) + randn() * noise.confNoise));
    obs.push({
      ...o,
      x: o.x + randn() * noise.posStd,
      y: o.y + randn() * noise.posStd,
      conf,
      isHazardGT: isHaz,
      dist3d: Math.hypot(dx, dy, o.z - agent.z),
    });
  }

  if (Math.random() < noise.fp * 3.0) {
    obs.push({
      id: `fp_${state.t}_${isProposed ? 'p' : 'd'}`, type: "unknown_ghost", name: "Nhiễu ảo (FP)",
      x: agent.x + (Math.random() - 0.5) * 2.8, y: agent.y + 2.0 + Math.random() * 4.5, z: 0.5,
      sx: 0.4, sy: 0.4, sz: 0.8, vx: 0, vy: 0, cv: 0.05, haz: 0.4, dyn: false,
      conf: 0.25 + Math.random() * 0.25, isHazardGT: false, dist3d: 3.5,
    });
  }

  let sel;
  if (!isProposed) {
    sel = new Set([...obs].sort((a, b) => a.dist3d - b.dist3d).slice(0, k).map(o => o.id));
  } else {
    const scored = obs.map(o => {
      const rv = computeRisk(agent, o);
      const uv = 1.0 - o.conf;
      let sv = 1.0 * rv * (1.0 + 0.6 * uv * (rv > 0.30 ? 1 : 0)) + 0.15 * o.conf * o.cv;
      if (agent.prevSel.has(o.id) && sv >= 0.12) {
        sv += (agent.persistence[o.id] || 1) < 2 ? 0.27 : 0.18;
      }
      return { id: o.id, sv };
    }).sort((a, b) => b.sv - a.sv);
    sel = new Set(scored.slice(0, k).map(o => o.id));

    const np = {};
    for (const id of sel) np[id] = (agent.persistence[id] || 0) + 1;
    agent.persistence = np;
  }

  const obsHaz = new Set(obs.filter(o => o.isHazardGT).map(o => o.id));
  const miss = new Set([...obsHaz].filter(id => !sel.has(id)));
  const rec = obsHaz.size === 0 ? 1.0 : (obsHaz.size - miss.size) / obsHaz.size;

  agent.frames++;
  agent.sumRec += rec;
  if (agent.frames > 1) agent.sumFl += jaccardFlicker(agent.prevSel, sel);
  agent.prevSel = sel;
  agent.obs = obs;
  agent.sel = sel;
  agent.miss = miss;

  // Closed-loop planning: Agent ONLY sees the k selected objects!
  const perceived = obs.filter(o => sel.has(o.id));
  planReactiveVelocity(agent, perceived, state.width);
}

function stepPhysics(subDt) {
  const { width } = state;
  // Move dynamic objects
  for (const o of state.objects) {
    if (o.dyn) {
      o.x += o.vx * subDt;
      o.y += o.vy * subDt;
      if (o.x < 0.4 || o.x > width - 0.4) { o.vx *= -1; o.x = Math.max(0.4, Math.min(width - 0.4, o.x)); }
    }
  }

  // Smooth velocity blending + real GT collision check for both agents
  for (const ag of [state.agentD, state.agentP]) {
    const blend = Math.min(1.0, subDt * 8.0);
    ag.vx += (ag.targetVx - ag.vx) * blend;
    ag.vy += (ag.targetVy - ag.vy) * blend;
    ag.x = Math.max(0.38, Math.min(width - 0.38, ag.x + ag.vx * subDt));
    ag.y += ag.vy * subDt;

    if (ag.hitFlash > 0) ag.hitFlash = Math.max(0, ag.hitFlash - subDt);

    // Check physical ground-truth collision
    for (const o of state.objects) {
      if (!isHeightCollidable(ag, o)) continue;
      const dx = Math.abs(ag.x - o.x) - (o.sx / 2 + ag.radius * 0.85);
      const dy = Math.abs(ag.y - o.y) - (o.sy / 2 + ag.radius * 0.85);
      if (dx < 0 && dy < 0 && !ag.hitSet.has(o.id)) {
        ag.hitSet.add(o.id);
        ag.collisions++;
        ag.hitFlash = 0.55;
      }
    }
  }
  ensureChunksAhead();
}

function stepAlgorithm() {
  const k = parseInt(document.getElementById("kSelect").value, 10);
  const noise = NOISE_CFG[document.getElementById("noiseSelect").value];
  state.t++;

  for (const o of state.objects) {
    if (o.dyn) { o.trail.push([o.x, o.y]); if (o.trail.length > 14) o.trail.shift(); }
  }
  for (const ag of [state.agentD, state.agentP]) {
    ag.trail.push([ag.x, ag.y]);
    if (ag.trail.length > 32) ag.trail.shift();
  }

  updateAgentAlgorithm(state.agentD, false, k, noise);
  updateAgentAlgorithm(state.agentP, true, k, noise);

  state.historyD.push((state.agentD.sumRec / state.agentD.frames) * 100);
  state.historyP.push((state.agentP.sumRec / state.agentP.frames) * 100);
  if (state.historyD.length > 160) { state.historyD.shift(); state.historyP.shift(); }
}

// --- Rendering ---
function resize(c) {
  const r = c.getBoundingClientRect();
  if (c.width !== r.width || c.height !== r.height) { c.width = r.width; c.height = r.height; }
}

function getTransform(c, agent) {
  const scale = Math.min(c.width / (state.width + 3.0), c.height / 14.5);
  return {
    scale,
    toScreen: (x, y) => [c.width * 0.5 + (x - state.width / 2) * scale, c.height * 0.78 - (y - agent.y) * scale],
    toWorld: (sx, sy) => [state.width / 2 + (sx - c.width * 0.5) / scale, agent.y + (c.height * 0.78 - sy) / scale],
  };
}

function drawStyledObject(ctx, o, ox, oy, scale, inObs, isSel, isMiss, isHaz) {
  const w = Math.max(10, o.sx * scale);
  const h = Math.max(10, o.sy * scale);
  ctx.save();
  ctx.globalAlpha = isSel || isMiss ? 1.0 : (inObs ? 0.38 : 0.14);

  if (o.type === "stair_edge") {
    ctx.fillStyle = "#f97316"; ctx.fillRect(ox - w / 2, oy - h / 2, w, h);
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1.5;
    for (let s = -h / 2 + 3; s < h / 2; s += 4) { ctx.beginPath(); ctx.moveTo(ox - w / 2, oy + s); ctx.lineTo(ox + w / 2, oy + s); ctx.stroke(); }
  } else if (o.type === "low_barrier" || o.type === "overhead_sign") {
    ctx.fillStyle = o.type === "overhead_sign" ? "#a855f7" : "#eab308";
    ctx.fillRect(ox - w / 2, oy - h / 2, w, h);
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 2; ctx.strokeRect(ox - w / 2, oy - h / 2, w, h);
  } else if (o.type === "ceiling_light") {
    ctx.fillStyle = "#fde047"; ctx.beginPath(); ctx.arc(ox, oy, 5, 0, Math.PI * 2); ctx.fill();
  } else if (o.type === "floor_mat") {
    ctx.strokeStyle = "#64748b"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5;
    ctx.strokeRect(ox - w / 2, oy - h / 2, w, h); ctx.setLineDash([]);
  } else if (o.type === "pedestrian") {
    ctx.fillStyle = isHaz ? "#ef4444" : "#22c55e";
    ctx.beginPath(); ctx.ellipse(ox, oy, w * 0.55, h * 0.42, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f8fafc"; ctx.beginPath(); ctx.arc(ox, oy, 3.2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fca5a5"; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + o.vx * scale * 0.7, oy - o.vy * scale * 0.7); ctx.stroke();
  } else {
    ctx.fillStyle = isHaz ? "#ef4444" : "#22c55e";
    ctx.fillRect(ox - w / 2, oy - h / 2, w, h);
  }

  if (isSel) {
    ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(ox, oy, Math.max(w, h) * 0.65 + 5, 0, Math.PI * 2); ctx.stroke();
  }
  if (isMiss) {
    ctx.strokeStyle = "#ff2e2e"; ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(ox - 8, oy - 8); ctx.lineTo(ox + 8, oy + 8);
    ctx.moveTo(ox + 8, oy - 8); ctx.lineTo(ox - 8, oy + 8);
    ctx.stroke();
  }
  if (isSel || isMiss) {
    ctx.fillStyle = isMiss ? "#fca5a5" : "#f8fafc";
    ctx.font = isMiss ? "bold 10.5px system-ui" : "10px system-ui";
    ctx.fillText(isMiss ? `SÓT: ${o.name}` : o.name, ox + w / 2 + 5, oy + 3);
  }
  ctx.restore();
}

function drawPanel(canvasId, agent, showCone) {
  const c = document.getElementById(canvasId);
  resize(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  // Red collision flash background tint
  if (agent.hitFlash > 0) {
    ctx.fillStyle = `rgba(239, 68, 68, ${agent.hitFlash * 0.45})`;
    ctx.fillRect(0, 0, c.width, c.height);
  }

  const { scale, toScreen } = getTransform(c, agent);
  const { width } = state;

  // Floor grid
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.fillStyle = "#475569"; ctx.font = "10px system-ui";
  const startM = Math.floor(agent.y / 2) * 2 - 2;
  for (let m = startM; m <= startM + 18; m += 2) {
    const [lx, ly] = toScreen(0, m), [rx] = toScreen(width, m);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ly); ctx.stroke();
    ctx.fillText(`${m}m`, lx - 26, ly + 3);
  }

  // Walls
  const [wl] = toScreen(0, agent.y), [wr] = toScreen(width, agent.y);
  ctx.strokeStyle = "#64748b"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(wl, 0); ctx.lineTo(wl, c.height); ctx.moveTo(wr, 0); ctx.lineTo(wr, c.height); ctx.stroke();

  // FOV & Collision Cone
  const [sx, sy] = toScreen(agent.x, agent.y);
  const spd = Math.hypot(agent.vx, agent.vy);
  const heading = spd > 0.12 ? Math.atan2(agent.vy, agent.vx) : Math.PI / 2;

  ctx.fillStyle = "rgba(250, 204, 21, 0.06)";
  ctx.strokeStyle = "rgba(250, 204, 21, 0.28)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(sx, sy);
  ctx.arc(sx, sy, agent.maxRange * scale, -heading - agent.fov, -heading + agent.fov);
  ctx.closePath(); ctx.fill(); ctx.stroke();

  if (showCone) {
    const coneHalf = 38 * Math.PI / 180;
    ctx.fillStyle = "rgba(56, 189, 248, 0.13)";
    ctx.strokeStyle = "rgba(56, 189, 248, 0.7)"; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(sx, sy);
    ctx.arc(sx, sy, 6.5 * scale, -heading - coneHalf, -heading + coneHalf);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    // Personal Safety Bubble (2.2m) to prevent side-swipe blind spots
    ctx.strokeStyle = "rgba(74, 222, 128, 0.55)"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(sx, sy, 2.2 * scale, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Agent walked trail
  if (agent.trail.length > 1) {
    ctx.strokeStyle = "rgba(245, 158, 11, 0.45)"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    agent.trail.forEach((p, i) => { const [tx, ty] = toScreen(p[0], p[1]); i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty); });
    ctx.stroke();
  }

  // Planned avoidance vector
  ctx.strokeStyle = "#4ade80"; ctx.lineWidth = 2.2; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(sx, sy);
  ctx.lineTo(sx + agent.targetVx * scale * 1.4, sy - agent.targetVy * scale * 1.4);
  ctx.stroke(); ctx.setLineDash([]);

  // Pedestrian trails & objects
  const obsMap = Object.fromEntries(agent.obs.map(o => [o.id, o]));
  for (const o of state.objects) {
    if (o.dyn && o.trail.length > 1) {
      ctx.strokeStyle = "rgba(248, 113, 113, 0.3)"; ctx.lineWidth = 1.8;
      ctx.beginPath();
      o.trail.forEach((p, i) => { const [tx, ty] = toScreen(p[0], p[1]); i === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty); });
      ctx.stroke();
    }
    const [ox, oy] = toScreen(o.x, o.y);
    if (oy < -40 || oy > c.height + 40) continue;
    const obsObj = obsMap[o.id];
    drawStyledObject(ctx, o, ox, oy, scale, Boolean(obsObj), agent.sel.has(o.id), agent.miss.has(o.id), obsObj ? obsObj.isHazardGT : isFutureHazard(agent, o));
  }

  // Stylized Agent
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(Math.PI / 2 - heading);
  ctx.fillStyle = agent.hitFlash > 0 ? "#ef4444" : "#f59e0b";
  ctx.beginPath(); ctx.ellipse(0, 0, 12, 7, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#fef3c7";
  ctx.beginPath(); ctx.arc(0, -3, 4.8, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  if (agent.hitFlash > 0) {
    ctx.fillStyle = "#fecaca"; ctx.font = "bold 13px system-ui";
    ctx.fillText("💥 VA CHẠM!", sx - 38, sy - 18);
  }
}

function drawSparkline() {
  const c = document.getElementById("sparkCanvas");
  resize(c);
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (state.historyP.length < 2) return;

  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1; ctx.fillStyle = "#64748b"; ctx.font = "9px system-ui";
  [50, 75, 100].forEach(v => {
    const y = c.height - (v / 100) * (c.height - 14) - 7;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke();
    ctx.fillText(v + "%", 3, y - 2);
  });

  const drawLine = (arr, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * (c.width - 10) + 5;
      const y = c.height - (v / 100) * (c.height - 14) - 7;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawLine(state.historyD, "#38bdf8");
  drawLine(state.historyP, "#4ade80");
}

function renderUI() {
  const aD = state.agentD, aP = state.agentP;
  document.getElementById("frameCount").textContent = aP.frames;

  const db = document.getElementById("distStatus");
  db.textContent = aD.hitFlash > 0 ? "💥 VA CHẠM!" : aD.status;
  db.className = "badge " + (aD.hitFlash > 0 ? "b-red" : aD.statusKind);

  const pb = document.getElementById("propStatus");
  pb.textContent = aP.hitFlash > 0 ? "💥 VA CHẠM!" : aP.status;
  pb.className = "badge " + (aP.hitFlash > 0 ? "b-red" : aP.statusKind);

  const recD = (aD.sumRec / Math.max(1, aD.frames) * 100).toFixed(1);
  const recP = (aP.sumRec / Math.max(1, aP.frames) * 100).toFixed(1);
  const flD = (aD.sumFl / Math.max(1, aD.frames - 1)).toFixed(3);
  const flP = (aP.sumFl / Math.max(1, aP.frames - 1)).toFixed(3);

  const winR = parseFloat(recP) >= parseFloat(recD);
  const winF = parseFloat(flP) <= parseFloat(flD);
  const winC = aP.collisions <= aD.collisions;

  document.getElementById("metricsBody").innerHTML = `
    <tr><td><b>💥 Va chạm thật (↓)</b></td><td class="${!winC?'win':'lose'}">${aD.collisions} lần</td><td class="${winC?'win':'lose'}">${aP.collisions} lần</td></tr>
    <tr><td>Quãng đường đi</td><td>${aD.y.toFixed(0)}m</td><td>${aP.y.toFixed(0)}m</td></tr>
    <tr><td>Hazard Recall (↑)</td><td class="${!winR?'win':'lose'}">${recD}%</td><td class="${winR?'win':'lose'}">${recP}%</td></tr>
    <tr><td>Flicker F_t (↓)</td><td class="${!winF?'win':'lose'}">${flD}</td><td class="${winF?'win':'lose'}">${flP}</td></tr>
  `;

  const byId = Object.fromEntries(aP.obs.map(o => [o.id, o]));
  const items = [];
  for (const id of aP.miss) items.push(`<li class="li-miss"><span>✕ SÓT: ${byId[id]?.name || id}</span><span>NGUY HIỂM</span></li>`);
  for (const id of aP.sel) {
    const o = byId[id];
    items.push(`<li class="${o?.isHazardGT ? 'li-ok' : 'li-safe'}"><span>✓ ${o?.name || id}</span><span>${o?.isHazardGT ? 'Đang né' : 'Định hướng'}</span></li>`);
  }
  document.getElementById("propList").innerHTML = items.join("");

  drawPanel("canvasDist", aD, false);
  drawPanel("canvasProp", aP, true);
  drawSparkline();
}

["canvasDist", "canvasProp"].forEach((cid, idx) => {
  document.getElementById(cid).addEventListener("click", (e) => {
    const c = e.target;
    const rect = c.getBoundingClientRect();
    const ag = idx === 0 ? state.agentD : state.agentP;
    const { toWorld } = getTransform(c, ag);
    const [wx, wy] = toWorld(e.clientX - rect.left, e.clientY - rect.top);
    const sp = SPECS.pedestrian;
    state.objects.push({
      id: `user_ped_${state.nextId++}`, type: "pedestrian", name: "Người (Mày thả!)",
      x: Math.max(0.4, Math.min(state.width - 0.4, wx)), y: wy, z: sp.z,
      sx: sp.size[0], sy: sp.size[1], sz: sp.size[2],
      vx: 0, vy: -1.3, cv: sp.cv, haz: sp.haz, dyn: true, trail: [[wx, wy]],
    });
  });
});

let playing = true, lastFrame = performance.now(), algAccum = 0, speed = 1.0;
document.getElementById("playBtn").onclick = (e) => { playing = !playing; e.target.textContent = playing ? "Tạm dừng" : "Chạy tiếp"; lastFrame = performance.now(); };
document.getElementById("resetBtn").onclick = () => { initWorld(); stepAlgorithm(); renderUI(); };
document.getElementById("modeSelect").onchange = () => { initWorld(); stepAlgorithm(); renderUI(); };
document.getElementById("speedSelect").onchange = (e) => { speed = parseFloat(e.target.value); };

function loop(now) {
  const realDt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  if (playing) {
    const simDt = realDt * speed;
    stepPhysics(simDt);
    algAccum += simDt;
    if (algAccum >= 0.1) {
      stepAlgorithm();
      algAccum %= 0.1;
    }
    renderUI();
  }
  requestAnimationFrame(loop);
}

initWorld();
stepAlgorithm();
renderUI();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def main() -> None:
    out_path = Path("results/infinite_demo.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML_INFINITE, encoding="utf-8")
    print(f"Closed-Loop 60 FPS Infinite Demo generated at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
