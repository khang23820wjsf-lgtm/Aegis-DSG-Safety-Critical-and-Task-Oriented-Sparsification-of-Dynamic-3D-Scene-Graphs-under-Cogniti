"""Build a self-contained Human-in-the-Loop Cognitive Load & Reaction-Time User Study App.

Generates `user_study/aegis_human_eval.html` using REAL Meta Project Aria Digital Twin
egocentric RGB frames (`preview_rgb.mp4` + exact 3D GT boxes), comparing 3 visual AR modes:
  - Condition A: Raw Uncompressed Perception (~80-90 bounding boxes -> high visual clutter)
  - Condition B: Distance Baseline (k=5 closest raw objects)
  - Condition C: Aegis-Core v2.0 (k=5 with Atomic Hazard Preservation & Union-Find Coarsening)

The generated single-file HTML5 app runs offline in any browser (no server needed) and collects:
  1. Hazard Identification Reaction Time (ms, via performance.now())
  2. Hazard Localization Accuracy (% clicks inside exact GT hazard bounding box)
  3. Official 6-subscale NASA-TLX Cognitive Load Score (0-100) per condition
  4. Anonymized JSON export ready for `user_study/analyze_study.py` (paired t-test & Wilcoxon)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aegis_core import AegisConfig, AegisDSG
from src.baselines import DistanceSparsifier
from src.real_data.loader_aria_adt import DEFAULT_ADT_SEQ_DIR, load_aria_adt_sequence


def _rotate_box_cw90(box: Tuple[int, int, int, int], raw_w: int = 1408, raw_h: int = 1408) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, raw_h - y2), max(0, x1), min(raw_h - 1, raw_h - y1), min(raw_w - 1, x2))


def _encode_jpg_b64(img_bgr: np.ndarray, size: int = 620, quality: int = 82) -> str:
    resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_tag(img: np.ndarray, x1: int, y1: int, tag: str, col: Tuple[int, int, int]) -> None:
    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.85, 2
    (tw, th), _ = cv2.getTextSize(tag, font, scale, thick)
    tx = int(np.clip(x1, 8, max(8, w_img - tw - 16)))
    ty = max(th + 14, y1 - 8)
    cv2.rectangle(img, (tx - 4, ty - th - 8), (tx + tw + 8, ty + 6), col, -1)
    cv2.putText(img, tag, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def render_three_conditions(
    img_bgr: np.ndarray,
    frame,
    raw_boxes: Dict[str, Tuple[int, int, int, int]],
    dist_sel: set,
    aegis_entities: list,
) -> Dict[str, str]:
    """Render Raw, Distance (k=5), and Aegis-Core v2.0 (k=5) overlays on the exact same real frame."""
    h_img, w_img = img_bgr.shape[:2]

    # Condition A: Raw Uncompressed Perception (cluttered)
    raw_img = img_bgr.copy()
    for nid, box in raw_boxes.items():
        x1, y1, x2, y2 = _rotate_box_cw90(box, w_img, h_img)
        cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 215, 255), 2)
        if nid in frame.graph_obs.nodes:
            lbl = str(frame.graph_obs.nodes[nid].get("label", ""))[:14]
            if (x2 - x1) > 45 and (y2 - y1) > 45:
                cv2.putText(raw_img, lbl, (x1 + 4, max(24, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 225, 255), 1, cv2.LINE_AA)

    # Condition B: Distance Baseline (k=5)
    dist_img = (img_bgr.astype(np.float32) * 0.58).astype(np.uint8)
    for nid in dist_sel:
        if nid in raw_boxes:
            x1, y1, x2, y2 = _rotate_box_cw90(raw_boxes[nid], w_img, h_img)
            dist_img[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
    for nid in dist_sel:
        if nid in raw_boxes and nid in frame.graph_obs.nodes:
            x1, y1, x2, y2 = _rotate_box_cw90(raw_boxes[nid], w_img, h_img)
            d = frame.graph_obs.nodes[nid]
            dm = float(np.linalg.norm(np.array(d["pos"][:2]) - frame.agent.pos[:2]))
            col = (230, 140, 40)
            cv2.rectangle(dist_img, (x1, y1), (x2, y2), col, 4)
            _draw_tag(dist_img, x1, y1, f"{d.get('label', 'obj')} | {dm:.1f}m", col)

    # Condition C: Aegis-Core v2.0 (k=5)
    aegis_img = (img_bgr.astype(np.float32) * 0.52).astype(np.uint8)
    covered = set().union(*(e.member_ids for e in aegis_entities)) if aegis_entities else set()
    for nid in covered:
        if nid in raw_boxes:
            x1, y1, x2, y2 = _rotate_box_cw90(raw_boxes[nid], w_img, h_img)
            aegis_img[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
    for ent in aegis_entities:
        boxes_ent = [_rotate_box_cw90(raw_boxes[m], w_img, h_img) for m in ent.member_ids if m in raw_boxes]
        if not boxes_ent:
            continue
        bx1, by1 = min(b[0] for b in boxes_ent), min(b[1] for b in boxes_ent)
        bx2, by2 = max(b[2] for b in boxes_ent), max(b[3] for b in boxes_ent)
        dm = float(np.linalg.norm(ent.pos[:2] - frame.agent.pos[:2]))
        if ent.is_atomic_hazard:
            col = (220, 20, 255)
            tag = f"[!] {ent.label.upper()} | {dm:.1f}m"
            thick = 6
        elif ent.is_super_node:
            col = (0, 140, 255)
            tag = f"{ent.label} | {dm:.1f}m"
            thick = 4
        else:
            col = (50, 85, 245) if ent.risk > 0.28 else (80, 195, 80)
            tag = f"{ent.label} | {dm:.1f}m"
            thick = 4
        cv2.rectangle(aegis_img, (bx1, by1), (bx2, by2), col, thick)
        _draw_tag(aegis_img, bx1, by1, tag, col)

    return {
        "raw": _encode_jpg_b64(raw_img),
        "distance": _encode_jpg_b64(dist_img),
        "aegis": _encode_jpg_b64(aegis_img),
    }


def extract_real_trials(num_scenes: int = 4) -> List[dict]:
    """Select real Meta Aria keyframes with clear, verifiable ground-truth atomic hazards (knife/scissors)."""
    frames, bboxes_list, vid_indices = load_aria_adt_sequence(DEFAULT_ADT_SEQ_DIR, max_frames=150, stride=2)
    video_path = DEFAULT_ADT_SEQ_DIR / "ADT_Apartment_release_golden_skeleton_seq100_10s_sample_M1292_preview_rgb.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open Meta Aria preview video: {video_path}")

    dist_sp = DistanceSparsifier()
    aegis = AegisDSG(AegisConfig(tau_min=0.15))
    candidates: List[Tuple[int, dict]] = []

    for idx, (frame, raw_boxes, vid_idx) in enumerate(zip(frames, bboxes_list, vid_indices)):
        d_sel = dist_sp.select(frame.graph_obs, frame.agent, 5)
        a_ents = aegis.compress(frame.graph_obs, frame.agent, 5)

        # Find visible atomic hazards (knife / scissors) with reasonably visible 2D bounding boxes
        atomics = []
        for nid, d in frame.graph_obs.nodes(data=True):
            if d.get("layer") != "object" or nid not in raw_boxes:
                continue
            if d.get("is_atomic_hazard", False):
                x1, y1, x2, y2 = _rotate_box_cw90(raw_boxes[nid], 1408, 1408)
                area = (x2 - x1) * (y2 - y1)
                if area >= 2200:
                    atomics.append((area, nid, str(d.get("label", "hazard")), [x1 / 1408.0, y1 / 1408.0, x2 / 1408.0, y2 / 1408.0]))

        if atomics:
            atomics.sort(reverse=True)
            _, tgt_id, tgt_lbl, tgt_box = atomics[0]
            candidates.append((
                idx,
                {
                    "frame_idx": idx,
                    "vid_idx": vid_idx,
                    "frame": frame,
                    "raw_boxes": raw_boxes,
                    "dist_sel": d_sel,
                    "aegis_ents": a_ents,
                    "target_label": tgt_lbl,
                    "target_box": tgt_box,
                    "raw_count": len(raw_boxes),
                },
            ))

    if len(candidates) < num_scenes:
        raise RuntimeError(f"Found only {len(candidates)} candidate scenes with visible atomic hazards.")

    # Pick evenly spaced distinct frames across the sequence so scenes look genuinely different
    pick_indices = np.linspace(0, len(candidates) - 1, num_scenes, dtype=int)
    trials: List[dict] = []
    for scene_i, ci in enumerate(pick_indices):
        item = candidates[int(ci)][1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["vid_idx"])
        ok, img_bgr = cap.read()
        if not ok:
            continue
        imgs = render_three_conditions(img_bgr, item["frame"], item["raw_boxes"], item["dist_sel"], item["aegis_ents"])
        for cond_key, cond_name in (
            ("raw", f"Raw Perception ({item['raw_count']} boxes)"),
            ("distance", "Distance Baseline (k=5)"),
            ("aegis", "Aegis-Core v2.0 (k=5)"),
        ):
            trials.append({
                "trial_key": f"scene{scene_i + 1}_{cond_key}",
                "scene_id": scene_i + 1,
                "condition": cond_key,
                "condition_label": cond_name,
                "target_label": item["target_label"],
                "target_box": [round(v, 4) for v in item["target_box"]],
                "raw_count": item["raw_count"],
                "image_b64": imgs[cond_key],
            })
    cap.release()
    return trials


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Aegis-DSG Human-in-the-Loop Cognitive Load & Reaction-Time Study</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f1f5f9;
    --muted: #94a3b8; --accent: #38bdf8; --danger: #f43f5e; --ok: #22c55e;
  }
  * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; justify-content: center; padding: 20px; }
  .wrap { width: 100%; max-width: 860px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 24px 28px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
  h1 { margin: 0 0 8px; font-size: 1.55rem; color: #fff; }
  h2 { margin: 0 0 12px; font-size: 1.2rem; color: var(--accent); }
  p, li { color: #cbd5e1; line-height: 1.55; font-size: 0.96rem; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; background: rgba(56,189,248,0.16); color: var(--accent); border: 1px solid rgba(56,189,248,0.35); margin-bottom: 10px; }
  .btn { background: var(--accent); color: #0f172a; border: none; padding: 11px 22px; border-radius: 9px; font-weight: 700; font-size: 0.96rem; cursor: pointer; transition: 0.15s; }
  .btn:hover { filter: brightness(1.1); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-secondary { background: #334155; color: #f8fafc; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin: 12px 0; }
  input[type="text"], select { background: #0f172a; color: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; font-size: 0.95rem; }
  .stage-box { position: relative; width: 100%; max-width: 620px; margin: 0 auto; aspect-ratio: 1/1; background: #020617; border: 2px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; user-select: none; }
  .stage-box img { width: 100%; height: 100%; object-fit: contain; cursor: crosshair; display: block; }
  .fixation { font-size: 4rem; font-weight: 700; color: var(--accent); }
  .progress-bar { height: 8px; background: #0f172a; border-radius: 999px; overflow: hidden; margin-bottom: 14px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #22c55e); width: 0%; transition: width 0.25s; }
  .prompt-banner { background: rgba(244,63,94,0.14); border: 1px solid rgba(244,63,94,0.45); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; font-weight: 600; color: #fecdd3; text-align: center; }
  .tlx-grid { display: grid; grid-template-columns: 1fr; gap: 14px; margin: 16px 0; }
  .tlx-item { background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .tlx-item label { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; font-size: 0.92rem; }
  .tlx-item input[type="range"] { width: 100%; accent-color: var(--accent); }
  .tlx-hint { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { border-bottom: 1px solid var(--border); padding: 10px 8px; text-align: center; font-size: 0.92rem; }
  th { color: var(--accent); }
  .hidden { display: none !important; }
  .feedback-dot { position: absolute; width: 22px; height: 22px; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; border: 3px solid #fff; }
</style>
</head>
<body>
<div class="wrap">

  <!-- SCREEN 1: INTRO & CONSENT -->
  <div id="screen-intro" class="card">
    <div class="badge">Aegis-DSG • Human-in-the-Loop Evaluation (IEEE RA-L / HRI Protocol)</div>
    <h1>AR Hazard Detection & Cognitive Load Study (~3 mins)</h1>
    <p><strong>Mục tiêu / Purpose:</strong> Đánh giá thời gian phản xạ (Reaction Time) và tải nhận thức (NASA-TLX) của người dùng khi quan sát 3 chế độ hiển thị AR trên dữ liệu thật từ kính <strong>Meta Project Aria</strong>: <em>Raw Perception (80+ box)</em>, <em>Distance (k=5)</em>, và <em>Aegis-Core v2.0 (k=5)</em>.</p>
    <ul>
      <li>Bạn sẽ xem <strong>12 khung hình thật</strong> (ngẫu nhiên thứ tự).</li>
      <li>Ở mỗi ảnh, hãy <strong>click nhanh và chính xác nhất có thể vào VẬT NGUY HIỂM SẮC NHỌN (con dao / kéo)</strong> nếu thấy, hoặc bấm nút <em>"Không thấy / Không chắc"</em> nếu quá rối không tìm ra.</li>
      <li>Cuối bài, bạn chấm nhanh bảng <strong>NASA-TLX</strong> (tải nhận thức) cho từng chế độ hiển thị và tải file kết quả <code>.json</code> gửi lại cho nhóm nghiên cứu.</li>
    </ul>
    <div class="row">
      <label>Mã người tham gia (Participant ID):
        <input id="pid-input" type="text" placeholder="VD: P01" value="P01" style="width:110px; margin-left:6px;" />
      </label>
      <label>Kinh nghiệm AR/VR:
        <select id="exp-input" style="margin-left:6px;">
          <option value="none">Chưa từng (None)</option>
          <option value="some">Ít (Beginner)</option>
          <option value="high">Thường xuyên (Experienced)</option>
        </select>
      </label>
    </div>
    <div class="row">
      <label style="cursor:pointer;">
        <input id="consent-box" type="checkbox" checked /> Tôi đồng ý tham gia khảo sát ẩn danh phục vụ nghiên cứu khoa học.
      </label>
    </div>
    <button id="btn-start" class="btn">Bắt đầu bài test (Start Study) →</button>
  </div>

  <!-- SCREEN 2: TRIALS -->
  <div id="screen-trial" class="card hidden">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <span id="trial-counter" style="font-weight:700; color:var(--accent);">Trial 1 / 12</span>
      <span id="trial-cond-tag" style="font-size:0.85rem; color:var(--muted);">Condition: Blinded</span>
    </div>
    <div class="progress-bar"><div id="progress-fill" class="progress-fill"></div></div>
    <div id="prompt-banner" class="prompt-banner">
      CLICK NHANH VÀO VẬT NGUY HIỂM SẮC NHỌN (DAO / KÉO) TRONG ẢNH! / Click the SHARP HAZARD (Knife/Scissors) as fast as possible!
    </div>
    <div class="stage-box" id="stage-box">
      <div id="fixation-cross" class="fixation">+</div>
      <img id="trial-img" class="hidden" alt="Meta Aria AR Trial Frame" />
    </div>
    <div class="row" style="justify-content:center; margin-top:14px;">
      <button id="btn-not-found" class="btn btn-secondary hidden">Không tìm thấy / Quá rối (Cannot Locate Hazard)</button>
    </div>
  </div>

  <!-- SCREEN 3: NASA-TLX QUESTIONNAIRE (3 CONDITIONS) -->
  <div id="screen-tlx" class="card hidden">
    <div class="badge">NASA-TLX Cognitive Load Assessment</div>
    <h2 id="tlx-title">Đánh giá tải nhận thức — Chế độ 1 / 3</h2>
    <p id="tlx-desc"></p>
    <div id="tlx-preview-wrap" style="text-align:center; margin:10px 0;">
      <img id="tlx-preview" style="max-width:260px; border-radius:10px; border:1px solid var(--border);" />
    </div>
    <div class="tlx-grid" id="tlx-sliders"></div>
    <div class="row" style="justify-content:flex-end;">
      <button id="btn-tlx-next" class="btn">Tiếp tục (Next) →</button>
    </div>
  </div>

  <!-- SCREEN 4: RESULTS & JSON EXPORT -->
  <div id="screen-done" class="card hidden">
    <div class="badge" style="background:rgba(34,197,94,0.18); color:#4ade80; border-color:rgba(34,197,94,0.4);">Completed • Thank You!</div>
    <h1>Kết quả đo thực tế của bạn (Your Study Summary)</h1>
    <p>Hãy bấm nút tải file <code>.json</code> bên dưới và gửi cho nhóm nghiên cứu để chạy kiểm định thống kê gộp (Paired t-test & Wilcoxon).</p>
    <table>
      <thead>
        <tr>
          <th>Chế độ hiển thị (Mode)</th>
          <th>Mean Reaction Time (ms ↓)</th>
          <th>Hazard Accuracy (% ↑)</th>
          <th>NASA-TLX Load (0–100 ↓)</th>
        </tr>
      </thead>
      <tbody id="summary-tbody"></tbody>
    </table>
    <div class="row" style="margin-top:20px;">
      <button id="btn-download" class="btn">⬇ Tải file kết quả (.json) / Download Results</button>
      <button class="btn btn-secondary" onclick="location.reload()">Chạy lại với người mới (New Participant)</button>
    </div>
  </div>

</div>

<script>
const TRIAL_POOL = __TRIALS_JSON__;

const TLX_DIMS = [
  { id: "mental", name: "1. Mental Demand (Áp lực trí óc)", desc: "Mức độ phải căng mắt tập trung, suy nghĩ để tìm vật nguy hiểm?" },
  { id: "physical", name: "2. Visual/Physical Effort (Mỏi mắt / căng thẳng)", desc: "Mức độ rối mắt, khó chịu thị giác khi nhìn giao diện?" },
  { id: "temporal", name: "3. Temporal Demand (Áp lực thời gian)", desc: "Bạn có thấy bị vội vàng / mất quá nhiều thời gian để tìm?" },
  { id: "performance", name: "4. Failure / Uncertainty (Thiếu tự tin — cao là tệ)", desc: "Mức độ bạn cảm thấy KHÔNG chắc chắn với cú click của mình? (0 = Rất chắc chắn, 100 = Đoán bừa)" },
  { id: "effort", name: "5. Effort (Công sức bỏ ra)", desc: "Tổng công sức phải bỏ ra để hoàn thành nhiệm vụ?" },
  { id: "frustration", name: "6. Frustration (Mức độ ức chế / khó chịu)", desc: "Mức độ khó chịu, rối rắm khi nhìn các khung hình này?" }
];

const COND_META = [
  { key: "raw", title: "Chế độ A: Raw Uncompressed (80+ khung vàng dày đặc)", desc: "Chế độ hiển thị toàn bộ ~80–90 khung nhận diện thô từ kính Meta Aria." },
  { key: "distance", title: "Chế độ B: Distance Baseline (5 khung màu xanh dương/cam)", desc: "Chế độ chỉ giữ lại 5 vật thể gần nhất theo khoảng cách." },
  { key: "aegis", title: "Chế độ C: Aegis-Core v2.0 (5 khung + cảnh báo tím [!] KNIFE)", desc: "Chế độ gom cụm vật tĩnh và đánh dấu nổi bật vật nguy hiểm nguyên tử." }
];

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

let trials = [];
let currTrialIdx = 0;
let trialStartMs = 0;
let canClick = false;
let trialRecords = [];
let tlxStep = 0;
let tlxScores = {};
let participantMeta = {};

document.getElementById("btn-start").addEventListener("click", () => {
  if (!document.getElementById("consent-box").checked) {
    alert("Vui lòng tích vào ô đồng ý tham gia khảo sát.");
    return;
  }
  participantMeta = {
    participant_id: (document.getElementById("pid-input").value || "P00").trim(),
    ar_experience: document.getElementById("exp-input").value,
    timestamp_iso: new Date().toISOString(),
    user_agent: navigator.userAgent
  };
  trials = shuffle(TRIAL_POOL);
  currTrialIdx = 0;
  trialRecords = [];
  document.getElementById("screen-intro").classList.add("hidden");
  document.getElementById("screen-trial").classList.remove("hidden");
  runNextTrial();
});

function runNextTrial() {
  if (currTrialIdx >= trials.length) {
    document.getElementById("screen-trial").classList.add("hidden");
    document.getElementById("screen-tlx").classList.remove("hidden");
    tlxStep = 0;
    renderTlxStep();
    return;
  }
  const t = trials[currTrialIdx];
  document.getElementById("trial-counter").textContent = `Trial ${currTrialIdx + 1} / ${trials.length}`;
  document.getElementById("progress-fill").style.width = `${(currTrialIdx / trials.length) * 100}%`;

  const fix = document.getElementById("fixation-cross");
  const img = document.getElementById("trial-img");
  const btnNF = document.getElementById("btn-not-found");
  fix.classList.remove("hidden");
  img.classList.add("hidden");
  btnNF.classList.add("hidden");
  canClick = false;

  setTimeout(() => {
    img.src = t.image_b64;
    fix.classList.add("hidden");
    img.classList.remove("hidden");
    btnNF.classList.remove("hidden");
    trialStartMs = performance.now();
    canClick = true;
  }, 550);
}

document.getElementById("trial-img").addEventListener("click", (ev) => {
  if (!canClick) return;
  canClick = false;
  const rtMs = Math.round(performance.now() - trialStartMs);
  const rect = ev.target.getBoundingClientRect();
  const nx = (ev.clientX - rect.left) / rect.width;
  const ny = (ev.clientY - rect.top) / rect.height;

  const t = trials[currTrialIdx];
  const [x1, y1, x2, y2] = t.target_box;
  const pad = 0.045; // fair 4.5% finger/mouse click margin around small knife bbox
  const isHit = (nx >= x1 - pad && nx <= x2 + pad && ny >= y1 - pad && ny <= y2 + pad);

  trialRecords.push({
    trial_order: currTrialIdx + 1,
    trial_key: t.trial_key,
    scene_id: t.scene_id,
    condition: t.condition,
    target_label: t.target_label,
    rt_ms: rtMs,
    click_xy: [Number(nx.toFixed(4)), Number(ny.toFixed(4))],
    hit: isHit,
    gave_up: false
  });

  // Brief visual dot feedback
  const dot = document.createElement("div");
  dot.className = "feedback-dot";
  dot.style.left = `${nx * 100}%`;
  dot.style.top = `${ny * 100}%`;
  dot.style.background = isHit ? "#22c55e" : "#f43f5e";
  document.getElementById("stage-box").appendChild(dot);
  setTimeout(() => { dot.remove(); currTrialIdx++; runNextTrial(); }, 320);
});

document.getElementById("btn-not-found").addEventListener("click", () => {
  if (!canClick) return;
  canClick = false;
  const rtMs = Math.round(performance.now() - trialStartMs);
  const t = trials[currTrialIdx];
  trialRecords.push({
    trial_order: currTrialIdx + 1,
    trial_key: t.trial_key,
    scene_id: t.scene_id,
    condition: t.condition,
    target_label: t.target_label,
    rt_ms: rtMs,
    click_xy: null,
    hit: false,
    gave_up: true
  });
  currTrialIdx++;
  runNextTrial();
});

function renderTlxStep() {
  const cm = COND_META[tlxStep];
  document.getElementById("tlx-title").textContent = `NASA-TLX (${tlxStep + 1} / 3): ${cm.title}`;
  document.getElementById("tlx-desc").textContent = cm.desc;
  const sample = TRIAL_POOL.find(x => x.condition === cm.key);
  document.getElementById("tlx-preview").src = sample.image_b64;

  const container = document.getElementById("tlx-sliders");
  container.innerHTML = "";
  TLX_DIMS.forEach(d => {
    const div = document.createElement("div");
    div.className = "tlx-item";
    div.innerHTML = `
      <label><span>${d.name}</span><span id="val-${d.id}">50</span></label>
      <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:6px;">${d.desc}</div>
      <input type="range" min="0" max="100" step="5" value="50" id="sl-${d.id}"
        oninput="document.getElementById('val-${d.id}').textContent = this.value" />
      <div class="tlx-hint"><span>0 (Rất thấp / Rất dễ)</span><span>100 (Rất cao / Rất căng)</span></div>
    `;
    container.appendChild(div);
  });
}

document.getElementById("btn-tlx-next").addEventListener("click", () => {
  const cm = COND_META[tlxStep];
  const vals = {};
  let sum = 0;
  TLX_DIMS.forEach(d => {
    const v = Number(document.getElementById(`sl-${d.id}`).value);
    vals[d.id] = v;
    sum += v;
  });
  tlxScores[cm.key] = { subscales: vals, mean_tlx: Number((sum / TLX_DIMS.length).toFixed(1)) };
  tlxStep++;
  if (tlxStep < COND_META.length) {
    renderTlxStep();
  } else {
    showSummary();
  }
});

function showSummary() {
  document.getElementById("screen-tlx").classList.add("hidden");
  document.getElementById("screen-done").classList.remove("hidden");
  const tbody = document.getElementById("summary-tbody");
  tbody.innerHTML = "";
  const summary = {};
  COND_META.forEach(cm => {
    const recs = trialRecords.filter(r => r.condition === cm.key);
    const meanRt = Math.round(recs.reduce((s, r) => s + r.rt_ms, 0) / Math.max(1, recs.length));
    const acc = Number(((recs.filter(r => r.hit).length / Math.max(1, recs.length)) * 100).toFixed(1));
    const tlx = tlxScores[cm.key].mean_tlx;
    summary[cm.key] = { mean_rt_ms: meanRt, accuracy_pct: acc, nasa_tlx: tlx };
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><strong>${cm.key.toUpperCase()}</strong></td><td>${meanRt} ms</td><td>${acc}%</td><td>${tlx} / 100</td>`;
    tbody.appendChild(tr);
  });

  const payload = {
    meta: participantMeta,
    summary,
    nasa_tlx: tlxScores,
    trials: trialRecords
  };

  document.getElementById("btn-download").onclick = () => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `aegis_study_${participantMeta.participant_id}.json`;
    a.click();
  };
}
</script>
</body>
</html>
"""


def main() -> None:
    print("Building Human-in-the-Loop Cognitive Load & Reaction-Time Study App from REAL Meta Aria frames...")
    trials = extract_real_trials(num_scenes=4)
    out_path = Path("user_study/aegis_human_eval.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__TRIALS_JSON__", json.dumps(trials))
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved self-contained study app ({len(trials)} randomized real Meta Aria trials, {out_path.stat().st_size/1024:.1f} KB) to:")
    print(f"  {out_path.resolve()}")


if __name__ == "__main__":
    main()
