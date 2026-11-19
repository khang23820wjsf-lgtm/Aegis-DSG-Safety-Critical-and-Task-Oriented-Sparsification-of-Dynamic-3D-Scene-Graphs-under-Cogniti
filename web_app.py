"""Aegis-DSG Unified Local Web Simulator Hub (Zero Extra Web Dependencies).

Run:
    python web_app.py
Then open http://127.0.0.1:8000 in Chrome/Edge.

Features 3 interactive tabs:
  1. Real-Photo 3D Perception Simulator (Upload any room photo -> YOLOv8 + MiDaS + Aegis-v2)
  2. Phase B 60 FPS Closed-Loop Infinite World (Coarsening + DWA + Clio*/D-Lite*)
  3. MuJoCo 4.0 x Real 3DSSG 360° Rigid-Body 3D Viewer
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np

from run_real_photo_aegis import (
    build_3d_scene_graph_from_photo,
    estimate_depth_map,
    visualize_real_photo_aegis,
)


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
PHOTOS_DIR = ROOT_DIR / "data" / "real_photos"

_YOLO_CACHE = None


def get_yolo():
    global _YOLO_CACHE
    if _YOLO_CACHE is None:
        from ultralytics import YOLO
        _YOLO_CACHE = YOLO("yolov8n.pt")
    return _YOLO_CACHE


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Aegis-DSG — Web Giả Lập Nhận Thức Không Gian 3D</title>
<style>
  * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  nav { background: #1e293b; border-bottom: 1px solid #334155; padding: 0 18px; display: flex; align-items: center; justify-content: space-between; height: 52px; flex-shrink: 0; }
  .brand { font-weight: 800; font-size: 15px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
  .brand span { background: rgba(56,189,248,0.18); color: #7dd3fc; font-size: 11px; padding: 2px 8px; border-radius: 10px; }
  .tabs { display: flex; gap: 6px; }
  .tab-btn { background: transparent; color: #94a3b8; border: 1px solid transparent; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }
  .tab-btn:hover { color: #f8fafc; background: #334155; }
  .tab-btn.active { background: #0284c7; color: white; }
  .tab-panel { flex: 1; display: none; min-height: 0; overflow: hidden; }
  .tab-panel.active { display: flex; flex-direction: column; }
  iframe { flex: 1; width: 100%; height: 100%; border: none; background: #0b1120; }

  /* Tab 1: Real Photo Lab */
  .photo-lab { flex: 1; display: grid; grid-template-columns: 320px 1fr; gap: 12px; padding: 12px; min-height: 0; overflow: hidden; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
  .card h2 { margin: 0; font-size: 14px; color: #f8fafc; }
  .dropzone { border: 2px dashed #475569; border-radius: 8px; padding: 18px 12px; text-align: center; cursor: pointer; background: #0f172a; transition: 0.15s; }
  .dropzone:hover { border-color: #38bdf8; background: #132036; }
  .sample-btns { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  button.btn { background: #0284c7; color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }
  button.btn:hover { background: #0369a1; }
  button.btn:disabled { opacity: 0.5; cursor: wait; }
  button.sample { background: #334155; color: #e2e8f0; border: 1px solid #475569; padding: 6px 8px; border-radius: 5px; cursor: pointer; font-size: 12px; }
  button.sample:hover { background: #475569; }
  .slider-row { display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; }
  .slider-row input { width: 100%; accent-color: #38bdf8; }
  .result-stage { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: auto; position: relative; }
  .result-stage img { max-width: 100%; max-height: 100%; border-radius: 6px; box-shadow: 0 10px 25px rgba(0,0,0,0.45); }
  .status-bar { font-size: 12px; color: #94a3b8; background: #0f172a; padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; line-height: 1.5; }
</style>
</head>
<body>
<nav>
  <div class="brand">AEGIS-DSG SIMULATOR <span>PHASE A + B + REAL 3D</span></div>
  <div class="tabs">
    <button class="tab-btn active" onclick="openTab('photo', this)">📸 Ảnh Phòng Thật (YOLOv8 + MiDaS)</button>
    <button class="tab-btn" onclick="openTab('phaseb', this)">🎮 Đóng vòng 60 FPS (Coarsening)</button>
    <button class="tab-btn" onclick="openTab('mujoco', this)">🧊 Vật lý cứng 3D MuJoCo × 3DSSG</button>
  </div>
</nav>

<!-- TAB 1: Real Photo Upload & Live Aegis-v2 -->
<div id="tab-photo" class="tab-panel active">
  <div class="photo-lab">
    <div class="card">
      <h2>1. Chọn hoặc Tải Ảnh Phòng Thật</h2>
      <div class="dropzone" onclick="document.getElementById('fileInput').click()">
        <div style="font-size:24px;">📷</div>
        <div style="font-size:13px; font-weight:600; margin-top:4px;">Bấm chọn ảnh từ máy mày</div>
        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Hỗ trợ JPG / PNG (phòng ngủ, hành lang, quán cafe...)</div>
        <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="onFilePicked(event)" />
      </div>

      <div style="font-size:12px; color:#94a3b8;">Hoặc test nhanh bằng ảnh mẫu có sẵn:</div>
      <div class="sample-btns">
        <button class="sample" onclick="runSample('cluttered_office.jpg')">Phòng khách</button>
        <button class="sample" onclick="runSample('cafe_tables.jpg')">Ảnh mẫu 2</button>
        <button class="sample" onclick="runSample('living_room.jpg')">Ảnh mẫu 3</button>
        <button class="sample" onclick="runSample('room_chairs.jpg')">Ảnh mẫu 4</button>
      </div>

      <h2>2. Tham số Aegis-v2</h2>
      <div class="slider-row">
        <label>Ngân sách nhận thức tối đa (k): <b id="kVal">5</b> thực thể</label>
        <input type="range" id="kSlider" min="1" max="10" value="5" oninput="document.getElementById('kVal').textContent=this.value" />
      </div>
      <div class="slider-row">
        <label>Ngưỡng sàn loại rác (τ_min): <b id="tauVal">0.20</b></label>
        <input type="range" id="tauSlider" min="0.05" max="0.50" step="0.05" value="0.20" oninput="document.getElementById('tauVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>

      <button id="runBtn" class="btn" onclick="runCurrent()">🚀 Chạy Aegis-v2 trên ảnh này</button>

      <div id="statusBox" class="status-bar">
        Sẵn sàng! Bấm chọn ảnh từ máy mày hoặc bấm <b>Phòng khách</b> để xem YOLOv8 + MiDaS 3D + Aegis-v2 xử lý trực tiếp.
      </div>
    </div>

    <div class="result-stage">
      <img id="resultImg" src="/results/real_photo_aegis_v2.png" alt="Kết quả Aegis-v2 trên ảnh thật" />
    </div>
  </div>
</div>

<!-- TAB 2: Phase B 60 FPS Infinite Closed-Loop -->
<div id="tab-phaseb" class="tab-panel">
  <iframe src="/results/phase_b_demo.html"></iframe>
</div>

<!-- TAB 3: MuJoCo 4.0 x Real 3DSSG 3D Viewer -->
<div id="tab-mujoco" class="tab-panel">
  <iframe src="/results/mujoco_3dssg_viewer.html"></iframe>
</div>

<script>
let currentPayload = { sample: "cluttered_office.jpg" };

function openTab(id, btn) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + id).classList.add("active");
  btn.classList.add("active");
}

function onFilePicked(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    currentPayload = { image_b64: reader.result, filename: file.name };
    runCurrent();
  };
  reader.readAsDataURL(file);
}

function runSample(name) {
  currentPayload = { sample: name };
  runCurrent();
}

async function runCurrent() {
  const btn = document.getElementById("runBtn");
  const status = document.getElementById("statusBox");
  const k = parseInt(document.getElementById("kSlider").value, 10);
  const tau = parseFloat(document.getElementById("tauSlider").value);

  btn.disabled = true;
  status.innerHTML = "⏳ Đang chạy <b>YOLOv8 + MiDaS 3D Depth + Aegis-v2</b>... (mất khoảng 2–5 giây)";

  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...currentPayload, k, tau_min: tau }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Lỗi server");
    document.getElementById("resultImg").src = "data:image/png;base64," + data.image_b64;
    status.innerHTML = `✅ <b>Hoàn tất trong ${data.elapsed_s}s!</b><br/>• Phát hiện thô: <b>${data.raw_count}</b> vật thể<br/>• Đầu ra Aegis-v2: <b>k ≤ ${k}</b> (ngưỡng sàn S ≥ ${tau.toFixed(2)})`;
  } catch (err) {
    status.innerHTML = `❌ Lỗi: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class AegisHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, code: int, content_type: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_bytes(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return

        if path.startswith("/results/"):
            rel = path[len("/results/"):]
            target = (RESULTS_DIR / rel).resolve()
            if target.exists() and target.is_file() and RESULTS_DIR in target.parents:
                ext = target.suffix.lower()
                mime = {
                    ".html": "text/html; charset=utf-8",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".json": "application/json",
                }.get(ext, "application/octet-stream")
                self._send_bytes(200, mime, target.read_bytes())
                return

        self._send_bytes(404, "text/plain", b"Not Found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._send_bytes(404, "application/json", b'{"error":"Not Found"}')
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            k = int(body.get("k", 5))
            tau_min = float(body.get("tau_min", 0.20))

            t0 = time.perf_counter()
            if "image_b64" in body:
                raw_b64 = body["image_b64"].split(",")[-1]
                arr = np.frombuffer(base64.b64decode(raw_b64), dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                sample_name = Path(str(body.get("sample", "cluttered_office.jpg"))).name
                img_bgr = cv2.imread(str(PHOTOS_DIR / sample_name))

            if img_bgr is None:
                raise ValueError("Không đọc được file ảnh đầu vào.")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            yolo = get_yolo()
            depth_map = estimate_depth_map(img_rgb)
            g_obs, bboxes_2d = build_3d_scene_graph_from_photo(img_rgb, yolo, depth_map)
            raw_count = sum(1 for _, d in g_obs.nodes(data=True) if d.get("layer") == "object")

            tmp_out = RESULTS_DIR / "_web_live_out.png"
            visualize_real_photo_aegis(img_rgb, g_obs, bboxes_2d, k, tmp_out, tau_min=tau_min)
            png_b64 = base64.b64encode(tmp_out.read_bytes()).decode("ascii")

            resp = json.dumps({
                "image_b64": png_b64,
                "raw_count": raw_count,
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }).encode("utf-8")
            self._send_bytes(200, "application/json", resp)
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send_bytes(500, "application/json", err)

    def log_message(self, format: str, *args) -> None:
        pass  # Keep console clean


def main() -> None:
    host = "127.0.0.1"
    for port in (8765, 8888, 9090, 5050):
        try:
            server = ThreadingHTTPServer((host, port), AegisHandler)
            print(f"Aegis-DSG Web Simulator running at http://{host}:{port}", flush=True)
            server.serve_forever()
            return
        except OSError:
            continue
    raise RuntimeError("Could not bind any local port.")


if __name__ == "__main__":
    main()
