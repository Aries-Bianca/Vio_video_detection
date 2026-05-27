// video_scanner.js v2 – Capture frame từ <video> trên mạng xã hội
// Giải pháp CORS: <video> element là cross-origin nhưng Canvas.drawImage() hoạt động
// vì video đã được trình duyệt load vào context của trang (không phải extension).
// Điều này khác với fetch() – Canvas API được phép đọc pixel của video đang phát.

(function () {
'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const SCAN_INTERVAL_MS  = 3000;   // Quét mỗi 3 giây
const SEQUENCE_LENGTH   = 10;     // Giảm từ 20 → 10: khớp VIDEO_SEQ_LEN backend
const FRAME_W           = 112;    // Khớp VIDEO_IMAGE_W notebook
const FRAME_H           = 112;    // Khớp VIDEO_IMAGE_H notebook
const FRAME_INTERVAL_MS = 120;    // 10 frame × 120ms ≈ 1.2 giây capture window

// ── State ─────────────────────────────────────────────────────────────────────
let activeVideo    = null;
let intervalId     = null;
let isScanning     = false;
let contextAlive   = true;
let settings       = { autoScanVideo: false, violenceThreshold: 0.70 };
let apiUrl         = 'http://localhost:8000';
const overlayMap   = new WeakMap();

// Canvas dùng riêng cho video – không share với image_scanner
const videoCanvas  = document.createElement('canvas');
videoCanvas.width  = FRAME_W;
videoCanvas.height = FRAME_H;
const videoCtx     = videoCanvas.getContext('2d', { willReadFrequently: true });

// ── Extension guard ───────────────────────────────────────────────────────────
function alive() {
  try { return !!chrome.runtime?.id; }
  catch { return false; }
}

function safeMsg(msg) {
  if (!alive()) { contextAlive = false; stop(); return Promise.resolve(null); }
  return chrome.runtime.sendMessage(msg).catch(e => {
    if (e.message?.includes('Extension context')) { contextAlive = false; stop(); }
    return null;
  });
}

// ── Tìm video đang phát ───────────────────────────────────────────────────────
// Canvas.drawImage() hoạt động được với video cross-origin vì video được trình
// duyệt load vào DOM của trang – không phải fetched bởi extension.
// Tuy nhiên nếu video có attribute crossorigin="anonymous" và server không set
// CORS header, canvas sẽ bị "tainted". Ta xử lý bằng try/catch trong captureFrame().
function findBestVideo() {
  const all = [...document.querySelectorAll('video')];
  if (!all.length) return null;

  // Ưu tiên: đang phát, có kích thước thực, không phải preview nhỏ
  const playing = all.filter(v =>
    !v.paused &&
    !v.ended &&
    v.readyState >= 2 &&
    v.videoWidth  >= 200 &&
    v.videoHeight >= 200
  );

  const pool = playing.length ? playing : all.filter(v =>
    v.videoWidth >= 200 && v.videoHeight >= 200
  );

  if (!pool.length) return null;

  // Lấy video diện tích lớn nhất (video chính, không phải preview)
  return pool.reduce((best, v) =>
    (v.videoWidth * v.videoHeight) > (best.videoWidth * best.videoHeight) ? v : best
  );
}

// ── Capture một frame từ video ────────────────────────────────────────────────
function captureFrame(videoEl) {
  try {
    // drawImage() hoạt động khi:
    // 1. Video cùng origin với trang (Facebook/YouTube self-hosted)
    // 2. Video từ CDN không có crossorigin attribute → browser không enforce CORS
    // 3. Video blob: URL (stream) → luôn cho phép
    videoCtx.drawImage(videoEl, 0, 0, FRAME_W, FRAME_H);
    return videoCanvas.toDataURL('image/jpeg', 0.75);
  } catch (e) {
    // Trường hợp canvas bị "tainted" (video có crossorigin + server block CORS)
    // Xảy ra hiếm vì hầu hết CDN không set crossorigin attribute trên <video>
    return null;
  }
}

// ── Thu thập SEQUENCE_LENGTH frames liên tiếp ────────────────────────────────
async function captureSequence(videoEl) {
  const frames = [];

  for (let i = 0; i < SEQUENCE_LENGTH; i++) {
    // Kiểm tra video vẫn đang chạy
    if (videoEl.paused || videoEl.ended || videoEl.readyState < 2) break;

    const frame = captureFrame(videoEl);
    if (frame) {
      frames.push(frame);
    } else if (frames.length === 0 && i === 0) {
      // Frame đầu tiên đã lỗi → video bị taint hoàn toàn, dừng lại
      break;
    }
    // Đợi FRAME_INTERVAL_MS để lấy frame ở thời điểm khác nhau
    await new Promise(r => setTimeout(r, FRAME_INTERVAL_MS));
  }

  return frames;
}

// ── Gửi frames về server ─────────────────────────────────────────────────────
async function sendFrames(frames) {
  try {
    const res = await fetch(`${apiUrl}/predict_frames`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frames, threshold: settings.violenceThreshold }),
      signal:  AbortSignal.timeout(20000),
    });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

// ── Overlay ───────────────────────────────────────────────────────────────────
function getOverlay(videoEl) {
  if (overlayMap.has(videoEl)) return overlayMap.get(videoEl);

  const overlay = document.createElement('div');
  overlay.className = 'vd-video-overlay';
  overlay.innerHTML = `
    <div class="vd-video-badge scanning">
      <span class="vd-dot"></span>
      <span class="vd-label">Đang quét...</span>
      <span class="vd-conf"></span>
    </div>`;

  const parent = videoEl.parentElement;
  if (parent) {
    const pos = getComputedStyle(parent).position;
    if (pos === 'static') parent.style.position = 'relative';
    parent.appendChild(overlay);
  }

  overlayMap.set(videoEl, overlay);
  return overlay;
}

function updateOverlay(videoEl, result) {
  const overlay = getOverlay(videoEl);
  const badge   = overlay.querySelector('.vd-video-badge');
  const dot     = overlay.querySelector('.vd-dot');
  const lbl     = overlay.querySelector('.vd-label');
  const conf    = overlay.querySelector('.vd-conf');

  if (!result) {
    badge.className = 'vd-video-badge scanning';
    dot.className   = 'vd-dot';
    lbl.textContent = 'Đang quét...';
    conf.textContent = '';
    return;
  }

  const isV = result.has_violence;
  const isU = result.overall_label === 'Uncertain';
  const cls = isV ? 'violence' : isU ? 'uncertain' : 'safe';

  badge.className  = `vd-video-badge ${cls}`;
  dot.className    = `vd-dot ${cls}`;
  lbl.textContent  = isV ? '🚨 Bạo lực' : isU ? '⚠️ Không chắc' : '✅ Bình thường';
  conf.textContent = `${(result.overall_confidence * 100).toFixed(0)}%`;
}

function removeOverlay(videoEl) {
  if (!overlayMap.has(videoEl)) return;
  overlayMap.get(videoEl)?.remove();
  overlayMap.delete(videoEl);
}

// ── Vòng quét chính ───────────────────────────────────────────────────────────
async function scanTick() {
  if (isScanning || !contextAlive) return;
  isScanning = true;

  try {
    const video = findBestVideo();

    if (!video) {
      if (activeVideo) { removeOverlay(activeVideo); activeVideo = null; }
      return;
    }

    // Video mới xuất hiện
    if (video !== activeVideo) {
      if (activeVideo) removeOverlay(activeVideo);
      activeVideo = video;
    }

    updateOverlay(video, null); // "Đang quét..."

    const frames = await captureSequence(video);

    // Không đủ frame tối thiểu (video bị taint hoặc quá ngắn)
    if (frames.length < Math.ceil(SEQUENCE_LENGTH * 0.5)) {
      removeOverlay(video);
      return;
    }

    const result = await sendFrames(frames);
    updateOverlay(video, result);

    // Gửi kết quả về background → popup
    if (result && alive()) {
      safeMsg({ type: 'VIDEO_RESULT', result });
    }

  } finally {
    isScanning = false;
  }
}

// ── Start / Stop ──────────────────────────────────────────────────────────────
function start() {
  if (intervalId) return;
  intervalId = setInterval(scanTick, SCAN_INTERVAL_MS);
  scanTick();
}

function stop() {
  clearInterval(intervalId);
  intervalId = null;
  if (activeVideo) { removeOverlay(activeVideo); activeVideo = null; }
}

// ── Settings ──────────────────────────────────────────────────────────────────
async function loadSettings() {
  if (!alive()) return;
  try {
    const d = await chrome.storage.sync.get(['autoScanVideo', 'violenceThreshold', 'apiUrl']);
    settings.autoScanVideo     = d.autoScanVideo ?? false;
    settings.violenceThreshold = parseFloat(d.violenceThreshold) || 0.70;
    apiUrl = d.apiUrl || 'http://localhost:8000';
  } catch { return; }

  settings.autoScanVideo ? start() : stop();
}

// ── Messages ──────────────────────────────────────────────────────────────────
if (alive()) {
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!contextAlive) return;

    if (msg.type === 'SETTINGS_UPDATED') { loadSettings(); return; }

    if (msg.type === 'SCAN_VIDEO_NOW') {
      scanTick();
      return;
    }

    if (msg.type === 'GET_VIDEO_STATUS') {
      sendResponse({
        hasVideo: !!findBestVideo(),
        scanning: isScanning,
      });
      return true;
    }
  });
}

// ── SPA navigation ────────────────────────────────────────────────────────────
let lastHref = location.href;
new MutationObserver(() => {
  if (!contextAlive) return;
  if (location.href !== lastHref) {
    lastHref = location.href;
    stop();
    if (settings.autoScanVideo) setTimeout(start, 2500);
  }
}).observe(document.documentElement, { childList: true, subtree: true });

// ── Init ──────────────────────────────────────────────────────────────────────
loadSettings();

})();
