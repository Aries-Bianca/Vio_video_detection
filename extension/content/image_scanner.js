// image_scanner.js v2 – Nhận diện ảnh bạo lực trên mạng xã hội
//
// VẤN ĐỀ CORS với ảnh:
//   Canvas.drawImage(imgEl) → SecurityError nếu ảnh là cross-origin (Facebook CDN,
//   YouTube CDN, TikTok CDN đều block cross-origin canvas read).
//
// GIẢI PHÁP:
//   Dùng background script làm proxy fetch. Background script chạy ở extension
//   context → có thể fetch bất kỳ URL nào (theo host_permissions trong manifest).
//   Flow: image_scanner gửi URL → background fetch → trả về base64 → ViT inference.
//   Cách này đảm bảo 100% lấy được ảnh từ mọi CDN mạng xã hội.

(function () {
'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const IMG_SIZE       = 224;    // Khớp VIT_IMG_SIZE notebook v3
const SCAN_DELAY_MS  = 1000;   // Debounce sau khi DOM thay đổi
const BATCH_SIZE     = 8;      // Số ảnh gửi mỗi batch
const MIN_DIM        = 120;    // Bỏ qua ảnh nhỏ hơn 120px (icon, avatar)

// ── State ─────────────────────────────────────────────────────────────────────
let contextAlive = true;
let settings     = { autoScanImage: false, violenceThreshold: 0.70 };
let apiUrl       = 'http://localhost:8000';
let scanTimer    = null;
let observer     = null;
const scanned    = new Set();   // URL đã xử lý trong session

// ── Extension guard ───────────────────────────────────────────────────────────
function alive() {
  try { return !!chrome.runtime?.id; }
  catch { return false; }
}

// ── Selectors theo nền tảng ──────────────────────────────────────────────────
// Selector chọn ảnh nội dung (post, feed, thumbnail) – không phải icon/avatar nhỏ.
function getSelectors() {
  const host = location.hostname;

  if (host.includes('facebook.com')) return [
    'div[role="main"] img[src*="fbcdn.net"]',
    'div[data-pagelet] img[src*="fbcdn"]',
    'img[src*="fbcdn"][width]',
  ];

  if (host.includes('youtube.com')) return [
    'ytd-thumbnail img[src*="ytimg.com"]',
    'img.yt-img-shadow[src*="ytimg"]',
    'ytd-rich-item-renderer img',
  ];

  if (host.includes('tiktok.com')) return [
    'img[src*="tiktokcdn"]',
    'img[src*="muscdn"]',
  ];

  if (host.includes('threads.net')) return [
    'img[src*="cdninstagram"]',
    'article img[src]',
  ];

  return [];
}

// ── Thu thập URL ảnh chưa quét ───────────────────────────────────────────────
function collectImageUrls() {
  const selectors = getSelectors();
  if (!selectors.length) return [];

  const urls = [];
  for (const sel of selectors) {
    for (const img of document.querySelectorAll(sel)) {
      // Bỏ qua: đã quét, chưa load, quá nhỏ, không có src
      if (img.dataset.vdScanned) continue;
      if (!img.complete) continue;
      if (img.naturalWidth < MIN_DIM || img.naturalHeight < MIN_DIM) continue;

      const url = img.currentSrc || img.src;
      if (!url || url.startsWith('data:') || scanned.has(url)) continue;

      img.dataset.vdScanned = '1';
      scanned.add(url);
      urls.push({ url, el: img });

      if (urls.length >= BATCH_SIZE) return urls;
    }
  }
  return urls;
}

// ── Fetch ảnh qua background (bypass CORS) ───────────────────────────────────
// Background script chạy ở extension context, có quyền fetch mọi URL
// được khai báo trong host_permissions. Đây là cách duy nhất đáng tin cậy
// để lấy ảnh cross-origin từ Facebook CDN, YouTube CDN, TikTok CDN.
async function fetchImageViaBackground(url) {
  if (!alive()) return null;
  try {
    const res = await chrome.runtime.sendMessage({
      type: 'FETCH_IMAGE',
      url,
    });
    return res?.b64 || null;
  } catch {
    return null;
  }
}

// ── Gửi batch lên server ─────────────────────────────────────────────────────
async function analyzeImages(b64List) {
  try {
    const res = await fetch(`${apiUrl}/predict_images`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        images:    b64List,
        threshold: settings.violenceThreshold,
      }),
      signal: AbortSignal.timeout(20000),
    });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

// ── Áp kết quả lên ảnh ───────────────────────────────────────────────────────
function applyBadge(imgEl, pred) {
  // Xoá badge cũ
  imgEl.parentElement?.querySelector('.vd-img-badge')?.remove();

  if (pred.label === 'non-violent') return;   // sạch → không cần badge

  const isV  = pred.label === 'violent';
  const badge = document.createElement('div');
  badge.className = `vd-img-badge ${isV ? 'violence' : 'uncertain'}`;
  badge.innerHTML = `
    <span>${isV ? '🚨' : '⚠️'}</span>
    <span>${isV ? 'Bạo lực' : 'Không chắc'}</span>
    <span class="vd-img-conf">${(pred.confidence * 100).toFixed(0)}%</span>`;
  badge.title = `ViDetect · ${pred.label} · ${(pred.confidence * 100).toFixed(1)}%`;

  const parent = imgEl.parentElement;
  if (parent) {
    if (getComputedStyle(parent).position === 'static') {
      parent.style.position = 'relative';
    }
    parent.appendChild(badge);
  }

  // Blur ảnh bạo lực confidence cao, click badge để bỏ blur
  if (isV && pred.confidence > 0.82) {
    imgEl.style.filter    = 'blur(10px)';
    imgEl.style.transition = 'filter 0.3s ease';
    badge.style.cursor    = 'pointer';
    badge.title += ' · Click để xem';
    badge.addEventListener('click', () => {
      imgEl.style.filter = 'none';
      badge.remove();
    }, { once: true });
  }
}

// ── Vòng quét chính ──────────────────────────────────────────────────────────
async function scanBatch() {
  if (!contextAlive) return;

  const candidates = collectImageUrls();
  if (!candidates.length) return;

  // Fetch tất cả ảnh song song qua background (bypass CORS)
  const fetched = await Promise.all(
    candidates.map(async ({ url, el }) => {
      const b64 = await fetchImageViaBackground(url);
      return b64 ? { b64, el } : null;
    })
  );

  const valid = fetched.filter(Boolean);
  if (!valid.length) return;

  const results = await analyzeImages(valid.map(v => v.b64));
  if (!results?.predictions) return;

  // Áp badge
  results.predictions.forEach((pred, i) => {
    if (valid[i]) applyBadge(valid[i].el, pred);
  });

  // Báo về background → popup
  if (alive()) {
    chrome.runtime.sendMessage({
      type:           'IMAGE_RESULT',
      count:          valid.length,
      violence_count: results.violence_count,
    }).catch(() => {});
  }
}

// ── Debounce ──────────────────────────────────────────────────────────────────
function scheduleScan() {
  clearTimeout(scanTimer);
  scanTimer = setTimeout(scanBatch, SCAN_DELAY_MS);
}

// ── MutationObserver ─────────────────────────────────────────────────────────
function startScan() {
  if (observer) return;
  observer = new MutationObserver(mutations => {
    if (!contextAlive) return;
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.tagName === 'IMG' || node.querySelector?.('img')) {
          scheduleScan();
          return;
        }
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  scanBatch();  // quét ngay ảnh hiện có
}

function stopScan() {
  observer?.disconnect();
  observer = null;
  clearTimeout(scanTimer);
}

// ── Settings ──────────────────────────────────────────────────────────────────
async function loadSettings() {
  if (!alive()) return;
  try {
    const d = await chrome.storage.sync.get(['autoScanImage', 'violenceThreshold', 'apiUrl']);
    settings.autoScanImage     = d.autoScanImage ?? false;
    settings.violenceThreshold = parseFloat(d.violenceThreshold) || 0.70;
    apiUrl = d.apiUrl || 'http://localhost:8000';
  } catch { return; }

  settings.autoScanImage ? startScan() : stopScan();
}

// ── Messages ──────────────────────────────────────────────────────────────────
if (alive()) {
  chrome.runtime.onMessage.addListener((msg) => {
    if (!contextAlive) return;
    if (msg.type === 'SETTINGS_UPDATED') loadSettings();
    if (msg.type === 'SCAN_IMAGES_NOW') {
      scanned.clear();
      document.querySelectorAll('[data-vd-scanned]').forEach(el => delete el.dataset.vdScanned);
      scanBatch();
    }
  });
}

// ── SPA navigation ────────────────────────────────────────────────────────────
let lastHref = location.href;
new MutationObserver(() => {
  if (!contextAlive) return;
  if (location.href !== lastHref) {
    lastHref = location.href;
    scanned.clear();
    document.querySelectorAll('[data-vd-scanned]').forEach(el => delete el.dataset.vdScanned);
    if (settings.autoScanImage) setTimeout(scanBatch, 2000);
  }
}).observe(document.documentElement, { childList: true, subtree: true });

// ── Init ──────────────────────────────────────────────────────────────────────
loadSettings();

})();
