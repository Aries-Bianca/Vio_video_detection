// Content Script v3 – ViDetect
// 1. Quét bình luận thù ghét (PhoBERT)
// 2. Phân tích video đang phát trên trang (MoBiLSTM)

(function () {
  'use strict';

  let settings    = { autoScan: false, hideHate: false, violenceThreshold: 0.70 };
  let observer    = null;
  let scanQueue   = new Set();
  let scanning    = false;
  let contextAlive = true;

  // ── Context guard ──────────────────────────────────────────────────────────
  function isContextAlive() {
    try { return !!chrome.runtime?.id; } catch { return false; }
  }

  async function safeSendMessage(msg) {
    if (!isContextAlive()) { contextAlive = false; stopAll(); return null; }
    try {
      return await chrome.runtime.sendMessage(msg);
    } catch (err) {
      if (err.message?.includes('Extension context') || err.message?.includes('message port')) {
        contextAlive = false; stopAll();
      }
      return null;
    }
  }

  async function safeStorageGet(keys) {
    if (!isContextAlive()) return {};
    try { return await chrome.storage.sync.get(keys); } catch { return {}; }
  }

  function stopAll() { stopCommentObserver(); stopAllVideoScanners(); }

  // ═════════════════════════════════════════════════════════════════════════
  // PHẦN 1 – COMMENT SCANNER (giữ nguyên từ v2)
  // ═════════════════════════════════════════════════════════════════════════
  function getCommentSelectors() {
    const h = location.hostname;
    if (h.includes('facebook.com')) return ['[data-ad-comet-preview="message"]','[dir="auto"] > span'];
    if (h.includes('youtube.com'))  return ['#content-text.ytd-comment-renderer'];
    if (h.includes('tiktok.com'))   return ['[data-e2e="comment-level-1"] p'];
    return [];
  }

  async function scanComment(el) {
    if (!contextAlive || el.dataset.vihateScanned) return;
    el.dataset.vihateScanned = '1';
    const text = el.innerText?.trim();
    if (!text || text.length < 3 || text.length > 500) return;
    const res = await safeSendMessage({ type: 'PREDICT_TEXT', text });
    if (res?.ok) applyCommentLabel(el, res.data);
  }

  function applyCommentLabel(el, data) {
    if (data.label_id === 0) return;
    if (data.label_id === 2 && settings.hideHate) {
      const c = el.closest('[data-ad-comet-preview="message"]')?.parentElement?.parentElement || el.parentElement;
      c.style.display = 'none'; return;
    }
    if (el.parentElement?.querySelector('.vihate-badge')) return;
    const badge = document.createElement('span');
    badge.className = `vihate-badge vihate-${data.label_id === 1 ? 'offensive' : 'hate'}`;
    badge.title = `${data.label_id === 1 ? 'Xúc phạm' : 'Thù ghét'} · ${(data.confidence*100).toFixed(0)}%`;
    badge.textContent = data.label_id === 1 ? '⚠️ Xúc phạm' : '🚨 Thù ghét';
    el.after(badge);
  }

  function startCommentObserver() {
    if (observer) return;
    const sels = getCommentSelectors();
    if (!sels.length) return;
    observer = new MutationObserver(mutations => {
      for (const m of mutations)
        for (const node of m.addedNodes) {
          if (node.nodeType !== 1) continue;
          sels.forEach(s => {
            const els = node.matches?.(s) ? [node] : [...node.querySelectorAll(s)];
            els.forEach(el => scanQueue.add(el));
          });
        }
      processCommentQueue();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    sels.forEach(s => document.querySelectorAll(s).forEach(el => scanQueue.add(el)));
    processCommentQueue();
  }

  function stopCommentObserver() { observer?.disconnect(); observer = null; }

  async function processCommentQueue() {
    if (scanning || !scanQueue.size) return;
    scanning = true;
    for (const el of [...scanQueue]) {
      scanQueue.delete(el);
      await scanComment(el);
      await sleep(150);
    }
    scanning = false;
  }

  // ═════════════════════════════════════════════════════════════════════════
  // PHẦN 2 – VIDEO SCANNER
  // Chiến lược: tìm <video> trên trang, capture frame định kỳ bằng Canvas,
  // gom đủ SEQUENCE_LENGTH frame rồi gửi server phân tích.
  // ═════════════════════════════════════════════════════════════════════════
  const VIDEO_SEQ_LEN  = 20;     // phải khớp với model
  const IMG_SIZE       = 112;    // phải khớp với model
  const CAPTURE_MS     = 500;    // capture 1 frame mỗi 500ms → 20 frame = ~10 giây video
  const COOLDOWN_MS    = 15000;  // sau khi predict xong, chờ 15s trước khi predict lại

  // Map: videoElement → trạng thái scanner của video đó
  const videoScanners  = new Map();
  let videoObserver    = null;

  function startVideoScanner(video) {
    if (videoScanners.has(video)) return;
    if (video.readyState < 2) {
      video.addEventListener('canplay', () => startVideoScanner(video), { once: true });
      return;
    }

    const state = {
      frames: [],
      captureInterval: null,
      cooldown: false,
      overlay: null,
    };
    videoScanners.set(video, state);

    // Tạo overlay badge trên video
    state.overlay = createVideoOverlay(video);

    // Bắt đầu capture frame định kỳ khi video đang phát
    state.captureInterval = setInterval(() => {
      if (!contextAlive) { clearInterval(state.captureInterval); return; }
      if (video.paused || video.ended || state.cooldown) return;
      captureFrame(video, state);
    }, CAPTURE_MS);
  }

  function stopVideoScanner(video) {
    const state = videoScanners.get(video);
    if (!state) return;
    clearInterval(state.captureInterval);
    state.overlay?.remove();
    videoScanners.delete(video);
  }

  function stopAllVideoScanners() {
    videoScanners.forEach((_, video) => stopVideoScanner(video));
    videoObserver?.disconnect();
    videoObserver = null;
  }

  // Capture 1 frame từ <video> bằng Canvas → base64 JPEG
  function captureFrame(video, state) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width  = IMG_SIZE;
      canvas.height = IMG_SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
      const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
      state.frames.push(b64);

      // Cập nhật overlay: đang thu thập
      updateOverlay(state.overlay, 'collecting', state.frames.length, null);

      // Đủ frame → gửi phân tích
      if (state.frames.length >= VIDEO_SEQ_LEN) {
        const framesToSend = [...state.frames];
        state.frames = [];        // reset buffer
        state.cooldown = true;
        analyzeFrames(framesToSend, state, video);
      }
    } catch (e) {
      // CORS hoặc tainted canvas (video cross-origin) — phổ biến trên YouTube
      updateOverlay(state.overlay, 'cors', 0, null);
      clearInterval(state.captureInterval);
    }
  }

  async function analyzeFrames(frames, state, video) {
    updateOverlay(state.overlay, 'analyzing', VIDEO_SEQ_LEN, null);
    const res = await safeSendMessage({
      type: 'PREDICT_VIDEO_FRAMES',
      frames,
      threshold: settings.violenceThreshold,
    });

    if (res?.ok) {
      const data = res.data;
      updateOverlay(state.overlay, 'result', VIDEO_SEQ_LEN, data);
    } else {
      updateOverlay(state.overlay, 'error', 0, null);
    }

    // Cooldown trước khi collect lại
    await sleep(COOLDOWN_MS);
    state.cooldown = false;
  }

  // ── Overlay UI trên video ──────────────────────────────────────────────────
  function createVideoOverlay(video) {
    // Tìm container của video để đặt overlay
    const container = video.parentElement;
    if (!container) return null;

    const prev = container.querySelector('.vihate-video-overlay');
    if (prev) return prev;

    const overlay = document.createElement('div');
    overlay.className = 'vihate-video-overlay';
    overlay.innerHTML = `<span class="vihate-vo-badge">🛡️ ViDetect đang giám sát...</span>`;

    // Đảm bảo container có position để overlay đặt được
    const pos = getComputedStyle(container).position;
    if (pos === 'static') container.style.position = 'relative';
    container.appendChild(overlay);
    return overlay;
  }

  function updateOverlay(overlay, state, frameCount, data) {
    if (!overlay) return;
    const badge = overlay.querySelector('.vihate-vo-badge');
    if (!badge) return;

    badge.className = 'vihate-vo-badge';

    switch (state) {
      case 'collecting':
        badge.textContent = `🛡️ Thu thập ${frameCount}/${VIDEO_SEQ_LEN} frame...`;
        badge.classList.add('collecting');
        break;
      case 'analyzing':
        badge.textContent = '🔍 Đang phân tích...';
        badge.classList.add('analyzing');
        break;
      case 'result':
        if (data.has_violence) {
          badge.textContent = `🚨 Phát hiện bạo lực (${(data.violence_ratio*100).toFixed(0)}%)`;
          badge.classList.add('violence');
        } else {
          badge.textContent = `✅ Bình thường`;
          badge.classList.add('safe');
        }
        break;
      case 'cors':
        badge.textContent = '⚠️ Video được bảo vệ – không thể đọc frame';
        badge.classList.add('warn');
        break;
      case 'error':
        badge.textContent = '❌ Không kết nối được server';
        badge.classList.add('error');
        break;
    }
  }

  // ── Tìm tất cả <video> hiện có và mới thêm vào ───────────────────────────
  function attachToAllVideos() {
    document.querySelectorAll('video').forEach(v => startVideoScanner(v));
  }

  function startVideoObserver() {
    if (videoObserver) return;
    videoObserver = new MutationObserver(mutations => {
      for (const m of mutations)
        for (const node of m.addedNodes) {
          if (node.nodeType !== 1) continue;
          if (node.tagName === 'VIDEO') startVideoScanner(node);
          node.querySelectorAll?.('video').forEach(v => startVideoScanner(v));
        }
    });
    videoObserver.observe(document.body, { childList: true, subtree: true });
    attachToAllVideos();
  }

  // ═════════════════════════════════════════════════════════════════════════
  // SETTINGS & INIT
  // ═════════════════════════════════════════════════════════════════════════
  async function loadSettings() {
    const data = await safeStorageGet(['autoScan','hideHate','violenceThreshold','videoScan']);
    settings = {
      autoScan:           data.autoScan           ?? false,
      hideHate:           data.hideHate           ?? false,
      violenceThreshold:  parseFloat(data.violenceThreshold) || 0.70,
      videoScan:          data.videoScan          ?? true,
    };

    if (settings.autoScan) startCommentObserver(); else stopCommentObserver();
    if (settings.videoScan) startVideoObserver();  else stopAllVideoScanners();
  }

  // ── Toast notifications ───────────────────────────────────────────────────
  function showToast(result, text) {
    const existing = document.querySelector('.vihate-toast');
    if (existing) existing.remove();
    const colors = {0:'#22d3a0',1:'#f59e0b',2:'#f43f5e'};
    const names  = {0:'Bình thường',1:'Xúc phạm',2:'Thù ghét'};
    const icons  = {0:'✅',1:'⚠️',2:'🚨'};
    const toast  = document.createElement('div');
    toast.className = 'vihate-toast';
    toast.innerHTML = `
      <div class="vihate-toast-header" style="color:${colors[result.label_id]}">
        ${icons[result.label_id]} ${names[result.label_id]}
        <span style="margin-left:auto;font-size:11px;opacity:.7">${(result.confidence*100).toFixed(1)}%</span>
      </div>
      <div class="vihate-toast-text">"${text}"</div>`;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('visible'));
    setTimeout(() => { toast.classList.remove('visible'); setTimeout(() => toast.remove(), 300); }, 4000);
  }

  function showErrorToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'vihate-toast vihate-toast-error';
    toast.textContent = '❌ ' + msg;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('visible'));
    setTimeout(() => { toast.classList.remove('visible'); setTimeout(() => toast.remove(), 300); }, 3000);
  }

  // ── Message listener ───────────────────────────────────────────────────────
  if (isContextAlive()) {
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (!contextAlive) return;
      if (msg.type === 'SHOW_RESULT')      showToast(msg.result, msg.text);
      if (msg.type === 'SHOW_ERROR')       showErrorToast(msg.message);
      if (msg.type === 'SETTINGS_UPDATED') loadSettings();
      sendResponse({ ok: true });
      return true;
    });
  }

  // URL change (SPA)
  let lastUrl = location.href;
  new MutationObserver(() => {
    if (!contextAlive || location.href === lastUrl) return;
    lastUrl = location.href;
    document.querySelectorAll('[data-vihate-scanned]').forEach(el => delete el.dataset.vihateScanned);
    setTimeout(() => { if (settings.videoScan) attachToAllVideos(); }, 2000);
  }).observe(document.body, { childList: true, subtree: true });

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  loadSettings();
})();
