// background.js v3 – Service Worker
// Vai trò chính:
//   1. FETCH_IMAGE: fetch ảnh từ CDN mạng xã hội (bypass CORS cho content script)
//   2. PREDICT relay: forward text predict từ content.js
//   3. VIDEO_RESULT / IMAGE_RESULT: relay kết quả lên popup
//   4. Context menu: kiểm tra text được bôi đen

// ── FETCH_IMAGE – đây là lý do tồn tại của background script ─────────────────
// Content script bị giới hạn bởi CORS của trang web.
// Background script chạy ở extension context, fetch theo host_permissions trong manifest,
// không bị CORS của trang áp dụng → có thể tải ảnh từ mọi CDN mạng xã hội.
async function fetchImageAsBase64(url) {
  try {
    const res = await fetch(url, {
      // Không set Origin header để tránh trigger CORS preflight
      mode: 'no-cors',
    });

    // no-cors trả về opaque response – không đọc được body
    // Dùng cors mode nhưng không set credentials
    throw new Error('retry_with_cors');
  } catch {
    // Thử lại với cors mode (hoạt động với YouTube, TikTok CDN)
    try {
      const res = await fetch(url, { credentials: 'omit' });
      if (!res.ok) return null;
      const buf  = await res.arrayBuffer();
      const b64  = btoa(String.fromCharCode(...new Uint8Array(buf)));
      const mime = res.headers.get('content-type') || 'image/jpeg';
      return `data:${mime};base64,${b64}`;
    } catch {
      return null;
    }
  }
}

// ── Context menu ──────────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       'analyzeText',
    title:    '🛡️ Kiểm tra ngôn ngữ thù ghét',
    contexts: ['selection'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== 'analyzeText') return;
  const text = info.selectionText?.trim();
  if (!text || !tab?.id) return;

  const { apiUrl = 'http://localhost:8000' } = await chrome.storage.sync.get('apiUrl');
  try {
    const res  = await fetch(`${apiUrl}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();
    chrome.tabs.sendMessage(tab.id, {
      type:   'SHOW_RESULT',
      result: data,
      text:   text.slice(0, 60) + (text.length > 60 ? '…' : ''),
    });
  } catch {
    chrome.tabs.sendMessage(tab.id, {
      type:    'SHOW_ERROR',
      message: 'Không kết nối được API server',
    });
  }
});

// ── Message handler ───────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  // ── FETCH_IMAGE: content script yêu cầu fetch ảnh qua background ─────────
  if (msg.type === 'FETCH_IMAGE') {
    fetchImageAsBase64(msg.url)
      .then(b64 => sendResponse({ b64 }))
      .catch(()  => sendResponse({ b64: null }));
    return true;  // async
  }

  // ── PREDICT: relay text hate speech từ content.js ─────────────────────────
  if (msg.type === 'PREDICT') {
    chrome.storage.sync.get('apiUrl').then(({ apiUrl = 'http://localhost:8000' }) =>
      fetch(`${apiUrl}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text: msg.text }),
      })
        .then(r => r.json())
        .then(d  => sendResponse({ ok: true, data: d }))
        .catch(e => sendResponse({ ok: false, error: e.message }))
    );
    return true;
  }

  // ── VIDEO_RESULT: relay từ video_scanner → popup ──────────────────────────
  if (msg.type === 'VIDEO_RESULT') {
    // Broadcast lên popup (nếu đang mở)
    chrome.runtime.sendMessage({ type: 'VIDEO_RESULT', result: msg.result }).catch(() => {});
    // Cập nhật stats
    chrome.storage.local.get(['statVideoTotal', 'statVideoViolence']).then(d =>
      chrome.storage.local.set({
        statVideoTotal:    (d.statVideoTotal    || 0) + 1,
        statVideoViolence: (d.statVideoViolence || 0) + (msg.result?.has_violence ? 1 : 0),
      })
    );
    return false;
  }

  // ── IMAGE_RESULT: relay từ image_scanner → popup ──────────────────────────
  if (msg.type === 'IMAGE_RESULT') {
    chrome.runtime.sendMessage({
      type:           'IMAGE_SCAN_DONE',
      count:          msg.count          || 0,
      violence_count: msg.violence_count || 0,
    }).catch(() => {});
    return false;
  }
});
