// popup.js v3.0 – ViDetect
const $ = id => document.getElementById(id);

const TEXT_LABELS = {
  0:{name:'Bình thường',cls:'clean',    icon:'✅',desc:'Không vi phạm'},
  1:{name:'Xúc phạm',  cls:'offensive',icon:'⚠️',desc:'Ngôn từ xúc phạm'},
  2:{name:'Thù ghét',  cls:'hate',     icon:'🚨',desc:'Ngôn ngữ thù ghét'},
};

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    $('tab-' + tab.dataset.tab).classList.add('active');
    if (tab.dataset.tab === 'stats') loadStats();
    if (tab.dataset.tab === 'video') refreshVideoStatus();
  });
});

// ── Status ────────────────────────────────────────────────────────────────────
async function initStatus() {
  const { apiUrl = 'http://localhost:8000' } = await chrome.storage.sync.get('apiUrl');
  try {
    const res = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(3000) });
    $('statusDot').classList.toggle('offline', !res.ok);
  } catch { $('statusDot').classList.add('offline'); }
}

// ════════════════════════════════════════════════════════════════
// TEXT
// ════════════════════════════════════════════════════════════════
$('inputText').addEventListener('input', () => {
  const l = $('inputText').value.length;
  $('charCount').textContent = l;
  $('charCount').style.color = l>450?'var(--hate)':l>350?'var(--offensive)':'var(--muted)';
});

$('analyzeBtn').onclick = async () => {
  const text = $('inputText').value.trim();
  if (!text) return;
  setLoad('analyzeBtn','btnText','btnSpinner', true);
  hideErr('apiError'); $('resultCard').classList.remove('show');
  const { apiUrl = 'http://localhost:8000' } = await chrome.storage.sync.get('apiUrl');
  try {
    const res  = await fetch(`${apiUrl}/predict`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text}), signal:AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(`Server lỗi ${res.status}`);
    const data = await res.json();
    showTextResult(data);
    await incStat('text', data.label_id);
  } catch(e) { showErr('apiError', `❌ ${e.message}`); }
  finally    { setLoad('analyzeBtn','btnText','btnSpinner', false); }
};

function showTextResult(d) {
  const info = TEXT_LABELS[d.label_id];
  $('resultHeader').className  = `result-header ${info.cls}`;
  $('resultIcon').textContent  = info.icon;
  $('resultLabel').textContent = info.name;
  $('resultLabel').className   = `result-label ${info.cls}`;
  $('resultDesc').textContent  = info.desc;
  $('resultConf').textContent  = `${(d.confidence*100).toFixed(1)}%`;
  $('resultConf').style.color  = `var(--${info.cls})`;
  [0,1,2].forEach(i => {
    const p = (d.probs[i]*100).toFixed(1);
    $(`bar${i}`).style.width = p+'%'; $(`pct${i}`).textContent = p+'%';
  });
  $('resultCard').classList.add('show');
}

// ════════════════════════════════════════════════════════════════
// VIDEO
// ════════════════════════════════════════════════════════════════
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'VIDEO_RESULT' && $('tab-video').classList.contains('active'))
    showVideoResult(msg.result);
  if (msg.type === 'IMAGE_SCAN_DONE' && $('tab-image').classList.contains('active'))
    showImageResult(msg);
});

async function refreshVideoStatus() {
  const [tab] = await chrome.tabs.query({active:true,currentWindow:true});
  if (!tab?.id) return;
  try {
    const res = await chrome.tabs.sendMessage(tab.id, {type:'GET_VIDEO_STATUS'});
    if (res?.result) showVideoResult(res.result);
    else if (res?.scanning) setVideoIdle('🔍','Đang quét video...','');
    else setVideoIdle('🎬','Chưa có video đang phát','Phát video trên Facebook, YouTube, TikTok');
  } catch { setVideoIdle('🎬','Chưa có video đang phát','Mở trang có video để bắt đầu'); }
}

function setVideoIdle(icon, title, sub) {
  $('videoStatusCard').innerHTML = `<div class="vstatus-idle">
    <div style="font-size:26px;margin-bottom:6px">${icon}</div>
    <div style="font-size:13px;font-weight:600;color:var(--text)">${title}</div>
    ${sub?`<div style="font-size:11px;color:var(--muted);margin-top:3px">${sub}</div>`:''}
  </div>`;
}

function showVideoResult(data) {
  if (!data) return;
  const isV = data.has_violence, isU = data.overall_label==='Uncertain';
  const cls  = isV?'violence':isU?'uncertain':'nonviolence';
  const icon = isV?'🚨':isU?'⚠️':'✅';
  const lbl  = isV?'Phát hiện Bạo lực':isU?'Không chắc chắn':'Bình thường';
  const conf = (data.overall_confidence*100).toFixed(1);
  const segs = data.segments||[];
  const tl   = segs.length
    ? segs.map(s=>`<div class="timeline-seg ${s.label==='Violence'?'violence':s.label==='Uncertain'?'uncertain':'nonviolence'}" title="${s.label} · ${(s.confidence*100).toFixed(0)}%"></div>`).join('')
    : `<div class="timeline-seg ${cls}" style="flex:1"></div>`;

  $('videoStatusCard').innerHTML = `
    <div class="result-header ${cls}" style="border-radius:8px 8px 0 0;padding:11px 13px">
      <span style="font-size:20px">${icon}</span>
      <div><div class="result-label ${cls}">${lbl}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:1px">Video đang phát</div></div>
      <span class="result-conf" style="color:var(--${cls})">${conf}%</span>
    </div>
    <div style="background:var(--bg3);border:1px solid var(--border);border-top:none;border-radius:0 0 8px 8px;padding:11px 13px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
        <div class="vsumm-mini"><div class="vsumm-val violence">${(data.violence_ratio*100).toFixed(1)}%</div><div class="vsumm-lbl">Tỷ lệ bạo lực</div></div>
        <div class="vsumm-mini"><div class="vsumm-val neutral">${segs.length||'—'}</div><div class="vsumm-lbl">Đoạn đã quét</div></div>
      </div>
      <div style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px">Timeline</div>
      <div class="timeline-bar">${tl}</div>
      <div class="timeline-legend" style="margin-top:6px">
        <span><span class="legend-dot" style="background:var(--hate)"></span>Bạo lực</span>
        <span><span class="legend-dot" style="background:var(--clean)"></span>Bình thường</span>
        <span><span class="legend-dot" style="background:var(--offensive)"></span>Không chắc</span>
      </div>
    </div>`;
}

$('scanNowBtn').onclick = async () => {
  const [tab] = await chrome.tabs.query({active:true,currentWindow:true});
  if (!tab?.id) return;
  setVideoIdle('🔍','Đang quét video...','');
  try {
    await chrome.tabs.sendMessage(tab.id, {type:'SCAN_VIDEO_NOW'});
    setTimeout(refreshVideoStatus, 5500);
  } catch { showErr('videoTabError','❌ Không gửi được lệnh – trang này có hỗ trợ không?'); }
};

// ════════════════════════════════════════════════════════════════
// IMAGE
// ════════════════════════════════════════════════════════════════
let sessionImgTotal = 0, sessionImgViolence = 0;

function showImageResult(msg) {
  sessionImgTotal   += msg.count || 0;
  sessionImgViolence += msg.violence_count || 0;
  $('statImgTotal').textContent   = sessionImgTotal;
  $('statImgViolence').textContent = sessionImgViolence;

  const safeCount = (msg.count||0) - (msg.violence_count||0);
  $('imageStatusCard').innerHTML = `
    <div style="padding:12px 14px">
      <div style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">Kết quả lần quét gần nhất</div>
      <div class="img-result-grid">
        <div class="img-stat-card"><div class="img-stat-val total">${msg.count||0}</div><div class="img-stat-lbl">Ảnh đã quét</div></div>
        <div class="img-stat-card"><div class="img-stat-val safe">${safeCount}</div><div class="img-stat-lbl">Bình thường</div></div>
        <div class="img-stat-card"><div class="img-stat-val danger">${msg.violence_count||0}</div><div class="img-stat-lbl">Bạo lực</div></div>
      </div>
      ${msg.violence_count > 0
        ? `<div style="margin-top:10px;padding:8px 10px;background:var(--hate-bg);border:1px solid rgba(244,63,94,.3);border-radius:6px;font-size:12px;color:var(--hate)">
             🚨 Đã blur ${msg.violence_count} ảnh bạo lực trên trang. Click badge đỏ để xem.
           </div>`
        : `<div style="margin-top:10px;padding:8px 10px;background:var(--clean-bg);border:1px solid rgba(34,211,160,.2);border-radius:6px;font-size:12px;color:var(--clean)">
             ✅ Không phát hiện ảnh bạo lực
           </div>`
      }
    </div>`;
}

$('scanImagesNowBtn').onclick = async () => {
  const [tab] = await chrome.tabs.query({active:true,currentWindow:true});
  if (!tab?.id) return;
  $('imageStatusCard').innerHTML = `<div class="vstatus-idle">
    <div style="font-size:26px;margin-bottom:6px">🔍</div>
    <div style="font-size:13px;font-weight:600;color:var(--accent2)">Đang quét ảnh...</div>
    <div style="font-size:11px;color:var(--muted);margin-top:3px">Đang thu thập và phân tích ảnh trên trang</div>
  </div>`;
  try {
    await chrome.tabs.sendMessage(tab.id, {type:'SCAN_IMAGES_NOW'});
  } catch { showErr('imageTabError','❌ Không gửi được lệnh – trang này có hỗ trợ không?'); }
};

// ════════════════════════════════════════════════════════════════
// STATS
// ════════════════════════════════════════════════════════════════
async function incStat(type, value) {
  if (type === 'text') {
    const keys = ['statClean','statOffensive','statHate'];
    const d = await chrome.storage.local.get(keys);
    if (keys[value] !== undefined) { d[keys[value]]=(d[keys[value]]||0)+1; await chrome.storage.local.set(d); }
  }
}
async function loadStats() {
  const d = await chrome.storage.local.get(['statClean','statOffensive','statHate','statVideoTotal','statVideoViolence']);
  const c=d.statClean||0,o=d.statOffensive||0,h=d.statHate||0;
  $('statTotal').textContent=c+o+h; $('statClean').textContent=c;
  $('statOffensive').textContent=o; $('statHate').textContent=h;
  $('statVideoTotal').textContent=d.statVideoTotal||0;
  $('statVideoViolence').textContent=d.statVideoViolence||0;
  $('statImgTotal').textContent=sessionImgTotal;
  $('statImgViolence').textContent=sessionImgViolence;
}
$('clearStats').onclick = async () => {
  await chrome.storage.local.set({statClean:0,statOffensive:0,statHate:0,statVideoTotal:0,statVideoViolence:0});
  sessionImgTotal=0; sessionImgViolence=0;
  loadStats();
};

// ════════════════════════════════════════════════════════════════
// SETTINGS
// ════════════════════════════════════════════════════════════════
async function loadSettings() {
  const d = await chrome.storage.sync.get(['apiUrl','autoScan','hideHate','autoScanVideo','autoScanImage','violenceThreshold']);
  $('apiUrl').value            = d.apiUrl            || 'http://localhost:8000';
  $('autoScan').checked        = d.autoScan          || false;
  $('hideHate').checked        = d.hideHate          || false;
  $('autoScanVideo').checked   = d.autoScanVideo     || false;
  $('autoScanImage').checked   = d.autoScanImage     || false;
  $('violenceThreshold').value = d.violenceThreshold || '0.70';
}
$('saveSettings').onclick = async () => {
  await chrome.storage.sync.set({
    apiUrl:            $('apiUrl').value.trim(),
    autoScan:          $('autoScan').checked,
    hideHate:          $('hideHate').checked,
    autoScanVideo:     $('autoScanVideo').checked,
    autoScanImage:     $('autoScanImage').checked,
    violenceThreshold: $('violenceThreshold').value.trim(),
  });
  $('saveSettings').textContent = '✓ Đã lưu!';
  setTimeout(() => $('saveSettings').textContent = '💾 Lưu cài đặt', 1500);
  const [tab] = await chrome.tabs.query({active:true,currentWindow:true});
  if (tab?.id) chrome.tabs.sendMessage(tab.id, {type:'SETTINGS_UPDATED'}).catch(()=>{});
  initStatus();
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoad(btnId, textId, spinnerId, on) {
  $(btnId).disabled=$(btnId).disabled||false; $(btnId).disabled=on;
  $(textId).style.display=on?'none':'block';
  $(spinnerId).style.display=on?'block':'none';
}
function showErr(id,msg){$(id).textContent=msg;$(id).classList.add('show');}
function hideErr(id){$(id).classList.remove('show');}

// ── Init ──────────────────────────────────────────────────────────────────────
loadSettings();
initStatus();
