/* BroClips — shared page code: talking to the app, small helpers, toasts, confetti, and the header's live status pill
   (every page asks /api/state every 2 seconds). Needs icons.js first (ic(), hydrateIcons()). */
const $ = (sel, el) => (el || document).querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

/* Every POST is JSON (the app refuses anything else, so other websites cannot drive it). */
async function api(url, body) {
  const opt = body === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  try {
    const r = await fetch(url, opt);
    let j;
    try { j = await r.json(); } catch (e) { j = {ok: false, error: `BroClips gave a strange answer (${r.status}).`}; }
    if (!r.ok && j.ok === undefined) j.ok = false;
    return j;
  } catch (e) {
    return {ok: false, offline: true,
      error: 'BroClips is not running. Start it again with the BroClips shortcut on your Desktop.'};
  }
}

const LANGS = [['auto', 'Find out by itself'], ['ar', 'Arabic (العربية)'], ['en', 'English'], ['fr', 'French'],
  ['es', 'Spanish'], ['de', 'German'], ['tr', 'Turkish'], ['pt', 'Portuguese'], ['it', 'Italian'], ['ru', 'Russian'],
  ['hi', 'Hindi'], ['ur', 'Urdu'], ['id', 'Indonesian'], ['ja', 'Japanese'], ['ko', 'Korean'], ['zh', 'Chinese']];

/* <option>s for a select; a saved value this page does not know is kept as an option. */
function opts(list, cur) {
  if (cur && !list.some(([v]) => v === cur)) list = [[cur, cur], ...list];
  return list.map(([v, t]) => `<option value="${esc(v)}" ${v === cur ? 'selected' : ''}>${esc(t)}</option>`).join('');
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(+sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = String(sec % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

function fmtDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d) ? '' : d.toLocaleDateString(undefined, {day: 'numeric', month: 'short', year: 'numeric'});
}

/* Toasts: bottom-right, stacked, click to close. bad = an error (stays longer). */
function toast(msg, bad) {
  let box = $('#toasts');
  if (!box) {
    box = document.createElement('div');
    box.id = 'toasts'; box.setAttribute('role', 'status'); box.setAttribute('aria-live', 'polite');
    document.body.appendChild(box);
  }
  const t = document.createElement('div');
  t.className = 'toast' + (bad ? ' bad' : '');
  t.title = 'Click to close';
  t.innerHTML = `<span class="ti">${ic(bad ? 'triangle-alert' : 'check')}</span><span></span>`;
  t.lastChild.textContent = msg;
  const close = () => { if (!t.isConnected) return; t.classList.add('out'); setTimeout(() => t.remove(), 230); };
  t.onclick = close;
  box.appendChild(t);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(close, bad ? 10000 : 4500);
}

/* "Sure? Click again" for buttons that remove something. Returns true on the second click. */
function confirmClick(btn, text) {
  if (btn.dataset.sure) return true;
  btn.dataset.sure = '1';
  btn.dataset.old = btn.innerHTML;
  btn.classList.add('sure');
  btn.innerHTML = ic('triangle-alert') + `<span>${esc(text || 'Sure? Click again')}</span>`;
  setTimeout(() => {
    if (btn.isConnected && btn.dataset.sure) {
      delete btn.dataset.sure; btn.innerHTML = btn.dataset.old; btn.classList.remove('sure');
    }
  }, 4000);
  return false;
}

/* A one-time celebration (a tiny canvas, no library). Skipped when the PC asks for less motion. */
function confetti() {
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const c = document.createElement('canvas');
  c.className = 'confetti';
  document.body.appendChild(c);
  const g = c.getContext('2d'), k = Math.min(2, window.devicePixelRatio || 1);
  const W = c.width = innerWidth * k, H = c.height = innerHeight * k;
  const colors = ['#ff4d8d', '#8b5cf6', '#22d3ee', '#22c55e', '#f59e0b', '#ffffff', '#f472b6'];
  const P = [];
  for (let i = 0; i < 170; i++) {
    const left = i % 2 === 0;
    P.push({x: (left ? .08 : .92) * W, y: H * .62, vx: (left ? 1 : -1) * (3 + Math.random() * 10) * k,
      vy: (-11 - Math.random() * 11) * k, w: (6 + Math.random() * 7) * k, h: (9 + Math.random() * 8) * k,
      r: Math.random() * 6.3, vr: (Math.random() - .5) * .4, c: colors[i % colors.length], round: Math.random() < .3});
  }
  const t0 = performance.now();
  (function frame(t) {
    const life = (t - t0) / 1000;
    g.clearRect(0, 0, W, H);
    for (const p of P) {
      p.vy += .34 * k; p.vx *= .991; p.vy *= .991; p.x += p.vx; p.y += p.vy; p.r += p.vr;
      g.save();
      g.globalAlpha = Math.max(0, 1 - Math.max(0, life - 1.7) / 1.1);
      g.translate(p.x, p.y); g.rotate(p.r); g.fillStyle = p.c;
      if (p.round) { g.beginPath(); g.arc(0, 0, p.w / 2.2, 0, 6.29); g.fill(); }
      else g.fillRect(-p.w / 2, -p.h / 2, p.w, p.h * Math.max(.15, Math.abs(Math.cos(p.r * 1.7))));
      g.restore();
    }
    if (life < 2.9) requestAnimationFrame(frame); else c.remove();
  })(t0);
  setTimeout(() => c.remove(), 3400);       // also when the window is hidden (no animation frames then)
}

/* The status pill in the header: "Making “Lesson 3” · Listening 41 %", "Waiting for … to close", "Ready". */
const KIND_VERB = {make: 'Making', find_again: 'New moments for', short_edit: 'Re-making', assets: 'Downloading',
                   cutout: 'Cutting out', thumbai: 'AI thumbnails for'};
const STEP_SHORT = {1: 'Getting ready', 2: 'Listening', 3: 'Finding moments', 4: 'Cutting Shorts', 5: 'Long video',
                    6: 'Titles & thumbnails'};
const noEmoji = s => String(s || '').replace(/^[^\p{L}\p{N}“"«(]+/u, '').trim();

function workText(w) {
  const cur = w.current, n = w.queue || 0;
  if (cur) {
    const title = noEmoji(cur.title) || 'a job';
    if (cur.stopping) return `Stopping “${title}”…`;
    const verb = KIND_VERB[cur.kind] || 'Working on';
    const label = (cur.of === 6 && STEP_SHORT[cur.step]) || noEmoji(cur.label).replace(/[….\s]+$/, '') || 'Starting';
    return `${verb} “${title}” · ${label} ${cur.pct || 0} %` + (n ? ` · ${n} more waiting` : '');
  }
  if (w.waiting_for) return `Waiting until ${w.waiting_for} closes` + (n > 1 ? ` · ${n} jobs` : '');
  if (n) return 'Starting…';
  return 'Ready';
}

let STATE = null;
async function pollState() {
  const s = await api('/api/state');
  const w = $('#work'), pill = $('#statuspill');
  if (s.offline) {
    if (w) w.textContent = 'BroClips is not running — start it again from the Desktop shortcut.';
    if (pill) pill.className = 'status off';
    return;
  }
  STATE = s;
  if (w && s.work) {
    const wk = s.work, txt = workText(wk);
    if (w.textContent !== txt) w.textContent = txt;
    w.classList.toggle('busy', !!wk.busy);
    if (pill) {
      pill.className = 'status' + (wk.current ? ' busy' : wk.waiting_for ? ' wait' : wk.busy ? ' busy' : '');
      pill.style.setProperty('--p', wk.current ? (wk.current.pct || 0) : 0);
      pill.title = (wk.text || txt) + ' — the work line: BroClips does one job at a time, in this order.';
    }
  }
  const rn = $('#runnow');
  if (rn) rn.hidden = !(s.work && s.work.waiting_for);
  if (window.onState) window.onState(s);
}

async function runNow() {
  await api('/api/jobs/run_now', {});
  pollState();
}

async function openFolder(which, project, video) {
  const r = await api('/api/open_folder', project ? {project, video} : {which});
  if (!r.ok) toast(r.error, true);
}

/* Pictures fade in over a shimmer; a picture that cannot load shows the placeholder behind it. */
function imgIn(img) { img.classList.add('in'); if (img.parentNode) img.parentNode.classList.remove('skel'); }
function imgErr(img) { const f = img.parentNode; if (f) { f.classList.remove('skel'); f.classList.add('noimg'); } }

/* Tiles that work like radio buttons: the arrow keys move the choice. */
document.addEventListener('keydown', e => {
  const t = e.target, d = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[e.key];
  if (!d || !t || !t.getAttribute || t.getAttribute('role') !== 'radio') return;
  const g = t.closest('[role=radiogroup]');
  if (!g) return;
  const items = [...g.querySelectorAll('[role=radio]')], n = items[(items.indexOf(t) + d + items.length) % items.length];
  e.preventDefault();
  n.focus(); n.click();
});

/* Header nav: mark the page we are on. */
function markNav() {
  const here = location.pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/';
  document.querySelectorAll('.seg a').forEach(a => {
    const to = new URL(a.href, location.href).pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/';
    const on = to === here || (to === '/' && here === '/index');
    a.classList.toggle('on', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
}

hydrateIcons();
markNav();
document.addEventListener('DOMContentLoaded', () => { hydrateIcons(); pollState(); setInterval(pollState, 2000); });
