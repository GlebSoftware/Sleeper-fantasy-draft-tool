/* draftadvisor - draft overlay.
 *
 * Runs in the extension's isolated world (chrome.storage + cross-origin fetch available) on the
 * ESPN draft-room page, and standalone in selftest.html. Three independent ways to learn about
 * picks, all live at once:
 *   1. websocket  - frames forwarded by hook.js (MAIN world), parsed here.
 *   2. dom        - MutationObserver + 1 s sweep for known player names in a pick-ish context.
 *   3. manual     - the search box. Always works; the safety net.
 * Everything (taken, mine, settings, panel geometry) is persisted on every change, so a reload
 * in the middle of a draft loses nothing.
 */
(function () {
  "use strict";
  if (window.__draftadvisorOverlay) return;
  window.__draftadvisorOverlay = true;

  // ---------------------------------------------------------------- constants
  var API_BASE_DEFAULT = "https://sleeper-draft-advisor.vercel.app";
  var STORE_KEY = "draftadvisor.session.v1";
  var POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"];
  var PICK_VERBS = { SELECTED: 1, SELECT: 1, PICK: 1, PICKED: 1, DRAFTED: 1, DRAFT: 1, AUTOPICK: 1, AUTODRAFT: 1 };
  var PICK_CTX = /(R?\d+[.,]\s*\d+)|round|pick|selected by|drafted by/i;
  var DEBOUNCE_MS = 400, MIN_INTERVAL_MS = 2000, INDEX_TOP = 600;

  // ---------------------------------------------------------------- state
  var S = {
    taken: [],            // [{espn_id, name, src, ts}] most recent first
    mine: [],             // [{espn_id, name}]
    made: 0,              // deepest overall pick number seen on the board, recognised or not
    settings: {
      api: API_BASE_DEFAULT, scoring: "ppr", teams: 12, rounds: 16,
      superflex: false, myTeamId: "", mySlot: "", accessCode: "", domScan: true, domDisappear: false
    },
    ui: { top: 80, left: null, right: 24, w: 380, h: 620, collapsed: false, posOpen: { QB: 1, RB: 1, WR: 1, TE: 1, K: 0, DEF: 0 } }
  };
  var layer = "manual";        // which detection layer last produced a pick
  var index = [], byEspn = new Map(), byName = new Map(), byLast = new Map(), indexError = "";
  var advice = null, adviceError = "", loading = false, usingLocal = false;
  var debug = [], verbs = Object.create(null), candidates = [];
  var els = {}, root = null;

  // ---------------------------------------------------------------- utils
  function log(kind, text) {
    debug.unshift({ kind: kind, text: String(text).slice(0, 300), ts: Date.now() });
    if (debug.length > 40) debug.length = 40;
    if (els.debug) renderDebug();
  }
  function normName(s) {
    return String(s || "").toLowerCase().replace(/[.'`’]/g, "")
      .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, "").replace(/[^a-z0-9 ]+/g, " ")
      .replace(/\s+/g, " ").trim();
  }
  function num(v, d) { var n = parseFloat(v); return isFinite(n) ? n : (d === undefined ? null : d); }
  function fmt(v, dp) { var n = num(v); return n === null ? "-" : n.toFixed(dp === undefined ? 1 : dp); }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function posOf(p) { var x = String(p || "").toUpperCase(); return x === "DST" || x === "D/ST" ? "DEF" : x; }
  /** How deep the board is, which is NOT how many picks we recognised. The server needs this
   *  separately: a pick we could not name still moved the draft on, and a board it thinks is
   *  shallower than it is puts the draft back in an earlier round, where every starting slot looks
   *  open and the best available player (usually a running back) wins every time. */
  function noteMade(no) {
    var n = parseInt(no, 10);
    var cap = (parseInt(S.settings.teams, 10) || 12) * (parseInt(S.settings.rounds, 10) || 16);
    if (!isFinite(n) || n < 1 || n > cap) return;
    if (n > S.made) { S.made = n; saveState(); }
  }

  // ---------------------------------------------------------------- storage
  function hasChromeStorage() {
    try { return typeof chrome !== "undefined" && chrome.storage && chrome.storage.local; } catch (e) { return false; }
  }
  function saveState() {
    var blob = { taken: S.taken, mine: S.mine, made: S.made, settings: S.settings, ui: S.ui };
    try {
      if (hasChromeStorage()) { var o = {}; o[STORE_KEY] = blob; chrome.storage.local.set(o); return; }
    } catch (e) { /* fall through */ }
    try { localStorage.setItem(STORE_KEY, JSON.stringify(blob)); } catch (e) { /* ignore */ }
  }
  function applyState(blob) {
    if (!blob || typeof blob !== "object") return;
    if (Array.isArray(blob.taken)) S.taken = blob.taken;
    if (Array.isArray(blob.mine)) S.mine = blob.mine;
    if (isFinite(blob.made)) S.made = parseInt(blob.made, 10) || 0;
    if (blob.settings) for (var k in S.settings) if (blob.settings[k] !== undefined) S.settings[k] = blob.settings[k];
    if (blob.ui) for (var j in S.ui) if (blob.ui[j] !== undefined) S.ui[j] = blob.ui[j];
    if (!S.settings.api) S.settings.api = API_BASE_DEFAULT;
  }
  function loadState(done) {
    try {
      if (hasChromeStorage()) {
        chrome.storage.local.get([STORE_KEY], function (res) { applyState(res && res[STORE_KEY]); done(); });
        return;
      }
    } catch (e) { /* fall through */ }
    try { applyState(JSON.parse(localStorage.getItem(STORE_KEY) || "null")); } catch (e) { /* ignore */ }
    done();
  }

  // ---------------------------------------------------------------- taken / mine sets
  function keyOf(p) {
    if (p.espn_id !== null && p.espn_id !== undefined && p.espn_id !== "") return "id:" + p.espn_id;
    return "nm:" + normName(p.name);
  }
  function isTaken(p) {
    var k = keyOf(p), nk = p.name ? "nm:" + normName(p.name) : null;
    for (var i = 0; i < S.taken.length; i++) {
      var t = S.taken[i];
      if (keyOf(t) === k) return true;
      if (nk && t.name && "nm:" + normName(t.name) === nk) return true;
      if (p.espn_id != null && t.espn_id != null && String(t.espn_id) === String(p.espn_id)) return true;
    }
    return false;
  }
  function addTaken(p, src, mineToo) {
    var espn_id = (p.espn_id === undefined || p.espn_id === "" ) ? null : p.espn_id;
    var name = p.name || null;
    if (espn_id != null && !name) { var known = byEspn.get(String(espn_id)); if (known) name = known.name; }
    if (name && espn_id == null) { var k2 = byName.get(normName(name)); if (k2 && k2.espn_id != null) espn_id = k2.espn_id; }
    var rec = { espn_id: espn_id, name: name, src: src, ts: Date.now() };
    if (isTaken(rec)) {
      if (mineToo && !S.mine.some(function (m) { return keyOf(m) === keyOf(rec); })) {
        S.mine.unshift({ espn_id: espn_id, name: name }); saveState(); refresh();
      }
      return false;
    }
    S.taken.unshift(rec);
    if (mineToo) S.mine.unshift({ espn_id: espn_id, name: name });
    if (src === "websocket" || src === "dom" || src === "manual") layer = src === "manual" ? layer : src;
    if (src === "manual" && layer === "manual") layer = "manual";
    saveState(); renderHeader(); renderTaken(); refresh();
    return true;
  }
  function removeTaken(i) {
    var rec = S.taken[i]; if (!rec) return;
    S.taken.splice(i, 1);
    S.mine = S.mine.filter(function (m) { return keyOf(m) !== keyOf(rec); });
    saveState(); renderHeader(); renderTaken(); refresh();
  }

  /** Extra headers: the deployment is public unless DRAFTADVISOR_ACCESS_CODE is set on it. */
  function headers(extra) {
    var h = extra || {};
    if (S.settings.accessCode) h["X-Access-Code"] = S.settings.accessCode;
    return h;
  }

  // ---------------------------------------------------------------- player index
  function ingestIndex(rows) {
    index = []; byEspn = new Map(); byName = new Map(); byLast = new Map();
    (rows || []).forEach(function (r) {
      var eid = r.espn_id !== undefined ? r.espn_id : (r.espn !== undefined ? r.espn : (r.ids && r.ids.espn));
      var p = {
        player_id: r.player_id, espn_id: (eid === undefined || eid === null || eid === "") ? null : eid,
        name: r.name, position: posOf(r.position), team: r.team,
        points: num(r.points), adp: num(r.adp), vorp: num(r.vorp), bye: r.bye
      };
      if (!p.name) return;
      index.push(p);
      if (p.espn_id != null) byEspn.set(String(p.espn_id), p);
      var nk = normName(p.name);
      if (nk && !byName.has(nk)) byName.set(nk, p);
      var parts = nk.split(" ");
      if (parts.length > 1) {
        var last = parts[parts.length - 1];
        if (!byLast.has(last)) byLast.set(last, []);
        byLast.get(last).push(p);
      }
    });
    index.sort(function (a, b) { return (b.points || 0) - (a.points || 0); });
  }
  function loadIndex() {
    var url = S.settings.api.replace(/\/+$/, "") + "/api/projections?top=" + INDEX_TOP;
    return fetch(url, { credentials: "omit", headers: headers() })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (rows) {
        ingestIndex(rows); indexError = "";
        log("index", "player index: " + index.length + " players, " + byEspn.size + " with espn ids");
        try { if (hasChromeStorage()) chrome.storage.local.set({ "draftadvisor.index.v1": rows }); } catch (e) {}
        renderSearch(); renderHeader();
      })
      .catch(function (e) {
        indexError = String(e && e.message || e);
        log("index", "player index FAILED: " + indexError + " (websocket ids still work; manual search is empty)");
        try {
          if (hasChromeStorage()) chrome.storage.local.get(["draftadvisor.index.v1"], function (res) {
            var rows = res && res["draftadvisor.index.v1"];
            if (rows && rows.length) { ingestIndex(rows); log("index", "using cached index (" + index.length + ")"); renderSearch(); renderHeader(); }
          });
        } catch (e2) {}
        renderHeader();
      });
  }

  // ---------------------------------------------------------------- layer 1: websocket frames
  function intsIn(text) {
    var m = String(text).match(/-?\d+/g) || [];
    return m.map(function (x) { return parseInt(x, 10); }).filter(function (x) { return isFinite(x); });
  }
  function plausibleId(id) {
    if (id === -1 || id === 0 || !isFinite(id)) return false;
    if (id < 0) return id <= -16000;                 // D/ST are -16000 - proTeamId
    return id > 999;                                  // real ESPN player ids are 5-7 digits
  }
  function handleFrame(text, url) {
    text = String(text || "");
    var trimmed = text.trim();
    if (!trimmed) return;
    var toks = trimmed.split(/\s+/);
    var verb = (toks[0] || "").toUpperCase().replace(/[^A-Z]/g, "");
    if (verb) { verbs[verb] = (verbs[verb] || 0) + 1; }
    log("ws", trimmed.slice(0, 200));

    if (PICK_VERBS[verb]) {
      var teamId = parseInt(toks[1], 10), playerId = parseInt(toks[2], 10), overall = parseInt(toks[3], 10);
      noteMade(overall);
      if (plausibleId(playerId)) {
        var mineToo = !!(S.settings.myTeamId !== "" && String(teamId) === String(S.settings.myTeamId));
        var added = addTaken({ espn_id: playerId }, "websocket", mineToo);
        layer = "websocket";
        renderHeader();
        log("pick", verb + " team=" + teamId + " player=" + playerId + " overall=" + (isFinite(overall) ? overall : "?") +
          (added ? "" : " (duplicate)"));
        return;
      }
    }
    // JSON frames: look for an obvious player id field.
    if (trimmed.charAt(0) === "{" || trimmed.charAt(0) === "[") {
      try {
        var seen = [];
        (function walk(o, d) {
          if (!o || d > 6) return;
          if (Array.isArray(o)) { o.forEach(function (x) { walk(x, d + 1); }); return; }
          if (typeof o !== "object") return;
          Object.keys(o).forEach(function (k) {
            if (/^(playerid|player_id|id)$/i.test(k) && plausibleId(parseInt(o[k], 10))) seen.push(parseInt(o[k], 10));
            walk(o[k], d + 1);
          });
        })(JSON.parse(trimmed), 0);
        seen.forEach(function (pid) {
          if (byEspn.has(String(pid))) { addTaken({ espn_id: pid }, "websocket"); layer = "websocket"; renderHeader(); }
        });
        if (seen.length) return;
      } catch (e) { /* not JSON after all */ }
    }
    // Unverified verb: any frame with >= 3 integers, one of which is a known ESPN player id,
    // becomes a candidate the owner can accept with one click instead of being dropped.
    var ints = intsIn(trimmed);
    if (ints.length >= 3) {
      var hit = ints.filter(function (i) { return plausibleId(i) && byEspn.has(String(i)); });
      if (hit.length) {
        var pid2 = hit[0];
        if (!isTaken({ espn_id: pid2 })) {
          candidates.unshift({ id: pid2, frame: trimmed.slice(0, 160), name: (byEspn.get(String(pid2)) || {}).name || String(pid2) });
          if (candidates.length > 20) candidates.length = 20;
          log("cand", "candidate pick? " + ((byEspn.get(String(pid2)) || {}).name || pid2) + "  <<  " + trimmed.slice(0, 120));
        }
      }
    }
  }
  window.addEventListener("message", function (ev) {
    var d = ev.data;
    if (!d || typeof d !== "object" || !d.__draftadvisor) return;
    if (d.__draftadvisor === "ws-frame") handleFrame(d.text, d.url);
    else if (d.__draftadvisor === "ws-hook") log("hook", "websocket hook installed");
    else if (d.__draftadvisor === "ws-open") log("hook", "socket open: " + String(d.url).slice(0, 140));
    else if (d.__draftadvisor === "ws-close") log("hook", d.text + ": " + String(d.url).slice(0, 120));
    else if (d.__draftadvisor === "ws-send") log("send", String(d.text).slice(0, 120));
  });
  window.__draftadvisorHandleFrame = handleFrame; // used by selftest.html

  // ------------------------------------------------- layer 2a: ESPN draft-room detector
  // Built from a live inspection of https://fantasy.espn.com/football/draft (2026 season).
  // The pick train is <ul class="picklist"> with <li class="picklist--pick"> children; each holds
  // .pick-number ("PICK 12") and, once the pick is made, the player's headshot image whose URL
  // carries ESPN's numeric player id: .../i/headshots/nfl/players/full/{playerId}.png
  // That id is exact, so it beats name matching (suffixes, defenses, duplicates).
  var HEADSHOT_ID = /\/full\/(\d+)\.png/;
  var espnCfg = { slot: null, teams: null, read: false };

  /** ESPN writes rosters as "C. McCaffrey": match on last name, disambiguated by the first initial
   *  and, when that is still ambiguous, by the highest projection (the drafted one, in practice). */
  function matchAbbrev(text) {
    var t = normName(text);
    if (!t) return null;
    var exact = byName.get(t);
    if (exact) return exact;
    var parts = t.split(" ");
    if (parts.length < 2) return null;
    var last = parts[parts.length - 1], initial = parts[0].charAt(0);
    var cands = byLast.get(last) || [];
    if (!cands.length) return null;
    var narrowed = cands.filter(function (p) { return normName(p.name).charAt(0) === initial; });
    var pool = narrowed.length ? narrowed : cands;
    return pool.slice().sort(function (a, b) { return (b.points || 0) - (a.points || 0); })[0];
  }

  /** My roster, read straight from ESPN's roster panel - ground truth, no slot arithmetic needed.
   *
   *  This is the single most load-bearing read in the extension: without it the server sees an empty
   *  roster, treats every starting slot as open and recommends the best player alive. The DOM is
   *  ESPN's, so it is read through a list of selectors and, inside a row, id-first (the headshot URL
   *  carries the exact ESPN player id) before falling back to the abbreviated name. */
  var ROSTER_SELECTORS = [".roster tr[data-idx]", ".roster tbody tr", "[class*='roster'] tbody tr"];
  function espnRosterRows() {
    for (var i = 0; i < ROSTER_SELECTORS.length; i++) {
      try {
        var rows = document.querySelectorAll(ROSTER_SELECTORS[i]);
        if (rows && rows.length) return rows;
      } catch (e) { /* selector unsupported: try the next */ }
    }
    return [];
  }
  var rosterMiss = 0;
  function espnMyRoster() {
    var out = [], rows = espnRosterRows(), filled = 0;
    try {
      for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        if (row.querySelector(".player-column__empty")) continue;          // unfilled slot
        var p = null, m = HEADSHOT_ID.exec(row.innerHTML || "");
        if (m) p = byEspn.get(String(m[1])) || null;                       // exact: ESPN's own id
        var cell = row.querySelector(".player-column, .player-name, a[title]");
        var name = cell ? (cell.getAttribute("title") || cell.textContent || "").trim() : "";
        if (!p && name && !/^empty$/i.test(name)) p = matchAbbrev(name);
        if (!p && !m && !name) continue;                                   // header or spacer row
        filled++;
        if (p) out.push(p);
      }
    } catch (e) { log("espn", "roster read failed: " + e); }
    // say so once when the panel is there but unreadable: an empty roster is never silently fine
    if (filled > out.length && rosterMiss !== filled - out.length) {
      rosterMiss = filled - out.length;
      log("espn", "roster panel: " + out.length + " of " + filled + " rows matched a player");
    }
    return out;
  }

  /** My slot, from ESPN's live "You're on the clock in: N Picks / Round R, Pick P" banner.
   *  Snake: an odd round counts forward, an even round backward. Exact, and it works mid-draft. */
  function espnSlotFromClock() {
    try {
      var txt = document.body.innerText || "";
      var m = txt.match(/round\s*(\d{1,2})\s*,\s*pick\s*(\d{1,2})/i);
      var teams = espnCfg.teams || parseInt(S.settings.teams, 10) || 12;
      if (!m) return null;
      var rnd = parseInt(m[1], 10), idx = parseInt(m[2], 10);
      if (!rnd || !idx || idx > teams) return null;
      return (rnd % 2 === 1) ? idx : (teams - idx + 1);
    } catch (e) { return null; }
  }

  var espnScoring = null;
  /** The league's real scoring rules straight from ESPN (same-origin fetch, the user's own cookies).
   *  ESPN freezes draft PICKS during a draft but settings are served normally, so this is reliable. */
  function loadEspnScoring() {
    try {
      var q = espnUrlParams();
      if (!q.leagueId || !q.seasonId || espnScoring) return;
      var url = "https://fantasy.espn.com/apis/v3/games/ffl/seasons/" + encodeURIComponent(q.seasonId) +
                "/segments/0/leagues/" + encodeURIComponent(q.leagueId) + "?view=mSettings";
      fetch(url, { credentials: "include" }).then(function (r) { return r.ok ? r.json() : null; }).then(function (j) {
        var body = Array.isArray(j) ? j[0] : j;
        var items = body && body.settings && body.settings.scoringSettings &&
                    body.settings.scoringSettings.scoringItems;
        if (items && items.length) {
          espnScoring = items;
          log("espn", "league scoring loaded from ESPN: " + items.length + " rules");
          refresh();
        }
      }).catch(function (e) { log("espn", "could not read league scoring: " + e); });
    } catch (e) { /* never block the overlay */ }
  }

  function espnUrlParams() {
    try {
      var q = new URLSearchParams(location.search);
      return { leagueId: q.get("leagueId"), seasonId: q.get("seasonId"),
               teamId: q.get("teamId"), memberId: q.get("memberId") };
    } catch (e) { return {}; }
  }
  /** My slot from the pre-draft banner ("Your first pick: Round 1, Pick 4"), and the league size
   *  from the pick train (round 1 runs up to the .picklist--divider). Both are best effort. */
  function espnAutoConfig() {
    try {
      loadEspnScoring();
      if (espnCfg.teams == null) {
        var items = document.querySelectorAll("ul.picklist > li");
        var n = 0;
        for (var i = 0; i < items.length; i++) {
          if (items[i].className.indexOf("picklist--divider") >= 0) break;
          if (items[i].className.indexOf("picklist--pick") >= 0) n++;
        }
        if (n >= 4 && n <= 20) {
          espnCfg.teams = n;
          if (S.settings.teams !== n) { S.settings.teams = n; saveState(); log("espn", "league size from the pick train: " + n + " teams"); }
        }
      }
      var live = espnSlotFromClock();
      if (live && espnCfg.slot !== live) { espnCfg.slot = live; log("espn", "your slot from the live clock: " + live); }
      // my roster, straight off ESPN's panel - this is what makes "needs" correct
      var roster = espnMyRoster();
      for (var r = 0; r < roster.length; r++) {
        var rp = roster[r];
        if (!S.mine.some(function (m) { return String(m.espn_id) === String(rp.espn_id) || normName(m.name || "") === normName(rp.name); })) {
          S.mine.push({ espn_id: rp.espn_id, name: rp.name });
          addTaken({ espn_id: rp.espn_id, name: rp.name }, "espn", false);
          log("espn", "your roster: " + rp.name);
          saveState(); refresh();
        }
      }
      if (espnCfg.slot == null) {
        var txt = document.body.innerText || "";
        var at = txt.toLowerCase().indexOf("your first pick");
        var m = at >= 0 ? txt.slice(at, at + 90).match(/pick\s*[:#]?\s*(\d{1,2})\s*$|,\s*pick\s*(\d{1,2})/i) : null;
        if (m) {
          espnCfg.slot = parseInt(m[1] || m[2], 10);
          log("espn", "your slot from the draft banner: " + espnCfg.slot);
        }
      }
    } catch (e) { /* never let auto-config break the scan */ }
  }
  /** Snake order: is overall pick `no` mine, given my slot and the league size? */
  function isMyPickNumber(no) {
    var t = espnCfg.teams || S.settings.teams, slot = espnCfg.slot;
    if (!t || !slot || !no) return false;
    var rnd = Math.floor((no - 1) / t) + 1, idx = (no - 1) % t + 1;
    return (rnd % 2 === 1) ? idx === slot : idx === (t - slot + 1);
  }
  /** Player id + name out of one pick-train item / activity row, or null when it holds no player. */
  function espnPlayerIn(el) {
    var id = null, name = null;
    var imgs = el.querySelectorAll("img[src]");
    for (var i = 0; i < imgs.length; i++) {
      var m = HEADSHOT_ID.exec(imgs[i].getAttribute("src") || "");
      if (m) { id = m[1]; break; }
    }
    // the pick item's own title is the OWNER's real name, so never read it as the player;
    // a player name comes from a nested [title] / .player-column / a known name in the text
    var t = el.querySelector(".player-column, .player-name, a[title]");
    if (t) name = (t.getAttribute && t.getAttribute("title")) || (t.textContent || "").trim() || null;
    if (!name) {
      var txt = (el.textContent || "").split("\n");
      for (var j = 0; j < txt.length; j++) {
        var cand = txt[j].trim();
        if (cand.length > 4 && cand.length < 40 && byName.has(normName(cand))) { name = cand; break; }
      }
    }
    if (!id && !name) return null;
    if (id && !byEspn.has(String(id)) && !name) return null;      // unknown id, nothing to show
    return { espn_id: id, name: name };
  }
  function espnScan() {
    if (!document.body) return 0;
    espnAutoConfig();
    var hits = 0;
    try {
      var picks = document.querySelectorAll("ul.picklist li.picklist--pick");
      for (var i = 0; i < picks.length; i++) {
        var li = picks[i];
        var p = espnPlayerIn(li);
        if (!p) {                                                // no player we can name ...
          if (HEADSHOT_ID.test((li.innerHTML || ""))) {          // ... but a headshot means it was made
            var un = ((li.querySelector(".pick-number") || {}).textContent || "").match(/(\d+)/);
            noteMade(un ? parseInt(un[1], 10) : null);
          }
          continue;
        }
        var numTxt = (li.querySelector(".pick-number") || {}).textContent || "";
        var nm = numTxt.match(/(\d+)/);
        var no = nm ? parseInt(nm[1], 10) : null;
        noteMade(no);
        var mine = isMyPickNumber(no);
        if (addTaken({ espn_id: p.espn_id, name: p.name }, "espn", mine)) {
          hits++; layer = "espn"; renderHeader();
          log("espn", "pick " + (no || "?") + ": " + (p.name || ("id " + p.espn_id)) + (mine ? " (yours)" : ""));
        }
      }
      // the activity feed also carries picks (its own filter calls them "Picks")
      var msgs = document.querySelectorAll("li.message, .message");
      for (var k = 0; k < msgs.length && k < 80; k++) {
        var mp = espnPlayerIn(msgs[k]);
        if (mp && addTaken({ espn_id: mp.espn_id, name: mp.name }, "espn", false)) {
          hits++; layer = "espn"; renderHeader();
          log("espn", "activity feed: " + (mp.name || ("id " + mp.espn_id)));
        }
      }
    } catch (e) { log("espn", "scan error: " + e); }
    return hits;
  }
  window.__draftadvisorEspnScan = espnScan;                       // used by selftest.html

  // ---------------------------------------------------------------- layer 2: DOM scan
  var domSeen = new Map(), domTimer = null, sweepScheduled = false;
  function pickContext(node) {
    var el = node.parentElement, hops = 0;
    while (el && hops < 4) {
      var t = (el.textContent || "").slice(0, 400);
      if (PICK_CTX.test(t)) return true;
      el = el.parentElement; hops++;
    }
    return false;
  }
  function domSweep() {
    sweepScheduled = false;
    if (!document.body) return;
    espnScan();                                   // precise ESPN selectors first (ids, pick numbers)
    if (!S.settings.domScan || !byName.size) return;
    var found = new Set(), budget = 25000;
    try {
      var w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
      var n;
      while ((n = w.nextNode()) && budget-- > 0) {
        var raw = n.nodeValue;
        if (!raw) continue;
        var t = raw.trim();
        if (t.length < 5 || t.length > 60 || t.indexOf(" ") < 0) continue;
        var p = byName.get(normName(t));
        if (!p) continue;
        found.add(normName(t));
        if (!isTaken(p) && pickContext(n)) {
          if (addTaken({ espn_id: p.espn_id, name: p.name }, "dom")) {
            layer = "dom"; renderHeader();
            log("dom", "pick from DOM context: " + p.name);
          }
        }
      }
    } catch (e) { log("dom", "sweep error: " + e); }
    // "disappeared from a list it was in" - opt-in, because ESPN's player list is virtualised
    // and scrolling alone removes rows from the DOM.
    found.forEach(function (k) {
      var r = domSeen.get(k) || { seen: 0, gone: 0 };
      r.seen++; r.gone = 0; domSeen.set(k, r);
    });
    if (S.settings.domDisappear) {
      domSeen.forEach(function (r, k) {
        if (found.has(k) || r.seen < 3) return;
        r.gone++;
        if (r.gone === 2) {
          var p = byName.get(k);
          if (p && !isTaken(p) && addTaken({ espn_id: p.espn_id, name: p.name }, "dom")) {
            layer = "dom"; renderHeader(); log("dom", "disappeared from list: " + p.name);
          }
        }
      });
    }
  }
  function scheduleSweep() {
    if (sweepScheduled) return;
    sweepScheduled = true;
    setTimeout(domSweep, 500);
  }
  function startDomScan() {
    try {
      if (document.body) new MutationObserver(scheduleSweep).observe(document.body, { childList: true, subtree: true, characterData: true });
    } catch (e) { /* ignore */ }
    domTimer = setInterval(function () { if (S.settings.domScan) domSweep(); }, 1000);
  }

  // ---------------------------------------------------------------- API
  var reqTimer = null, lastReq = 0, inflight = false, pending = false;
  function refresh(force) {
    if (reqTimer) clearTimeout(reqTimer);
    var wait = Math.max(DEBOUNCE_MS, MIN_INTERVAL_MS - (Date.now() - lastReq));
    if (force) wait = 0;
    reqTimer = setTimeout(doRequest, wait);
  }
  function requestBody() {
    return {
      taken: S.taken.map(function (t) { return { espn_id: t.espn_id === undefined ? null : t.espn_id, name: t.name || null }; }),
      mine: S.mine.map(function (m) { return { espn_id: m.espn_id === undefined ? null : m.espn_id, name: m.name || null }; }),
      scoring: S.settings.scoring, teams: parseInt(S.settings.teams, 10) || 12,
      rounds: parseInt(S.settings.rounds, 10) || 16, superflex: !!S.settings.superflex,
      slot: parseInt(S.settings.mySlot, 10) || espnCfg.slot || null,   // typed slot wins over the banner
      made: S.made || null,                                            // board depth, independent of what we named
      espn_scoring_items: espnScoring                                   // the league's real rules, read off ESPN
    };
  }
  function doRequest() {
    if (inflight) { pending = true; return; }
    inflight = true; loading = true; lastReq = Date.now(); renderHeader();
    var url = S.settings.api.replace(/\/+$/, "") + "/api/extension/advice";
    fetch(url, {
      method: "POST", credentials: "omit",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(requestBody())
    })
      .then(function (r) {
        return r.text().then(function (txt) {
          if (!r.ok) throw new Error("HTTP " + r.status + " " + txt.slice(0, 200));
          try { return JSON.parse(txt); } catch (e) { throw new Error("bad JSON from API: " + txt.slice(0, 120)); }
        });
      })
      .then(function (j) { advice = j; adviceError = ""; usingLocal = false; })
      .catch(function (e) {
        adviceError = String(e && e.message || e);
        advice = localAdvice(); usingLocal = true;
        log("api", "advice failed: " + adviceError + " - falling back to the local projection list");
      })
      .then(function () {
        inflight = false; loading = false;
        renderAll();
        if (pending) { pending = false; refresh(); }
      });
  }
  /** Last-resort advice computed in the browser from the player index, so the panel is never blank. */
  function localAdvice() {
    if (!index.length) return null;
    var avail = index.filter(function (p) { return !isTaken(p); });
    var by = {};
    POSITIONS.forEach(function (pos) {
      by[pos] = avail.filter(function (p) { return p.position === pos; }).slice(0, 5);
    });
    var top = avail.slice(0, 5);
    return {
      overall: top, by_position: by,
      suggestion: top[0] ? {
        player_id: top[0].player_id, name: top[0].name, position: top[0].position, team: top[0].team,
        action: "draft", why: "Local fallback: highest projected points still on the board (the advice API is unreachable)."
      } : null,
      needs: [], counts: { taken: S.taken.length, resolved: 0, mine: S.mine.length }, unresolved: [], ms: 0
    };
  }

  // ---------------------------------------------------------------- panel
  function buildPanel() {
    var host = document.createElement("div");
    host.id = "draftadvisor-host";
    host.style.cssText = "all:initial;position:static";
    (document.body || document.documentElement).appendChild(host);
    root = host.attachShadow ? host.attachShadow({ mode: "open" }) : host;

    var style = document.createElement("style");
    style.textContent = window.DRAFTADVISOR_STYLES || "";
    root.appendChild(style);

    var panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = [
      '<div class="hdr" data-drag="1">',
      '  <span class="dot" id="dot"></span><span class="title">draftadvisor</span>',
      '  <span class="layer" id="layer">manual</span>',
      '  <span class="count" id="count"></span>',
      '  <button id="refresh" title="Re-ask the API now">Refresh</button>',
      '  <button id="collapse" title="Collapse">-</button>',
      "</div>",
      '<div class="body">',
      '  <div id="msgs"></div>',
      '  <div id="sug"></div>',
      '  <div id="roster"></div>',
      "  <h3>Top 5 available</h3><div id=\"overall\"></div>",
      '  <div id="bypos"></div>',
      "  <h3>Manual entry (always works)</h3>",
      '  <div class="row"><input id="q" placeholder="Type a player name, click to mark taken" autocomplete="off">',
      '    <label><input type="checkbox" id="mineToggle" style="width:auto"> my pick</label></div>',
      '  <div id="results"></div>',
      '  <h3>Taken (<span id="tcount">0</span>) - click x to undo</h3>',
      '  <div class="taken" id="taken"></div>',
      "  <details id=\"setwrap\"><summary>Settings</summary><div id=\"settings\"></div></details>",
      '  <details id="dbgwrap"><summary>Debug - raw frames &amp; verbs</summary><div id="dbgbody"></div></details>',
      "</div>"
    ].join("");
    root.appendChild(panel);

    els.panel = panel;
    ["dot", "layer", "count", "refresh", "collapse", "msgs", "sug", "roster", "overall", "bypos", "q", "mineToggle",
      "results", "taken", "tcount", "settings", "dbgbody", "dbgwrap"].forEach(function (id) {
        els[id] = root.getElementById ? root.getElementById(id) : panel.querySelector("#" + id);
      });
    els.debug = els.dbgbody;

    applyGeometry();
    wireHeader(panel);
    wireSearch();
    buildSettings();
  }
  function applyGeometry() {
    var p = els.panel, u = S.ui;
    p.style.width = u.w + "px";
    p.style.height = u.h + "px";
    p.style.top = u.top + "px";
    if (u.left !== null && u.left !== undefined) { p.style.left = u.left + "px"; p.style.right = "auto"; }
    else { p.style.right = (u.right || 24) + "px"; p.style.left = "auto"; }
    p.classList.toggle("collapsed", !!u.collapsed);
    if (els.collapse) els.collapse.textContent = u.collapsed ? "+" : "-";
  }
  function wireHeader(panel) {
    var hdr = panel.querySelector(".hdr"), dragging = false, ox = 0, oy = 0;
    hdr.addEventListener("mousedown", function (e) {
      if (e.target.tagName === "BUTTON") return;
      dragging = true;
      var r = panel.getBoundingClientRect();
      ox = e.clientX - r.left; oy = e.clientY - r.top;
      e.preventDefault();
    });
    window.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      S.ui.left = Math.max(0, e.clientX - ox); S.ui.top = Math.max(0, e.clientY - oy);
      panel.style.left = S.ui.left + "px"; panel.style.right = "auto"; panel.style.top = S.ui.top + "px";
    });
    window.addEventListener("mouseup", function () { if (dragging) { dragging = false; saveState(); } });
    try {
      new ResizeObserver(function () {
        if (S.ui.collapsed) return;
        var r = panel.getBoundingClientRect();
        if (r.width > 100) { S.ui.w = Math.round(r.width); S.ui.h = Math.round(r.height); saveState(); }
      }).observe(panel);
    } catch (e) { /* ignore */ }
    els.collapse.addEventListener("click", function () {
      S.ui.collapsed = !S.ui.collapsed; applyGeometry(); saveState();
    });
    els.refresh.addEventListener("click", function () { refresh(true); });
  }

  // ---------------------------------------------------------------- rendering
  function renderHeader() {
    if (!els.dot) return;
    els.dot.className = "dot " + layer;
    els.layer.textContent = layer + (loading ? "" : "");
    var b = (advice && advice.board) || null;
    var clock = b && b.on_the_clock ? " / pick #" + b.on_the_clock + " (rd " + b.round + ")" : "";
    els.count.innerHTML = (loading ? '<span class="spin"></span> ' : "") +
      esc(S.taken.length + " taken / " + S.mine.length + " mine" + clock + (index.length ? "" : " / no index"));
    if (els.tcount) els.tcount.textContent = String(S.taken.length);
  }
  function renderMsgs() {
    var out = "";
    if (adviceError) {
      out += '<div class="err">Advice API error: ' + esc(adviceError) +
        (/\b401\b/.test(adviceError) ? "<br>That deployment wants an access code - put it in Settings." : "") +
        (usingLocal ? "<br>Showing the local projection ranking instead (no positional need logic)." : "") +
        ' <button id="retry">Retry</button></div>';
    }
    if (indexError && !index.length) {
      out += '<div class="warn">Player index unavailable (' + esc(indexError) +
        "). WebSocket ids still work; manual search has no names to offer.</div>";
    }
    var w = (advice && advice.warnings) || [];
    for (var wi = 0; wi < w.length; wi++) out += '<div class="warn">' + esc(w[wi]) + "</div>";
    if (!S.settings.domScan) out += '<div class="warn">DOM scan is off - only the websocket hook and manual entry are live.</div>';
    els.msgs.innerHTML = out;
    var r = els.msgs.querySelector("#retry");
    if (r) r.addEventListener("click", function () { refresh(true); });
  }
  function cardRow(c, i) {
    return "<tr><td class=\"num\">" + (i + 1) + '</td><td class="n" title="' + esc(c.name) + '">' + esc(c.name) +
      '</td><td><span class="pos ' + esc(posOf(c.position)) + '">' + esc(posOf(c.position)) + "</span></td><td>" +
      esc(c.team || "") + '</td><td class="num">' + fmt(c.points) + '</td><td class="num">' + fmt(c.vorp) +
      '</td><td class="num">' + fmt(c.adp) + "</td></tr>";
  }
  function table(cards) {
    if (!cards || !cards.length) return '<div class="muted">nothing available</div>';
    return "<table><thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Tm</th><th>Pts</th><th>VORP</th><th>ADP</th></tr></thead><tbody>" +
      cards.slice(0, 5).map(cardRow).join("") + "</tbody></table>";
  }
  function renderAdvice() {
    var a = advice;
    if (!a) {
      els.sug.innerHTML = '<div class="sug empty"><div class="lbl">next pick</div><div class="nm">' +
        (loading ? "asking the API..." : "no advice yet") + "</div></div>";
      els.overall.innerHTML = '<div class="muted">-</div>';
      els.bypos.innerHTML = "";
      return;
    }
    var s = a.suggestion;
    els.sug.innerHTML = s
      ? '<div class="sug"><div class="lbl">next pick' + (usingLocal ? " (local fallback)" : "") + '</div><div class="nm">' +
        esc(s.name) + '</div><div class="meta">' + esc(posOf(s.position)) + " - " + esc(s.team || "") +
        (s.action ? " - " + esc(s.action) : "") + '</div><div class="why">' + esc(s.why || "") + "</div>" +
        ((a.needs && a.needs.length) ? '<div class="meta">needs: ' + esc(a.needs.join(", ")) + "</div>" : "") + "</div>"
      : '<div class="sug empty"><div class="lbl">next pick</div><div class="nm">API returned no suggestion</div></div>';
    if (els.roster) {
      var rr = a.roster;
      if (rr && rr.players && rr.players.length) {
        var counts = Object.keys(rr.counts || {}).map(function (k) { return k + " " + rr.counts[k]; }).join(", ");
        els.roster.innerHTML = '<div class="meta">your roster (' + esc(counts) + "): " +
          esc(rr.players.map(function (x) { return x.name; }).join(", ")) + "</div>";
      } else {
        els.roster.innerHTML = '<div class="meta">your roster: empty - every position counts as a need</div>';
      }
    }
    els.overall.innerHTML = table(a.overall);
    var bp = a.by_position || {};
    els.bypos.innerHTML = POSITIONS.map(function (pos) {
      return "<details data-pos=\"" + pos + '"' + (S.ui.posOpen[pos] ? " open" : "") + "><summary>" + pos +
        "</summary>" + table(bp[pos]) + "</details>";
    }).join("");
    Array.prototype.forEach.call(els.bypos.querySelectorAll("details"), function (d) {
      d.addEventListener("toggle", function () { S.ui.posOpen[d.getAttribute("data-pos")] = d.open ? 1 : 0; saveState(); });
    });
    if (a.unresolved && a.unresolved.length) {
      els.overall.insertAdjacentHTML("afterend",
        '<div class="muted">unmatched by the server: ' + esc(a.unresolved.slice(0, 8).join(", ")) + "</div>");
    }
  }
  function renderTaken() {
    if (!els.taken) return;
    if (els.tcount) els.tcount.textContent = String(S.taken.length);
    els.taken.innerHTML = S.taken.length
      ? S.taken.map(function (t, i) {
          var mine = S.mine.some(function (m) { return keyOf(m) === keyOf(t); });
          return '<div class="t"><span>' + esc(t.name || ("espn#" + t.espn_id)) + "</span>" +
            (mine ? '<span class="src" style="color:#7aa2ff">MINE</span>' : "") +
            '<span class="src">' + esc(t.src) + "</span>" +
            '<span class="x" data-i="' + i + '" title="undo">x</span></div>';
        }).join("")
      : '<div class="muted">no picks recorded yet</div>';
    Array.prototype.forEach.call(els.taken.querySelectorAll(".x"), function (x) {
      x.addEventListener("click", function () { removeTaken(parseInt(x.getAttribute("data-i"), 10)); });
    });
  }
  function renderSearch() { if (els.q) els.q.placeholder = index.length ? ("Search " + index.length + " players - click to mark taken") : "Player index unavailable - type a full name + Enter"; }
  function wireSearch() {
    els.q.addEventListener("input", function () { showResults(els.q.value); });
    els.q.addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      var first = els.results.querySelector(".r");
      if (first) { first.click(); return; }
      var v = els.q.value.trim();
      if (v.length > 2) { addTaken({ name: v }, "manual", els.mineToggle.checked); els.q.value = ""; els.results.innerHTML = ""; }
    });
  }
  function showResults(q) {
    var nq = normName(q);
    if (!nq) { els.results.innerHTML = ""; return; }
    var hits = index.filter(function (p) { return normName(p.name).indexOf(nq) >= 0; }).slice(0, 10);
    els.results.className = "res";
    els.results.innerHTML = hits.length
      ? hits.map(function (p, i) {
          return '<div class="r" data-i="' + i + '"><span class="pos ' + p.position + '">' + p.position + "</span>" +
            "<span>" + esc(p.name) + "</span><span class=\"muted\">" + esc(p.team || "") + " " + fmt(p.points, 0) + " pts" +
            (isTaken(p) ? " - TAKEN (click to undo)" : "") + "</span></div>";
        }).join("")
      : '<div class="r muted">no match - press Enter to add "' + esc(q) + '" by name anyway</div>';
    Array.prototype.forEach.call(els.results.querySelectorAll(".r[data-i]"), function (el) {
      el.addEventListener("click", function () {
        var p = hits[parseInt(el.getAttribute("data-i"), 10)];
        if (!p) return;
        if (isTaken(p)) {
          for (var i = 0; i < S.taken.length; i++) {
            var t = S.taken[i];
            if ((t.espn_id != null && String(t.espn_id) === String(p.espn_id)) ||
                (t.name && normName(t.name) === normName(p.name))) { removeTaken(i); break; }
          }
        } else {
          addTaken({ espn_id: p.espn_id, name: p.name }, "manual", els.mineToggle.checked);
        }
        els.q.value = ""; els.results.innerHTML = "";
      });
    });
  }
  function buildSettings() {
    els.settings.innerHTML = [
      '<div class="row"><label>scoring</label><select id="s-scoring"><option value="ppr">PPR</option>',
      '<option value="half_ppr">Half PPR</option><option value="std">Standard</option></select>',
      '<label>teams</label><input id="s-teams" type="number" min="2" max="20" style="width:56px">',
      '<label>rounds</label><input id="s-rounds" type="number" min="1" max="30" style="width:56px"></div>',
      '<div class="row"><label><input type="checkbox" id="s-sflex" style="width:auto"> superflex</label>',
      '<label>my ESPN team id</label><input id="s-team" style="width:56px" placeholder="opt">',
      '<label>my slot</label><input id="s-slot" type="number" min="1" max="20" style="width:48px" placeholder="auto"></div>',
      '<div class="row"><label>access code</label><input id="s-code" placeholder="only if the deployment asks for one"></div>',
      '<div class="row"><label>API</label><input id="s-api"></div>',
      '<div class="row"><label><input type="checkbox" id="s-dom" style="width:auto"> DOM scan</label>',
      '<label><input type="checkbox" id="s-gone" style="width:auto"> treat disappearing names as picks (risky)</label></div>',
      '<div class="row"><button id="s-clear">Clear this draft</button><span class="muted">wipes taken + mine</span></div>'
    ].join("");
    var g = function (id) { return els.settings.querySelector("#" + id); };
    g("s-scoring").value = S.settings.scoring;
    g("s-teams").value = S.settings.teams;
    g("s-slot").value = S.settings.mySlot || "";
    g("s-rounds").value = S.settings.rounds;
    g("s-sflex").checked = !!S.settings.superflex;
    g("s-team").value = S.settings.myTeamId;
    g("s-api").value = S.settings.api;
    g("s-code").value = S.settings.accessCode;
    g("s-dom").checked = !!S.settings.domScan;
    g("s-gone").checked = !!S.settings.domDisappear;
    function bind(id, key, kind) {
      g(id).addEventListener("change", function () {
        var el = g(id);
        S.settings[key] = kind === "bool" ? el.checked : (kind === "int" ? (parseInt(el.value, 10) || S.settings[key]) : el.value);
        saveState(); renderMsgs();
        if (key === "api") { loadIndex(); }
        refresh(true);
      });
    }
    bind("s-scoring", "scoring"); bind("s-teams", "teams", "int"); bind("s-rounds", "rounds", "int");
    bind("s-slot", "mySlot");
    bind("s-sflex", "superflex", "bool"); bind("s-team", "myTeamId"); bind("s-api", "api");
    bind("s-code", "accessCode");
    bind("s-dom", "domScan", "bool"); bind("s-gone", "domDisappear", "bool");
    g("s-clear").addEventListener("click", function () {
      S.taken = []; S.mine = []; S.made = 0; domSeen = new Map(); saveState(); renderTaken(); renderHeader(); refresh(true);
    });
  }
  function renderDebug() {
    if (!els.debug) return;
    var vs = Object.keys(verbs).sort(function (a, b) { return verbs[b] - verbs[a]; })
      .map(function (v) { return v + "x" + verbs[v]; }).join("  ") || "(no frames seen)";
    var cand = candidates.length
      ? '<div class="cand">candidate picks (click to accept):</div>' + candidates.map(function (c, i) {
          return '<div class="r cand" data-c="' + i + '" style="cursor:pointer">+ ' + esc(c.name) + "  <<  " + esc(c.frame) + "</div>";
        }).join("")
      : "";
    els.debug.innerHTML = '<div class="dbg"><b>verbs seen:</b> ' + esc(vs) + "</div>" + cand +
      '<div class="dbg">' + debug.map(function (d) {
        return "[" + d.kind + "] " + esc(d.text);
      }).join("\n") + "</div>";
    Array.prototype.forEach.call(els.debug.querySelectorAll("[data-c]"), function (el) {
      el.addEventListener("click", function () {
        var c = candidates[parseInt(el.getAttribute("data-c"), 10)];
        if (!c) return;
        addTaken({ espn_id: c.id }, "websocket");
        candidates = candidates.filter(function (x) { return x !== c; });
        layer = "websocket"; renderHeader(); renderDebug();
      });
    });
  }
  function renderAll() { renderHeader(); renderMsgs(); renderAdvice(); renderTaken(); renderDebug(); }

  // ---------------------------------------------------------------- boot
  function boot() {
    loadState(function () {
      buildPanel();
      renderAll();
      loadIndex().then(function () { renderAll(); refresh(true); });
      startDomScan();
      refresh(true);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.__draftadvisor = {
    state: S, addTaken: addTaken, handleFrame: handleFrame, refresh: refresh,
    layer: function () { return layer; }, index: function () { return index; }
  };
})();
