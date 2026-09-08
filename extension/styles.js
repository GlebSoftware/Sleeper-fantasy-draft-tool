/* draftadvisor - overlay stylesheet, injected inside the panel's Shadow DOM so ESPN's CSS
 * cannot reach it and ours cannot leak out. Kept as a JS string (not a .css file) so the
 * overlay works identically in the extension and in selftest.html with no fetch or bundler. */
window.DRAFTADVISOR_STYLES = `
:host { all: initial; }
* { box-sizing: border-box; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.panel {
  position: fixed; z-index: 2147483647; top: 80px; right: 24px; width: 380px; height: 620px;
  min-width: 300px; min-height: 120px; max-width: 96vw; max-height: 96vh;
  background: #10131a; color: #e8edf6; border: 1px solid #2b3446; border-radius: 10px;
  box-shadow: 0 10px 40px rgba(0,0,0,.6); display: flex; flex-direction: column;
  font-size: 12px; line-height: 1.35; overflow: hidden; resize: both;
}
.panel.collapsed { height: auto !important; min-height: 0; resize: none; }
.panel.collapsed .body { display: none; }
.hdr { display: flex; align-items: center; gap: 6px; padding: 7px 9px; background: #171c27;
  border-bottom: 1px solid #2b3446; cursor: move; user-select: none; flex: 0 0 auto; }
.title { font-weight: 700; letter-spacing: .3px; color: #fff; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #6b7488; flex: 0 0 auto; }
.dot.websocket { background: #35d07f; } .dot.dom { background: #f2b544; } .dot.manual { background: #7aa2ff; }
.layer { color: #b9c4d6; text-transform: uppercase; font-size: 10px; letter-spacing: .5px; }
.count { color: #8e9ab1; margin-left: auto; white-space: nowrap; }
.body { overflow: auto; padding: 8px 9px 12px; flex: 1 1 auto; }
button { background: #232b3b; color: #e8edf6; border: 1px solid #35405a; border-radius: 5px;
  padding: 3px 7px; font-size: 11px; cursor: pointer; }
button:hover { background: #2d3750; }
button.primary { background: #2f6df6; border-color: #2f6df6; color: #fff; }
input, select { background: #0b0e14; color: #e8edf6; border: 1px solid #35405a; border-radius: 5px;
  padding: 4px 6px; font-size: 11px; width: 100%; }
.sug { background: #14361f; border: 1px solid #2b7a4b; border-radius: 8px; padding: 9px 10px; margin: 2px 0 10px; }
.sug .lbl { font-size: 10px; letter-spacing: 1px; color: #86e0ab; text-transform: uppercase; }
.sug .nm { font-size: 20px; font-weight: 800; color: #fff; margin: 2px 0; }
.sug .meta { color: #b6d8c4; }
.sug .why { color: #d6efe0; margin-top: 4px; }
.sug.empty { background: #1a1f2b; border-color: #333e55; }
.sug.empty .nm { color: #97a3b8; font-size: 15px; }
h3 { margin: 12px 0 4px; font-size: 11px; letter-spacing: .8px; text-transform: uppercase; color: #94a2ba; font-weight: 700; }
table { width: 100%; border-collapse: collapse; }
td, th { padding: 2px 3px; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
th { color: #7c8aa3; font-weight: 600; font-size: 10px; }
tbody tr:nth-child(odd) { background: #161b25; }
td.n { font-weight: 600; color: #fff; max-width: 130px; }
td.num { text-align: right; color: #b9c4d6; font-variant-numeric: tabular-nums; }
.pos { display: inline-block; min-width: 26px; text-align: center; border-radius: 3px; padding: 0 3px;
  font-size: 10px; font-weight: 700; color: #0b0e14; }
.pos.QB{background:#f2748f}.pos.RB{background:#5fd3a6}.pos.WR{background:#6fb7ff}
.pos.TE{background:#f5b971}.pos.K{background:#c9a7f0}.pos.DEF{background:#9fb0c6}
details { border-top: 1px solid #232b3b; }
details > summary { cursor: pointer; padding: 4px 0; color: #cdd7e6; font-weight: 600; list-style: none; }
details > summary::-webkit-details-marker { display: none; }
details > summary:before { content: "\\25B8 "; color: #6d7b93; }
details[open] > summary:before { content: "\\25BE "; }
.row { display: flex; gap: 5px; align-items: center; margin: 5px 0; }
.row label { color: #8e9ab1; font-size: 10px; white-space: nowrap; }
.taken { max-height: 150px; overflow: auto; }
.taken .t { display: flex; gap: 6px; align-items: center; padding: 2px 0; border-bottom: 1px solid #1c2331; }
.taken .t .src { color: #6d7b93; font-size: 10px; }
.taken .t .x { margin-left: auto; cursor: pointer; color: #ff8080; padding: 0 4px; }
.res { border: 1px solid #35405a; border-radius: 5px; margin-top: 3px; max-height: 170px; overflow: auto; }
.res .r { padding: 4px 6px; cursor: pointer; display: flex; gap: 6px; align-items: center; }
.res .r:hover { background: #223052; }
.err { background: #3a1620; border: 1px solid #8c2f45; color: #ffc9d4; padding: 7px 8px; border-radius: 6px; margin: 4px 0; }
.warn { background: #2e2612; border: 1px solid #7a5f22; color: #f2dfae; padding: 6px 8px; border-radius: 6px; margin: 4px 0; }
.dbg { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 10px; color: #9fb0c6;
  max-height: 170px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
.dbg .cand { color: #ffd479; }
.spin { display: inline-block; width: 10px; height: 10px; border: 2px solid #46536e; border-top-color: #7aa2ff;
  border-radius: 50%; animation: sp .7s linear infinite; vertical-align: -1px; }
@keyframes sp { to { transform: rotate(360deg); } }
.muted { color: #8e9ab1; }
.grip { position: absolute; right: 2px; bottom: 2px; width: 12px; height: 12px; cursor: nwse-resize; }
`;
