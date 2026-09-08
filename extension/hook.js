/* draftadvisor - MAIN-world WebSocket hook.
 *
 * Runs in the page's own JS world (manifest content_scripts "world": "MAIN") at document_start,
 * before the ESPN draft-room bundle opens its socket. It wraps window.WebSocket, listens to every
 * frame of every socket the page opens, and forwards the raw text to the overlay (isolated world)
 * with window.postMessage. It parses nothing: all parsing lives in overlay.js so it can be fixed
 * without touching this file.
 */
(function () {
  "use strict";
  if (window.__draftadvisorHooked) return;
  window.__draftadvisorHooked = true;

  var MAX = 4000; // never forward more than this many characters of one frame

  function post(kind, url, text) {
    try {
      window.postMessage(
        { __draftadvisor: kind, url: String(url || ""), text: String(text).slice(0, MAX), ts: Date.now() },
        "*"
      );
    } catch (e) { /* ignore */ }
  }

  function forward(kind, url, data) {
    try {
      if (typeof data === "string") { post(kind, url, data); return; }
      if (data instanceof ArrayBuffer) { post(kind, url, new TextDecoder().decode(new Uint8Array(data))); return; }
      if (typeof Blob !== "undefined" && data instanceof Blob) {
        if (typeof data.text === "function") data.text().then(function (t) { post(kind, url, t); }, function () {});
        return;
      }
      if (data && data.buffer) { post(kind, url, new TextDecoder().decode(data)); return; }
      post(kind, url, String(data));
    } catch (e) { /* ignore */ }
  }

  var Native = window.WebSocket;
  if (!Native) return;

  function attach(ws, url) {
    try {
      ws.addEventListener("message", function (ev) { forward("ws-frame", url, ev.data); });
      ws.addEventListener("open", function () { post("ws-open", url, "OPEN"); });
      ws.addEventListener("close", function () { post("ws-close", url, "CLOSE"); });
      ws.addEventListener("error", function () { post("ws-close", url, "ERROR"); });
    } catch (e) { /* ignore */ }
    // Defensive: also see frames if the page assigns .onmessage and something swallows listeners.
    try {
      var handler = null;
      Object.defineProperty(ws, "onmessage", {
        configurable: true,
        enumerable: true,
        get: function () { return handler; },
        set: function (fn) {
          handler = fn;
          ws.addEventListener("message", function (ev) {
            try { if (typeof handler === "function") handler.call(ws, ev); } catch (e) { /* page's problem */ }
          });
          // Replace the accessor with a plain slot so we only wrap once.
          Object.defineProperty(ws, "onmessage", { configurable: true, enumerable: true, writable: true, value: null });
        }
      });
    } catch (e) { /* ignore */ }
    try {
      var origSend = ws.send.bind(ws);
      ws.send = function (data) { forward("ws-send", url, data); return origSend(data); };
    } catch (e) { /* ignore */ }
  }

  var Wrapped;
  try {
    Wrapped = new Proxy(Native, {
      construct: function (target, args) {
        var ws = new target(args[0], args[1]);
        attach(ws, args[0]);
        return ws;
      }
    });
  } catch (e) {
    Wrapped = function (url, protocols) {
      var ws = protocols === undefined ? new Native(url) : new Native(url, protocols);
      attach(ws, url);
      return ws;
    };
    Wrapped.prototype = Native.prototype;
    ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (k) { Wrapped[k] = Native[k]; });
  }

  try {
    Object.defineProperty(window, "WebSocket", { value: Wrapped, writable: true, configurable: true });
  } catch (e) {
    window.WebSocket = Wrapped;
  }
  window.__draftadvisorNativeWebSocket = Native;
  post("ws-hook", "", "HOOK INSTALLED");
})();
