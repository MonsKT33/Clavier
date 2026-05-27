#!/usr/bin/env python3
"""
Clavier Local Server  v3
- Downloads Verovio JS+WASM once to ./static/ on first run
- Serves Verovio from localhost (eliminates CDN issues)
- Proxies Anthropic API calls (API key stays server-side)
- Renders kern live with Verovio Python on demand
- TRANSPARENT PROXY CACHE: API responses are saved to ./cache/ by content
  hash so that repeat visitors (interviewers) never hit the Anthropic API.
  Populate the cache once by loading each demo piece with your API key;
  after that the app runs at zero cost indefinitely.

Usage:
    pip install flask flask-cors requests verovio
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 server.py
    open http://localhost:5001

Cache management:
    cache/ is created automatically.
    To force a fresh API call for a piece, delete its file(s) from cache/.
    To wipe everything: rm -rf cache/
"""

import os, json, hashlib
from pathlib import Path

# Python 3.x from python.org ships without macOS system CA certificates.
# certifi is installed; point both ssl and requests at its bundle.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE",      certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass  # certifi not available; proceed and hope system certs work

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests as req

app       = Flask(__name__, static_folder=".")
CORS(app)
API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
BASE      = Path(__file__).parent
STATIC    = BASE / "static"
KERN_DIR  = BASE / "kern"
CACHE_DIR = BASE / "cache"          # ← new: persisted API response cache

_render_cache = {}   # {piece_id: [svg_str, ...]}
_vrv = None          # single reused Verovio toolkit — fonts load once only

def _get_vrv():
    """Return a Verovio toolkit with resource path explicitly set.
    Creates once and reuses: fonts only load on first call.
    Creating a new toolkit() per request causes intermittent font failures."""
    global _vrv
    if _vrv is not None:
        return _vrv
    try:
        import verovio, os, sys
        _vrv = verovio.toolkit()

        # Search for Verovio's data/ directory (contains Bravura.json, Leipzig.json)
        candidates = []
        base = os.path.dirname(os.path.abspath(verovio.__file__))
        candidates += [
            os.path.join(base, "data"),                      # most common layout
            os.path.join(os.path.dirname(base), "data"),     # if __file__ is inside pkg
        ]
        try:
            import importlib.resources as ir
            candidates.append(str(ir.files("verovio").joinpath("data")))
        except Exception:
            pass
        for sp in sys.path:                                   # search all site-packages
            candidates.append(os.path.join(sp, "verovio", "data"))

        for path in candidates:
            if os.path.isdir(path) and any(
                f.endswith(".json") for f in os.listdir(path)
            ):
                _vrv.setResourcePath(path)
                print(f"  [Vrv] resource path set: {path}")
                break
        else:
            print("  [Vrv] WARNING: resource path not found — font errors may occur")

    except Exception as e:
        print(f"  [Vrv] ERROR creating toolkit: {e}")
    return _vrv

KERN_FILES = {
    "wtc1f19":    "wtc1f19.krn",
    "wtc1f20":    "wtc1f20.krn",
    "wtc1f24":    "wtc1f24.krn",
    "chopin-n72": "nocturne72-1.krn",
    "schub-op90": "op90-03.krn",
    "brahms-bal": "ballad10-1.krn",
    "brahms-w1":  "op39-01.krn",
    "beet-moon":  "beethoven-op27n2-1.krn",
    "beet-path":  "beethoven-op13-1.krn",
    "mozart-k331":"sonata11-1a.krn",
}

VRV_BASE  = "https://www.verovio.org/javascript/latest/"
VRV_FILES = ["verovio-toolkit-wasm.js"]  # WASM is embedded in the JS (Emscripten SINGLE_FILE build)

# ── Download Verovio on first run ─────────────────────────────────────────────
def ensure_verovio():
    STATIC.mkdir(exist_ok=True)
    for fname in VRV_FILES:
        dest = STATIC / fname
        if dest.exists():
            print(f"  ✓ {fname} ({dest.stat().st_size // 1024}KB cached)")
            continue
        print(f"  Downloading {fname} from verovio.org …", flush=True)
        r = req.get(VRV_BASE + fname, timeout=120, stream=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"    ✓ {len(r.content) // 1024}KB")

# ── Static files ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE, "Clavier_wasm.html")

@app.route("/static/<path:filename>")
def static_file(filename):
    return send_from_directory(STATIC, filename)

# ── Kern files ─────────────────────────────────────────────────────────────────
@app.route("/kern/<piece_id>")
def get_kern(piece_id):
    f = KERN_DIR / KERN_FILES.get(piece_id, "")
    if not f.exists():
        return f"Kern not found: {f}", 404
    return Response(f.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain")

# ── Score rendering (live, via Verovio Python) ────────────────────────────────
VRV_OPTS = dict(scale=45, pageWidth=2200, pageHeight=2970,
                breaks="auto", spacingStaff=6, spacingSystem=16)

def _clean_kern(raw: str) -> str:
    """Normalise kern text before passing to Verovio."""
    # Strip UTF-8 BOM
    text = raw.lstrip("\ufeff")
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text

@app.route("/render/<piece_id>")
def render_piece(piece_id):
    """Render all pages to SVG using Python Verovio (supports kern).
    Cached in memory — subsequent requests for the same piece are instant."""
    if piece_id in _render_cache:
        return jsonify(_render_cache[piece_id])
    vrv = _get_vrv()
    if vrv is None:
        return jsonify({"error": "verovio not installed — run: pip install verovio"}), 500
    f = KERN_DIR / KERN_FILES.get(piece_id, "")
    if not f.exists():
        return jsonify({"error": f"Kern file not found: {f}  (copy kern files into ./kern/)"}), 404
    kern = _clean_kern(f.read_text(encoding="utf-8", errors="replace"))
    vrv.setOptions(json.dumps(VRV_OPTS))
    ok = vrv.loadData(kern)
    if not ok:
        try:   vrv_log = vrv.getLog().strip()[-800:]
        except: vrv_log = "(getLog unavailable)"
        msg = f"Verovio parse failed for {piece_id}.\nLog:\n{vrv_log}"
        print(f"  ✗ {piece_id}: {vrv_log[-300:]}")
        return jsonify({"error": msg}), 500
    n = vrv.getPageCount()
    if n == 0:
        return jsonify({"error": f"{piece_id}: loadData ok but getPageCount() = 0"}), 500
    pages = [vrv.renderToSVG(i) for i in range(1, n + 1)]
    _render_cache[piece_id] = pages
    print(f"  ✓ {piece_id}: {n} pages")
    return jsonify(pages)

@app.route("/page-count/<piece_id>")
def page_count(piece_id):
    try:
        import verovio
        f = KERN_DIR / KERN_FILES.get(piece_id, "")
        if not f.exists(): return jsonify({"count":0,"ready":False})
        vrv = _get_vrv()
        if vrv is None: return jsonify({"count":0,"ready":False,"error":"verovio not installed"})
        vrv.setOptions(json.dumps(VRV_OPTS))
        vrv.loadData(f.read_text(encoding="utf-8", errors="replace"))
        return jsonify({"count": vrv.getPageCount(), "ready": True})
    except Exception as e:
        return jsonify({"count":0,"ready":False,"error":str(e)})

@app.route("/svg/<piece_id>/<int:page>")
def get_svg(piece_id, page):
    try:
        import verovio
        f = KERN_DIR / KERN_FILES.get(piece_id, "")
        if not f.exists(): return "Kern not found", 404
        vrv = _get_vrv()
        if vrv is None: return jsonify({"count":0,"ready":False,"error":"verovio not installed"})
        vrv.setOptions(json.dumps(VRV_OPTS))
        vrv.loadData(f.read_text(encoding="utf-8", errors="replace"))
        n = vrv.getPageCount()
        if page < 1 or page > n: return f"Page {page} out of range (1–{n})", 404
        return Response(vrv.renderToSVG(page, {}), mimetype="image/svg+xml")
    except Exception as e:
        return str(e), 500

# ── Proxy cache helpers ────────────────────────────────────────────────────────
def _cache_key(body: dict) -> str:
    """Stable 16-char hex key from model + message text content.

    The same agent prompt for the same piece will always produce the same key,
    so a populated cache means zero Anthropic calls for repeat visitors.
    """
    parts = [body.get("model", ""), body.get("system", "")]
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return digest[:16]

def _cache_stats() -> str:
    """Return a short summary of what's in the cache directory."""
    files = list(CACHE_DIR.glob("*.json"))
    total_kb = sum(f.stat().st_size for f in files) // 1024
    return f"{len(files)} entries, {total_kb}KB"

# ── Anthropic API proxy (with transparent caching) ────────────────────────────
@app.route("/api/messages", methods=["POST"])
def proxy():
    if not API_KEY and not any(CACHE_DIR.glob("*.json")):
        return jsonify({"error":{"message":"ANTHROPIC_API_KEY not set and cache is empty. "
                                            "export ANTHROPIC_API_KEY=sk-ant-... and restart."}}), 500

    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error":{"message":"Proxy: could not parse request body as JSON"}}), 400

    # ── Cache lookup ──────────────────────────────────────────────────────────
    key        = _cache_key(body)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        print(f"  [cache] HIT  {key}")
        return Response(cache_file.read_bytes(), status=200, mimetype="application/json")

    # ── No cache hit: forward to Anthropic ───────────────────────────────────
    if not API_KEY:
        return jsonify({"error":{"message":
            f"Cache miss (key={key}) and ANTHROPIC_API_KEY not set. "
            "Set the key and reload this piece to populate the cache."}}), 500

    print(f"  [cache] MISS {key} — calling Anthropic …")
    try:
        for attempt in range(3):
            try:
                r = req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type":    "application/json",
                             "x-api-key":        API_KEY,
                             "anthropic-version":"2023-06-01",
                             "Connection":        "close"},
                    json=body, timeout=120)
                break
            except req.exceptions.ConnectionError as ce:
                if attempt == 2:
                    raise
                print(f"  [proxy] connection error attempt {attempt+1}: {ce} — retrying")
                import time; time.sleep(1)

        ct = r.headers.get("content-type", "")
        if "json" in ct:
            # Save successful responses to cache
            if r.status_code == 200:
                CACHE_DIR.mkdir(exist_ok=True)
                cache_file.write_bytes(r.content)
                print(f"  [cache] SAVE {key}  ({len(r.content)//1024}KB) — {_cache_stats()}")
            return Response(r.content, status=r.status_code, mimetype="application/json")

        # Anthropic returned non-JSON (e.g. CDN error page) — wrap it
        return jsonify({"error":{"message":f"Anthropic HTTP {r.status_code}: {r.text[:300]}"}}), r.status_code

    except req.exceptions.Timeout:
        return jsonify({"error":{"message":"Request to Anthropic timed out (120 s)"}}), 504
    except Exception as e:
        return jsonify({"error":{"message":f"Proxy exception: {e}"}}), 500

# ── Cache info endpoint (optional, handy during setup) ────────────────────────
@app.route("/cache-status")
def cache_status():
    files  = list(CACHE_DIR.glob("*.json"))
    return jsonify({
        "entries":   len(files),
        "total_kb":  sum(f.stat().st_size for f in files) // 1024,
        "keys":      [f.stem for f in sorted(files)],
    })

if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True)
    print("Clavier Local Server  v3")
    print("─" * 40)
    print(f"API key : {'✓ set' if API_KEY else '⚠ NOT SET — export ANTHROPIC_API_KEY=sk-ant-...'}")
    print(f"Cache   : {CACHE_DIR}  ({_cache_stats()})")
    print("Checking Verovio files …")
    ensure_verovio()
    print(f"\nOpen: http://0.0.0.0:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
