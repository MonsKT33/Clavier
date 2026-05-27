#!/usr/bin/env python3
"""
Clavier Cache Builder
======================
Pre-populates cache/ with Anthropic API responses for every piece in the
corpus so that server.py can serve the app at zero API cost to visitors.

Replicates the exact agent pipeline from Clavier_wasm.html:
  Phase 1 (parallel): Agent 2 (harmonic analysis) + Eval (sources)
  Phase 2 (parallel): Agent 3 (overview/tags)    + Sim (similarity)
  Phase 3 (sequential): Soul (programme note, needs top Sim result)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 build_cache.py                         # cache all 10 pieces
    python3 build_cache.py --pieces beet-path mozart-k331 chopin-n72
    python3 build_cache.py --dry-run               # show what would be called
    python3 build_cache.py --skip-cached           # skip pieces already in cache

Requirements:
    pip install requests
    Kern files must be in ./kern/  (same as server.py expects)
"""

import os, sys, json, hashlib, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
CACHE_DIR = Path(__file__).parent / "cache"
KERN_DIR  = Path(__file__).parent / "kern"
MODEL     = "claude-sonnet-4-6"

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

PIECES = [
    {"id":"wtc1f19",    "composer":"Bach, J.S.",      "title":"WTC I — Fugue 19 in A major",  "opus":"BWV 864b",      "era":"Baroque",   "key":"A major",  "sig":"9/8"},
    {"id":"wtc1f20",    "composer":"Bach, J.S.",      "title":"WTC I — Fugue 20 in A minor",  "opus":"BWV 865b",      "era":"Baroque",   "key":"A minor",  "sig":"4/4"},
    {"id":"wtc1f24",    "composer":"Bach, J.S.",      "title":"WTC I — Fugue 24 in B minor",  "opus":"BWV 869b",      "era":"Baroque",   "key":"B minor",  "sig":"4/4"},
    {"id":"chopin-n72", "composer":"Chopin, F.",      "title":"Nocturne in E minor",          "opus":"Op.72 No.1",    "era":"Romantic",  "key":"E minor",  "sig":"4/4"},
    {"id":"schub-op90", "composer":"Schubert, F.",    "title":"Impromptu in G♭ major",        "opus":"D.899 Op.90/3", "era":"Romantic",  "key":"G♭ major", "sig":"4/2"},
    {"id":"brahms-bal", "composer":"Brahms, J.",      "title":"Ballade No.1 in D minor",      "opus":"Op.10 No.1",    "era":"Romantic",  "key":"D minor",  "sig":"4/4"},
    {"id":"brahms-w1",  "composer":"Brahms, J.",      "title":"Waltz in B major",             "opus":"Op.39 No.1",    "era":"Romantic",  "key":"B major",  "sig":"3/4"},
    {"id":"beet-moon",  "composer":"Beethoven, L.v.", "title":"Sonata 'Moonlight' — I",       "opus":"Op.27 No.2",    "era":"Classical", "key":"C♯ minor", "sig":"2/2"},
    {"id":"beet-path",  "composer":"Beethoven, L.v.", "title":"Sonata 'Pathétique' — I",      "opus":"Op.13",         "era":"Classical", "key":"C minor",  "sig":"4/4"},
    {"id":"mozart-k331","composer":"Mozart, W.A.",    "title":"Sonata K.331 — Thema",         "opus":"K.331",         "era":"Classical", "key":"A major",  "sig":"6/8"},
]
PIECE_MAP = {p["id"]: p for p in PIECES}

# ── System prompts (copied verbatim from Clavier_wasm.html) ──────────────────
A2_SYS = """You are Agent 2 in Clavier — the Harmonic Analyzer. Input: full Humdrum kern score.
Return ONLY valid JSON, no markdown, no code fences:
{"key":string,"time_signature":string,"total_bars":number,"formal_sections":[{"label":string,"bar_start":number,"bar_end":number}],"key_regions":[{"key":string,"bar_start":number,"bar_end":number}],"modulations":[{"bar":number,"from_key":string,"to_key":string,"type":string,"pivot_chord":string|null}],"cadences":[{"bar":number,"beat":number,"type":string,"key":string}],"chord_labels":[{"bar":number,"beat":number,"roman_numeral":string}],"themes":[{"id":string,"bar_start":number,"bar_end":number,"description":string}]}
chord_labels: structurally significant chords only — max 2 per bar (strong beats + cadence/modulation chords). Roman numeral only, no chord name needed.
Cadence types: PAC, IAC, HC, DC, Deceptive, Evaded, Plagal.
Modulation types: diatonic,chromatic_mediant,neapolitan,enharmonic,modal_mixture,tonicization.
Scholarly. Correct Roman numerals with accidentals where needed."""

A3_SYS = """You are Agent 3 in Clavier. Given Agent 2's harmonic analysis, write a 2-sentence overview and assign cross-piece tags.
Return ONLY valid JSON, no markdown:
{"overview":string,"cross_piece_tags":[string]}
overview: the piece's harmonic character and most distinctive feature.
cross_piece_tags: 4-8 tags from: fugue prelude nocturne impromptu ballade sonata_form ternary binary theme_and_variations chromatic_mediant neapolitan modal_mixture enharmonic_modulation circle_of_fifths pedal_point counterpoint arpeggiated_bass alberti_bass lyrical tragic dramatic meditative intimate melancholic"""

EVAL_SYS = """Musicology research assistant. Find 3 authoritative published analyses. Priority: peer-reviewed journals, then academic books, then Grove Music.
Return ONLY valid JSON: {"sources":[{"title":string,"author":string,"year":number|null,"type":"academic"|"book"|"criticism","publication":string,"url":string,"description":string}]}"""

SIM_SYS = """Music similarity analyst. Given the current piece, do two tasks:
1) Rate similarity (0-100) to each corpus piece.
2) From training knowledge, group corpus pieces by shared elements.
Return ONLY valid JSON, no markdown:
{"similarities":[{"id":string,"score":number,"reason":string}],"corpus_by_cadences":[{"cadence_type":string,"pieces":[string]}],"corpus_by_modulations":[{"modulation_type":string,"pieces":[string]}],"corpus_by_tags":[{"tag":string,"pieces":[string]}]}
reason≤12 words. Only groups with ≥1 piece. Use exact corpus IDs."""

SOUL_SYS = """You are the Soul Writer for Clavier. Write three short paragraphs about a piano piece for a pianist without conservatory training who wants to understand what makes this piece distinctive and how it connects to a similar work.

You receive Agent 2's full harmonic analysis JSON for the current piece, plus metadata about the most harmonically similar piece in the corpus.

Write in programme-note style: analytically precise, bar-cited where possible from Agent 2's JSON, but accessible enough that the pianist can explain it to a non-musician audience.

Return ONLY valid JSON, no markdown:
{"signature":string,"significance":string,"connection":string}

signature (2–3 sentences): What makes this piece harmonically and structurally itself. Cite specific bar numbers, key regions, cadence types, or modulations directly from Agent 2's JSON. Be factually accurate — do not invent musical events not present in the data.

significance (1–2 sentences): Why that harmonic character matters — what emotional or expressive effect it creates. Plain language only; no jargon a non-musician could not follow.

connection (2–3 sentences): How this piece's defining character echoes in or was inspired by the most similar piece. Draw on your musicological knowledge of the similar piece. Do not fabricate bar numbers for the similar piece since it has not been analysed — instead describe the shared technique or expressive instinct precisely. Phrase claims about the similar piece as informed observations, not certainties."""

# ── Cache key (must match server.py exactly) ──────────────────────────────────
def cache_key(body: dict) -> str:
    parts = [body.get("model", ""), body.get("system", "")]
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]

# ── Anthropic call ─────────────────────────────────────────────────────────────
def call_api(body: dict, label: str, dry_run=False) -> dict:
    """POST to Anthropic, save to cache, return parsed response JSON."""
    import requests
    key  = cache_key(body)
    dest = CACHE_DIR / f"{key}.json"

    if dest.exists():
        print(f"    ✓ {label} — cache hit ({key})")
        return json.loads(dest.read_bytes())

    if dry_run:
        print(f"    ~ {label} — would call API (key={key})")
        return {}

    print(f"    → {label} … ", end="", flush=True)
    t0 = time.time()

    for attempt in range(3):
        try:
            import requests as req
            r = req.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type":    "application/json",
                         "x-api-key":        API_KEY,
                         "anthropic-version":"2023-06-01",
                         "Connection":        "close"},
                json=body, timeout=180)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                print(f"FAILED ({e})")
                raise
            print(f"retry {attempt+1}… ", end="", flush=True)
            time.sleep(2 ** attempt)

    elapsed = time.time() - t0
    dest.write_bytes(r.content)
    resp = r.json()
    tokens = resp.get("usage", {})
    print(f"done in {elapsed:.1f}s  (in={tokens.get('input_tokens','?')} out={tokens.get('output_tokens','?')})  saved → {key}.json")
    return resp

# ── JSON extraction (mirrors frontend parseJ) ─────────────────────────────────
def extract_json(resp: dict):
    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None

# ── Request body builders (must match callA2/callA3/etc. in the HTML exactly) ─
def body_a2(kern: str) -> dict:
    return {
        "model": MODEL, "max_tokens": 8192, "system": A2_SYS,
        "messages": [{"role": "user", "content": f"Analyze this complete kern score:\n\n{kern}"}]
    }

def body_eval(piece: dict) -> dict:
    return {
        "model": MODEL, "max_tokens": 2000, "system": EVAL_SYS,
        "messages": [{"role": "user",
                      "content": f"Identify 3 authoritative published analyses of: "
                                 f"{piece['composer']} — {piece['title']} {piece['opus']}"}]
    }

def body_a3(a2data: dict) -> dict:
    return {
        "model": MODEL, "max_tokens": 2000, "system": A3_SYS,
        "messages": [{"role": "user", "content": json.dumps(a2data, separators=(',', ':'), ensure_ascii=False)}]
    }

def body_sim(a2data: dict, piece: dict) -> dict:
    others   = [p for p in PIECES if p["id"] != piece["id"]]
    mod_types = ", ".join(dict.fromkeys(m["type"] for m in a2data.get("modulations", []))) or "none"
    cad_types = ", ".join(dict.fromkeys(c["type"] for c in a2data.get("cadences", [])))    or "none"
    content = (
        f'Current: "{piece["composer"]} — {piece["title"]}" ({piece["era"]}, {piece["key"]})\n'
        f"Modulation types: {mod_types} | Cadence types: {cad_types}\n"
        f"Corpus IDs:\n" +
        "\n".join(f'"{p["id"]}": {p["composer"]} — {p["title"]} ({p["era"]})' for p in others)
    )
    return {"model": MODEL, "max_tokens": 2000, "system": SIM_SYS,
            "messages": [{"role": "user", "content": content}]}

def body_soul(a2data: dict, piece: dict, top_sim: dict) -> dict:
    top_piece = PIECE_MAP[top_sim["id"]]
    content = (
        f"Current piece: {piece['composer']} — {piece['title']} ({piece['opus']}), "
        f"{piece['era']}, {piece['key']}\n\n"
        f"Agent 2 harmonic analysis:\n{json.dumps(a2data, separators=(',', ':'), ensure_ascii=False)}\n\n"
        f"Most similar piece in corpus: {top_piece['composer']} — {top_piece['title']} "
        f"({top_piece['opus']}), {top_piece['era']}, {top_piece['key']}\n"
        f"Similarity score: {top_sim['score']}%\nSimilarity reason: {top_sim['reason']}"
    )
    return {"model": MODEL, "max_tokens": 1500, "system": SOUL_SYS,
            "messages": [{"role": "user", "content": content}]}

# ── Per-piece pipeline ────────────────────────────────────────────────────────
def cache_piece(piece_id: str, dry_run=False, skip_cached=False) -> bool:
    piece = PIECE_MAP.get(piece_id)
    if not piece:
        print(f"  ✗ Unknown piece id: {piece_id}")
        return False

    kern_path = KERN_DIR / KERN_FILES.get(piece_id, "")
    if not kern_path.exists():
        print(f"  ✗ Kern file missing: {kern_path}")
        return False

    kern = kern_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")

    # Check if already fully cached (all 5 agents)
    if skip_cached:
        bodies = [body_a2(kern), body_eval(piece)]
        all_hit = all((CACHE_DIR / f"{cache_key(b)}.json").exists() for b in bodies)
        if all_hit:
            print(f"  ↷ {piece_id} — A2+Eval already cached, skipping (use without --skip-cached to force)")
            return True

    print(f"\n  ── {piece['composer']} — {piece['title']} ({piece['opus']}) ──")

    # Phase 1: A2 + Eval in parallel (independent)
    print(f"  Phase 1: Agent 2 + Eval")
    a2_resp = ev_resp = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a2 = ex.submit(call_api, body_a2(kern),     "Agent 2", dry_run)
        fut_ev = ex.submit(call_api, body_eval(piece),  "Eval",    dry_run)
        a2_resp = fut_a2.result()
        ev_resp = fut_ev.result()

    a2data = extract_json(a2_resp)
    if not a2data:
        if not dry_run:
            print(f"  ✗ Agent 2 returned no valid JSON — skipping phases 2+3 for {piece_id}")
            return False
        # In dry-run, continue with empty placeholder
        a2data = {"modulations": [], "cadences": [], "key": piece["key"]}

    # Phase 2: A3 + Sim in parallel (both need A2 output)
    print(f"  Phase 2: Agent 3 + Sim")
    a3_resp = sim_resp = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a3  = ex.submit(call_api, body_a3(a2data),        "Agent 3", dry_run)
        fut_sim = ex.submit(call_api, body_sim(a2data, piece), "Sim",     dry_run)
        a3_resp  = fut_a3.result()
        sim_resp = fut_sim.result()

    sim_data = extract_json(sim_resp)
    if not sim_data or not sim_data.get("similarities"):
        if not dry_run:
            print(f"  ✗ Sim returned no similarities — skipping Soul for {piece_id}")
            return False
        sim_data = {"similarities": [{"id": PIECES[0]["id"], "score": 80, "reason": "placeholder"}]}

    # Phase 3: Soul (needs Sim's top match)
    print(f"  Phase 3: Soul")
    top_sim = sorted(sim_data["similarities"], key=lambda x: x["score"], reverse=True)[0]
    call_api(body_soul(a2data, piece, top_sim), "Soul", dry_run)

    return True

# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-populate Clavier cache/ with all Anthropic API responses.")
    parser.add_argument("--pieces", nargs="+", metavar="ID",
        help=f"Piece IDs to cache (default: all). Options: {', '.join(KERN_FILES)}")
    parser.add_argument("--dry-run", action="store_true",
        help="Print what would be called without hitting the API.")
    parser.add_argument("--skip-cached", action="store_true",
        help="Skip pieces whose A2+Eval responses are already cached.")
    args = parser.parse_args()

    if not args.dry_run and not API_KEY:
        print("✗ ANTHROPIC_API_KEY not set.\n  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("✗ requests not installed.\n  pip install requests")
        sys.exit(1)

    CACHE_DIR.mkdir(exist_ok=True)
    target_ids = args.pieces or list(KERN_FILES.keys())
    invalid    = [i for i in target_ids if i not in KERN_FILES]
    if invalid:
        print(f"✗ Unknown piece IDs: {', '.join(invalid)}")
        print(f"  Valid IDs: {', '.join(KERN_FILES)}")
        sys.exit(1)

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"\nClavier Cache Builder  {mode}")
    print(f"{'─' * 50}")
    print(f"Pieces : {', '.join(target_ids)}")
    print(f"Cache  : {CACHE_DIR}")
    existing = len(list(CACHE_DIR.glob("*.json")))
    print(f"Cached : {existing} files already in cache/")
    print(f"Model  : {MODEL}")
    print(f"{'─' * 50}")

    t_start  = time.time()
    ok = fail = 0
    for piece_id in target_ids:
        success = cache_piece(piece_id, dry_run=args.dry_run, skip_cached=args.skip_cached)
        if success: ok += 1
        else:        fail += 1

    elapsed = time.time() - t_start
    new_files = len(list(CACHE_DIR.glob("*.json"))) - existing
    print(f"\n{'─' * 50}")
    print(f"Done in {elapsed:.0f}s — {ok} pieces succeeded, {fail} failed, {new_files} new cache files")
    if not args.dry_run and ok:
        print(f"\nNext step: remove your API key from the environment.")
        print(f"The server will now serve all cached responses at zero cost.")
        print(f"\n  Verify: curl http://localhost:5001/cache-status")

if __name__ == "__main__":
    main()
