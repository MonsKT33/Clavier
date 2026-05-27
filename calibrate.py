"""
calibrate.py — Clavier Calibration Harness (Day 5 / Fork 1)

Compares Agent 2's analysis against DCML Mozart Piano Sonatas ground truth
on three dimensions: global key, cadences (F1 ± 1 bar), chord labels (3 resolutions).

Prerequisites:
    pip install anthropic pandas

    # Clone the DCML Mozart corpus into data/dcml_mozart:
    git clone https://github.com/DCMLab/mozart_piano_sonatas data/dcml_mozart

    # Confirm your kern files are in the path set by KERN_DIR below.
    # KernScores Mozart sonatas are typically named sonata01-1.krn etc.

    export ANTHROPIC_API_KEY=sk-...

Usage:
    python calibrate.py                          # runs first 5 pieces in PIECE_MAP
    python calibrate.py --pieces K279-1 K283-1   # specific pieces
    python calibrate.py --all                     # all 20 pieces in PIECE_MAP
    python calibrate.py --dry-run                 # parse ground truth only, skip Agent 2
"""

import os, json, re, argparse, time
from pathlib import Path
import pandas as pd
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_MODEL = "claude-sonnet-4-6"
DCML_DIR        = Path("data/dcml_mozart")
KERN_DIR        = Path("kern/mozart_sonata")       # ← adjust to your local kern folder
OUTPUT_DIR      = Path("calibration_output")

# ── Agent 2 system prompt ─────────────────────────────────────────────────────
# PASTE THE EXACT SYSTEM PROMPT FROM server.py HERE.
# This must match what the app sends so harness outputs are comparable to
# what you see in the browser. Without this, scores won't reflect real app fidelity.

AGENT2_SYSTEM = """You are Agent 2 in Clavier — the Harmonic Analyzer. Input: full Humdrum kern score.
Return ONLY valid JSON, no markdown, no code fences:
{"key":string,"time_signature":string,"total_bars":number,"formal_sections":[{"label":string,"bar_start":number,"bar_end":number}],"key_regions":[{"key":string,"bar_start":number,"bar_end":number}],"modulations":[{"bar":number,"from_key":string,"to_key":string,"type":string,"pivot_chord":string|null}],"cadences":[{"bar":number,"beat":number,"type":string,"key":string}],"chord_labels":[{"bar":number,"beat":number,"roman_numeral":string}],"themes":[{"id":string,"bar_start":number,"bar_end":number,"description":string}]}
chord_labels: structurally significant chords only — max 2 per bar (strong beats + cadence/modulation chords). Roman numeral only, no chord name needed.
Cadence types: PAC, IAC, HC, DC, Deceptive, Evaded, Plagal.
Modulation types: diatonic,chromatic_mediant,neapolitan,enharmonic,modal_mixture,tonicization.
Scholarly. Correct Roman numerals with accidentals where needed."""

# ── Piece map: DCML ID → kern filename ───────────────────────────────────────
# These are calibration pieces, NOT the app's 10-piece corpus.
# Kern files for Mozart sonatas come from KernScores:
#   http://kern.humdrum.org/search?s=composer&q=mozart  (filter for piano sonatas)
# Download and place in KERN_DIR. KernScores naming for Mozart sonatas is typically
# sonata01-1.krn (K279 mvt1), sonata05-1.krn (K283 mvt1), etc.
# Verify filenames against your local download before running.
#
# The DCML Mozart corpus (data/dcml_mozart) covers all 18 sonatas.
# K.331 is the one piece that overlaps with the current app corpus (mozart-k331).
# For the app's other 9 pieces (Bach, Beethoven, Schubert, Chopin, Brahms),
# DCML ground truth exists in separate corpora — add them here as you expand.

PIECE_MAP = {
    # K279 Sonata No.1 in C major
    "K279-1": "sonata01-1.krn",
    "K279-2": "sonata01-2.krn",
    "K279-3": "sonata01-3.krn",
    # K280 Sonata No.2 in F major
    "K280-1": "sonata02-1.krn",
    "K280-2": "sonata02-2.krn",
    "K280-3": "sonata02-3.krn",
    # K283 Sonata No.5 in G major
    "K283-1": "sonata05-1.krn",
    "K283-2": "sonata05-2.krn",
    "K283-3": "sonata05-3.krn",
    # K310 Sonata No.8 in A minor
    "K310-1": "sonata08-1.krn",
    "K310-2": "sonata08-2.krn",
    "K310-3": "sonata08-3.krn",
    # K330 Sonata No.10 in C major
    "K330-1": "sonata10-1.krn",
    # K331 Sonata No.11 in A major (theme & variations + Rondo alla Turca)
    "K331-1": "sonata11-1.krn",
    "K331-3": "sonata11-3.krn",   # Rondo alla Turca — not sonata form: good edge case
    # K332 Sonata No.12 in F major
    "K332-1": "sonata12-1.krn",
    # K333 Sonata No.13 in B♭ major
    "K333-1": "sonata13-1.krn",
    # K545 Sonata No.16 in C major ("Sonata facile")
    "K545-1": "sonata16-1.krn",
    # K457 Sonata No.14 in C minor
    "K457-1": "sonata14-1.krn",
}

# Both DCML labels and Agent 2 labels → canonical form for comparison.
# DCML uses short codes; Agent 2 uses "Deceptive"/"Evaded"/"Plagal" as full words
# (its prompt lists "PAC, IAC, HC, DC, Deceptive, Evaded, Plagal" — so both DC and
# Deceptive are valid; we normalise everything to short canonical codes.
CANONICAL_CADENCE = {
    # DCML codes
    "PAC": "PAC", "IAC": "IAC", "HC": "HC",
    "DC":  "DC",  "EC":  "EC",  "PC": "PC",
    # Agent 2 full-word variants
    "Deceptive": "DC",
    "Evaded":    "EC",
    "Plagal":    "PC",
}

# ── DCML ground truth parser ──────────────────────────────────────────────────

def load_dcml_ground_truth(dcml_id: str) -> dict | None:
    """
    Parse DCML TSVs for one movement into a normalised ground-truth dict.

    DCML Mozart corpus structure (verify against your clone):
      data/dcml_mozart/harmonies/{dcml_id}.harmonies.tsv
      data/dcml_mozart/cadences/{dcml_id}.tsv   (may be embedded in harmonies)

    Key DCML columns used:
      mn          — measure number (displayed bar number, matches Agent 2 "bar")
      globalkey   — piece key: uppercase = major, lowercase = minor (e.g. "C", "d")
      localkey    — local tonal context per chord
      numeral     — Roman numeral (e.g. "I", "V", "ii")
      form        — chord form modifier ("M", "m", "o", "+", "%")
      figbass     — figured bass (e.g. "7", "65")
      cadence     — cadence type if applicable, else NaN
    """
    harm_path = DCML_DIR / "harmonies" / f"{dcml_id}.harmonies.tsv"
    if not harm_path.exists():
        print(f"    ⚠ DCML file not found: {harm_path}")
        print(f"      Check DCML_DIR ({DCML_DIR}) and that the corpus is cloned.")
        return None

    df = pd.read_csv(harm_path, sep="\t")

    # Global key: first non-null globalkey value
    global_key_raw = df["globalkey"].dropna().iloc[0] if "globalkey" in df.columns else "?"

    # Chords: one row per chord event, keyed by measure number
    chord_cols = [c for c in ["mn", "numeral", "form", "figbass", "localkey"] if c in df.columns]
    chords = df[chord_cols].dropna(subset=["numeral"]).to_dict("records")

    # Cadences: rows where cadence column is not NaN
    cad_df = df[df["cadence"].notna()] if "cadence" in df.columns else pd.DataFrame()
    cadences = cad_df[["mn", "cadence", "localkey"]].to_dict("records") if not cad_df.empty else []

    return {
        "global_key_raw": global_key_raw,
        "global_key": dcml_key_to_human(global_key_raw),
        "chords": chords,
        "cadences": cadences,
    }


def dcml_key_to_human(dcml_key: str) -> str:
    """Convert DCML key notation to human-readable. 'C' → 'C major', 'd' → 'D minor'."""
    if not dcml_key:
        return "?"
    # DCML: uppercase = major, lowercase = minor
    # May have accidentals: "bb" = B♭ major, "f#" = F# major, "bb" ambiguous
    # Simple approach: if first char is uppercase → major, else → minor
    root = dcml_key[0].upper() + dcml_key[1:].replace("b", "♭").replace("#", "♯")
    mode = "major" if dcml_key[0].isupper() else "minor"
    return f"{root} {mode}"


# ── Agent 2 runner ────────────────────────────────────────────────────────────

def run_agent2(kern_content: str, retries: int = 4) -> dict:
    """Call Agent 2 directly with the same model and system prompt as the app.
    Retries on 529 overloaded errors with exponential backoff."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=8192,
                system=AGENT2_SYSTEM,
                messages=[{"role": "user", "content": f"Analyze this complete kern score:\n\n{kern_content}"}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < retries - 1:
                wait = 2 ** attempt * 10   # 10s, 20s, 40s, 80s
                print(f"  ⚠ API overloaded (529) — retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise


# ── Scorers ───────────────────────────────────────────────────────────────────

def score_key(agent2: dict, gt: dict) -> dict:
    """
    Score global key. Exact match on root + mode.
    Returns {"score": 0|1, "agent2": str, "dcml": str, "match": bool}
    """
    a2_key  = normalise_key(agent2.get("key", ""))
    gt_key  = normalise_key(gt["global_key"])
    match   = (a2_key == gt_key)
    return {
        "score": 1.0 if match else 0.0,
        "match": match,
        "agent2": agent2.get("key", "—"),
        "dcml":   gt["global_key"],
    }


def normalise_key(key_str: str) -> str:
    """Normalise key string for comparison: 'D minor' → 'd minor', 'C major' → 'c major'."""
    s = key_str.strip().lower()
    s = s.replace("♭", "b").replace("♯", "#")
    return s


def score_cadences(agent2: dict, gt: dict, tolerance: int = 1) -> dict:
    """
    Precision / Recall / F1 for cadence detection, ±tolerance bars.
    Also classifies each hit/miss as a disagreement record for the iteration loop.
    """
    a2_cads = {c["bar"]: CANONICAL_CADENCE.get(c["type"], c["type"]) for c in agent2.get("cadences", [])}
    gt_cads = {int(c["mn"]): CANONICAL_CADENCE.get(c["cadence"], c["cadence"]) for c in gt["cadences"]}

    tp = fp = fn = 0
    disagreements = []

    # Ground truth pass: did Agent 2 find it?
    matched_a2_bars = set()
    for gt_bar, gt_type in gt_cads.items():
        found = False
        for offset in range(-tolerance, tolerance + 1):
            probe = gt_bar + offset
            if probe in a2_cads:
                tp += 1
                matched_a2_bars.add(probe)
                found = True
                a2_type = a2_cads[probe]
                dcml_type = gt_cads[gt_bar]
                if a2_type != dcml_type:
                    disagreements.append({
                        "dimension": "cadence_type",
                        "bar": gt_bar,
                        "agent2": a2_type,
                        "dcml": dcml_type,
                        "severity": "medium",
                        "evidence": f"DCML labels m.{gt_bar} as {gt_type}; Agent 2 says {a2_type} at m.{probe}",
                    })
                break
        if not found:
            fn += 1
            disagreements.append({
                "dimension": "cadence_missed",
                "bar": gt_bar,
                "agent2": None,
                "dcml": gt_cads[gt_bar],
                "severity": "high",
                "evidence": f"DCML has {gt_cads[gt_bar]} at m.{gt_bar}; Agent 2 has nothing within ±{tolerance} bars",
            })

    # False positives: Agent 2 cadences not in ground truth
    for a2_bar in a2_cads:
        if a2_bar not in matched_a2_bars:
            near_gt = any(abs(a2_bar - g) <= tolerance for g in gt_cads)
            if not near_gt:
                fp += 1
                disagreements.append({
                    "dimension": "cadence_extra",
                    "bar": a2_bar,
                    "agent2": a2_cads[a2_bar],
                    "dcml": None,
                    "severity": "low",
                    "evidence": f"Agent 2 adds {a2_cads[a2_bar]} at m.{a2_bar}; DCML has no cadence nearby",
                })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "disagreements":  disagreements,
    }


def score_chord_labels(agent2: dict, gt: dict, tolerance: int = 1) -> dict:
    """
    Cadence-anchored chord check.

    Old approach compared Agent 2's structural chord picks against DCML's
    first-chord-per-bar, producing noise (different moments, same bar).

    New approach: for each DCML cadence bar, check whether Agent 2 has
    any chord label within ±tolerance bars. If it does, check whether the
    root matches the DCML chord at that cadence point.

    Metrics:
      cadence_coverage   — % of DCML cadence bars where Agent 2 has any
                           chord label nearby (did it label something here?)
      cadence_root_match — % of covered bars where the root degree matches
                           (is what it labeled correct?)
      n_cadence_bars     — total DCML cadence bars evaluated
    """
    # Agent 2 chords indexed by bar, all entries (not just first)
    a2_by_bar: dict[int, list[str]] = {}
    for c in agent2.get("chord_labels", []):
        rn = c.get("roman_numeral") or c.get("chord") or c.get("label")
        if not rn or "bar" not in c:
            continue
        bar = int(c["bar"])
        a2_by_bar.setdefault(bar, []).append(rn)

    # DCML cadence bars with their chord context
    # Use the DCML harmonies at the cadence bar to get the expected chord
    gt_chord_by_bar: dict[int, str] = {}
    for c in gt["chords"]:
        bar = int(c["mn"])
        if c.get("numeral") and bar not in gt_chord_by_bar:
            gt_chord_by_bar[bar] = build_dcml_rn(c)

    cadence_bars = [int(c["mn"]) for c in gt["cadences"]]
    if not cadence_bars:
        return {"cadence_coverage": 0.0, "cadence_root_match": 0.0,
                "n_cadence_bars": 0, "disagreements": []}

    covered = root_matched = 0
    disagreements = []

    for gt_bar in sorted(cadence_bars):
        # Collect all Agent 2 chord labels within ±tolerance bars
        nearby_rns = []
        for offset in range(-tolerance, tolerance + 1):
            nearby_rns.extend(a2_by_bar.get(gt_bar + offset, []))

        gt_rn = gt_chord_by_bar.get(gt_bar, "?")
        gt_root = extract_root(gt_rn) if gt_rn != "?" else "?"

        if not nearby_rns:
            # Agent 2 has no chord label anywhere near this cadence
            disagreements.append({
                "dimension": "chord_at_cadence_missing",
                "bar": gt_bar,
                "agent2": None,
                "dcml": gt_rn,
                "severity": "medium",
                "evidence": f"m.{gt_bar}: Agent 2 has no chord label within ±{tolerance} bars of DCML cadence ({gt_rn})",
            })
            continue

        covered += 1

        # Check if any nearby Agent 2 chord matches the expected root
        a2_roots = [extract_root(rn) for rn in nearby_rns]
        root_match = gt_root in a2_roots

        if root_match:
            root_matched += 1
        else:
            best_a2 = nearby_rns[0]  # closest (first within window)
            disagreements.append({
                "dimension": "chord_at_cadence_root_mismatch",
                "bar": gt_bar,
                "agent2": nearby_rns,
                "dcml": gt_rn,
                "severity": "low",
                "evidence": (
                    f"m.{gt_bar}: DCML cadence chord={gt_rn} (root {gt_root}), "
                    f"Agent 2 nearby={nearby_rns} (roots {a2_roots})"
                ),
            })

    n = len(cadence_bars)
    return {
        "cadence_coverage":   round(covered       / n, 3),
        "cadence_root_match": round(root_matched   / covered, 3) if covered else 0.0,
        "n_cadence_bars":     n,
        "disagreements":      disagreements,
    }


# ── Roman numeral helpers ─────────────────────────────────────────────────────

def build_dcml_rn(chord_row: dict) -> str:
    """Reconstruct a Roman numeral string from DCML chord row fields."""
    numeral = chord_row.get("numeral") or "?"
    form    = chord_row.get("form")    or ""
    figbass = chord_row.get("figbass") or ""
    # Coerce pandas NaN (float) to empty string
    if not isinstance(form,    str): form    = ""
    if not isinstance(figbass, str): figbass = ""
    form_map = {"M": "", "m": "", "o": "o", "+": "+", "%": "%", "": ""}
    suffix = form_map.get(form, form) + figbass
    return f"{numeral}{suffix}".strip()


def extract_root(rn: str) -> str:
    """Extract just the scale-degree numeral. 'bVII7' → 'VII', 'ii65' → 'II'."""
    m = re.match(r"[#b♭♯]*([IiVv]+)", rn)
    return m.group(1).upper() if m else rn.upper()


def rn_to_root_quality(rn: str) -> str:
    """Reduce RN to root + quality. 'V7' → 'V-M', 'ii6' → 'II-m', 'viio' → 'VII-d'."""
    root = extract_root(rn)
    # Infer quality: uppercase numeral = major, lowercase = minor
    m = re.match(r"[#b♭♯]*([IiVv]+)(.*)", rn)
    if not m:
        return rn
    numeral_raw = m.group(1)
    suffix = m.group(2)
    if "o" in suffix and "%" not in suffix:
        quality = "d"   # diminished
    elif "+" in suffix:
        quality = "a"   # augmented
    elif "%" in suffix:
        quality = "hd"  # half-diminished
    elif numeral_raw == numeral_raw.lower():
        quality = "m"   # minor
    else:
        quality = "M"   # major
    return f"{root}-{quality}"


def normalise_rn_full(rn: str) -> str:
    """Normalise full RN for comparison: strip whitespace, normalise accidentals."""
    return rn.strip().replace(" ", "").replace("♭", "b").replace("♯", "#").replace("b", "b")


# ── Per-piece calibration run ─────────────────────────────────────────────────

def run_calibration(piece_id: str, dry_run: bool = False) -> dict:
    kern_filename = PIECE_MAP.get(piece_id)
    if not kern_filename:
        return {"piece": piece_id, "error": f"No entry in PIECE_MAP for {piece_id}"}

    kern_path = KERN_DIR / kern_filename
    if not kern_path.exists():
        return {"piece": piece_id, "error": f"Kern file not found: {kern_path}"}

    # Load ground truth
    print(f"  Loading DCML ground truth...")
    gt = load_dcml_ground_truth(piece_id)
    if gt is None:
        return {"piece": piece_id, "error": "DCML ground truth not found — check DCML_DIR"}

    # Run Agent 2 (skip in dry-run mode)
    if dry_run:
        print(f"  [dry-run] Skipping Agent 2 call.")
        return {"piece": piece_id, "dry_run": True, "dcml_key": gt["global_key"],
                "cadences_in_gt": len(gt["cadences"]), "chords_in_gt": len(gt["chords"])}

    print(f"  Running Agent 2 ({ANTHROPIC_MODEL})...")
    kern_content = kern_path.read_text()
    try:
        agent2_output = run_agent2(kern_content)
    except Exception as e:
        return {"piece": piece_id, "error": f"Agent 2 failed: {e}"}

    # Score
    key_result      = score_key(agent2_output, gt)
    cadence_result  = score_cadences(agent2_output, gt)
    chord_result    = score_chord_labels(agent2_output, gt)

    all_disagreements = cadence_result.pop("disagreements") + chord_result.pop("disagreements")
    # Sort by severity: high → medium → low
    sev_order = {"high": 0, "medium": 1, "low": 2}
    all_disagreements.sort(key=lambda d: sev_order.get(d["severity"], 3))

    return {
        "piece": piece_id,
        "scores": {
            "key":          key_result,
            "cadences":     cadence_result,
            "chord_labels": chord_result,
        },
        "disagreements": all_disagreements,
        "agent2_raw": agent2_output,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clavier calibration harness")
    parser.add_argument("--pieces", nargs="*", help="DCML piece IDs to test (e.g. K279-1 K283-1)")
    parser.add_argument("--all",     action="store_true", help="Test all pieces in PIECE_MAP")
    parser.add_argument("--dry-run", action="store_true", help="Parse DCML only, skip Agent 2 API calls")
    parser.add_argument("--output",  default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    # Validate prompt is set
    if not args.dry_run and "PASTE AGENT 2" in AGENT2_SYSTEM:
        print("ERROR: AGENT2_SYSTEM is still a placeholder.")
        print("Paste the Agent 2 system prompt from server.py into calibrate.py before running.")
        print("(Use --dry-run to test DCML parsing without Agent 2)")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    if args.all:
        pieces = list(PIECE_MAP.keys())
    elif args.pieces:
        pieces = args.pieces
    else:
        pieces = list(PIECE_MAP.keys())[:5]

    print(f"\nClavier Calibration Harness")
    print(f"Pieces: {', '.join(pieces)}")
    print(f"DCML dir: {DCML_DIR}")
    print(f"Kern dir: {KERN_DIR}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 55)

    results = []
    for piece_id in pieces:
        print(f"\n── {piece_id} ───────────────────────────────")
        result = run_calibration(piece_id, dry_run=args.dry_run)
        results.append(result)

        # Save individual scorecard
        (out_dir / f"{piece_id}.json").write_text(json.dumps(result, indent=2))

        # Print summary row
        if "error" in result:
            print(f"  ✗ ERROR: {result['error']}")
        elif "dry_run" in result:
            print(f"  DCML key: {result['dcml_key']}")
            print(f"  DCML cadences: {result['cadences_in_gt']}  |  chords: {result['chords_in_gt']}")
        else:
            s = result["scores"]
            key_icon = "✓" if s["key"]["match"] else "✗"
            print(f"  Key:        {key_icon}  Agent 2={s['key']['agent2']}  DCML={s['key']['dcml']}")
            print(f"  Cadence F1: {s['cadences']['f1']:.2f}  "
                  f"(P={s['cadences']['precision']:.2f} R={s['cadences']['recall']:.2f}  "
                  f"TP={s['cadences']['tp']} FP={s['cadences']['fp']} FN={s['cadences']['fn']})")
            cl = s['chord_labels']
            print(f"  Chords@cad: coverage={cl['cadence_coverage']:.2f}  "
                  f"root_match={cl['cadence_root_match']:.2f}  "
                  f"(n={cl['n_cadence_bars']} cadence bars)")
            print(f"  Disagreements: {len(result['disagreements'])}  "
                  f"({sum(1 for d in result['disagreements'] if d['severity']=='high')} high  "
                  f"{sum(1 for d in result['disagreements'] if d['severity']=='medium')} medium  "
                  f"{sum(1 for d in result['disagreements'] if d['severity']=='low')} low)")

        if not args.dry_run:
            time.sleep(1.5)  # Avoid rate-limit spikes

    # Aggregate report
    valid = [r for r in results if "error" not in r and "dry_run" not in r]
    if valid:
        avg_key          = sum(r["scores"]["key"]["score"]                       for r in valid) / len(valid)
        avg_cad_f1       = sum(r["scores"]["cadences"]["f1"]                     for r in valid) / len(valid)
        avg_ch_coverage  = sum(r["scores"]["chord_labels"]["cadence_coverage"]   for r in valid) / len(valid)
        avg_ch_root      = sum(r["scores"]["chord_labels"]["cadence_root_match"] for r in valid) / len(valid)

        print(f"\n{'═'*55}")
        print(f"AGGREGATE RESULTS  ({len(valid)} pieces)")
        print(f"  Key accuracy:           {avg_key:.1%}")
        print(f"  Cadence F1:             {avg_cad_f1:.2f}")
        print(f"  Chord coverage@cadence: {avg_ch_coverage:.1%}")
        print(f"  Chord root@cadence:     {avg_ch_root:.1%}")
        print(f"{'═'*55}")

        total_disagreements = sum(len(r["disagreements"]) for r in valid)
        high_disagreements  = sum(
            sum(1 for d in r["disagreements"] if d["severity"] == "high")
            for r in valid
        )
        print(f"  Total disagreements: {total_disagreements}  ({high_disagreements} high-severity)")

        summary = {
            "n_pieces":                    len(valid),
            "avg_key_accuracy":            avg_key,
            "avg_cadence_f1":              avg_cad_f1,
            "avg_chord_coverage_at_cad":   avg_ch_coverage,
            "avg_chord_root_at_cad":       avg_ch_root,
            "total_disagreements":         total_disagreements,
            "pieces": results,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
