# Clavier

**Every piano piece, understood.**

Clavier is a classical piano analysis tool that reads symbolic score files, extracts harmonic structure with AI, and surfaces what makes each piece musically distinctive — and which other pieces share its DNA.

Built in one week as a focused AI engineering project. Evals calibrated against the DCML Mozart Piano Sonatas corpus: **100% key accuracy, 0.59 cadence F1**.

---

## Demo

[Live demo →](https://clavier.onrender.com)

**Recommended pieces to explore:** Beethoven Pathétique (dramatic sonata form), Chopin Nocturne Op.72 (distinctive harmonic language), Mozart K.331 Thema (clean classical structure).

---

## What It Does

Select a piece → click **Analyze** → the 4-agent pipeline runs in under 30 seconds:

- **Agent 2 — Harmonic Analyzer** reads the raw Humdrum kern file and produces a bar-cited JSON analysis: key regions, formal sections, cadences, modulations, and chord labels
- **Agent 3 — Overview** synthesises the analysis into a 2-sentence summary and assigns cross-piece tags from a controlled vocabulary
- **Sim Agent** compares the piece's harmonic fingerprint against every other piece in the corpus and surfaces the closest matches with reasons
- **Soul Agent** uses the top similarity result to write a programme note: what makes this piece harmonically itself, why it matters, and how it echoes in the most similar piece

The score renders from the kern file via Verovio with full notation fidelity.

---

## Architecture

```
kern file → Agent 2 (harmonic analysis JSON)
                ├── Agent 3 (overview + tags)      } parallel
                └── Sim Agent (corpus similarity)  }
                        └── Soul Agent (programme note)
```

**Cost/quality tradeoffs by agent:**
- Agent 2: `claude-sonnet-4-6` — heavy reasoning, full kern input
- Agent 3: classification only, cheaper model sufficient
- Sim + Soul: `claude-sonnet-4-6` — synthesis and retrieval-augmented generation

**Transparent proxy cache:** API responses are saved to `cache/` by content hash. Repeat visitors never hit the Anthropic API — the demo runs at zero cost once populated.

---

## Stack

- **Backend:** Python, Flask
- **Score rendering:** Verovio (kern → SVG, server-side)
- **AI:** Anthropic Claude API (`claude-sonnet-4-6`)
- **Frontend:** React (single HTML file, no build step)
- **Deployment:** Render

---

## Evals

Calibrated against the [DCML Mozart Piano Sonatas corpus](https://github.com/DCMLab/mozart_piano_sonatas) — 18 sonatas with expert Roman numeral, cadence, and phrase annotations.

| Metric | Result |
|--------|--------|
| Global key accuracy | 100% |
| Cadence detection F1 (±1 bar) | 0.59 |
| Chord coverage at cadence points | 94.1% |
| Root match at cadence (±1 bar) | 76.6% |

The cadence F1 gap reflects a deliberate design choice: DCML annotates transitional phrase closes that Agent 2 correctly skips for the app's purpose (structural landmarks only).

---

## Running Locally

```bash
# Install dependencies
pip install flask flask-cors requests verovio certifi

# Set API key (only needed to populate cache for new pieces)
export ANTHROPIC_API_KEY=sk-ant-...

# Start server
python3 server.py

# Open
open http://localhost:5001
```

The `cache/` directory is committed — the app runs without an API key once the cache is populated.

**To add new pieces to the cache:**
```bash
python3 build_cache.py --pieces <piece_id>
```

**To get DCML eval data:**
```bash
git clone https://github.com/DCMLab/mozart_piano_sonatas data/dcml_mozart
```

---

## Corpus

10 pieces from the [KernScores](https://kern.humdrum.org) corpus:

| ID | Piece |
|----|-------|
| `beet-path` | Beethoven — Sonata 'Pathétique' Op.13, I |
| `beet-moon` | Beethoven — Sonata 'Moonlight' Op.27 No.2, I |
| `mozart-k331` | Mozart — Sonata K.331, Thema |
| `chopin-n72` | Chopin — Nocturne in E minor Op.72 No.1 |
| `schub-op90` | Schubert — Impromptu Op.90 No.3 |
| `brahms-bal` | Brahms — Ballade No.1 Op.10 No.1 |
| `brahms-w1` | Brahms — Waltz Op.39 No.1 |
| `wtc1f19` | Bach — WTC I Fugue 19 in A major |
| `wtc1f20` | Bach — WTC I Fugue 20 in A minor |
| `wtc1f24` | Bach — WTC I Fugue 24 in B minor |

---

## License

MIT
