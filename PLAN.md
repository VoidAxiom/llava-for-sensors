# PLAN.md — llava-for-sensors

**llava-for-sensors** is a portfolio project: a multimodal foundation model that bolts a small trainable fusion adapter onto a frozen **Qwen2-VL-2B** so time-series sensor data flows into the VLM's token space alongside vision and language — LLaVA-for-sensors. Trained locally on M2 Max (40GB usable). Thesis: **sensors + vision + text beats vision + text alone** on industrial fault prediction, measurable on CWRU bearing fault classification. The model is the artifact; the per-PR auditable build process is the portfolio meta-story.

This file is the **planning contract**. The headline figure design, doctrine, framework decision, and phased delivery are committed here before code begins. Linear (project `llava-for-sensors`, team `VoidAxiom`, prefix `VOI-`) mirrors this plan as **milestone per phase → parent issue per phase → subissue per packet**. Command center: [VOI-180](https://linear.app/voidaxiom/issue/VOI-180/command-center-llava-for-sensors).

---

## 1. Headline Figure Contract (the thesis, designed in advance)

The headline figure is the load-bearing claim of this project. It is committed here **before any model code is written**. If the chart isn't designable in advance, the experiment isn't designable in advance.

### 1.1 Chart

- **Type:** bar chart, 3 bars, error bars from bootstrap CI.
- **X-axis (categorical):** modality condition
  1. `sensors-only` — TS encoder + linear classification head. No VLM, no images, no text.
  2. `vision+text` — frozen Qwen2-VL-2B + LoRA + classification head. Image and synthetic technician note as input. No TS encoder.
  3. `all-three` — frozen VLM + LoRA + classification head + TS encoder + fusion adapter projecting sensor embeddings into VLM token space.
- **Y-axis (continuous, single primary metric):** **macro-F1** on held-out CWRU test split, 4-class fault classification (Normal, Inner Race Fault, Outer Race Fault, Ball Fault).
- **Bar value:** mean across N seeds.
- **Error bar:** 95% percentile bootstrap CI over the N seed-level F1s.
- **Significance annotation:** `*` for p<0.05, `**` for p<0.01 from a paired bootstrap test comparing `all-three` vs `vision+text` (the comparison the thesis turns on). The `sensors-only` vs `all-three` comparison is reported in the table, not on the chart, to avoid annotation clutter.

### 1.2 Dataset split

- **Source:** CWRU drive-end accelerometer data, 12 kHz sampling rate, 4 classes (Normal, IRF, ORF, BF).
- **Windowing:** non-overlapping 2048-sample windows (≈170 ms each at 12 kHz). Label inherited from the parent recording's class.
- **Split:** stratified random 80/10/10 train/val/test by window, fixed `seed=0` for the SPLIT (the split itself is deterministic across all conditions; only training-time randomness varies across seeds).
- **Stretch (Phase-7 optional):** cross-load split — train on motor loads 0/1/2 HP, test on 3 HP — to demonstrate generalization to unseen operating conditions. Reported as a secondary table in the tech report.

### 1.2.1 Implementation correction (added 2026-05-28)

The original `data/cwru.py::build_split` shipped in VOI-205 (PR #32) interpreted §1.2's "by window" specification as file-grouped (windows from the same `.mat` file stay in the same bucket, to prevent within-recording leakage from the rig's structural transfer function). This was a deliberate but **unauthorized** divergence from the §1.2 spec, which explicitly says "by window," not "by file."

With only 4 `.mat` files per class in the minimum CWRU set (one per motor load 0/1/2/3 HP at 0.007″ diameter), a file-grouped stratified split of 16 files into 80/10/10 buckets cannot guarantee class coverage in val AND test simultaneously. `seed=0` happened to allocate val to classes {outer_race, ball} only and test to classes {normal, inner_race} only, producing a `macro-F1 = 0.5` ceiling that masked all model behavior across the VOI-210 ablation runs.

The fix (VOI-205-fix-window-split, PR #40) reverts to the §1.2 contract: **stratified random 80/10/10 by window**. Window-level stratification of ~915 atoms across 4 classes (~228 per class) yields ~22-23 val + test samples per class, well above the macro-F1 / bootstrap-CI noise threshold from §1.3.

The within-recording leakage concern that motivated file-grouping is real but accepted as a known weakness of this CWRU setup. Mitigation requires cross-rig evaluation (train on CWRU, test on Paderborn or IMS), which is out of scope for this project but discussed as future work in TECH_REPORT.md.

### 1.3 Seeds & statistics

- **N = 5 seeds** per condition. Seeds: `{0, 1, 2, 3, 4}`. Each seed runs the full training loop end-to-end (model init + data shuffle + augmentation RNG).
- **CI:** 95% percentile bootstrap over the 5 seed values, 10,000 resamples. Bootstrap is more honest than t-CI at N=5.
- **Paired bootstrap p-value** for `all-three` vs `vision+text`: same 5 seeds, same data split, only the model architecture differs. Resample the per-seed difference 10,000× and compute two-sided p.
- **Pre-registered acceptance:** the headline claim "fusion wins" is supported iff `all-three` mean > `vision+text` mean AND paired bootstrap p < 0.05 AND the 95% CIs do not fully overlap.
- **Pre-registered failure protocol:** if the acceptance criterion is NOT met, the tech report says so honestly under "Negative result" with a section discussing likely causes (architecture, dataset, fusion strategy). No retroactive metric-shopping, no quietly dropping `sensors-only`. The chart ships as designed.

### 1.4 Implementation hooks

- The chart-rendering script (`eval/headline_figure.py`) is committed in **Phase 0** as a stub that takes a 3×N results array and emits the SVG. It runs in CI on every phase from Phase 4 onward with mock data until real numbers exist.
- The pre-registered acceptance criterion is encoded as a unit test in `eval/test_headline.py` — passes a known-good 3×N array, asserts the verdict logic produces "fusion wins"; passes a known-flat array, asserts it produces "no significant difference".

---

## 2. Doctrine & cadence

### 2.1 Director / impl / codex split

Claude is **director / spec author / scope-gate / final integrator** and does NOT write production code. The hook `hooks/write-scope-guard.mjs` enforces this — Claude can only Edit/Write in: `scripts/**`, `**/*.test.*`, `.claude/**`, `.codex/**`, `.codex-runs/**`, `hooks/**`, `docs/**`, `**/*.md`, `.gitignore` (root).

**Two scope adjustments needed in Phase 0** (because architecture-as-code and the committed knowledge graph are documentation deliverables, not production code):
- Add `architecture/**` (LikeC4 source) to Claude's allowlist.
- Add `.understand-anything/**` to Claude's allowlist.

Both changes go through Claude editing `hooks/write-scope-guard.mjs` (hooks/ IS Claude-writable) plus a CLAUDE.md update documenting the expanded territory.

Production code (`data/`, `models/`, `train/`, `eval/`, `demo/`, anything else outside the Claude-writable allowlist) goes through an `implementer` subagent per packet. The impl dispatches `codex exec` workers, runs gates, drives `/codex:review`, commits within the packet allowlist, pushes, opens the PR, drives the `@codex review` eye-emoji loop, and notifies Claude on REVIEWED-CLEAN or CLEAN-COMMENT-MANUAL. See `.claude/agents/implementer.md` for the full contract.

### 2.2 Linear hierarchy

```
Project: llava-for-sensors  (be1055c7-7f77-4cc5-8713-a595739a9198)
├── Milestone: Phase 0 — Setup, Headline Contract, Tooling
│   └── Issue: Phase 0 rollup (parent)
│       ├── Subissue: 0.1 Scaffold repo skeleton
│       ├── Subissue: 0.2 scripts/check-prereqs.sh
│       ├── ... (one subissue per packet)
├── Milestone: Phase 1 — Toy synthetic pipeline
│   └── Issue: Phase 1 rollup
│       └── Subissues: 1.1, 1.2, ...
├── ... (one milestone + parent issue per phase)
└── Issue (standalone, not on a milestone): VOI-180 Command Center
```

Branches: `sk/voi-<n>-<slug>` (matches the existing VoidAxiom convention from fraud-forecast). PR body MUST contain `Closes VOI-N` so the subissue auto-transitions to Done on merge. The phase-rollup parent issue is closed manually after all its subissues are Done.

The Linear hierarchy is **created after PLAN.md approval** (task #6) — that way it mirrors the approved plan, not a guess.

### 2.3 Per-phase gate

A phase is **Done** only when:
1. All packets in that phase are merged to `main`.
2. Relevant LikeC4 view(s) for the phase are updated and `likec4 build` exports re-committed.
3. `/understand` is re-run and the knowledge graph commit is current.
4. RUNNING_NOTES.md has a short phase-rollup entry (decisions, dead ends, memory/timing).
5. The Command Center issue (VOI-180) has a one-paragraph status comment.
6. Claude explicitly tells you "Phase N done" and you say "proceed to Phase N+1".

### 2.4 Doctrine escape valve

If a phase blocks for >2 calendar days on doctrine overhead rather than substance (e.g., excessive impl/codex round-trips for a simple scaffold), surface in RUNNING_NOTES.md and we'll decide whether to carve a narrow exception — most likely a `sandbox/` directory exempt from the write-scope guard for one-off scratch experiments. Doctrine relaxation is **per-decision**, never global.

---

## 3. Framework & Stack

### 3.1 Framework: PyTorch + MPS (fp16, no 4-bit in v1)

- **Primary:** PyTorch + MPS (Metal Performance Shaders backend).
- **Mixed precision:** bf16 where MPS supports it cleanly, fp16 otherwise.
- **Quantization:** NONE in v1. Qwen2-VL-2B at fp16 is ~4–5 GB; with the vision tower + activations + KV cache + LoRA + fusion adapter, comfortably fits in 40 GB. Dropping 4-bit dodges the bitsandbytes-doesn't-support-MPS landmine entirely.
- **Escape hatch:** if MPS hits a wall (attention kernel pathologies, ops that fall back to CPU at ruinous cost), retreat to MLX (`mlx-vlm` ships Qwen2-VL-2B at 4-bit). Documented in RUNNING_NOTES.md; not a default.

### 3.2 Stack (concrete versions to be locked in Phase 0)

- **Python:** ≥ 3.11 (managed via `uv`).
- **PyTorch:** latest stable supporting MPS at the time of Phase 0 (≥ 2.4).
- **VLM:** `Qwen/Qwen2-VL-2B-Instruct` (HuggingFace, fp16 weights).
- **PEFT:** for LoRA on the LLM layers (rank 8–16, target attention projections).
- **Time-series encoder:** PatchTST (preferred — well-maintained, simple) or Moment-small if PatchTST hits portability issues on MPS. Decided in Phase 2.
- **Gradio:** ≥ 4.0 for the demo.
- **LikeC4:** `@likec4/likec4` (Node CLI, requires Node ≥ 18).
- **Understand-Anything:** install per https://github.com/Lum1104/Understand-Anything.
- **git-lfs:** for the knowledge graph if it exceeds 10 MB.

These versions go into `pyproject.toml` (Python) and `package.json` (Node tooling) during Phase 0. `VERIFY_CMD`/`BUILD_CMD`/`INSPECT_CMD` in `.template.answers` and the corresponding `TODO`s in CLAUDE.md/AGENTS.md are filled in Phase 0 once tooling is concrete.

### 3.3 Reproducibility baseline

- Deterministic where MPS allows: `torch.use_deterministic_algorithms(True)` where supported, `PYTHONHASHSEED=0`, fixed seeds in dataset shuffle + DataLoader workers.
- A unit test in `eval/test_determinism.py` runs the same seed twice and asserts identical output. MPS nondeterminism that defeats this is acknowledged in RUNNING_NOTES.md per occurrence; we don't pretend.

---

## 4. Phase plan

Each phase below specifies:
- **(a) Code/data changes** — what artifacts ship
- **(b) LikeC4 views to update** — which `.c4` files change
- **(c) /understand re-run trigger** — when the knowledge graph regenerates
- **(d) Linear subissues** — the packet decomposition (created after PLAN.md approval)
- **(e) Phase exit gate** — concrete criteria for "done"

### Phase 0 — Setup, Headline Contract, Tooling

Mostly Claude-direct (Claude territory). One impl subagent for the `data/models/train/eval/demo/` skeleton with placeholder `__init__.py` files (which is production-code territory and needs codex-written stubs).

**(a) Code/data:**
- `scripts/check-prereqs.sh` — verifies node ≥18, npm, uv, python ≥3.11, git-lfs, `likec4` on PATH, `/understand` availability. Fails loudly with install instructions for anything missing. **Also includes a regression check** (added per user audit request) that no runtime code in `scripts/` or `hooks/` reads from `.template.answers` (which is outside Claude's writable scope and intentionally stale per §6). Audit at project creation: clean — only commentary mentions in `hooks/write-scope-guard.mjs`. See VOI-189 for the exact grep guard.
- `pyproject.toml` (Python project metadata; `uv`-managed deps).
- `package.json` (Node deps — just LikeC4).
- Directory skeleton: `data/`, `models/`, `train/`, `eval/`, `demo/`, `architecture/`, `docs/architecture/`, `.understand-anything/`. Each gets a placeholder `__init__.py` or `.gitkeep` so git tracks them.
- `eval/headline_figure.py` — chart-rendering stub per §1.4. Takes a 3×N results array, emits SVG.
- `eval/test_headline.py` — unit test for the pre-registered verdict logic per §1.4.
- `README.md` — short project intro + reviewer entry points (Linear command center, PLAN.md, TECH_REPORT.md, /architecture, knowledge graph).
- `RUNNING_NOTES.md` — scaffolded with Phase-0 section.
- `TECH_REPORT.md` — scaffolded (abstract / method / data / ablations / limitations / future work).
- `.gitignore` additions for Python/Node/MPS caches.
- `hooks/write-scope-guard.mjs` — add `architecture/**` and `.understand-anything/**` to Claude's allowlist.
- `CLAUDE.md` — document the expanded Claude territory.
- LikeC4 + Understand-Anything install (or escalation if blocked).

**(b) LikeC4 views:**
- `architecture/landscape.c4` — initial system landscape: user → demo → model → CWRU dataset → outputs.
- `architecture/container.c4` — scaffold with placeholders for data pipeline, training loop, eval harness, demo app (filled in later phases).
- `likec4 build` exports SVGs into `docs/architecture/`.

**(c) /understand:** initial run at the end of Phase 0; commits `.understand-anything/knowledge-graph.json`.

**(d) Linear subissues (planned):**
- 0.1: Repo skeleton + `__init__.py` placeholders (impl, codex-written stubs)
- 0.2: `scripts/check-prereqs.sh` + run it (Claude)
- 0.3: `pyproject.toml` + `package.json` (impl)
- 0.4: Install LikeC4 + Understand-Anything; escalate if blocked (Claude)
- 0.5: `architecture/landscape.c4` + `container.c4` skeleton + first `likec4 build` (Claude, after scope expansion)
- 0.6: Scope-guard expansion (Claude edits `hooks/write-scope-guard.mjs` + CLAUDE.md)
- 0.7: `eval/headline_figure.py` + `eval/test_headline.py` (impl — load-bearing; the verdict logic is the experiment's spine)
- 0.8: `README.md` + `RUNNING_NOTES.md` + `TECH_REPORT.md` skeletons (Claude)
- 0.9: First `/understand` run + commit graph (Claude)

**(e) Phase exit gate:**
- `check-prereqs.sh` returns clean.
- `likec4 build` produces valid SVGs in `docs/architecture/`.
- `eval/test_headline.py` passes with mock data.
- `.understand-anything/knowledge-graph.json` is committed.
- All planned Phase-0 subissues are Done.
- RUNNING_NOTES.md has a Phase-0 entry.

### Phase 1 — Toy synthetic pipeline (smoke-tests fusion-actually-helps)

The toy dataset's anomaly signature is designed so **no single modality is sufficient** — fusion must win on the toy set or the architecture is broken (cheap, early signal before CWRU investment).

**(a) Code/data:**
- `data/synthetic.py` — generates ~1000 samples. Each sample:
  - **Sensor window:** sine wave at one of 4 base frequencies, optionally with an injected anomaly. The anomaly is ambiguous *alone* (could be class A or B based on sensor signal alone).
  - **Image:** a small 224×224 schematic per class, but two classes share the same image — image alone is ambiguous between those two.
  - **Text:** a synthetic technician note like "vibration high, motor warm" — partial disambiguation but not complete alone.
  - **Label:** only the combination of all three uniquely identifies the class. A model using any single modality should plateau around ~50–75% F1; fusion should hit ~95%+.
- `data/dataset.py` — torch Dataset yielding `(sensor_window, image, text, label)`.
- `models/encoder.py` — toy 1D-CNN time-series encoder (3–5 layers, <2M params; the real PatchTST swap happens in Phase 2).
- `models/fusion.py` — toy cross-attention fusion adapter projecting sensor embeddings into VLM token space.
- `models/vlm.py` — Qwen2-VL-2B loader (fp16, frozen) + LoRA setup.
- `train/loop.py` — minimal training loop, batch=1–4, gradient accumulation.
- `eval/ablation.py` — runs all 3 modality conditions × N=5 seeds, produces a results CSV.
- Updated `eval/headline_figure.py` consuming the real CSV.

**(b) LikeC4 views:**
- `architecture/component.c4` — model internals: VLM backbone (frozen), TS encoder, fusion adapter, LoRA layers, tensor flow.
- Update `architecture/container.c4` — data pipeline + training loop + eval harness containers.
- `likec4 build` exports updated SVGs.

**(c) /understand:** re-run at end of Phase 1. Before Phase 2 dispatch, run `/understand-diff` to predict ripple effects.

**(d) Linear subissues (planned):**
- 1.1: `data/synthetic.py` + `data/dataset.py` (impl)
- 1.2: `models/encoder.py` + `models/fusion.py` (toy versions) (impl)
- 1.3: `models/vlm.py` — frozen Qwen2-VL-2B + LoRA wiring (impl)
- 1.4: `train/loop.py` — minimal end-to-end training (impl)
- 1.5: `eval/ablation.py` + headline figure with toy data (impl)
- 1.6: LikeC4 `component.c4` + `container.c4` update + `likec4 build` (Claude)
- 1.7: `/understand` re-run + commit graph + Phase 1 RUNNING_NOTES entry (Claude)

**(e) Phase exit gate (LOAD-BEARING):**
- The toy ablation runs end-to-end on M2 Max.
- The headline figure on the toy set shows: `all-three` macro-F1 > `vision+text` macro-F1, **AND** `all-three` > `sensors-only`, **AND** the gap is large (>15 pp on toy).
- If this fails, the architecture is broken. **STOP, do not proceed to Phase 2.** Diagnose: is the fusion adapter wired correctly? Are sensor embeddings reaching the LLM? Use `/understand-diff` + manual inspection.
- Memory & timing measurements logged: peak RAM, time-per-epoch, time-per-condition.

### Phase 2 — Real time-series encoder

**(a) Code/data:** Swap toy `models/encoder.py` for **PatchTST** (preferred) or Moment-small. Keep everything else identical. Retrain on toy dataset, verify the Phase 1 exit gate still passes with the real encoder.

**(b) LikeC4:** update `component.c4` — encoder swap.

**(c) /understand:** re-run at end.

**(d) Linear subissues:**
- 2.1: PatchTST integration + retrain on toy (impl)
- 2.2: LikeC4 + /understand update (Claude)

**(e) Exit gate:** Phase 1 gate still passes with the real encoder.

### Phase 3 — CWRU integration

**(a) Code/data:**
- `data/cwru.py` — download (or read local copy of) CWRU drive-end accelerometer data, preprocess into 2048-sample windows per §1.2, write to a torch-friendly format.
- `data/images.py` — pair each class with a representative equipment image (one per class initially; can be diversified later).
- `data/notes.py` — generate synthetic technician notes from label + metadata (deterministic template, not LLM-generated to avoid metered API spend).
- Updated `data/dataset.py` — CWRU mode.

**(b) LikeC4:** update `container.c4` — data pipeline now reads CWRU.

**(c) /understand:** re-run at end. Run `/understand-diff` before any non-trivial change inside this phase.

**(d) Linear subissues:**
- 3.1: CWRU download + preprocess (impl) — VOI-205
- 3.2: Image pairing + technician notes (impl) — VOI-206
- 3.3: CWRU mode in Dataset + smoke training run (impl) — VOI-207
- 3.4: LikeC4 + /understand update (Claude) — VOI-208
- 3.5: `scripts/budget-check.sh` — 15-run ablation timing extrapolation gate (impl) — VOI-223 [added per user request, blocks Phase 4 launch]

**(e) Exit gate:** CWRU Dataset returns valid `(sensor, image, text, label)` tuples; one short training run on CWRU completes without crashes; **`scripts/budget-check.sh` outputs GREEN (≤20h) or YELLOW (20–24h, documented in RUNNING_NOTES.md). If RED (>24h projected), Phase 3 is NOT done — redesign (shrink seeds, shrink epochs, simplify encoder, fewer conditions) before Phase 4 begins.**

### Phase 4 — Full training + headline ablation

The big one. End-to-end CWRU training × 3 conditions × 5 seeds = 15 training runs. Must fit in 24h on M2 Max.

**(a) Code/data:**
- Training run orchestration (probably a shell script under `scripts/` that loops conditions × seeds and writes a single results CSV).
- `eval/ablation.py` produces the final 3×5 results array.
- `eval/headline_figure.py` produces the final SVG.
- `eval/test_acceptance.py` runs the pre-registered acceptance criterion (§1.3) and outputs `PASS` / `FAIL: negative result` / `FAIL: degenerate`.
- Memory & timing logged exhaustively in RUNNING_NOTES.md.

**(b) LikeC4:** update `component.c4` if the training loop architecture changed materially; otherwise no-op.

**(c) /understand:** re-run at end.

**(d) Linear subissues:**
- 4.1: Multi-seed training orchestrator script (impl)
- 4.2: Full ablation run (impl — this is a long-running packet; impl monitors and reports timing)
- 4.3: Headline figure final render + acceptance test (impl)
- 4.4: Phase 4 RUNNING_NOTES + LikeC4 + /understand (Claude)

**(e) Exit gate:**
- 15 training runs complete.
- Headline figure SVG committed at `docs/architecture/headline.svg` (or similar canonical path).
- Acceptance test runs and gives a verdict (PASS or honest FAIL).
- Memory peak ≤ 40 GB across all runs (logged).
- Total wall time ≤ 24h (logged).

### Phase 5 — Gradio demo

**(a) Code/data:** `demo/app.py` — Gradio app that accepts a sensor CSV upload + image upload, runs inference, returns prediction + rationale text (from the VLM's natural-language output, prompted to explain).

**(b) LikeC4:** update `container.c4` — demo container connects to model.

**(c) /understand:** re-run at end.

**(d) Linear subissues:**
- 5.1: Gradio app skeleton + UI (impl)
- 5.2: Inference path + rationale prompt template (impl)
- 5.3: Polish + screenshot for README (impl)
- 5.4: LikeC4 + /understand update (Claude)

**(e) Exit gate:** demo runs locally, accepts inputs, returns a prediction + rationale. Screenshot in README.

### Phase 6 — Tech report + final polish

**(a) Code/data:** No production code changes. All Claude-territory.
- `TECH_REPORT.md` — full mini-arXiv writeup. Abstract, introduction, related work (LLaVA, time-series foundation models), method (with embedded LikeC4 component view), data (CWRU + synthetic), ablations (headline figure + table), limitations, future work (cross-load split per §1.2 stretch, C-MAPSS RUL framing, 4-bit quantization).
- `README.md` polish — reviewer entry points, Gradio demo screenshot, headline figure embedded, links to Linear command center + knowledge graph.
- All LikeC4 SVGs embedded in TECH_REPORT.md + README.md at appropriate sections.

**(b) LikeC4:** no architecture changes; just re-export for any tweaked views.

**(c) /understand:** final re-run; this is the graph reviewers see when they clone.

**(d) Linear subissues:**
- 6.1: TECH_REPORT.md draft (Claude — direct edit, no impl)
- 6.2: README polish + screenshot embedding (Claude)
- 6.3: Final LikeC4 review + /understand commit (Claude)

**(e) Exit gate:** README + TECH_REPORT readable end-to-end by someone who's never seen the repo. Linear Command Center final status comment posted. All phase milestones closed.

### Phase 7 — Optional flourishes (only if ≥3 days budget remain)

If Phases 0–6 land with at least 3 calendar days of budget left, pursue these in order:

1. **4-bit quantization** (demonstrates quant-aware fine-tuning as a portfolio chop). Likely requires retreating to MLX (`mlx-vlm` Qwen2-VL-2B 4-bit). Re-run the headline ablation; report delta vs fp16 in tech report.
2. **Cross-load CWRU split** per §1.2 stretch — train on 0/1/2 HP, test on 3 HP. Reported as secondary table in tech report.
3. **C-MAPSS feasibility note** — small experiment showing the fusion adapter framework generalizes to RUL regression. Reported as a paragraph in "Future work", not a full experiment.

**Phase 7 is genuinely optional.** Don't let it scope-creep into Phases 1–6.

---

## 5. Deliverables (final state at end of Phase 6)

- `README.md` — entry point with headline figure embedded, Gradio demo screenshot, links.
- `PLAN.md` — this file, kept current (decisions ledger appended as they're made).
- `RUNNING_NOTES.md` — lab journal: per-phase entries, decisions, dead ends, memory & timing measurements, links to LikeC4 views and knowledge-graph nodes.
- `TECH_REPORT.md` — mini-arXiv-style writeup with embedded LikeC4 SVGs.
- `data/`, `models/`, `train/`, `eval/`, `demo/` — production code, all impl-built per the per-packet PR trail.
- `architecture/*.c4` — LikeC4 source for landscape + container + component views.
- `docs/architecture/*.svg` — LikeC4 exports, embedded in README + TECH_REPORT.
- `.understand-anything/knowledge-graph.json` — committed codebase graph, current as of the last phase.
- `scripts/check-prereqs.sh` — reproducibility script.
- Linear project `llava-for-sensors` — every phase a milestone with closed parent issue + all closed subissues. Command center VOI-180 final status comment.
- GitHub repo `VoidAxiom/llava-for-sensors` (public) — every packet a merged PR with `Closes VOI-N`. Squash-merged. Codex review trail visible per PR.

---

## 6. Open questions / known unknowns (revisit at phase boundaries)

- **CWRU access mechanism.** The dataset is hosted at Case Western (free, registration-walled). Phase-3 packet must either bake the download in or document the manual fetch step. TBD which.
- **Image pairing fidelity.** v1 uses one representative image per class. If the toy-set smoke test (Phase 1) shows the image modality contributes <expected, Phase 3 may need to diversify per-class images (e.g., crops of the actual CWRU test rig from photos in the dataset documentation).
- **Synthetic technician note generation.** Phase 3 uses deterministic templates (no metered API). If the text modality contributes ~zero in ablation, consider expanding the template variety in a Phase-3 follow-up packet.
- **MPS nondeterminism.** Some ops are nondeterministic on MPS regardless of `use_deterministic_algorithms(True)`. The Phase 1 determinism test will surface this; document occurrences in RUNNING_NOTES.md and weaken the determinism claim if needed (don't pretend).
- **`.template.answers` staleness.** The file is outside Claude's writable scope (the hook treats it as production code). It records the template generation inputs and is unused at runtime. Audit at project creation confirmed: only commentary mentions exist in `hooks/write-scope-guard.mjs` (lines 7, 18, 63) — no `fs.readFile`, no shell `source`, no runtime read. Values were baked in by `init.sh` at generation time. Leave stale unless re-running `init.sh` becomes necessary; if so, update through an impl packet. **Regression protection:** `scripts/check-prereqs.sh` (P0.2 / VOI-189) includes a grep guard that fails if any non-comment runtime reference to `.template.answers` is introduced, so the staleness cannot silently become a footgun.

---

## 7. Glossary

- **VOI-N** — Linear issue identifier (team VoidAxiom).
- **Packet** — one unit of work = one Linear subissue = one branch = one PR. Owned by one `implementer` subagent in its own worktree, all code via `codex exec`.
- **Phase** — a Linear milestone + parent issue grouping related packets. Phases ship in order.
- **Headline figure** — the 3-bar modality-ablation chart per §1.1. The load-bearing claim of the project.
- **`all-three`** — the modality condition using sensors + vision + text via the fusion adapter. The thesis claim is that this beats `vision+text`.
- **LikeC4** — diagrams-as-code system (https://likec4.dev); `.c4` source files in `architecture/`, rendered SVGs in `docs/architecture/`.
- **Understand-Anything** — explorable codebase knowledge graph (https://github.com/Lum1104/Understand-Anything); committed `.understand-anything/knowledge-graph.json`.
- **Doctrine** — the Claude-as-director / impl-as-worker / codex-as-writer split enforced by `hooks/write-scope-guard.mjs`. See §2.1.
