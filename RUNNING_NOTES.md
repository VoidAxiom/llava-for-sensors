# RUNNING_NOTES.md — lab journal

Decisions, dead ends, memory/timing measurements, environmental quirks. The raw material from which [`TECH_REPORT.md`](./TECH_REPORT.md) is distilled in Phase 6. Append-only; never rewrite history. Each phase gets a section.

---

## Phase 0 — Setup, Headline Contract, Tooling

### Provenance

- Project scaffolded from the `orc-temp` template (the agent-orchestration shell — `.claude/`, `.codex/`, `hooks/`, `scripts/`, `CLAUDE.md`, `AGENTS.md`, `PLAN.md` skeleton).
- Project renamed `zawarudo` → `llava-for-sensors` at template instantiation.
- Linear project `llava-for-sensors` (id `be1055c7-7f77-4cc5-8713-a595739a9198`) in team VoidAxiom, prefix `VOI-`. Command center: [VOI-180](https://linear.app/voidaxiom/issue/VOI-180).
- Linear hierarchy created up front: 8 milestones (Phase 0..7), 8 phase parent issues (VOI-181..188), 38 packet subissues mirroring [`PLAN.md` §4](./PLAN.md).

### Decisions

- **Headline figure pre-registered before any model code.** [`PLAN.md` §1](./PLAN.md): 3 modality conditions × 5 seeds, macro-F1 on CWRU 4-class, 95% percentile bootstrap CI, paired bootstrap p for `all-three` vs `vision+text`. Pre-registered acceptance: "fusion wins" iff mean delta positive AND p<0.05 AND CIs non-overlapping. Pre-registered failure protocol: report the negative result honestly under "Negative result" in the tech report; no metric-shopping.
- **Framework: PyTorch + MPS (fp16, no 4-bit in v1).** 4-bit / MLX retreat deferred to Phase 7 if needed. See [`PLAN.md` §3.1](./PLAN.md) for the tradeoff.
- **`architecture/` and `.understand-anything/` are Claude-exclusive in the impl's blocklist mode.** They are documentation deliverables (LikeC4 source, committed knowledge graph), not production code; the impl cannot write them via `codex exec`. Mirrors how `docs/` is handled. Landed in VOI-191 (P0.5, PR #1).

### Tooling install state (as of 2026-05-26)

`bash scripts/check-prereqs.sh` on the dev M2 Max:

| Tool | Status | Notes |
| -- | -- | -- |
| node 26.0.0 | ✓ ≥18 | |
| npm 11.12.1 | ✓ | |
| uv 0.10.9 | ✓ | |
| python3 3.10.9 | ✗ — system; need ≥ 3.11 | uv-managed Python instead — see "P0.4 verification" below |
| git-lfs 3.7.1 | ✓ | Installed via `brew install git-lfs && git lfs install` |
| likec4 1.57.0 | ✓ (global) | Installed via `npm install -g likec4` — see "P0.4 verification" |
| `/understand` | ℹ skill-side | Loaded via the understand-anything plugin in the Claude Code session |

System `python3` remains 3.10.9; per the 2026-05-26 user directive ("use uv for custom python version and reqs"), uv owns the Python toolchain. `pyproject.toml` declares `requires-python >= 3.11` and `[tool.uv] python-preference = "managed"`; the P0.3 impl ran `uv python install 3.12` so all Python work goes through uv, not the system interpreter.

### P0.4 verification — LikeC4 + Understand-Anything live

Done 2026-05-26 as a Claude-direct packet (VOI-190); no production-code artifact, so this lab-journal entry IS the deliverable.

- **LikeC4** — `npm install -g likec4` → `/opt/homebrew/bin/likec4` v1.57.0. `likec4 validate architecture` is green on the P0.6 model; `likec4 gen mermaid` produced the renders in `docs/architecture/`. Project-local install will follow when P0.3's `package.json` lands (it pins `likec4 ^1.57`); `scripts/check-prereqs.sh` prefers `node_modules/.bin/likec4` over the global once available.
- **Understand-Anything** — the `/understand` family of skills is loaded in this Claude Code session (`understand-anything:understand`, `understand-anything:understand-diff`, `understand-anything:understand-dashboard`, `understand-anything:understand-knowledge`, `understand-anything:understand-onboard`, etc. all visible in the available-skills list at session start). End-to-end smoke run is deferred to P0.9, which commits the first `.understand-anything/knowledge-graph.json` on the real repo. No environmental blockers detected.
- **PNG / SVG export** — LikeC4 1.57 has no native SVG export, and `likec4 export png` requires a Playwright dep (`@tanstack/ai`) not on this env; graphviz `dot` is not installed. Mermaid is what we ship for embedding (GitHub renders inline). Documented in detail in PR #5 (VOI-192) description.

No escalation per PLAN.md §6 — the tooling is reachable and functional.

### Blockers

- **~~2026-05-26 — codex worker quota exhausted.~~** _Resolved 2026-05-26._ The default `gpt-5.3-codex-spark` hit its subscription quota on the first impl dispatch (VOI-224); user authorized switching to `gpt-5.5` (same model `/codex:review` uses; both subscription-covered). The codex-run.sh default was flipped in VOI-227 (P0.10 cleanup). Impl path unblocked; VOI-224 (P0.1 scaffold), VOI-225 (P0.3 pyproject) shipped on gpt-5.5; VOI-193 (P0.7 headline figure) next.

### Template defects (resolved 2026-05-26 in VOI-227)

The `zawarudo` template-name leaks identified during Phase 0 landed as a single cleanup PR (VOI-227 / P0.10):

- `scripts/worktree-new.sh` — `.zawarudo-worktrees` → `.llava-for-sensors-worktrees`.
- `scripts/codex-review.sh` — reviewer-prompt header rewritten with a real llava-for-sensors one-liner.
- `scripts/codex-run.sh` — `CODEX_MODEL` default flipped from `gpt-5.3-codex-spark` to `gpt-5.5` (operational, not template-related; bundled because the edit is one-line and review-cycle batching saves a PR).

One existing worktree (the impl's VOI-225 pyproject worktree) still lives at the legacy `.zawarudo-worktrees/` path — torn down after VOI-225 merges. New worktrees provisioned after VOI-227 will land under `.llava-for-sensors-worktrees/`.

### Memory / timing — not yet measured

Nothing material to log until the toy training loop lights up in Phase 1.

### Phase 0 — DONE (2026-05-26, 10 packets + 2 META PRs)

All Phase 0 packets merged to `main`. Final ledger:

| Packet | Linear | PR | Type | Notes |
| -- | -- | -- | -- | -- |
| P0.1 scaffold dirs | VOI-224 | #4 | impl | data/, models/, train/, eval/, demo/ with empty `__init__.py` |
| P0.2 check-prereqs.sh | VOI-189 | #2 | Claude | node/npm/uv/python/git-lfs/likec4 verifier + .template.answers regression guard |
| P0.3 pyproject + package.json | VOI-225 | #7 + #8 | impl + Claude | uv-managed python ≥3.11, likec4 ^1.57; .gitignore split out as Claude follow-up |
| P0.4 LikeC4 + Understand-Anything verify | VOI-190 | #9 | Claude | likec4 1.57.0 global install; /understand skill loaded |
| P0.5 scope-guard expansion | VOI-191 | #1 | Claude | architecture/** + .understand-anything/** added to Claude allowlist + CLAUDE_ONLY_DIRS |
| P0.6 architecture skeleton | VOI-192 | #5 | Claude | landscape.c4 + container.c4 + Mermaid renders (likec4 1.57 has no native SVG; Mermaid renders inline on GitHub) |
| P0.7 headline figure + tests | VOI-193 | #11 | impl | `compute_headline` (3xN array → verdict), pre-registered logic, deterministic SVG render; 9 pytest cases pass |
| P0.8 doc skeletons | VOI-194 | #3 | Claude | README + RUNNING_NOTES (this file) + TECH_REPORT mini-arXiv scaffold |
| P0.9 first /understand | VOI-195 | #14 | Claude | 58-node / 57-edge / 6-layer / 11-tour-step graph; dashboard render verified end-to-end |
| P0.10 template cleanup | VOI-227 | #6 | Claude | scripts/worktree-new + codex-review prompt + codex-run CODEX_MODEL default → gpt-5.5 |
| META parallelization gate | VOI-229 | #12 | Claude | PreToolUse hook on ScheduleWakeup; blocks idle wake when Todo packets are dispatchable |
| META deliver-working-product doctrine | VOI-230 | #13 | Claude | CLAUDE.md §"Deliver a working product" — packet acceptance requires runtime verification, not just test-pass |

**Pre-registered headline-figure contract live (PLAN.md §1).** `eval/headline_figure.py` + `eval/test_headline.py` encode the bar-chart + bootstrap-CI + paired-bootstrap-p + verdict logic. Mock SVG renders with 3 bars, axis labels, `**` significance annotation. Phase 4 ablation results will route through this same code path.

**Architecture-as-code live.** `architecture/landscape.c4` + `container.c4` validated; Mermaid renders in `docs/architecture/` embed inline in GitHub markdown. LikeC4 1.57 has no native SVG; Mermaid is the chosen embed format (documented in PR #5 description).

**Committed knowledge graph live.** `.understand-anything/knowledge-graph.json` (65 kB, 58 nodes, 57 edges, 6 layers, 11 tour steps) opens in `/understand-dashboard` at http://127.0.0.1:5173. Auto-update baseline (`fingerprints.json`) committed so subsequent commits trigger incremental updates correctly. Phase exit gates require regenerating the graph at the end of each phase.

**Parallelization gate live.** `scripts/queue-scan.sh` + `hooks/schedule-wakeup-guard.mjs` deny `ScheduleWakeup` when there are dispatchable Phase-N Todo packets whose dependencies are content-available (merged-to-main OR on an in-flight branch via the stacked-PR pattern). Tripped exactly once in this session — on a branch-misname false-positive that I resolved by renaming `sk/voi-228-...` → `sk/voi-229-...` via the GitHub branch-rename API.

**Deliver-working-product doctrine live.** Every future packet spec.md needs a `### Runtime verification` subsection with EXACT commands + EXACT observable outputs. Phase exit gate gets a new step 7 ("Claude has personally run the integrated phase deliverable end-to-end"). Cited as the rationale for VOI-195's dashboard render check and VOI-193's mock-SVG visual confirmation in this Phase 0.

### Phase 0 timing

- Session start: ~2026-05-25 23:14 UTC.
- Phase 0 close (PR #14 merge): ~2026-05-26 04:20 UTC.
- ~5 hours wall, ~14 squash-merges (11 packets + 3 in-flight fix rounds for VOI-193).
- Codex-worker quota stall: ~10 min before the user authorized the gpt-5.5 model switch.
- Branch-misname mishap: ~5 min to resolve via GitHub rename API (preserved PR linkage cleanly).

### Phase 0 → Phase 1 — awaiting user "proceed"

Per PLAN.md §2.3 step 6: phase boundaries cross only when Claude explicitly tells the user "Phase N done" and the user replies "proceed to Phase N+1". Phase 1 packets (VOI-196..202 — toy synthetic dataset, encoder/fusion/VLM stubs, train loop, ablation, LikeC4 + /understand updates) are queued in Linear; spec-author work on their packet.md files starts on the proceed signal.

---

## Phase 1 — toy synthetic pipeline (smoke-tests fusion-actually-helps)

**Status:** Done — the load-bearing acceptance gate passed by 45 pp.

### What landed

| Packet | PR | Summary |
|---|---|---|
| P1.1 | #17 | `data/synthetic.py` + `data/dataset.py` — 4-class toy dataset with cross-modal-required signature (sensor splits `{0,1}\|{2,3}`; image+text both split `{0,2}\|{1,3}`) plus the oracle logistic-regression test that asserts no single modality reaches F1 ≥ 0.90. |
| P1.2 | #18 | `models/encoder.py` (toy 1D-CNN, `(B, 2048) → (B, 32, 512)`) + `models/fusion.py` (16-query cross-attention adapter, `(B, 32, 512) → (B, 16, 1536)`). |
| P1.3 | #16 | `models/vlm.py` — `Qwen/Qwen2-VL-2B-Instruct` loaded fp16 on MPS, every param frozen, LoRA r=8/α=16 applied on every LLM `q_proj`/`v_proj`. CPU fallback uses fp32. <10 GB OOM smoke verified. |
| P1.4 | #19 | `train/loop.py` — `train_one_run` with AdamW + CosineAnnealingLR, gradient accumulation, JSONL logging, best-by-val-F1 checkpointing. |
| P1.5 | #21 | `eval/models.py` (3 ablation factories), `eval/ablation.py` (3 × 5 driver), `eval/headline.py` (acceptance gate). |
| P1.6 | #20 | `architecture/component.c4` (model-internals view) + 4 new Mermaid renders under `docs/architecture/`. META: `checkpoints/`, `logs/`, `.claude/scheduled_tasks.lock`, root `/index.mmd`, `/likec4.json` added to `.gitignore`. |
| P1.7 | _this entry_ | RUNNING_NOTES + `/understand` regen (this file + a follow-up commit). |

### Phase 1 (e) acceptance — the load-bearing gate

5 seeds × 3 modality conditions, on the 1000-sample toy dataset, n_epochs=5:

| Condition | mean macro-F1 | 95% CI | notes |
|---|---|---|---|
| `sensors-only` | 0.436 | [0.41, 0.45] | sensor partition alone — bounded by within-pair ambiguity |
| `vision+text` | 0.416 | [0.37, 0.45] | redundant axes; bounded by within-pair ambiguity |
| `all-three` | **0.867** | **[0.60, 1.00]** | 4/5 seeds at 1.000; seed 4 stuck at 0.333 (dead init) |

Paired bootstrap `all-three` vs `vision+text`: `p = 0.0002`. Gap_vt = **45.1 pp**, gap_so = **43.1 pp**. Both above the >15 pp PLAN.md §Phase 1 (e) gate. `verdict = fusion_wins`, `acceptance = passed`.

Headline figure rendered to `docs/figures/headline.svg` (referenced from the project explainer site at `docs/index.html`). _Forward-reference: `headline.svg` is being added on PR #11 (VOI-193 follow-up) and `index.html` on PR #22 (`sk/docs-project-explainer`); neither is on `origin/main` at the time this rollup is written. If this PR squash-merges before #11 and #22, the path is dangling until those land — preserved here as the historical record of where the Phase 1 deliverables exist locally._

### Phase 1 timing

- Session start: 2026-05-26 ~10:19 UTC (P1.1/1.2/1.3 dispatched in parallel).
- Phase 1 close (P1.5 merge): 2026-05-26 ~23:48 UTC.
- ~13 h wall time. Parallel fan-out at Phase entry (P1.1/1.2/1.3 disjoint surfaces — `data/`, `models/encoder+fusion`, `models/vlm`); serial fan-in through P1.4 → P1.5 → P1.6 → P1.7.
- The 5-seed × 5-epoch toy ablation itself took ~3 h on M2 Max (10:07 → 13:07 EDT).

### Phase 1 memory / timing — partially instrumented (carry to Phase 2)

PLAN.md §1.4(e) calls for: _"Memory & timing measurements logged: peak RAM, time-per-epoch, time-per-condition."_ Reporting honestly per CLAUDE.md §"Anti-rot" — what's in the code, what's missing, what to surface where:

- **time-per-condition:** ~3 h total ablation wall time / (3 conditions × 5 seeds) ≈ 12 min mean per (condition, seed) on M2 Max. **Instrumented:** `eval/ablation.py:39,49` writes `wall_time_s` per (condition, seed) row into the output CSV. Not aggregated into a roll-up table in this entry — the raw CSV is the source of truth.
- **time-per-epoch:** _not instrumented._ No `time.perf_counter()` / `time.time()` calls around the epoch loop in `train/loop.py`. Carry to Phase 2 — the PatchTST encoder swap will change per-epoch cost materially and per-epoch wall time becomes useful.
- **peak RAM:** **already instrumented**, just not surfaced here. `train/loop.py:250` defines `_memory_bytes()` which calls `torch.mps.driver_allocated_memory()`; `train/loop.py:131-132` updates `peak_memory_bytes` per training step; `train/loop.py:173` returns it in `TrainResult`. **Gap:** `eval/ablation.py` doesn't propagate `result.peak_memory_bytes` into the ablation CSV, so the per-condition peak isn't visible in the Phase 1 rollup. Carry to Phase 2 — add a `peak_memory_mb` column to the ablation CSV writer so the per-condition peak appears alongside the per-condition wall time.

Tickets to file at Phase 2 entry: `train/loop.py` epoch-timing instrumentation; `eval/ablation.py` propagation of `peak_memory_bytes` from `TrainResult` into the per-row CSV (don't re-add instrumentation — surface what's already there). Not blocking Phase 1 closure (the headline gate is what closes the phase); flagging so the next phase doesn't re-discover the gap.

### Decisions & near-misses (worth remembering)

- **r17 false-positive revert.** A GitHub `@codex review` flagged a [P1] that passing both `input_ids` and `inputs_embeds` to Qwen2-VL would XOR-raise. The "fix" (r17) did the image-token scatter manually in `AllThreeModel.forward` and dropped `input_ids` / `pixel_values` from the VLM call. A 15-min sanity on r17 showed `all-three` at 0.100 (below chance) — the manual-scatter approach broke the working path. Auditing `transformers==5.9.0` source: the outer `Qwen2VLModel.forward` accepts both arguments cleanly (uses `input_ids` to find image-pad mask, scatters into `inputs_embeds`, then calls the inner LLM with `input_ids=None`). The XOR check at `modeling_qwen2_vl.py:809` is in the inner `Qwen2VLTextModel.forward` and never fires because the outer model strips `input_ids` before the inner call. Reverted r17 (`21837c0`). **Lesson:** verify external review claims against canonical source before dispatching a fix. Codified in CLAUDE.md and the `@codex review` 👍-skip rule (PR #23).
- **Seed-4 dead initialization.** One of five `all-three` seeds stayed at val_f1 = 0.333 across all 5 epochs (zero learning). Real noise, not a code bug. The mean still clears the gate by 28 pp.
- **Toy dataset's deliberately-redundant axes.** Image and text both partition `{0,2}|{1,3}`. So `vision+text` alone caps at ~0.50 by construction — its low F1 doesn't mean vision is broken; it means the synthetic dataset is designed so only sensor + (vision or text) gives the orthogonal second axis. The real vision-pathway stress test is Phase 3 (CWRU bearing photos where each fault class has a distinct equipment image).
- **GitHub repo flipped private.** Mid-Phase 1. Workflow continues; codex bot needs a fresh environment for PRs opened after the flip (PR #23 is currently blocked on this).

### Phase 1 → Phase 2 — ready to dispatch

Phase 2 packets:
- **VOI-203 (P2.1)** — swap toy `models/encoder.py` for [PatchTST](https://arxiv.org/abs/2211.14730) (preferred) or Moment-small fallback. Preserve the `(B, 2048) → (B, 32, 512)` interface so `models/fusion.py` is unaffected. Re-run the Phase 1 ablation gate with the real encoder.
- **VOI-204 (P2.2)** — `architecture/component.c4` re-labels the encoder; `/understand` re-run; Phase 2 RUNNING_NOTES entry.

Specs drafted in `.codex-runs/voi-203/spec.md` and `.codex-runs/voi-204/spec.md`.

## Phase 2 — _(queued — real time-series encoder swap)_

## Phase 3 — _(scheduled — CWRU integration)_

## Phase 4 — _(scheduled — full training + headline ablation)_

## Phase 5 — _(scheduled — Gradio demo)_

## Phase 6 — _(scheduled — tech report + final polish)_

## Phase 7 — _(conditional — optional flourishes)_
