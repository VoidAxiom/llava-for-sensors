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

### Phase 0 — DONE (2026-05-26, 11 packets + 1 meta + 1 doctrine)

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

## Phase 1 — _(scheduled — toy synthetic pipeline)_

_To be filled in as Phase 1 progresses._

## Phase 2 — _(scheduled — real time-series encoder swap)_

## Phase 3 — _(scheduled — CWRU integration)_

## Phase 4 — _(scheduled — full training + headline ablation)_

## Phase 5 — _(scheduled — Gradio demo)_

## Phase 6 — _(scheduled — tech report + final polish)_

## Phase 7 — _(conditional — optional flourishes)_
