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
| python3 3.10.9 | ✗ — need ≥ 3.11 | `brew install python@3.12` pending |
| git-lfs | ✗ missing | `brew install git-lfs && git lfs install` pending |
| likec4 | ✗ missing | Will install via P0.4 (global or via `package.json` once P0.3 lands) |
| `/understand` | ℹ skill-side | Loaded via the understand-anything plugin in the Claude Code session |

Python upgrade + git-lfs install are user-actionable; will be resolved before Phase 1 (toy training pipeline) lights up.

### Blockers

- **2026-05-26 — codex worker quota exhausted.** First dispatched impl subagent (VOI-224 P0.1 scaffold) hit the `GPT-5.3-Codex-Spark` subscription quota immediately: _"You've hit your usage limit for GPT-5.3-Codex-Spark. Switch to another model now, or try again at May 31st, 2026 2:53 AM."_ Worktree provisioned, branch clean at the merged P0.5 base, nothing committed. **All impl packets (P0.1, P0.3, P0.7) blocked until the quota resets or an alternative `CODEX_MODEL` is authorized.** Continuing autonomously on Claude-direct packets (P0.4 LikeC4 install, P0.6 architecture skeleton, P0.8 doc skeletons — this file —, P0.9 first `/understand` run) in the meantime.

### Template defects (queued for cleanup)

Non-functional cosmetic leaks from the `zawarudo` template that didn't get renamed at instantiation. None blocks delivery; will be batched into a Claude-direct cleanup commit at the next phase boundary:

- `scripts/worktree-new.sh` uses `.zawarudo-worktrees` instead of `.llava-for-sensors-worktrees` per [`CLAUDE.md`](./CLAUDE.md) doctrine.
- `scripts/codex-review.sh` reviewer-prompt header still names "zawarudo (TODO — one-line description …)".

### Memory / timing — not yet measured

Nothing material to log until the toy training loop lights up in Phase 1.

---

## Phase 1 — _(scheduled — toy synthetic pipeline)_

_To be filled in as Phase 1 progresses._

## Phase 2 — _(scheduled — real time-series encoder swap)_

## Phase 3 — _(scheduled — CWRU integration)_

## Phase 4 — _(scheduled — full training + headline ablation)_

## Phase 5 — _(scheduled — Gradio demo)_

## Phase 6 — _(scheduled — tech report + final polish)_

## Phase 7 — _(conditional — optional flourishes)_
