# llava-for-sensors

A multimodal foundation model that bolts a small trainable **fusion adapter** onto a frozen **Qwen2-VL-2B** so time-series sensor data flows into the VLM's token space alongside vision and language — LLaVA, but for sensors. Trained end-to-end locally on an M2 Max (40 GB usable).

**Thesis.** Sensors + vision + text beats vision + text alone on industrial fault prediction. Measurable on CWRU bearing fault classification across 4 classes. The headline-figure ablation is pre-registered in [`PLAN.md` §1](./PLAN.md) before any model code is written.

**Status.** Phase 0 — tooling, headline-figure contract, repo provisioning. See the [command center](https://linear.app/voidaxiom/issue/VOI-180) for current state.

---

## Reviewer entry points

| What you want | Where to look |
| -- | -- |
| The thesis and the pre-registered headline figure | [`PLAN.md` §1](./PLAN.md) |
| Phased delivery, milestones, exit gates | [`PLAN.md` §4](./PLAN.md) |
| Lab journal — decisions, dead ends, memory/timing | [`RUNNING_NOTES.md`](./RUNNING_NOTES.md) |
| Mini-arXiv writeup (final) | [`TECH_REPORT.md`](./TECH_REPORT.md) — Phase 6 deliverable |
| Architecture diagrams (LikeC4 sources → SVG exports) | `architecture/` → `docs/architecture/` (populated in P0.6) |
| Committed knowledge graph (Understand-Anything) | `.understand-anything/knowledge-graph.json` (populated in P0.9; regenerated end-of-phase) |
| Per-PR audit trail (the meta-story) | [PR list](https://github.com/VoidAxiom/llava-for-sensors/pulls?q=is%3Apr+sort%3Acreated-asc) |
| Linear command center | [VOI-180](https://linear.app/voidaxiom/issue/VOI-180) (project tracker, links to all phase issues + packet subissues) |

## Headline figure (placeholder)

The pre-registered ablation chart — 3 modality conditions (sensors-only, vision+text, all-three) × 5 seeds, macro-F1 on CWRU test split, 95% bootstrap CI, paired bootstrap p for `all-three` vs `vision+text` — lands as a stub in `eval/headline_figure.py` (P0.7) and is produced for real in Phase 4 (P4.3).

> _Final SVG will be embedded here by P4.4._

## Demo (placeholder)

A local Gradio app — upload a sensor CSV + equipment image → predicted fault class + rationale text — ships in Phase 5.

> _Screenshot will be embedded here by P5.3._

## Doctrine in one line

The model is the artifact; the per-PR auditable build process — Claude as director, codex-exec workers as the only writers of production code, per-PR Codex review gate, head-pinned squash-merges to `main` — is the portfolio meta-story. See [`CLAUDE.md`](./CLAUDE.md) for the contract.

## Repro

Local-dev only — single environment, M2 Max with 40 GB usable, no cloud GPUs. Setup checklist:

```bash
bash scripts/check-prereqs.sh   # node ≥18, npm, uv, python ≥3.11, git-lfs, likec4, /understand
```

Detailed install steps and verified-tooling notes live in [`RUNNING_NOTES.md`](./RUNNING_NOTES.md). The headline ablation must fit ≤40 GB peak memory and ≤24h total wall time on M2 Max per [`PLAN.md` §4 Phase 4](./PLAN.md).

## CWRU dataset — manual fetch

The CWRU Bearing Data Center provides bearing vibration recordings used in Phase 3.
The dataset requires manual download due to a registration wall (scripted downloads are brittle).

**Step 1 — Register and download**

Visit https://engineering.case.edu/bearingdatacenter and download the drive-end
accelerometer `.mat` files for the following conditions:

| Class | Suggested files |
|-------|----------------|
| Normal | 97.mat, 98.mat, 99.mat, 100.mat |
| Inner Race Fault | 105.mat, 106.mat, 107.mat, 108.mat |
| Outer Race Fault | 130.mat, 131.mat, 132.mat, 133.mat |
| Ball Fault | 118.mat, 119.mat, 120.mat, 121.mat |

**Step 2 — Place files under `data/raw/cwru/`**

```text
data/raw/cwru/
  normal/
    97.mat
    ...
  inner_race/
    105.mat
    ...
  outer_race/
    130.mat
    ...
  ball/
    118.mat
    ...
```

**Step 3 — Build the split**

```bash
uv run python -m data.cwru build-split
```

This writes `data/processed/cwru/{train,val,test}.pt` — torch tensors loaded by
`data/dataset.py` (Phase 3, VOI-207).

The `data/raw/cwru/` directory is gitignored (too large; licensing prevents redistribution).
