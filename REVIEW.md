# REVIEW.md — Conventions for reviewing this project

Reviewer-facing companion to `CLAUDE.md`. `CLAUDE.md` tells authors what to do; this doc tells reviewers (LLM and human) what to look for, what severity to assign, and what to ignore.

## Scope

This project is **local-dev only**. Single environment on a single M2 Max — never staged, never deployed, never multi-tenant (see CLAUDE.md §Scope). "Production-realistic" applies to **schema correctness, simulator/fixture fidelity, algorithmic quality, latency under load, observability** — not to deployment, secrets management, multi-environment config, or backwards-compat hygiene.

## Severity calibration for THIS project

- **[P0]** — Drop-everything. Reserved for issues universal across any input: SQL injection, secret leak, data loss, crash-on-startup, scope-guard bypass that lets impl write Claude-only paths (see CLAUDE.md §Recurring failure classes #5). Rare.
- **[P1]** — Must fix this cycle. Examples: divergence from packet `spec.md` source-of-truth quote, hardcoded numbers asserted as computed values (see CLAUDE.md §Evidence), tests deleted or weakened across iterations (see implementer.md §Anti-gate-gaming), a substring-anywhere path matcher where an anchored one is required (CLAUDE.md §Recurring failure classes #5), pre-registered acceptance silently relaxed (PLAN.md §1 fixes the headline contract as `all-three > vision+text` AND p<0.05 AND non-overlapping CIs — changing any of those three conjuncts without an explicit PLAN.md update is [P1]).
- **[P2]** — Fix eventually. Examples: missing test for a non-load-bearing helper, a determinism gap that PLAN.md §"MPS nondeterminism" already concedes, a thinly-tested adapter that the next slow gate exercises end-to-end.
- **[P3]** — Nice to have. Naming, comment density, doc polish, minor typing tightening.

## Out of scope — DO NOT FLAG

Derived from CLAUDE.md §Scope ("Single environment", "Dev passwords in plain config", "No multi-env configurability", "No deployment automation", "No backwards-compat shims", "No reusability/library packaging") and the local-dev-only nature:

- **Production load / scaling / multi-tenancy** — N/A; one user, one machine.
- **Deployment portability** — no Docker, Helm, K8s, Terraform target.
- **Multi-environment config** — staging/prod don't exist; hardcoded values that are the same in every environment are correct.
- **Secrets management beyond `.env`** — `.env` is dev-defaults; Vault / KMS / sealed-secrets are out of scope.
- **API rate limiting / backoff** — the app makes zero metered calls in v1 (CLAUDE.md §Autonomy boundary).
- **Backwards-compat shims, deprecation paths, feature flags for "old behavior"** — migrations are forward-only.
- **`setup.py` / publish workflows / `__version__` discipline** — build metadata is for typecheck/lint/test, not distribution.
- **CUDA portability** — M2 Max only; MPS is the target. The CPU fallback is a test-env safety net, not a production path.
- **Observability beyond JSONL logs + RUNNING_NOTES.md** — no metrics endpoint, no tracing, no APM.

If a finding's rationale boils down to "this won't work on a different machine / in production / at scale," it's noise. Close with a one-liner citing CLAUDE.md §Scope.

## Project-specific anti-patterns to flag

The reviewer-side translation of CLAUDE.md §"Recurring failure classes" and §"Anti-rot":

- **Pattern:** A number the change displays or asserts is hardcoded / mocked / faked rather than computed by real in-app logic.
  **Severity:** [P1].
  **Rationale:** CLAUDE.md §Evidence: "every number/claim a change asserts must be really produced by real logic — never fabricated, hardcoded, or mocked." Also AGENTS.md §Test-value contract.
  **Suggested action:** Ask for the computation to be moved into the relevant module and unit-tested so "it is real" is enforced.

- **Pattern:** A test was deleted, `@skip`'d, `xfail`'d, or shrunk to pass a failing gate.
  **Severity:** [P1] (escalate to [P0] if a previously-load-bearing test).
  **Rationale:** Implementer contract §Anti-gate-gaming forbids weakening tests across iterations. The bytes of `**/*.test.*` and `**/*.spec.*` must not shrink.
  **Suggested action:** Either revert the weakening or, if the test was wrong, surface that explicitly so Claude (director) can rule.

- **Pattern:** Spec divergence — packet `spec.md` (or `PLAN.md` source-of-truth) quotes one thing, code does another.
  **Severity:** [P1] minimum. Quote the spec line and the code line side-by-side in the finding.
  **Rationale:** CLAUDE.md §Spec authoring — "When an impl `/codex:review` or `@codex review` flags a contradiction between the packet spec and the source, that is a CRITICAL signal." The source spec is the contract.
  **Suggested action:** Either fix the code to match the spec, OR (if the spec is wrong) escalate to Claude to rewrite the spec; never silently work around.

- **Pattern:** Path-matching logic uses substring-anywhere (e.g. `.includes('/x/')`) or extension-only (`/\.css$/.test(base)` standalone) — not anchored against the project root.
  **Severity:** [P1]. Often [P0] for scope-guard or merge-gate paths.
  **Rationale:** CLAUDE.md §Recurring failure classes #5 — the canonical example of a forgery-prone rail.
  **Suggested action:** Canonicalize via `path.resolve`, anchor to active write root, allow only by strict prefix from the project root.

- **Pattern:** Silent fallback that makes a metric pass for the wrong reason — e.g. an `if module is None: continue` that drops a load-bearing modality, an `except: pass` that swallows a missing dependency and routes through a degraded path, a config flag that disables a pipeline stage without raising.
  **Severity:** [P1].
  **Rationale:** CLAUDE.md §"Deliver a working product" — the deliverable is a live working system, not green tests. A silently-dropped modality lets the `all-three` ablation condition collapse to whichever sub-condition still works, and the headline figure reports the wrong attribution.
  **Suggested action:** Either remove the fallback and `raise`, or make the fallback explicit + tested + RUNNING_NOTES-documented. If you cite a specific incident, link the `RUNNING_NOTES.md` entry.

- **Pattern:** Determinism gap — RNG / clock / unordered-dict / unstable-sort introduced in render or layout logic.
  **Severity:** [P2] (MPS nondeterminism is PLAN.md-conceded), else [P1].
  **Rationale:** AGENTS.md §UI/styling — "no nondeterministic source in logic or render — use the project's seeded RNG; same seed ⇒ same output."
  **Suggested action:** Thread a seeded RNG / accept a `seed` kwarg / sort the keys before iteration.

- **Pattern:** A `@codex review` comment whose body is NEITHER (a) bare standalone for an initial review NOR (b) `@codex review` on its first line followed by an implementer.md §8e rationale block ("Changes since last review" + "Not changed deliberately") for a re-review. Free-form `@codex …` mentions in prose, thread replies, or fix-narration comments are NOT trigger forms and spawn phantom cloud tasks.
  **Severity:** [P2].
  **Rationale:** CLAUDE.md §GH connector hygiene defines exactly two accepted trigger forms (first-review-bare and re-review-with-rationale-block); everything else spawns phantom cloud tasks that narrate work which doesn't land in this repo. The re-review rationale block exists specifically to prevent the reviewer from re-raising findings the impl already addressed.
  **Suggested action:** If it's a fix-narration / acknowledgement, drop the `@<bot>` mention and resolve the thread. If it's a re-review, keep `@codex review` as the first line and add (or keep) the rationale block per implementer.md §8e — don't move the rationale to a separate comment.

- **Pattern:** A new automated gate added without a prior real failure motivating it.
  **Severity:** [P2].
  **Rationale:** CLAUDE.md §Anti-rot — "earned, not speculative". Speculative gates rot faster than they pay for themselves.
  **Suggested action:** Ask which past finding the gate would have caught. If none, drop it.

## When the spec is the contract

There's no top-level `spec/` directory yet; the equivalent for this project is **`PLAN.md` source-of-truth quotes** (Phase X (a)/(b)/(c) sections) and per-packet `.codex-runs/voi-N/spec.md` files (gitignored, but the impl's PR description embeds the key quotes). If a packet's PR diverges from the quoted contract:

- Quote the source line in the finding.
- Severity [P1] minimum.
- If the spec is genuinely ambiguous, file as [P2] "spec needs clarification" rather than guessing intent (see CLAUDE.md §"Self-check before dispatch").

## "Earned, not speculative" rule

Findings, like gates, must trace to concrete code behavior — not speculation about what might happen in a hypothetical environment. CLAUDE.md §Anti-rot frames this for authors; the reviewer-side version is: do not flag *"if a user did X under condition Y on platform Z, this might break"* unless X, Y, and Z are inside this project's stated scope. The local-dev / M2-Max / no-metered-API constraints close most "what-if" doors.

## When `tested` tag matters

Tests are load-bearing for:
- Any number a change displays or asserts (CLAUDE.md §Evidence — applies to every number, no exceptions).
- The packet's `## Acceptance` § "Runtime verification" command must exist and pass.
- The Phase-1-onward smoke-gate logic in `eval/headline.py` / `eval/test_ablation.py` — the gate IS the test.

Tests are NOT load-bearing for:
- Pure transcription of LikeC4 sources (`architecture/*.c4`) into Mermaid renders — the build is the test.
- One-line registry / index updates.
- Documentation prose.

Flag missing tests only where one of the load-bearing cases applies. A "thinly-tested helper" that the next slow integration smoke exercises end-to-end is acceptable; do not flag.

## Multi-reviewer behavior

- If your finding contradicts another reviewer's finding on the same PR, note this in your rationale.
- Convergence between reviewers is signal; divergence is hypothesis.
- Do not soften your finding to align with another reviewer. Surface disagreement explicitly.

## What lint/types/tests already catch

`ruff check` is wired for `data/`, `models/`, `train/`, `eval/` (configured in `pyproject.toml`). Type hints + `from __future__ import annotations` are present throughout for readability, but **no static type checker is gated** — there is no `mypy` / `pyright` / `pyre` step in `pyproject.toml` or any pre-commit / CI hook. Strict-typing issues that ruff doesn't catch (subtle generic variance, `Optional` not narrowed, `Any` leaking through a public signature, signature drift between caller and callee) are reviewer territory; flag them per the severity table above. `pytest` runs determinism, shape, and acceptance-gate tests on every PR via local impl gates (CLAUDE.md §Internal review loop). The merge gate also enforces: scope check (`scripts/impl-precommit-scope.sh`), codex-exec audit trail, head-pinned Codex verdict, zero unresolved review threads. **Don't waste reviewer cycles re-flagging what these already catch** — flag what they *miss*.

## What humans add that bots can't

- **Domain correctness on the architectural claim.** Whether `all-three > vision+text` reflects real fusion or a redundant-axis coincidence in the toy data — that's judgment, not pattern-matching.
- **Spec-vs-implementation reading.** Pulling the source-of-truth quote, comparing it to the diff, and ruling on subtle drift.
- **Cross-PR coherence.** Is this PR's design compatible with the next packet's pinned interface contract? Bots audit one PR at a time.
- **"Earned, not speculative" budget management.** Knowing when to delete a rotted gate, when to skip a check that no longer catches anything real.
- **Phase-gate judgment.** Did the runtime verification actually exercise the user-visible behavior, or is "tests pass" being passed off as "deliverable works" (see CLAUDE.md §"Deliver a working product").
