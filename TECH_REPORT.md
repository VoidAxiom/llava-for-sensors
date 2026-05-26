# LLaVA-for-Sensors: A Fusion Adapter for Time-Series + Vision + Language on a Frozen VLM

> **Status: skeleton.** This document is the Phase 6 deliverable. Each section currently holds a one-line stub describing what will land there. Real content is written from the lab journal in [`RUNNING_NOTES.md`](./RUNNING_NOTES.md) and the results of the Phase 4 headline ablation; the headline figure is pre-registered in [`PLAN.md` §1](./PLAN.md) before any model training.

## Abstract

_One paragraph. Thesis: a small trainable fusion adapter projecting time-series sensor embeddings into a frozen Qwen2-VL-2B's token space beats vision+text alone on CWRU bearing fault classification. Quantified by macro-F1 on a stratified test split over 5 seeds with bootstrap CIs and a paired bootstrap p-value._

## 1. Introduction

_The fault-prediction setting; why sensors + vision + text plausibly beats single-modality or vision+text; what a portfolio-scale local-only training run can and cannot prove._

## 2. Related work

_LLaVA (vision adapter on a frozen LLM), Time-LLM / Moment / PatchTST (time-series foundation encoders), Qwen2-VL, fault-prediction literature using CWRU. One-paragraph positioning relative to each._

## 3. Method

_Architecture: frozen Qwen2-VL-2B + LoRA + classification head + time-series encoder + fusion adapter that projects sensor embeddings to the VLM's token space. Token-space fusion details: how sensor tokens interleave with image and text tokens, position-encoding handling, attention mask. Trainable parameter count vs frozen count._

### 3.1 Time-series encoder

_Phase 1 starts with a toy 1D-CNN encoder; Phase 2 swaps to PatchTST (preferred) or Moment-small. Token shape after encoding._

### 3.2 Fusion adapter

_The single trainable projection from sensor-encoder output to VLM token space. Initialization, depth, parameter count._

### 3.3 LoRA on the VLM

_Rank, target modules, scaling. Why frozen base + LoRA + adapter is the right local-training-friendly choice on a 64 GB M2 Max._

## 4. Data

_CWRU drive-end accelerometer data, 12 kHz, 4 classes (Normal / IRF / ORF / BF). 2048-sample non-overlapping windows. Stratified 80/10/10 random split, fixed `seed=0` for the SPLIT (only training-time RNG varies across the 5 evaluation seeds). Equipment-image pairing strategy. Synthetic technician-note generation (deterministic per-class template)._

### 4.1 The CWRU split

_See [`PLAN.md` §1.2](./PLAN.md) for the pre-registered split rule._

### 4.2 Phase 1 toy dataset

_Synthetic, cross-modal-required-by-construction. No single modality should be sufficient — the toy set is the smoke gate for the architecture before CWRU investment._

## 5. Ablations

_The headline figure. Three conditions × five seeds. Mean macro-F1, 95% percentile bootstrap CI, paired bootstrap p for `all-three` vs `vision+text` (the comparison the thesis turns on)._

### 5.1 Headline figure

_The committed SVG, embedded here. Significance annotation per [`PLAN.md` §1.1](./PLAN.md): `*` for p<0.05, `**` for p<0.01._

### 5.2 Per-modality table

_All three pairwise comparisons reported in a table; `sensors-only` vs `all-three` lives here (not on the chart) to avoid annotation clutter._

### 5.3 Phase-7 cross-load split (conditional)

_If pursued: train on 0/1/2 HP, test on 3 HP, as a generalization-to-unseen-conditions secondary result._

## 6. Limitations

_Single-machine training, single fault dataset, fixed architecture choices. The negative-result protocol (if `all-three` does not beat `vision+text` significantly) is reported honestly here, not buried._

## 7. Future work

_Larger fault datasets (C-MAPSS, NASA bearing). Quantization paths if memory becomes the binding constraint. Cross-equipment generalization. Multi-sensor fusion (vibration + acoustic + thermal)._

## Reproducibility

_How to re-run the pipeline locally: `scripts/check-prereqs.sh`, `uv sync`, the training-orchestrator command from `train/`, the eval entry point. M2 Max with 64 GB (≈40 GB usable); training-time wall budget. The committed knowledge graph at [`.understand-anything/knowledge-graph.json`](./.understand-anything/) and the LikeC4 sources in [`architecture/`](./architecture/) give the static-architecture map; the per-PR audit trail on [GitHub](https://github.com/VoidAxiom/llava-for-sensors/pulls?q=is%3Apr+sort%3Acreated-asc) gives the dynamic build history._
