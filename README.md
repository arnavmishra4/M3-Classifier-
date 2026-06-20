<div align="center">

# 📊 M3 — Progression Classifier

**XGBoost + MLP · Part of [NeuroSight](../)**

*End-to-end clinical AI for GBM treatment monitoring and early detection*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/XGBoost-scikit--learn-orange?style=flat-square)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/MLP-PyTorch-red?style=flat-square)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

</div>

---

## The Clinical Problem

After chemoradiation, post-treatment GBM MRI often looks worse. **30–40% of the time this is pseudoprogression** — therapy-induced inflammation that mimics tumor growth on imaging. The clinical stakes are direct: a misread means pulling a patient off a treatment that is actually working.

The only current options are to wait 6–8 weeks for a repeat scan, or perform a biopsy. M3 solves this by consuming the biophysical shift between two longitudinal scans — computed by M2's Fisher-KPP PINN — and outputting a four-class progression label with a calibrated confidence score.

---

## At a Glance

| Property | Detail |
|---|---|
| **Task** | Four-class progression classification from longitudinal GBM MRI |
| **Input** | Δμ_D, Δμ_R, Δγ (M2 biophysical deltas) · Volumetric delta (M1) · Treatment metadata |
| **Output** | Four-class label + calibrated confidence score |
| **Architecture** | XGBoost ensemble + lightweight MLP |
| **Framework** | scikit-learn (XGBoost) + PyTorch (MLP) |
| **Fires on** | **Scan 2 only** — architecturally requires the delta between two scans |
| **Reported performance** | 84% accuracy, 0.81 AUC (OpenNeuro ds003416 + TCIA REMBRANDT) |

---

## Output Classes

| Class | Clinical Meaning | Recommended Action |
|---|---|---|
| **Pseudoprogression** | Imaging worsening is treatment-induced inflammation, not tumor growth | Continue current treatment protocol |
| **True Progression** | Tumor actively growing; therapeutic resistance confirmed | Reassess treatment immediately |
| **Treatment Working** | Tumor growing slower than biophysically predicted — treatment is effective | Maintain current protocol; monitor |
| **Treatment Failing** | Tumor growing faster than predicted — treatment is losing effect | Change treatment; escalate urgency |

---

## Why Biophysical Deltas — Not Spatial Comparison

The M2 PINN operates in normalized, non-dimensional time. Because the absolute end-time `t_end` is unknown, pixel-to-pixel spatial comparison between the Scan 1 forward prediction and the Scan 2 scan is not reliable.

M3 therefore classifies on the **parameter deltas between scans** rather than predicted-vs-actual spatial overlap. A jump in μ_R between Scan 1 and Scan 2 signals accelerating proliferation. A jump in μ_D signals invasion-dominant spread. These shifts carry the biological signal regardless of absolute timing.

This is a deliberate architectural decision — one documented in the M2 and M3 design specs — not a limitation workaround.

---

## Architecture

```
Scan 1 (stored in LangGraph patient state)
  └─ μ_D¹, μ_R¹, γ¹

Scan 2 (M2 output)
  └─ μ_D², μ_R², γ²

Feature Engineering
  ├─ Δμ_D  = μ_D² − μ_D¹      (diffusion shift)
  ├─ Δμ_R  = μ_R² − μ_R¹      (proliferation shift)
  ├─ Δγ    = γ² − γ¹           (Go-Grow Index shift)
  ├─ ΔVol_core                  (enhancing core volume delta, from M1)
  ├─ ΔVol_edema                 (peritumoral edema volume delta, from M1)
  └─ Treatment metadata         (RT dose, TMZ cycles, Δt, time since treatment end)
        │
        ▼
  ┌─────────────────────────────────┐
  │  XGBoost Ensemble               │  ← primary classifier
  │  + Lightweight MLP              │  ← calibration + ensemble refinement
  └─────────────────────────────────┘
        │
        ▼
  4-class label + calibrated confidence score (Platt scaling)
```

---

## Input Features

### Biophysical Deltas (from M2)

| Feature | Description |
|---|---|
| `delta_mu_D` | Change in normalized diffusion ratio between scans |
| `delta_mu_R` | Change in normalized proliferation ratio between scans |
| `delta_gamma` | Change in Go-Grow Index (μ_R / μ_D) between scans |

### Volumetric Deltas (from M1)

| Feature | Description |
|---|---|
| `delta_vol_core` | Absolute and % change in T1Gd enhancing core volume |
| `delta_vol_edema` | Absolute and % change in FLAIR peritumoral edema volume |

### Treatment Metadata

| Feature | Description |
|---|---|
| `rt_dose` | Radiotherapy dose administered (Gy) |
| `tmz_cycles` | Number of temozolomide cycles completed |
| `delta_t` | Time between Scan 1 and Scan 2 |
| `time_since_treatment_end` | Days elapsed since end of primary chemoradiation |

---

## Dataset

M3 is trained on a curated pool of longitudinal GBM cases with confirmed outcome labels. The effective training set is small by deep learning standards — uncertainty calibration is mandatory, and the model is designed with this constraint in mind.

### Training Sources

| Dataset | Cases | Role |
|---|---|---|
| **OpenNeuro ds003416** | ~40 two-scan pairs | Primary training set — pseudoprogression classifier with confirmed outcome labels |
| **TCIA REMBRANDT** | ~130 cases | Supplementary — clinical outcome follow-up data |
| **TCIA TCGA-GBM (longitudinal subset)** | ~80–120 usable | Auxiliary — post-treatment scan pairs; effective N lower after audit for two-scan completeness |

> **Effective N: ~150–200 cases with confirmed labels.** Many TCGA-GBM entries lack complete longitudinal follow-up — a dataset audit is required before training to identify usable two-scan pairs.

### Label Construction

Labels are derived from confirmed clinical outcomes, not radiologist reads alone:
- **Pseudoprogression** — imaging worsening followed by spontaneous improvement without treatment change
- **True Progression** — biopsy-confirmed or clinically confirmed tumor recurrence
- **Treatment Working / Failing** — determined by volumetric trajectory relative to PINN forward prediction and documented treatment response

### Data Limitations

This is a small-data regime. The model accounts for this through:
- **Platt scaling** for confidence calibration
- **Confidence gating** — cases below 0.55 confidence are escalated to human review; the agent does not emit a coin-flip classification
- **Uncertainty flagging** — cases in the 0.55–0.75 confidence band proceed with an explicit uncertainty flag passed to the NeuroBio Agent

---

## Confidence Gating

| Confidence | Agent Behavior |
|---|---|
| **≥ 0.75** | Classification emitted. Full M4 report generated. NeuroBio Agent runs on complete output. |
| **0.55 – 0.75** | Classification emitted with uncertainty flag. NeuroBio Agent treats label as tentative, weights literature search accordingly. |
| **< 0.55** | Halt. No classification report. Escalate to human review. NeuroBio Agent still runs on M2 biophysical parameters alone. |

---

## Integration Within NeuroSight

```
M1 (Scan 2 segmentation)
  └─ ΔVol_core, ΔVol_edema

M2 (Scan 1 → Scan 2 parameter shift)
  └─ Δμ_D, Δμ_R, Δγ

Treatment metadata (LangGraph patient state)
  └─ RT dose, TMZ cycles, Δt
        │
        ▼
     M3 (this repo)
        └─ 4-class label + confidence
              │
              ├─► M4 (Clinical RAG)
              │     └─ Full Scan 2 report grounded in NCCN + RANO
              │
              └─► NeuroBio Agent
                    └─ Mechanistic hypothesis from PubMed / bioRxiv / ClinicalTrials
```

M3 **only fires at Scan 2**. Scan 1 runs M1 and M2 only, stores parameters, and generates a forward prediction. Classification requires the delta — it is architecturally impossible on a single scan.

### Fallback Routing

If M2 fails to converge, the LangGraph agent automatically routes M3 to **radiomics-only features** (volumetric delta + treatment metadata, no biophysical parameters). An uncertainty flag is injected into the clinical report. The NeuroBio Agent is notified and adjusts its hypothesis accordingly. The pipeline does not crash.

---

## Repository Structure

```
m3_classifier/
├── train.py              # XGBoost + MLP training loop
├── model.py              # MLP architecture and ensemble logic
├── features.py           # Feature engineering from M1/M2 outputs
├── calibration.py        # Platt scaling for confidence calibration
├── dataset_audit.py      # Script to identify valid two-scan pairs from TCIA/OpenNeuro
├── evaluate.py           # Cross-validation, AUC, and per-class metrics
└── README.md
```

---

## Limitations

- **Small training set** — ~150–200 confirmed-label cases is limiting. Calibration and confidence gating are structural responses to this, not workarounds.
- **Scan 2 only** — classification on a single scan is architecturally impossible. The delta between two scans is the signal.
- **GBM-specific** — feature engineering, PINN parameter bounds, and label definitions are calibrated for GBM IDH-wildtype post-chemoradiation dynamics only.
- **PINN dependency** — if M2 does not converge, the strongest features (Δμ_D, Δμ_R, Δγ) are unavailable. Fallback to radiomics-only degrades classification quality and is flagged explicitly.

---

## References

- OpenNeuro ds003416 — longitudinal GBM pseudoprogression dataset
- TCIA REMBRANDT — GBM with clinical outcome follow-up
- TCIA TCGA-GBM — longitudinal subset with post-treatment imaging
- Zhang, Z. et al. "Biophysical parameter estimation using physics-informed neural networks." *Medical Image Analysis*, 2024
- *NeuroSight System Master Architecture and Build Documentation*, 2025 (`NeuroSight_Pleiades.docx`)

---

<div align="center">

**NeuroSight Pipeline**

[M1 — 3D Res-U-Net](../m1_segmentation) · [M2 — Fisher-KPP PINN](../m2_pinn) · **M3 — Progression Classifier (this repo)** · [M4 — Clinical RAG](../m4_rag) · [M5 — cfDNA Classifier](../m5_cfdna)

*Orchestrated by the NeuroBio Agent*

</div>
