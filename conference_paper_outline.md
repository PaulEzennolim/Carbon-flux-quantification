# AI4Science 2026 — Conference Paper Plan

**Working from:** existing dissertation results only (no covariate adapter — that is the journal paper)
**Format:** ACM `acmart`, `sigconf` style, single-column, **≥ 8 pages**
**Draft deadline:** end of June (submission deadline 15 July 2026)
**Single message (the spine):** *Pretrained time-series foundation models invert the classical cross-ecosystem transfer penalty in carbon-flux forecasting.*

---

## 0. Resolve before drafting (blocking)

1. **Model identity.** The repo loads `AutonLab/MOMENT-1-large` and cites Goswami et al. 2024 (MOMENT), but the text says "TEMPO-80M". These are different models. Confirm which one the code runs and use that name *everywhere* — title, abstract, body, references. This is non-negotiable for peer review.
2. **Double-blind?** Check the AI4Science CFP. If reviewing is double-anonymous, remove author/affiliation/ORCID for submission and avoid self-identifying citations (including pointing reviewers at this public repo or the named dissertation). Add author info only on acceptance.
3. **Parameter count.** "80M" does not match MOMENT-large (~385M). Get the real number from the loaded checkpoint and state it correctly.

---

## 1. Section-by-section structure (target ~8 single-column pages)

### Title + Abstract (~0.5 pg)
- Title in proper title case, no line breaks. Candidate: *"The Inverted Transfer Penalty: Time-Series Foundation Models Defy Cross-Ecosystem Transfer Decay in Carbon-Flux Forecasting."*
- Abstract ~150–200 words: problem (NEE forecasting matters, cross-ecosystem transfer poorly understood) → what you did (benchmark a foundation model, zero-shot and fine-tuned, vs XGBoost/RF/LSTM across contrasting FLUXNET ecosystems) → headline result (fine-tuned model: R² 0.728 on the cross-ecosystem *forest* site vs 0.599 on the same-ecosystem *wetland* site, a +21.6% inversion; baselines show the normal penalty of +40–61% RMSE) → significance.

### CCS Concepts + Keywords (~0.1 pg) — *required by the checklist*
- Generate CCS codes at https://dl.acm.org/ccs (paste codes verbatim). Likely: *Computing methodologies → Machine learning → Transfer learning*; *Applied computing → Earth and atmospheric sciences*.
- ≥ 3 keywords: time-series foundation models; transfer learning; carbon flux / net ecosystem exchange; FLUXNET; NEE forecasting.

### 1. Introduction (~1 pg)
- Why NEE forecasting matters (carbon cycle, climate).
- The gap: foundation models are promising but their *transfer behaviour across ecosystem boundaries* is unstudied.
- Classical transfer theory's prediction: same-ecosystem > cross-ecosystem.
- Your finding, stated up front: the opposite happens for the foundation model.
- Explicit contributions list (3–4 bullets): the inverted penalty; the negative-transfer quantification contrasting baselines vs the foundation model; honest mechanism + limitations.

### 2. Related Work (~0.75 pg)
- Time-series foundation models (MOMENT/Chronos/TimesFM — cite the one you used correctly).
- ML for carbon flux (FLUXCOM; Tramontana et al. 2016; Reichstein et al.).
- Transfer learning theory (Pan & Yang 2010; Zhuang et al. 2021).
- Keep it tight — position your contribution, don't survey.

### 3. Data and Methods (~1.5 pg)
- Sites: 5 wetland training sites + 2 held-out test sites (UK-AMo wetland, SE-Htm forest). Compact site table.
- Preprocessing: hourly aggregation, 336→96 sliding windows, per-site z-scoring with no leakage.
- Models: foundation model (zero-shot + fine-tuned, 20 epochs) vs XGBoost, RF, LSTM.
- **State the information-asymmetry honestly here**: baselines use 19 covariates; the foundation model is univariate (target only). This is the limitation that motivates the journal paper — name it, don't hide it.
- Metrics (R², RMSE, MAE) + statistics (1000-sample bootstrap CIs, paired t-tests, Bonferroni, Cohen's d).

### 4. Results (~2.5 pg — the core)
- **4.1 Overall benchmark** — primary R² table (all models × both sites). Figure: `all_models_comparison.png`.
- **4.2 The inverted transfer penalty** — the headline. Δ(forest − wetland) per model; transfer-penalty table (baselines +29–61% RMSE vs foundation model −21.6%). Figure: `transfer_learning/transfer_matrix_heatmap.png`.
- **4.3 Mechanism** — error-distribution comparison + the diurnal-regularity argument (forest's regular photosynthetic cycle aligns with pretraining; wetland's suppressed, irregular flux is out-of-distribution). Figure: dissertation Fig 5.3 (error distributions).
- **4.4 (optional, only if space) Negative transfer across configurations** — the 55% / 33-of-60 result, compressed to one short paragraph + a sentence. If pages are tight, fold the key number into 4.2 and drop the subsection.

### 5. Discussion (~0.75 pg)
- Implications: foundation models may behave unintuitively in geoscience; alignment with pretraining distribution can matter more than domain match.
- Limitations (be direct): information asymmetry; a single foundation model; only two test sites.
- One forward sentence pointing to the planned extension (covariate conditioning + global sites) **without** giving away results — sets up the journal paper, avoids self-scooping.

### 6. Conclusion (~0.3 pg)
- Restate the contribution in two sentences; one line of future work.

### Acknowledgments
- Po Yang (supervision); FLUXNET community + site PIs; AutonLab (model); ICOS.
- **AI-use disclosure** if any new text/figures/code in the paper were AI-generated (proportional disclosure; see ACM policy). Editing/polishing your own writing needs no disclosure.

### References (≥ 5; every entry has a year; authors span ≥ 3 countries; superscript [n] in text)
- You already have a strong seed list in the README (MOMENT, XGBoost, RF, LSTM, FLUXNET2015, Reichstein, Pan & Yang, Zhuang, Jung, Tramontana, Settles). That comfortably clears 5 and the 3-country spread.

---

## 2. Keep / cut decisions (from your 5 dissertation findings)

| Dissertation finding | Conference paper | Why |
|---|---|---|
| **F1 — Inverted transfer penalty** | **KEEP — the spine** | The novel, surprising, defensible result |
| **F2 — Negative-transfer quantification** | **KEEP (compressed)** | Directly contrasts baselines vs foundation model; strengthens F1 |
| Primary benchmark table | **KEEP** | Needed to ground every claim |
| F3 — Ensemble diversity collapse | **CUT** (→ journal) | A separate story; dilutes the single message |
| F4 — Learning-curve saturation | **CUT** (→ journal) | Belongs to the data-efficiency / active-learning thread |
| F5 — Seasonal uncertainty / active learning | **CUT** (→ journal) | A distinct contribution; keep your powder dry |

Cutting F3–F5 is deliberate: an 8-page paper with one sharp message beats a crammed one, and it leaves genuine new material for the journal paper so the two don't overlap.

**Assets to reuse (already in repo):**
- Figures: `all_models_comparison.png`, `transfer_learning/transfer_matrix_heatmap.png`, error-distribution figure (Fig 5.3).
- Tables: primary R² table, transfer-penalty table, compact site-characteristics table.
- Numbers: pull verbatim from `results/transfer_learning/` and the benchmark outputs — do not retype from memory.

---

## 3. Twelve-day schedule (≈ 18–30 June)

| Days | Task |
|---|---|
| 18–19 Jun | Resolve the 3 blockers; scaffold the ACM project; lock contribution statement + draft abstract; gather figures/tables |
| 20–22 Jun | Draft Methods + Results (the core) using real numbers from `results/` |
| 23–24 Jun | Draft Introduction + Related Work |
| 25 Jun | Draft Discussion + Conclusion; finalise abstract |
| 26 Jun | CCS codes, keywords, references (years/3-country check), author block + ORCIDs (if not blind) |
| 27 Jun | Full read-through; verify ≥ 8 pages; figures 300 DPI, no non-English characters; run the 文件1 checklist |
| 28 Jun | Buffer / self-review |
| 29–30 Jun | Send to Po & Gaoshan for polish |

---

## 4. Brief to paste into Claude Code (in your repo)

> Create a `paper/` directory. Scaffold an ACM `sigconf` LaTeX project there: `main.tex` using `\documentclass[sigconf]{acmart}`, plus `references.bib`. Build the section skeleton exactly as in `conference_paper_outline.md` (Intro, Related Work, Data & Methods, Results with subsections 4.1–4.4, Discussion, Conclusion, Acknowledgments, References).
>
> Then, reading the actual files in `results/` and `figures/`:
> - Populate the primary R² table and the transfer-penalty table with the real values from `results/transfer_learning/` and the benchmark CSVs.
> - Copy `all_models_comparison.png`, `transfer_learning/transfer_matrix_heatmap.png`, and the error-distribution figure into `paper/figures/` and reference them with `\includegraphics`.
> - Seed `references.bib` from the citations in `README.md`.
> - Before writing anything, grep the codebase to confirm whether the model is MOMENT or TEMPO, and use the correct name throughout.
>
> Compile with `latexmk -pdf main.tex` and report any errors. Do not invent any numbers — every figure and statistic must come from a file in the repo.

**Setup commands to start:**
```bash
cd Carbon-flux-quatification
git checkout -b paper-ai4science      # work on a branch
mkdir -p paper/figures
# (then open Claude Code and paste the brief above)
```

If you prefer Word over LaTeX, tell Claude Code to use `acm_submission_template.docx` instead and follow the 文件1 margin/style specs — but LaTeX `sigconf` will be far less fiddly for a paper with equations and floats.
