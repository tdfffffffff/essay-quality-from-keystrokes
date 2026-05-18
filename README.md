# Keystroke-Based Essay Quality Prediction

> Predicting essay quality from writers' keystroke logs alone, no final text required.

Ranked **Top 2% (38/1,877)** on the Kaggle public leaderboard with a final ensemble RMSE of **0.5767**.

**Tech stack:** Python · LightGBM · XGBoost · CatBoost · scikit-learn · Optuna · Polars · NumPy · pandas

---

## The Problem

Can we tell how good an essay is just by watching *how* it was written?

This project tackles that question directly. Given only raw keystroke logs recorded during essay drafting, with all alphanumeric characters anonymised as `"q"`, the goal is to predict each writer's final essay score (0–6 scale) without ever seeing the actual essay text.

This makes the task fundamentally harder than conventional automated essay scoring (AES): standard NLP approaches and pretrained language models cannot be applied because the readable text is gone. Quality signals must be recovered entirely from the temporal and behavioural trace of the writing process — typing speed, pause patterns, revision behaviour, cursor movement, and burst dynamics.

---

## Approach Overview

We built a **dual-modality pipeline** that analyses writing from two complementary perspectives:

**Modality 1 — What was written:** Reconstruct the final essay from the keystroke log (replaying every insert, delete, replace, move, and paste event) to recover lexical and structural signals that would otherwise be inaccessible.

**Modality 2 — How it was written:** Mine the raw keystroke sequence for process-level behavioural signals: pause distributions, burst lengths, revision patterns, cursor revisits, and typing speed trajectories, that are invisible in the final text.

These two modalities are then fused in a **two-stage stacking ensemble** that combines five base learners with a linear meta-learner, achieving better generalisation than any individual model.

---

## Pipeline Architecture

```
Raw Keystroke Logs
        │
        ├──► Essay Reconstruction ──► Lexical Features (TF-IDF, surface, grammar)
        │
        └──► Keystroke Log ──────────► Process Features (pauses, bursts, speed, ratios)
                                                │
                                        ┌───────┴────────────────────────────┐
                                        │         Stage 1: Base Models       │
                                        │  LightGBM · XGBoost · CatBoost     │
                                        │  LGBM Ordinal · XGB Ordinal        │
                                        │  Ridge (TF-IDF signal)             │
                                        └───────────────┬────────────────────┘
                                                        │ OOF predictions (6 columns)
                                                        ▼
                                        ┌───────────────────────────────────┐
                                        │         Stage 2: Meta-Learners    │
                                        │  Linear Regression (RMSE 0.5796)  │
                                        │  Ordinal Logistic (RMSE 0.5802)   │
                                        └───────────────┬───────────────────┘
                                                        │ Inverse-RMSE blending
                                                        ▼
                                              Final Ensemble RMSE: 0.5790
```

---

## Feature Engineering

Features are produced from two sources — the reconstructed essay and the raw keystroke log — yielding over 300 features per writer.

### From the Reconstructed Essay

**1. Text Surface Features** — character count, word count, paragraph count, average word length, lexical diversity, and prose density (unique to anonymised datasets).

**2. Multi-level Statistics** — distributional aggregates (count, min, max, median, std, q25, q75) at word, sentence, and paragraph level, motivated by research showing syntactic structural variation accounts for 45–49% of variance in argumentative essay scores beyond length alone.

**3. Grammar & Error Features** — regex-based counts of punctuation errors, incorrect spacing, and a consonant-run proxy for uncorrected typing errors that exploits keystroke-based reconstruction to surface artefacts absent from directly submitted essays.

**4. TF-IDF Features** — word n-gram (unigram–trigram) and character n-gram (2–5 gram) vectorisers, reduced to 64 components via TruncatedSVD, plus a Ridge meta-signal trained on the full uncompressed TF-IDF matrix to recover lexical co-occurrence patterns discarded during dimensionality reduction.

### From the Raw Keystroke Log

**1. Timing & Speed** — keys per second as a macro fluency proxy, and writing speed milestones recording time-to-reach word counts of 200/300/400/500 to reveal writing trajectory patterns.

**2. Pause Distributions** — inter-keystroke gaps binned into 8 duration brackets (micro < 100ms through very long > 3s) with cognitive interpretations, aggregated by linguistic boundary type (within-word, between-word, between-sentence, between-paragraph).

**3. Burst Analysis** — productive bursts (Input + Remove/Cut sequences with gaps ≤ 2000ms) and rhythmic bursts (Input-only sequences), with 10 summary statistics each for length and duration. Longer bursts indicate greater compositional automaticity.

**4. Cursor Revisit Features** — counts of how often each cursor position is revisited, grouped into buckets (1 through ≥ 6 visits), capturing non-linear composition behaviour associated with higher writing proficiency.

**5. Activity & Event Counts** — per-writer counts of each activity type and key event, plus behavioural diversity scores measuring the variety of key events used.

**6. Session-Normalised Ratios** — DI ratio, text-to-keystroke efficiency, words per minute, words per event, and events per second, normalising for session length to enable fair cross-writer comparison.

**7. Editing Run Statistics** — descriptive statistics across run length, run duration, and inter-run gap for consecutive Input and Remove/Cut sequences, distinguishing fluent progressive writers from fragmented heavily-monitored ones.

---

## Modelling

### Stage 1: Base Learners

| Model | Type | CV / OOF RMSE |
|---|---|---|
| CatBoost | Regressor | 0.5872 |
| LightGBM | Regressor | 0.5926 |
| XGBoost | Ordinal Classifier | 0.5935 |
| LightGBM | Ordinal Classifier | 0.5985 |
| XGBoost | Regressor | 0.8624 |
| Ridge | TF-IDF Signal | — |

All tree models tuned with Optuna minimising mean OOF RMSE across 10 stratified folds. XGBoost's weaker individual performance is tolerated because its level-wise growth produces errors uncorrelated with CatBoost and LightGBM's leaf-wise approaches — diversity the meta-learner exploits.

**Ordinal classifiers** reformulate scoring as a series of binary exceedance tasks (does score > 0.5? > 1.0? ... > 5.5?), recovering the ordered structure that regression models ignore.

**Cross-validation** uses 10-fold stratified KFold with strata = `(score × 2).astype(int)`, ensuring all 11 score levels (including underrepresented tails 1.0–2.0 and 5.0–6.0) appear in every fold.

### Stage 2: Meta-Learners

The 6-column OOF matrix (one column per base learner) feeds two meta-learners:

- **Linear Regression** — natural aggregation of well-calibrated continuous predictions; OOF RMSE 0.5796
- **Ordinal Logistic Regression** — operates in log-odds space, capturing calibration differences between base models at each score threshold; OOF RMSE 0.5802

Final predictions are blended using inverse-RMSE weighting (linear 0.5003, ordinal 0.4997), producing the final ensemble RMSE of **0.5790**.

---

## Results

| Model | CV RMSE | Δ vs. Baseline |
|---|---|---|
| Naive mean baseline | 1.024 | — |
| CatBoost Regressor | 0.5872 | −0.437 |
| LightGBM Regressor | 0.5926 | −0.432 |
| Linear Meta-Model | 0.5796 | −0.445 |
| **Final Ensemble** | **0.5790** | **−0.445** |

**Ablation study results:**

| Feature Group Removed | ΔRMSE | Additional Essays Mispredicted |
|---|---|---|
| Keystroke process features | +0.00634 | ~16 |
| Text-based features | +0.00394 | ~10 |
| Pause & burst features only | +0.00019 | ~1 |

Keystroke process features contribute the largest single improvement, confirming that behavioural signals capture information about *how* the essay was written that is not recoverable from the reconstructed text.

**Kaggle leaderboard:** Public score 0.5767 · Rank 38 / 1,877 entries · **Top 2.02%**

---

## Setup

```bash
git clone https://github.com/<your-username>/essay-quality-from-keystrokes.git
cd essay-quality-from-keystrokes
pip install -r requirements.txt
```

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality) and place the CSV files in `data/`.

Run the notebook end-to-end:

```bash
jupyter notebook notebooks/Keystroke_Based_Essay_Quality_Prediction.ipynb
```

---

## Key Design Decisions

**Why reconstruct essays rather than work from raw logs alone?** Reconstruction enables lexical and structural feature extraction that would otherwise require the final text. TF-IDF with Ridge regression proved to be the largest single feature contributor after keystroke process features, justifying the added complexity.

**Why include XGBoost despite its high individual RMSE?** Ensemble diversity matters more than individual model strength. XGBoost's level-wise growth produces errors uncorrelated with leaf-wise models, and the meta-learner automatically learned to down-weight its contribution, with its fitted coefficient reflecting its weaker performance.

**Why ordinal classification on top of regression?** Standard regression ignores the ordered structure of essay scores. Reformulating as a proportional-odds binary problem at both the base and meta level captures threshold-level structure — whether a score exceeds 3.5 vs 4.0 vs 4.5 — that regression treats as equivalent distances.

**Why 10-fold CV over 5-fold?** With 2,471 samples, 10-fold provides ~247 validation samples per fold versus ~494 for 5-fold, producing less noisy fold-level RMSE estimates and more reliable early-stopping signals. The bias-variance tradeoff favours 10-fold at this sample size.

---

*Data Science Project · Kaggle Top 2% (38/1,877)*
