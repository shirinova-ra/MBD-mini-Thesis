# Multi-label Sequence Classification on MBD-mini

## 1. Project Overview

This repository contains the full pipeline for a bank client modeling task based on the [MBD-mini dataset](https://huggingface.co/datasets/ai-lab/MBD-mini). The goal is to predict multiple product-related targets for each client using their historical transactional, dialog, and geo behavior, represented as time-ordered event sequences.

The project was developed as part of my Master's diploma thesis.

## 2. Problem Statement

This project addresses a multi-label classification task based on the sequence of client events. The history of transactions and interactions is represented as a time-ordered sequence, but the goal of the model is not to forecast future values of the series (time series forecasting). Instead, the objective is to assign each client’s entire sequence to one or more target classes (multi-label sequence / time series classification), corresponding to different banking product needs.

## 3. Data

- **Source**: [MBD-mini dataset](https://huggingface.co/datasets/ai-lab/MBD-mini)
- **Modalities**:
  - `clients`: static client attributes
  - `targets`: 4 binary product-related targets (multi-label)
  - `detail_trx`: transactional events with timestamps
  - `detail_dialog`: call-center / chat interactions
  - `detail_geo`: geo events and mobility patterns
- **CV setup**: multiple folds (`fold=0,1,3,4`) for robust cross-validation.

Raw data are downloaded automatically from Hugging Face and converted into cleaned Parquet files per fold.

## 4. Pipeline

High-level pipeline:

1. **ETL & preprocessing**
   - Download `detail.tar.gz`, `targets.tar.gz`, `client_split.tar.gz` from Hugging Face.
   - Extract archives into the `data/` folder.
   - For each dataset (clients, targets, detail_trx, detail_dialog, detail_geo) and each fold (`fold=0,1,3,4`), merge all Parquet shards into a single Parquet file with additional metadata (`fold_number`, `dataset_name`).

2. **Feature engineering**
   - Sort events by `event_time` for each `client_id`.
   - Build sequence-aware aggregations (counts, frequencies, recency, trends) over transactions, dialogs, and geo events.
   - Join dynamic features with static client attributes and multi-label targets.

3. **Modeling**
   - Baseline tabular multi-label model (e.g., LightGBM / RandomForest).
   - Optional sequence models (RNN / Transformer encoder) on event-level sequences.

4. **Evaluation**
   - Cross-validation across folds.
   - Per-target and averaged metrics (ROC-AUC, PR-AUC, F1).

## 5. Modeling

The main task is multi-label sequence classification:

- **Input**: a client-level sequence of events (transactions, dialogs, geo) and static client features.
- **Output**: a 4-dimensional binary target vector indicating product-related outcomes.
- **Baselines**:
  - Gradient-boosted trees on sequence-aware aggregations.
- **Sequence models (optional)**:
  - RNN / GRU / LSTM or Transformer encoder over ordered events, followed by a multi-label output layer (sigmoid activations).

## 6. Results

(Example placeholder — fill in after experiments.)

- Cross-validation on folds `0,1,3,4`
- Average ROC-AUC across 4 targets: **X.XX**
- Per-target ROC-AUC: `target_1: X.XX`, `target_2: X.XX`, `target_3: X.XX`, `target_4: X.XX`

## 7. Repository Structure

```bash
.
├── src/
│   ├── download_data.py         # ETL: download + fold-based Parquet aggregation
│   ├── build_features.py        # Feature engineering and sequence-aware aggregations
│   └── train_models.py          # Training and evaluation scripts
├── notebooks/
│   ├── 02_eda.ipynb             # Exploratory data analysis
│   └── 03_modeling.ipynb        # Experiments and model comparisons
├── data/
│   ├── raw/                     # (ignored) raw HF data
│   └── final_by_fold_named/     # cleaned Parquet files per fold
├── README.md
└── requirements.txt
