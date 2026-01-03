# EDA Report (Step 4)

## 1) Dataset Overview
- Dataset root: `data/raw/`
- Images folder: `data/raw/images/`
- Label source: `data/raw/train_labels.csv`
- Total rows in CSV: **1000**
- Missing files: **0**
- Corrupt/unreadable files: **0**
- Image column: **filename**
- Label column: **label** (grade 1–5)

## 2) Label Distribution (Grades 1–5)
Counts:
- Grade 1: **603** (60.3%)
- Grade 2: **212** (21.2%)
- Grade 3: **71** (7.1%)
- Grade 4: **66** (6.6%)
- Grade 5: **48** (4.8%)

### Interpretation
- The dataset is **highly imbalanced** toward Grade 1 (majority class).
- Minority grades (especially Grade 5) are significantly underrepresented.
- Expected impact: A model trained with standard cross-entropy is likely to bias predictions toward low grades unless imbalance mitigation is applied.

## 3) Derived 3-Group Distribution
Mapping used:
- Benign = {1,2}
- Borderline = {3}
- Malignant = {4,5}

Counts:
- benign: **815** (81.5%)
- borderline: **71** (7.1%)
- malignant: **114** (11.4%)

### Interpretation
- The 3-group setting is also imbalanced (benign dominates).
- Borderline is the smallest group and likely to have the lowest recall without targeted handling.

## 4) Image Characteristics

### 4.1 Sizes and mode
- All scanned images: **512 × 512**
- Mode: **L (grayscale)** for all images (1000/1000)

Implication:
- No resizing variability is expected at load time. Preprocessing can assume a fixed spatial size.

### 4.2 Intensity statistics (normalized to [0,1])
- min_global: **0.0**
- max_global: **1.0**
- mean(mean): **0.0565**
- mean(std): **0.0292**
- std(mean): **0.1362**

Interpretation:
- Images are generally **dark** (low average intensity), which is common in medical grayscale images.
- Normalization strategy should be consistent; later experiments should compare:
  - simple [0,1] scaling + standardization,
  - vs. per-image standardization.

## 5) Data Quality Issues
- Missing files: **0**
- Corrupt files: **0**
- Duplicate detection: **not computed** in this run (hashing disabled)

Implication:
- Dataset is clean and readable.
- Duplicate checking can still be worthwhile later to reduce leakage risk.

## 6) Implications for Modeling (Key Decisions)

1) **Splitting strategy**
   - Since we currently only have a single CSV (`train_labels.csv`), we must create **train/val/test splits**.
   - Splits should be **stratified by grade** (and also monitored at 3-group level) due to heavy imbalance.

2) **Imbalance handling**
   - Because Grade 1 dominates, we should evaluate imbalance mitigation:
     - class-weighted loss,
     - focal loss,
     - or balanced sampling strategies.
   - We should use metrics robust to imbalance: **macro-F1** and **balanced accuracy** (not just accuracy).

3) **Preprocessing**
   - All images are 512×512 grayscale → consistent input shape.
   - The low intensity mean suggests normalization choices matter; this should be defined in config and logged per run.

4) **Augmentations**
   - Start conservative (medical images): light geometric transforms (small rotations/translation) and mild intensity transforms.
   - Avoid aggressive flips/warps unless clinically justified.

5) **Evaluation focus**
   - Report both 5-class and 3-group metrics.
   - Pay attention to minority-grade recall and borderline performance.

## 7) Action Items for Next Steps
- Step 5 (splits): create `train/val/test` CSVs with stratification and log distributions.
- Step 6 (preprocessing): implement a config-driven dataset loader with grayscale handling and normalization.
- Step 7 (augmentations): add medically sensible augmentation policies and test their effect.
- Optional: run hash-based duplicate detection to reduce leakage risk.
