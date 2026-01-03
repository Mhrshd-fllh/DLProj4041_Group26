# Dataset Contract (Step 4.1)

This document defines the dataset assumptions, schema, label mapping, and invariants used across:
- dataset scanning (EDA),
- splitting,
- preprocessing,
- training,
- evaluation,
- inference.

If anything in the dataset format changes, this contract must be updated first.

---

## 1) Dataset Location & Access

### 1.1 Root directory
- Dataset root path (local, not committed to git):
  - `data/raw/` (recommended)
  - Actual path on my machine: **[FILL: e.g., D:\datasets\mammo\]**

### 1.2 File types
- Image extensions observed/expected: **[FILL: e.g., .png / .jpg / .jpeg / .dcm]**
- Color mode expected: **[FILL: grayscale / RGB / unknown]**
- Bit depth expected (if known): **[FILL: 8-bit / 16-bit / unknown]**

### 1.3 Data index / annotation source
How samples are referenced:
- [ ] A CSV file exists that lists image paths + labels
- [ ] Labels are encoded in folder names
- [ ] A JSON annotation file exists
- [ ] Other: **[FILL]**

Reference files (if any):
- Path(s): **[FILL: e.g., data/raw/labels.csv]**

---

## 2) Sample Schema (What is one sample?)

Each sample represents: **[FILL: e.g., a single mammography image / a view / a patient study]**

### 2.1 Mandatory fields per sample
A sample MUST provide:
- `image_path`: relative or absolute file path to an image
- `label_grade`: integer in **{1,2,3,4,5}** (5-class grades)

### 2.2 Optional fields (if available)
If provided by the dataset, we will store them and use them for leakage-safe splitting:
- `patient_id`: string/int
- `study_id`: string/int
- `laterality`: {L, R}
- `view`: **[FILL: e.g., CC/MLO/...]**
- `site/device`: acquisition device or hospital/site identifier

If these fields do NOT exist, we will note it and use best-effort splitting strategies.

---

## 3) Labels & Semantics

### 3.1 5-class grade definition
We assume grades are:
- Grade 1
- Grade 2
- Grade 3
- Grade 4
- Grade 5

Meaning of grades (if described by dataset source):
- Grade 1: **[FILL if known]**
- Grade 2: **[FILL if known]**
- Grade 3: **[FILL if known]**
- Grade 4: **[FILL if known]**
- Grade 5: **[FILL if known]**

If the dataset does not define these formally, we treat them as ordinal categories only.

### 3.2 Derived 3-group mapping
We also report predictions aggregated into 3 semantic groups:

- **Benign**: Grades **[FILL: default 1–2]**
- **Borderline / Suspicious**: Grades **[FILL: default 3]**
- **Malignant**: Grades **[FILL: default 4–5]**

Default mapping used unless the project spec states otherwise:
- Benign = {1, 2}
- Borderline = {3}
- Malignant = {4, 5}

---

## 4) Splitting Rules (Leakage Prevention)

### 4.1 Primary rule (preferred)
If `patient_id` exists:
- Splits MUST be created at **patient level**, meaning no patient appears in more than one of {train, val, test}.

### 4.2 Secondary rule
If no patient_id but `study_id` exists:
- Splits MUST be created at study level.

### 4.3 Fallback rule
If no patient/study identifiers exist:
- We will:
  1) stratify by label_grade,
  2) attempt duplicate detection (hashing) to reduce leakage risk,
  3) document limitations in the EDA report.

---

## 5) Image Handling Assumptions

### 5.1 Readability
- A sample is valid only if the image file can be read successfully.
- Unreadable/corrupted images are counted and excluded, with their paths logged.

### 5.2 Size and resizing
- We do NOT assume a fixed image size.
- Preprocessing later will standardize resolution using a config-driven pipeline.

### 5.3 Intensity range
- We do NOT assume intensities are normalized.
- Dataset scanning will compute global statistics (min/max/mean/std) to inform normalization.

### 5.4 Orientation / laterality
- We do NOT flip images unless justified by EDA.
- Any geometric transforms must be medically sensible and documented.

---

## 6) Data Quality Checks (Scanner MUST verify)

The dataset scanner (Step 4.2) MUST compute:
1) total sample count
2) grade distribution (1–5)
3) derived 3-group distribution
4) unreadable/corrupted count (+ list)
5) image size distribution (H, W)
6) intensity statistics (per-image mean/std and global)
7) duplicate detection (approx via hash) + count

---

## 7) Output Contracts (What gets produced)

### 7.1 EDA outputs (not committed)
The scanner produces under `outputs/eda/`:
- `dataset_summary.json`
- `class_distribution.csv`
- `image_size_stats.csv`
- `corrupt_files.csv` (if any)
- `duplicates.csv` (if any)
- Optional plots (png)

### 7.2 EDA report (committed)
We write `docs/eda_report.md` summarizing:
- evidence-based findings,
- issues,
- implications for modeling and trials.

---

## 8) Open Questions / To Fill

- Dataset source/description link: **[FILL]**
- Are images DICOM or pre-converted? **[FILL]**
- Do we have patient_id / study_id? **[FILL]**
- Confirm grade meanings (if provided): **[FILL]**
- Confirm 3-group mapping expected by spec: **[FILL]**
