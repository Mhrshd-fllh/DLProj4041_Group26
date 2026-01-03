## Split Plan (Step 5)
We generate train/val/test splits from `train_labels.csv` using a fixed random seed (42) and stratification on grade labels (1–5) with ratio 80/10/10. The split distributions are exported to `outputs/splits/<run_id>/` for auditability.
