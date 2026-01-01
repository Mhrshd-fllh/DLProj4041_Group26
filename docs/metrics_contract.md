# Metrics Contract

metrics.json MUST be a valid JSON file with at least:

## Required fields (minimum)
- run_id: string
- seed: int
- split: string (e.g., "val" or "test")
- primary_metric_name: string
- primary_metric_value: number
- timestamp_utc: string

## Recommended fields (to be added as the project grows)
- five_class: { macro_f1, balanced_acc, per_class: {...} }
- three_group: { macro_f1, balanced_acc, per_class: {...} }
- confusion_matrix_path: string
- calibration: { ece, reliability_diagram_path, temperature: number }
- resource: { device, train_time_sec, eval_time_sec }

## Notes
- Keep this file stable across trials so the orchestrator can compare runs reliably.
