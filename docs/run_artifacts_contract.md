# Run Artifacts Contract

Each run MUST create a dedicated folder under outputs/:

outputs/
  run_<RUN_ID>/
    config.yaml
    requirements.lock.txt
    run_meta.json
    metrics.json
    decision_note.md
    logs/ (optional)

## Required files
- config.yaml: exact configuration used for the run (copy of the input config).
- requirements.lock.txt: dependency lock snapshot taken at run time (pip freeze).
- run_meta.json: runtime metadata (timestamp, duration, device info, python/torch versions).
- metrics.json: evaluation summary for the run (schema in metrics_contract.md).
- decision_note.md: short rationale/notes (for agentic orchestration later).

## Naming
- RUN_ID should be unique and sortable (timestamp-based is recommended).

## Rules
- No data, outputs, or checkpoints are committed to git.
- Only docs/config/scripts/utilities are committed.
