from src.utils.run_logger import (
    copy_config,
    init_run,
    make_run_id,
    snapshot_pip_freeze,
    write_decision_note,
    write_metrics_dummy,
    write_run_meta,
)

CONFIG_PATH = "configs/train.yaml"
OUTPUT_ROOT = "outputs"
SEED = 42


def main():
    run_id = make_run_id(prefix="run")
    paths = init_run(output_root=OUTPUT_ROOT, run_id=run_id)

    copy_config(CONFIG_PATH, paths.config_copy)
    snapshot_pip_freeze(paths.requirements_lock)
    write_run_meta(paths.run_meta, extra={"purpose": "smoke_repro"})

    write_metrics_dummy(paths.metrics, run_id=run_id, seed=SEED, split="smoke")
    write_decision_note(paths.decision_note)

    print(f"[OK] Created run folder: {paths.run_dir}")


if __name__ == "__main__":
    main()
