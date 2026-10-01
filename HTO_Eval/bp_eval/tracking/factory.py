from bp_eval.tracking.base_tracker import ExperimentTracker
from bp_eval.tracking.csv_tracker import CSVTracker


def get_tracker(config: dict) -> ExperimentTracker:
    """Build the experiment tracker for this run.

    CSVTracker is the only backend: zero extra dependencies, writes a
    full run record under runs_dir/<experiment_id>/ plus one row per run
    in runs_dir/registry.csv.
    """
    return CSVTracker(runs_dir=config.get("runs_dir", "runs"))
