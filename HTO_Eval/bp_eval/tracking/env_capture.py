"""Capture the pieces of Part 4 that come from the surrounding environment
rather than from the model/config: git commit, exact run command, and a
dependency/environment reference.
"""
import subprocess
import sys
from pathlib import Path


def get_git_commit(repo_dir: str = ".") -> str:
    """Return the current commit hash, or a clear placeholder if there is
    no git repo yet (expected for a skeleton project before the real repo
    is created) or the working tree has uncommitted changes.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "no-git-repo"
        commit = result.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if dirty.stdout.strip():
            return f"{commit}-DIRTY (uncommitted changes present at run time)"
        return commit
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "no-git-repo"


def get_run_command() -> str:
    """The exact command line used to invoke this run."""
    return " ".join([sys.executable] + sys.argv)


def capture_environment(out_path: str) -> str:
    """Write `pip freeze` output to out_path and return a short reference
    string (line count) so registry.csv has something at-a-glance while
    the full list lives in the artifact.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        freeze_output = result.stdout
    except Exception as e:  # pragma: no cover - defensive
        freeze_output = f"# Could not capture environment: {e}\n"

    Path(out_path).write_text(
        f"# python {sys.version}\n" + freeze_output
    )
    n_packages = len([l for l in freeze_output.splitlines() if l.strip()])
    return f"python{sys.version_info.major}.{sys.version_info.minor}, {n_packages} packages (see environment.txt)"
