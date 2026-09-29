import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

import torch
import transformers

from src.paths import PROJECT_ROOT


def utc_now():
    from datetime import datetime, timezone

    return datetime.now(
        timezone.utc
    ).isoformat()


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _run_git(*arguments):
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def get_git_provenance():
    try:
        commit = _run_git(
            "rev-parse",
            "HEAD",
        )

        branch = _run_git(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        )

        status = _run_git(
            "status",
            "--porcelain",
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return {
            "git_available": False,
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
        }

    return {
        "git_available": True,
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
    }


def get_environment_metadata(device):
    return {
        "python_version": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(
            device
        ),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": _package_version(
            "accelerate"
        ),
        "numpy_version": _package_version(
            "numpy"
        ),
        "pandas_version": _package_version(
            "pandas"
        ),
        "scikit_learn_version": _package_version(
            "scikit-learn"
        ),
    }
