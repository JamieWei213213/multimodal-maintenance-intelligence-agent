"""Experiment tracking for evaluation runs.

Wraps MLflow behind a tiny interface so evaluation CLIs can log params, metrics,
prompt versions and report artifacts with one call, and so the whole thing is a
no-op when MLflow is not installed or ``MLFLOW_TRACKING_URI`` is unset. Every run
also records the git commit and the prompt registry versions, which together
make a metric reproducible.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from maintenance_intelligence.prompts import list_prompts

logger = logging.getLogger("maintenance_intelligence.experiments")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (bool, int, float)) and prefix:
        out[prefix] = float(value)


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """``{"retrieval": {"recall_at_10": 1.0}}`` -> ``{"retrieval.recall_at_10": 1.0}``."""
    out: dict[str, float] = {}
    _flatten("", metrics, out)
    return out


@dataclass
class ExperimentRun:
    name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    artifacts: list[Path]
    run_id: str | None = None
    backend: str = "none"


class ExperimentTracker:
    def __init__(self, tracking_uri: str | None = None, experiment: str = "maintenance-intelligence") -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment = experiment
        self._mlflow = None
        if self.tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment)
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; logging to stdout only")
            except Exception as exc:  # unreachable server etc.
                logger.warning("MLflow unavailable (%s); logging to stdout only", exc)

    @property
    def backend(self) -> str:
        return "mlflow" if self._mlflow else "none"

    def log_run(
        self,
        name: str,
        *,
        params: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: list[Path] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> ExperimentRun:
        full_params: dict[str, Any] = {"git_commit": git_commit() or "unknown"}
        full_params.update({f"prompt.{key}": value for key, value in list_prompts().items()})
        full_params.update(params or {})
        flat_metrics = flatten_metrics(metrics or {})
        run = ExperimentRun(name=name, params=full_params, metrics=flat_metrics, artifacts=list(artifacts or []), backend=self.backend)
        if self._mlflow is None:
            logger.info(json.dumps({"event": "experiment_run", "name": name, "params": full_params, "metrics": flat_metrics}, default=str))
            return run
        mlflow = self._mlflow
        with mlflow.start_run(run_name=name) as active:
            mlflow.log_params({key: str(value)[:500] for key, value in full_params.items()})
            if flat_metrics:
                mlflow.log_metrics(flat_metrics)
            if tags:
                mlflow.set_tags(dict(tags))
            for artifact in run.artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            run.run_id = active.info.run_id
        return run
