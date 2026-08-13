"""Model tracking and promotion, the local mirror of the Databricks setup.

On Databricks this is MLflow tracking plus a Unity Catalog registered model,
and batch jobs load the model by alias:

    models:/ml.churn.scorer@champion

Locally it is exactly the same MLflow API pointed at a file store under
outputs/mlruns, so the promotion pattern in this repo is the promotion pattern
you would ship. Nothing here is a simulation of MLflow; it is MLflow.

Promotion is guarded: a candidate only takes the @champion alias if it beats
the incumbent on the metric that matters, which keeps "we retrained, so we
shipped it" from being the release process.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager

from .config import MLRUNS, ensure_dirs

# MLflow is chatty on import and on every autolog decision.
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")

CHAMPION = "champion"
CHALLENGER = "challenger"


def _mlflow():
    """Import lazily: mlflow is slow to import and not every script needs it."""
    import mlflow  # noqa: PLC0415

    ensure_dirs()
    uri = MLRUNS.as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow


@contextmanager
def run(experiment: str, run_name: str):
    """An MLflow run scoped to an experiment, with the file store configured."""
    mlflow = _mlflow()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        yield active


def log_model(model, name: str, params: dict, metrics: dict, input_example=None,
              tags: dict | None = None, experiment: str | None = None) -> str:
    """Track a training run and register the model. Returns the new version."""
    import mlflow.sklearn  # noqa: PLC0415

    _mlflow()
    mlflow.set_experiment(experiment or name)
    with mlflow.start_run(run_name=f"train-{name}"):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if v is not None and _finite(v)})
        if tags:
            mlflow.set_tags(tags)
        info = mlflow.sklearn.log_model(
            model, "model", registered_model_name=name, input_example=input_example)
    version = _latest_version(name)
    print(f"  registered {name} version {version} "
          f"(run {info.run_id[:8]}, tracking {MLRUNS.name}/)")
    return version


def promote(name: str, version: str, metric: str, value: float,
            higher_is_better: bool = True) -> bool:
    """Give `version` the @champion alias only if it beats the incumbent.

    Returns True if the alias moved. The incumbent's score is read back from
    the run that produced it, so the comparison survives across processes.
    """
    mlflow = _mlflow()
    client = mlflow.MlflowClient()

    incumbent = champion_metric(name, metric)
    if incumbent is None:
        client.set_registered_model_alias(name, CHAMPION, version)
        print(f"  {name}@champion -> v{version} (first champion, {metric} {value:.4f})")
        return True

    better = value > incumbent if higher_is_better else value < incumbent
    if better:
        client.set_registered_model_alias(name, CHAMPION, version)
        print(f"  {name}@champion -> v{version} "
              f"({metric} {value:.4f} beats incumbent {incumbent:.4f})")
        return True

    client.set_registered_model_alias(name, CHALLENGER, version)
    print(f"  {name}@champion unchanged; v{version} parked as @challenger "
          f"({metric} {value:.4f} vs incumbent {incumbent:.4f})")
    return False


def load(name: str, alias: str = CHAMPION):
    """Load a model by alias, the way a batch job or an endpoint would."""
    import mlflow.sklearn  # noqa: PLC0415

    _mlflow()
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def champion_metric(name: str, metric: str) -> float | None:
    """The tracked metric of whatever currently holds @champion."""
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, CHAMPION)
    except Exception:
        return None
    try:
        data = client.get_run(version.run_id).data.metrics
    except Exception:
        return None
    return data.get(metric)


def champion_version(name: str) -> str | None:
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(name, CHAMPION).version
    except Exception:
        return None


def _latest_version(name: str) -> str:
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    return str(max(int(v.version) for v in versions))


def _finite(value) -> bool:
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False
