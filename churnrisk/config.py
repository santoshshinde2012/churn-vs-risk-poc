"""Shared constants and paths.

Everything that a reader might want to change lives here, so the pipeline
scripts stay about the ideas rather than about bookkeeping.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
POLICIES = ROOT / "policies"

# Sub-directories of outputs/, created on demand by ensure_dirs().
PLOTS = OUTPUTS / "plots"
METRICS = OUTPUTS / "metrics"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
INFERENCE = OUTPUTS / "inference"      # the local stand-in for inference tables
MLRUNS = OUTPUTS / "mlruns"            # the local stand-in for the UC model registry

SEED = 7

# ----------------------------------------------------------------- churn ----
CHURN_DAYS = 240                 # length of the simulated engagement history
CHURN_ACCOUNTS = 20000
CHURN_LABEL_WINDOW = 30          # days of silence that count as churned
CHURN_TRAIN_SNAPSHOTS = (120, 150)
CHURN_SCORE_SNAPSHOT = 200       # "today" for the nightly batch job
CHURN_MIN_HISTORY = 30           # below this an account is a cold start
CHURN_CALL_CAPACITY = 200        # how many customers the retention team can call
CHURN_HOLDOUT_SHARE = 0.15       # share that never receives an intervention
CHURN_CAMPAIGN_SNAPSHOT = 170    # the night 04_feedback_loops.py runs its campaign

# ------------------------------------------------------------------ risk ----
RISK_DAYS = 240                  # "today" is the last day of the stream
RISK_ACCOUNTS = 800
RISK_MATURATION_DAYS = 90        # a label is trustworthy only once this old
RISK_DRIFT_DAY = 200             # the fraudsters change tactics here
RISK_OOT_START = RISK_DAYS - RISK_MATURATION_DAYS   # first day of the blind window
RISK_NEW_ACCOUNT_DAYS = 30       # below this an account is a cold start

DAY_SECONDS = 86_400


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for path in (DATA, OUTPUTS, PLOTS, METRICS, TABLES, MODELS, INFERENCE, MLRUNS):
        path.mkdir(parents=True, exist_ok=True)
