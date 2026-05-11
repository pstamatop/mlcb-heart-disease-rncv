from typing import Dict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

THREADS = 4


def get_model_registry(random_state: int = 42) -> Dict[str, object]:
    missing = []
    if LGBMClassifier is None:
        missing.append("lightgbm")
    if XGBClassifier is None:
        missing.append("xgboost")
    if CatBoostClassifier is None:
        missing.append("catboost")

    if missing:
        install_tokens = " ".join(missing)
        raise ImportError(
            "Missing required model packages for Task 3.2: "
            f"{', '.join(missing)}. Install with: pip install {install_tokens}"
        )

    registry = {
        "lr_elasticnet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=5000,
            random_state=random_state,
        ),
        "gnb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "rf": RandomForestClassifier(random_state=random_state, n_jobs=THREADS),
        "lightgbm": LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=THREADS),
        "xgboost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=THREADS,
        ),
        "catboost": CatBoostClassifier(random_seed=random_state, verbose=0, thread_count=THREADS),
    }

    return registry
