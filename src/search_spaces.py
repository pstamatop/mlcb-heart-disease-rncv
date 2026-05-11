from typing import Dict

from scipy.stats import loguniform, randint, uniform

from .model_registry import CatBoostClassifier, LGBMClassifier, XGBClassifier


def get_param_spaces() -> Dict[str, Dict[str, object]]:
    """Backward-compatible alias to Optuna-typed spaces."""
    return get_param_spaces_optuna()


def get_param_spaces_optuna() -> Dict[str, Dict[str, object]]:
    """
    Typed search spaces for Optuna.

    Schema per parameter:
    - categorical: {"type": "categorical", "choices": [...]} 
    - float: {"type": "float", "low": a, "high": b, "log": bool}
    - int: {"type": "int", "low": a, "high": b, "step": optional}
    """
    spaces = {
        "lr_elasticnet": {
            "model__C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
            "model__l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
        },
        "gnb": {
            "model__var_smoothing": {"type": "float", "low": 1e-12, "high": 1e-7, "log": True},
        },
        "lda": {
            # ise numerically stable tuned variants; "eigen" can fail on some folds.
            "model__solver": {"type": "categorical", "choices": ["svd", "lsqr"]},
            "model__shrinkage": {"type": "categorical", "choices": [None, "auto", 0.1, 0.5, 0.9]},
        },
        "rf": {
            "model__n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
            "model__max_depth": {"type": "categorical", "choices": [None, 3, 5, 10, 15]},
            "model__min_samples_split": {"type": "int", "low": 2, "high": 20},
            "model__min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "model__max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        },
    }

    if LGBMClassifier is not None:
        spaces["lightgbm"] = {
            "model__n_estimators": {"type": "int", "low": 100, "high": 700, "step": 50},
            "model__learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "model__num_leaves": {"type": "int", "low": 15, "high": 127},
            "model__max_depth": {"type": "categorical", "choices": [-1, 3, 5, 10]},
            "model__subsample": {"type": "float", "low": 0.6, "high": 1.0, "log": False},
            "model__colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "log": False},
        }

    if XGBClassifier is not None:
        spaces["xgboost"] = {
            "model__n_estimators": {"type": "int", "low": 100, "high": 700, "step": 50},
            "model__learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "model__max_depth": {"type": "int", "low": 3, "high": 10},
            "model__subsample": {"type": "float", "low": 0.6, "high": 1.0, "log": False},
            "model__colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "log": False},
            "model__reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        }

    if CatBoostClassifier is not None:
        spaces["catboost"] = {
            "model__iterations": {"type": "int", "low": 200, "high": 1000, "step": 50},
            "model__learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "model__depth": {"type": "int", "low": 4, "high": 10},
            "model__l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 20.0, "log": True},
        }

    return spaces


def get_param_spaces_randomized() -> Dict[str, Dict[str, object]]:
    """Distribution-based spaces for RandomizedSearchCV."""
    spaces = {
        "lr_elasticnet": {
            "model__C": loguniform(1e-3, 1e2),
            "model__l1_ratio": uniform(0.0, 1.0),
        },
        "gnb": {
            "model__var_smoothing": loguniform(1e-12, 1e-7),
        },
        "lda": {
            "model__solver": ["svd", "lsqr"],
            "model__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
        },
        "rf": {
            "model__n_estimators": randint(100, 701),
            "model__max_depth": [None, 3, 5, 10, 15],
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 11),
            "model__max_features": ["sqrt", "log2", None],
        },
    }

    if LGBMClassifier is not None:
        spaces["lightgbm"] = {
            "model__n_estimators": randint(100, 701),
            "model__learning_rate": loguniform(1e-3, 0.3),
            "model__num_leaves": randint(15, 128),
            "model__max_depth": [-1, 3, 5, 10],
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        }

    if XGBClassifier is not None:
        spaces["xgboost"] = {
            "model__n_estimators": randint(100, 701),
            "model__learning_rate": loguniform(1e-3, 0.3),
            "model__max_depth": randint(3, 11),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(1e-3, 10.0),
        }

    if CatBoostClassifier is not None:
        spaces["catboost"] = {
            "model__iterations": randint(200, 1001),
            "model__learning_rate": loguniform(1e-3, 0.3),
            "model__depth": randint(4, 11),
            "model__l2_leaf_reg": loguniform(1e-3, 20.0),
        }

    return spaces
