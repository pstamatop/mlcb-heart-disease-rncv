from .model_registry import THREADS, get_model_registry
from .preprocessing import build_preprocessor, make_pipeline
from .search_spaces import get_param_spaces, get_param_spaces_optuna, get_param_spaces_randomized

STAGE_DEFAULT_REPEATED_CV = "default_repeated_cv"
STAGE_TUNED_FULL = "tuned_rncv_full_features"
STAGE_TUNED_SELECTED = "tuned_rncv_selected_features"
VALID_STAGES = {STAGE_DEFAULT_REPEATED_CV, STAGE_TUNED_FULL, STAGE_TUNED_SELECTED}

__all__ = [
    "THREADS",
    "STAGE_DEFAULT_REPEATED_CV",
    "STAGE_TUNED_FULL",
    "STAGE_TUNED_SELECTED",
    "VALID_STAGES",
    "build_preprocessor",
    "make_pipeline",
    "get_model_registry",
    "get_param_spaces",
    "get_param_spaces_optuna",
    "get_param_spaces_randomized",
]
