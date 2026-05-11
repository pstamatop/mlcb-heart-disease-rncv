import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .metrics_utils import bootstrap_median_ci, ci_overlaps, compute_metrics, extract_scores
from .model_specs import VALID_STAGES, make_pipeline

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception:  # pragma: no cover
    optuna = None
    TPESampler = None


@dataclass
class SearchResult:
    estimator: object
    best_params: Dict[str, object]
    best_inner_score: float


class RepeatedNestedCV:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimators: Dict[str, object],
        hyperparameter_spaces: Dict[str, Dict[str, object]],
        numeric_features: List[str],
        categorical_features: List[str],
        binary_features: Optional[List[str]] = None,
        ordinal_features: Optional[List[str]] = None,
        n_repeats: int = 10,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        search_strategy: str = "optuna",
        scoring: str = "mcc",
        random_state: int = 42,
        n_trials: int = 50,
        threshold: float = 0.5,
        feature_selector_factory: Optional[Callable[[], object]] = None,
        stage: str = "tuned_rncv_full_features",
    ):
        if stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid: {sorted(VALID_STAGES)}")
        if search_strategy not in {"optuna", "random", "none"}:
            raise ValueError("search_strategy must be one of: optuna, random, none")
        if scoring != "mcc":
            raise ValueError("Current implementation supports scoring='mcc' only.")
        # Hard-label metrics are currently computed from estimator.predict(...).
        # make non-default thresholds unsupported
        if float(threshold) != 0.5:
            raise ValueError(
                "threshold != 0.5 is not supported in RepeatedNestedCV.run() because hard labels "
                "are derived from estimator.predict(...). Use threshold=0.5."
            )

        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.estimators = estimators
        self.hyperparameter_spaces = hyperparameter_spaces
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.binary_features = list(binary_features or [])
        self.ordinal_features = list(ordinal_features or [])
        self.n_repeats = n_repeats
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.search_strategy = search_strategy
        self.scoring = scoring
        self.random_state = random_state
        self.n_trials = n_trials
        self.threshold = threshold
        self.feature_selector_factory = feature_selector_factory
        self.stage = stage

        self.outer_fold_results_: Optional[pd.DataFrame] = None
        self.outer_fold_predictions_: Optional[pd.DataFrame] = None
        self.summary_results_: Optional[pd.DataFrame] = None
        self.model_ranking_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[pd.DataFrame] = None
        self.feature_selection_frequency_transformed_: Optional[pd.DataFrame] = None
        self.feature_selection_frequency_original_: Optional[pd.DataFrame] = None

    def _build_pipeline(self, estimator):
        return make_pipeline(
            estimator=clone(estimator),
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            binary_features=self.binary_features,
            ordinal_features=self.ordinal_features,
            feature_selector_factory=self.feature_selector_factory,
        )

    def _sample_optuna_params(self, trial, search_space: Dict[str, object]) -> Dict[str, object]:
        params = {}
        for name, spec in search_space.items():
            if isinstance(spec, dict) and "type" in spec:
                p_type = spec["type"]
                if p_type == "categorical":
                    params[name] = trial.suggest_categorical(name, list(spec["choices"]))
                elif p_type == "float":
                    params[name] = trial.suggest_float(
                        name,
                        float(spec["low"]),
                        float(spec["high"]),
                        log=bool(spec.get("log", False)),
                    )
                elif p_type == "int":
                    if "step" in spec:
                        params[name] = trial.suggest_int(
                            name,
                            int(spec["low"]),
                            int(spec["high"]),
                            step=int(spec["step"]),
                        )
                    else:
                        params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
                else:
                    raise ValueError(f"Unsupported Optuna parameter type '{p_type}' for '{name}'.")
            else:
                unique_values = list(dict.fromkeys(spec))
                params[name] = trial.suggest_categorical(name, unique_values)
        return params

    def _sanitize_model_params(
        self,
        params: Dict[str, object],
        max_feature_k: Optional[int] = None,
    ) -> Dict[str, object]:
        """Resolve known incompatible parameter combinations."""
        clean = dict(params)
        solver = clean.get("model__solver")
        if solver == "svd" and "model__shrinkage" in clean:
            clean["model__shrinkage"] = None
        if "feature_selector__k" in clean and max_feature_k is not None:
            k_val = int(clean["feature_selector__k"])
            clean["feature_selector__k"] = max(1, min(k_val, int(max_feature_k)))
        return clean

    def _max_valid_feature_k(self, pipeline, X_train, y_train) -> Optional[int]:
        if "feature_selector" not in pipeline.named_steps:
            return None
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return None
        preproc = clone(preprocessor)
        preproc.fit(X_train, y_train)
        transformed = preproc.transform(X_train)
        return int(transformed.shape[1])

    def _get_transformed_feature_names(self, preprocessor, X_reference, y_reference) -> np.ndarray:
        try:
            return np.asarray(preprocessor.get_feature_names_out(), dtype=object)
        except Exception:
            # Fallback for transformers that do not expose get_feature_names_out.
            names: List[str] = []
            if not hasattr(preprocessor, "transformers_"):
                transformed = preprocessor.transform(X_reference)
                return np.asarray([f"feature_{i}" for i in range(transformed.shape[1])], dtype=object)

            for name, trans, cols in preprocessor.transformers_:
                if trans == "drop":
                    continue
                cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else [cols]
                if trans == "passthrough":
                    names.extend([str(c) for c in cols_list])
                    continue

                out_names = None
                if hasattr(trans, "get_feature_names_out"):
                    try:
                        out_names = trans.get_feature_names_out(cols_list)
                    except Exception:
                        try:
                            out_names = trans.get_feature_names_out()
                        except Exception:
                            out_names = None
                if out_names is not None:
                    names.extend([f"{name}__{str(n)}" for n in out_names])
                else:
                    names.extend([f"{name}__{str(c)}" for c in cols_list])

            transformed = preprocessor.transform(X_reference)
            width = int(transformed.shape[1])
            if len(names) != width:
                return np.asarray([f"feature_{i}" for i in range(width)], dtype=object)
            return np.asarray(names, dtype=object)

    def _fit_with_optuna(
        self,
        pipeline,
        search_space,
        X_train,
        y_train,
        seed,
        model_name: str,
        repeat: int,
        fold: int,
    ) -> SearchResult:
        if optuna is None or TPESampler is None:
            raise ImportError("Optuna is required for search_strategy='optuna'.")

        inner_cv = StratifiedKFold(n_splits=self.n_inner_splits, shuffle=True, random_state=seed)
        max_feature_k = self._max_valid_feature_k(pipeline, X_train, y_train)

        def objective(trial):
            params = self._sanitize_model_params(
                self._sample_optuna_params(trial, search_space),
                max_feature_k=max_feature_k,
            )
            scores = []
            for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train):
                X_in_train = X_train.iloc[inner_train_idx]
                y_in_train = y_train.iloc[inner_train_idx]
                X_in_valid = X_train.iloc[inner_valid_idx]
                y_in_valid = y_train.iloc[inner_valid_idx]

                model = clone(pipeline)
                model.set_params(**params)
                try:
                    model.fit(X_in_train, y_in_train)
                    y_pred = model.predict(X_in_valid)
                    scores.append(matthews_corrcoef(y_in_valid, y_pred))
                except Exception as exc:
                    # Keep failed trials in the study with a value of -1.0 (worst) and log the exception type/message
                    trial.set_user_attr("failed_exception_type", type(exc).__name__)
                    trial.set_user_attr("failed_exception_message", str(exc))
                    return -1.0

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=self.n_trials)

        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True,
        )
        best_model = None
        best_params = {}
        best_value = float("nan")
        candidate_failures: List[str] = []

        for t in sorted_trials:
            candidate_params = self._sanitize_model_params(t.params, max_feature_k=max_feature_k)
            candidate_model = clone(pipeline)
            candidate_model.set_params(**candidate_params)
            try:
                candidate_model.fit(X_train, y_train)
                best_model = candidate_model
                best_params = candidate_params
                best_value = float(t.value)
                break
            except Exception as exc:
                candidate_failures.append(f"{type(exc).__name__}: {exc}")
                continue

        if best_model is None:
            raise RuntimeError(
                "Optuna failed to produce any valid fitted model "
                f"for model='{model_name}', repeat={repeat}, fold={fold}, "
                f"search_space={search_space}. "
                f"Candidate failures: {candidate_failures[:5]}"
            )

        return SearchResult(
            estimator=best_model,
            best_params=best_params,
            best_inner_score=best_value,
        )

    def _fit_with_random_search(self, pipeline, search_space, X_train, y_train, seed) -> SearchResult:
        if not search_space:
            return self._fit_default(pipeline, X_train, y_train)
        inner_cv = StratifiedKFold(n_splits=self.n_inner_splits, shuffle=True, random_state=seed)
        scorer = make_scorer(matthews_corrcoef)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=self.n_trials,
            scoring=scorer,
            cv=inner_cv,
            random_state=seed,
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
        return SearchResult(
            estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_inner_score=float(search.best_score_),
        )

    def _fit_default(self, pipeline, X_train, y_train) -> SearchResult:
        model = clone(pipeline)
        model.fit(X_train, y_train)
        return SearchResult(estimator=model, best_params={}, best_inner_score=float("nan"))

    def _fit_model(
        self,
        pipeline,
        search_space,
        X_train,
        y_train,
        seed,
        model_name: str,
        repeat: int,
        fold: int,
    ) -> SearchResult:
        if self.search_strategy in {"optuna", "random"} and not search_space:
            return self._fit_default(pipeline, X_train, y_train)
        if self.search_strategy == "optuna":
            return self._fit_with_optuna(
                pipeline,
                search_space,
                X_train,
                y_train,
                seed,
                model_name=model_name,
                repeat=repeat,
                fold=fold,
            )
        if self.search_strategy == "random":
            return self._fit_with_random_search(pipeline, search_space, X_train, y_train, seed)
        return self._fit_default(pipeline, X_train, y_train)

    def run(self):
        rows = []
        pred_rows = []
        selected_feature_rows = []

        for repeat in range(self.n_repeats):
            outer_seed = self.random_state + repeat
            outer_cv = StratifiedKFold(
                n_splits=self.n_outer_splits,
                shuffle=True,
                random_state=outer_seed,
            )

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y), start=1):
                X_train = self.X.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                X_test = self.X.iloc[test_idx]
                y_test = self.y.iloc[test_idx]

                for model_index, (model_name, estimator) in enumerate(self.estimators.items()):
                    pipeline = self._build_pipeline(estimator)
                    search_space = self.hyperparameter_spaces.get(model_name, {})
                    inner_seed = self.random_state + repeat * 1000 + fold * 100 + model_index
                    result = self._fit_model(
                        pipeline=pipeline,
                        search_space=search_space,
                        X_train=X_train,
                        y_train=y_train,
                        seed=inner_seed,
                        model_name=model_name,
                        repeat=repeat + 1,
                        fold=fold,
                    )

                    selected_meta = self._extract_selected_feature_metadata(
                        estimator=result.estimator,
                        model_name=model_name,
                        repeat=repeat + 1,
                        fold=fold,
                        X_train=X_train,
                        y_train=y_train,
                    )
                    if selected_meta is not None:
                        selected_feature_rows.append(selected_meta)

                    y_pred = result.estimator.predict(X_test)
                    score_out = extract_scores(result.estimator, X_test)

                    metrics = compute_metrics(
                        y_true=y_test.to_numpy(),
                        y_pred=np.asarray(y_pred),
                        y_score=np.asarray(score_out.y_score),
                    )

                    row = {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "model": model_name,
                        "stage": self.stage,
                        "search_strategy": self.search_strategy,
                        "threshold": self.threshold,
                        "best_inner_score": result.best_inner_score,
                        "best_params": json.dumps(result.best_params, sort_keys=True),
                    }
                    row.update(metrics)
                    rows.append(row)

                    for local_i, sample_idx in enumerate(test_idx):
                        pred_rows.append(
                            {
                                "y_true": int(y_test.iloc[local_i]),
                                "y_pred": int(y_pred[local_i]),
                                "y_score": float(score_out.y_score[local_i])
                                if not np.isnan(score_out.y_score[local_i])
                                else np.nan,
                                "score_type": score_out.score_type,
                                "threshold": self.threshold,
                                "sample_index": int(sample_idx),
                                "repeat": repeat + 1,
                                "fold": fold,
                                "model": model_name,
                                "stage": self.stage,
                            }
                        )

        self.outer_fold_results_ = pd.DataFrame(rows)
        self.outer_fold_predictions_ = pd.DataFrame(pred_rows)
        self.selected_features_ = pd.DataFrame(selected_feature_rows)
        self.feature_selection_frequency_transformed_ = self._aggregate_feature_frequency(
            self.selected_features_, column="selected_transformed_feature_names"
        )
        self.feature_selection_frequency_original_ = self._aggregate_feature_frequency(
            self.selected_features_, column="selected_original_feature_names"
        )
        self.summary_results_ = self._summarize_results(self.outer_fold_results_)
        self.model_ranking_ = self.rank_models()
        return self.outer_fold_results_, self.summary_results_, self.outer_fold_predictions_

    def _derive_original_feature_name(self, transformed_name: str) -> str:
        feature_token = transformed_name.split("__", 1)[-1]
        all_original_features = (
            list(self.numeric_features)
            + list(self.binary_features)
            + list(self.ordinal_features)
            + list(self.categorical_features)
        )
        all_original_features = sorted(set(all_original_features), key=len, reverse=True)
        for original in all_original_features:
            if feature_token == original or feature_token.startswith(f"{original}_"):
                return original
        return feature_token

    def _extract_selected_feature_metadata(self, estimator, model_name: str, repeat: int, fold: int, X_train, y_train):
        if not hasattr(estimator, "named_steps"):
            return None
        if "feature_selector" not in estimator.named_steps:
            return None
        if "preprocessor" not in estimator.named_steps:
            return None

        preprocessor = estimator.named_steps["preprocessor"]
        selector = estimator.named_steps["feature_selector"]
        if not hasattr(selector, "get_support"):
            return None

        transformed_names = self._get_transformed_feature_names(preprocessor, X_train, y_train)
        support_mask = selector.get_support()
        selected_indices = np.where(support_mask)[0].astype(int)
        selected_transformed = transformed_names[selected_indices].tolist()
        selected_original = [self._derive_original_feature_name(name) for name in selected_transformed]
        selected_k = int(len(selected_indices))

        return {
            "model": model_name,
            "repeat": repeat,
            "fold": fold,
            "stage": self.stage,
            "selected_k": selected_k,
            "selected_indices": json.dumps(selected_indices.tolist()),
            "selected_transformed_feature_names": json.dumps(selected_transformed),
            "selected_original_feature_names": json.dumps(selected_original),
        }

    def _aggregate_feature_frequency(self, selected_df: pd.DataFrame, column: str) -> pd.DataFrame:
        if selected_df is None or selected_df.empty or column not in selected_df.columns:
            return pd.DataFrame(columns=["feature_name", "selection_count", "selection_frequency"])
        n_runs = int(len(selected_df))
        counts: Dict[str, int] = {}
        for raw in selected_df[column].tolist():
            if pd.isna(raw):
                continue
            names = json.loads(raw) if isinstance(raw, str) else list(raw)
            for name in set(names):
                counts[name] = counts.get(name, 0) + 1
        rows = [
            {
                "feature_name": name,
                "selection_count": int(cnt),
                "selection_frequency": float(cnt / n_runs) if n_runs > 0 else np.nan,
            }
            for name, cnt in counts.items()
        ]
        if not rows:
            return pd.DataFrame(columns=["feature_name", "selection_count", "selection_frequency"])
        return pd.DataFrame(rows).sort_values(
            by=["selection_frequency", "selection_count", "feature_name"], ascending=[False, False, True]
        ).reset_index(drop=True)

    def _summarize_results(self, fold_results: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [
            "mcc",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "f1",
            "recall",
            "specificity",
            "precision",
        ]
        grouped = fold_results.groupby(["model", "stage", "search_strategy"], as_index=False)
        rows = []
        for keys, g in grouped:
            model, stage, strategy = keys
            row = {"model": model, "stage": stage, "search_strategy": strategy}
            for metric_index, m in enumerate(metric_cols):
                vals = g[m].to_numpy(dtype=float)
                median, ci_low, ci_high = bootstrap_median_ci(
                    vals,
                    confidence_level=0.95,
                    n_bootstraps=2000,
                    random_state=self.random_state + metric_index,
                )
                row[f"{m}_median"] = float(median)
                row[f"{m}_mean"] = float(np.nanmean(vals))
                row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else np.nan
                row[f"{m}_ci_low"] = float(ci_low)
                row[f"{m}_ci_high"] = float(ci_high)
            rows.append(row)
        return pd.DataFrame(rows)

    def rank_models(
        self,
        primary_metric: str = "mcc",
        confidence_level: float = 0.95,
        n_bootstraps: int = 2000,
    ) -> pd.DataFrame:
        if self.outer_fold_results_ is None:
            raise ValueError("Run the evaluation before ranking models.")

        metric_col = primary_metric
        rows = []
        for model_name, g in self.outer_fold_results_.groupby("model"):
            vals = g[metric_col].to_numpy(dtype=float)
            median, ci_low, ci_high = bootstrap_median_ci(
                vals,
                confidence_level=confidence_level,
                n_bootstraps=n_bootstraps,
                random_state=self.random_state,
            )
            rows.append(
                {
                    "model": model_name,
                    f"{metric_col}_median": median,
                    f"{metric_col}_ci_low": ci_low,
                    f"{metric_col}_ci_high": ci_high,
                    "n_outer_points": int(np.sum(~np.isnan(vals))),
                }
            )

        ranking = pd.DataFrame(rows).sort_values(by=f"{metric_col}_median", ascending=False).reset_index(drop=True)
        if ranking.empty:
            return ranking

        top_ci = (ranking.loc[0, f"{metric_col}_ci_low"], ranking.loc[0, f"{metric_col}_ci_high"])
        overlap_flags = []
        for _, row in ranking.iterrows():
            this_ci = (row[f"{metric_col}_ci_low"], row[f"{metric_col}_ci_high"])
            overlap_flags.append(ci_overlaps(top_ci, this_ci))

        ranking["ci_overlap_with_top"] = overlap_flags

        secondary = self.outer_fold_results_.groupby("model", as_index=False).agg(
            roc_auc_median=("roc_auc", "median"),
            pr_auc_median=("pr_auc", "median"),
            balanced_accuracy_median=("balanced_accuracy", "median"),
            f1_median=("f1", "median"),
            recall_median=("recall", "median"),
            specificity_median=("specificity", "median"),
            precision_median=("precision", "median"),
        )
        ranking = ranking.merge(secondary, on="model", how="left")

        return ranking

    def _run_metadata(self) -> Dict[str, object]:
        has_feature_selector = self.feature_selector_factory is not None
        return {
            "stage": self.stage,
            "search_strategy": self.search_strategy,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "threshold": self.threshold,
            "n_repeats": self.n_repeats,
            "n_outer_splits": self.n_outer_splits,
            "n_inner_splits": self.n_inner_splits,
            "n_trials": self.n_trials,
            "model_names": sorted(list(self.estimators.keys())),
            "has_feature_selector": has_feature_selector,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def save_results(self, output_dir: Path | str):
        if self.outer_fold_results_ is None or self.outer_fold_predictions_ is None or self.summary_results_ is None:
            raise ValueError("No results available. Run the evaluation first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.outer_fold_results_.to_csv(out / "outer_fold_results.csv", index=False)
        self.outer_fold_predictions_.to_csv(out / "outer_fold_predictions.csv", index=False)
        self.summary_results_.to_csv(out / "summary_results.csv", index=False)

        ranking = self.model_ranking_ if self.model_ranking_ is not None else self.rank_models()
        ranking.to_csv(out / "model_ranking.csv", index=False)

        selected_features_df = self.selected_features_
        if selected_features_df is None:
            selected_features_df = pd.DataFrame(
                columns=[
                    "model",
                    "repeat",
                    "fold",
                    "stage",
                    "selected_k",
                    "selected_indices",
                    "selected_transformed_feature_names",
                    "selected_original_feature_names",
                ]
            )
        selected_features_df.to_csv(out / "selected_features_per_fold.csv", index=False)

        freq_transformed_df = self.feature_selection_frequency_transformed_
        if freq_transformed_df is None:
            freq_transformed_df = pd.DataFrame(columns=["feature_name", "selection_count", "selection_frequency"])
        freq_transformed_df.to_csv(out / "feature_selection_frequency_transformed.csv", index=False)

        freq_original_df = self.feature_selection_frequency_original_
        if freq_original_df is None:
            freq_original_df = pd.DataFrame(columns=["feature_name", "selection_count", "selection_frequency"])
        freq_original_df.to_csv(out / "feature_selection_frequency_original.csv", index=False)

        with (out / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self._run_metadata(), f, indent=2)
