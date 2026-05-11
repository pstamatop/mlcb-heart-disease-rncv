from typing import Callable, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features,
    categorical_features,
    binary_features=None,
    ordinal_features=None,
) -> ColumnTransformer:
    binary_features = list(binary_features or [])
    ordinal_features = list(ordinal_features or [])
    categorical_features = list(categorical_features or [])
    numeric_features = list(numeric_features or [])

    # Avoid accidental double assignment across groups.
    grouped = numeric_features + categorical_features + binary_features + ordinal_features
    if len(grouped) != len(set(grouped)):
        raise ValueError("Feature lists overlap across preprocessing groups.")

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # sklearn compatibility: sparse_output was added in newer versions.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("identity", FunctionTransformer(validate=False)),
        ]
    )
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("identity", FunctionTransformer(validate=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if binary_features:
        transformers.append(("bin", binary_pipe, binary_features))
    if ordinal_features:
        transformers.append(("ord", ordinal_pipe, ordinal_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def make_pipeline(
    estimator,
    numeric_features,
    categorical_features,
    binary_features=None,
    ordinal_features=None,
    feature_selector_factory: Optional[Callable[[], object]] = None,
) -> Pipeline:
    steps = [
        (
            "preprocessor",
            build_preprocessor(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                binary_features=binary_features,
                ordinal_features=ordinal_features,
            ),
        )
    ]
    if feature_selector_factory is not None:
        steps.append(("feature_selector", feature_selector_factory()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
