def get_real_transformed_feature_names(preprocessor, X_ref, X_transformed_ref):
    """Return transformed feature names from fitted ColumnTransformer internals."""
    try:
        names = preprocessor.get_feature_names_out()
        names = [str(n) for n in names]
        if len(names) == X_transformed_ref.shape[1]:
            return names
    except Exception:
        pass

    if not hasattr(preprocessor, "transformers_"):
        raise RuntimeError("Preprocessor is not fitted or has no transformers_.")

    out_names = []
    for t_name, trans, cols in preprocessor.transformers_:
        if t_name == "remainder" and trans == "drop":
            continue
        if trans == "drop":
            continue

        if isinstance(cols, slice):
            cols_list = list(X_ref.columns[cols])
        elif hasattr(cols, "tolist"):
            cols_list = list(cols.tolist())
        elif isinstance(cols, (list, tuple)):
            cols_list = list(cols)
        else:
            cols_list = [cols]

        cols_list = [str(c) for c in cols_list]

        if trans == "passthrough":
            out_names.extend(cols_list)
            continue

        if hasattr(trans, "named_steps"):
            last_step = trans.steps[-1][1]

            if hasattr(last_step, "categories_"):
                for col_name, cats in zip(cols_list, last_step.categories_):
                    for cat in cats:
                        out_names.append(f"{col_name}_{cat}")
                continue

            out_names.extend(cols_list)
            continue

        if hasattr(trans, "get_feature_names_out"):
            try:
                trans_names = trans.get_feature_names_out(cols_list)
                out_names.extend([str(n) for n in trans_names])
                continue
            except Exception:
                try:
                    trans_names = trans.get_feature_names_out()
                    out_names.extend([str(n) for n in trans_names])
                    continue
                except Exception:
                    pass

        out_names.extend(cols_list)

    if len(out_names) != X_transformed_ref.shape[1]:
        raise RuntimeError(
            "Could not reliably reconstruct transformed feature names. "
            f"Got {len(out_names)} names for {X_transformed_ref.shape[1]} columns."
        )

    return out_names
