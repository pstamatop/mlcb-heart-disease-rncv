def sample_optuna_params(trial, space):
    params = {}
    for name, spec in space.items():
        p_type = spec["type"]
        if p_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif p_type == "int":
            if "step" in spec:
                params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec["step"]))
            else:
                params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif p_type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported Optuna param type: {p_type}")
    return params
