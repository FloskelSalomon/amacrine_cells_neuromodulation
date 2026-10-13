import json, hashlib, subprocess, sys, platform, warnings, datetime as dt
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn

def save_with_meta(df, path, description, inputs, params):
    path = Path(path); df.to_pickle(path)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "script": sys.argv[0],
            "git_commit": commit, "inputs": inputs, "params": params, "shape": df.shape,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))

def _git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return commit, dirty


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _describe_model(model, X):
    """Summarize each fitted step (works for a Pipeline or a single estimator)."""
    steps = dict(model.named_steps) if hasattr(model, "named_steps") else {"model": model}
    summary = {}
    for name, est in steps.items():
        info = {"class": type(est).__name__, "params": est.get_params()}
        if hasattr(est, "explained_variance_ratio_"):  # PCA
            evr = est.explained_variance_ratio_
            info["explained_variance_ratio"] = np.round(evr, 4)
            info["cumulative_explained_variance"] = round(float(evr.sum()), 4)
        if hasattr(est, "converged_"):  # GaussianMixture
            info.update(converged=est.converged_, n_iter=est.n_iter_,
                        lower_bound=round(float(est.lower_bound_), 4))
        summary[name] = info

    # Training-set BIC and cluster sizes for the final GMM step
    final = list(steps.values())[-1]
    if hasattr(final, "bic"):
        X_in = model[:-1].transform(X) if hasattr(model, "named_steps") else X
        summary["train_bic"] = round(float(final.bic(X_in)), 2)
        labels, counts = np.unique(model.predict(X), return_counts=True)
        summary["train_cluster_sizes"] = dict(zip(labels.tolist(), counts.tolist()))
    return summary


def save_model_with_meta(model, path, description, inputs, params, X_train, extras=None):
    path = Path(path)
    joblib.dump(model, path)
    commit, dirty = _git_info()
    feature_names = getattr(model, "feature_names_in_", None)
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "description": description,
        "script": sys.argv[0],
        "git_commit": commit,
        "git_dirty": dirty,  # True = uncommitted changes when model was saved
        "inputs": inputs,
        "params": params,
        "train_shape": np.shape(X_train),
        "feature_names": list(feature_names) if feature_names is not None else None,
        "model": _describe_model(model, X_train),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "extras": extras or {},
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    return meta


def load_model_with_meta(path, check=True):
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if check:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        if sha != meta["sha256"]:
            raise ValueError(f"Hash mismatch for {path}: file {sha} vs meta {meta['sha256']}")
        if meta["versions"]["sklearn"] != sklearn.__version__:
            warnings.warn(f"Model saved with sklearn {meta['versions']['sklearn']}, "
                          f"you have {sklearn.__version__}")
    return joblib.load(path), meta