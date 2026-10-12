import json, hashlib, subprocess, sys, datetime as dt
from pathlib import Path

def save_with_meta(df, path, description, inputs, params):
    path = Path(path); df.to_pickle(path)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "script": sys.argv[0],
            "git_commit": commit, "inputs": inputs, "params": params, "shape": df.shape,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))