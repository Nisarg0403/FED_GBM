import sys, os, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
try:
    from app.utils.paths import (
        AGGREGATE_CSV, PER_SEED_CSV, PRIVACY_CSV, CONFIG_JSON,
        SPLIT_VERIFY_JSON, FEATURE_PROV_JSON, AGG_VERIFY_JSON
    )
except ModuleNotFoundError:
    from utils.paths import (
        AGGREGATE_CSV, PER_SEED_CSV, PRIVACY_CSV, CONFIG_JSON,
        SPLIT_VERIFY_JSON, FEATURE_PROV_JSON, AGG_VERIFY_JSON
    )

files = [AGGREGATE_CSV, PER_SEED_CSV, PRIVACY_CSV, CONFIG_JSON, SPLIT_VERIFY_JSON, FEATURE_PROV_JSON, AGG_VERIFY_JSON]
for f in files:
    status = 'OK' if os.path.exists(f) else 'MISSING'
    print(f"{status}: {os.path.basename(f)}")

agg = pd.read_csv(AGGREGATE_CSV)
print(f"Aggregate CSV: {len(agg)} rows, {len(agg.columns)} cols")

priv = pd.read_csv(PRIVACY_CSV)
print(f"Privacy CSV: {len(priv)} rows")

per = pd.read_csv(PER_SEED_CSV)
print(f"Per-seed CSV: {len(per)} rows")

with open(CONFIG_JSON) as f:
    cfg = json.load(f)
print(f"Config: {len(cfg['models'])} models, {len(cfg['seeds'])} seeds")

with open(SPLIT_VERIFY_JSON) as f:
    sv = json.load(f)
print(f"Split verification: identical_test_ids_across_all_seeds = {sv['identical_test_ids_across_all_seeds']}")

nan_in_agg = agg[["roc_auc_mean","pr_auc_mean","accuracy_mean","f1_mean"]].isna().sum().sum()
print(f"NaN in aggregate key metrics: {nan_in_agg}")

print("ALL OK")
