"""
MLP training script: fixed-arch baseline + HPO for all feature sets.

Two modes per experiment:
  fixed : 5-fold CV with fixed (256, 128, 64) architecture, refit on full train
  optuna : 30-trial Optuna TPE search, refit best config on full train

N_PARALLEL experiments run simultaneously. Skips completed experiments.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json, sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]  # repo root (script lives in scripts/)
sys.path.insert(0, str(ROOT))
from src.data_utils import load_experiment

# config

MODELS_DIR = ROOT / 'models' / 'mlp'
RESULTS_DIR = ROOT / 'results' / 'mlp'
SUMMARY_CSV = RESULTS_DIR / 'mlp_summary.csv'

TARGET_IDX = 2
RANDOM_STATE = 42
CV_FOLDS = 5
N_TRIALS = 30
N_PARALLEL = 4

FIXED_MLP = dict(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu', solver='adam',
    alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
    max_iter=500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

HIDDEN_CONFIGS = [
    (64,), (128,), (256,), (512,),
    (128, 64), (256, 128), (512, 256),
    (256, 128, 64), (512, 256, 128),
]

BASELINE_SETS = [('baseline', 2)]
NON_BASELINE = [
    ('geo_decorr', 5),
    ('geo_all', 6),
    ('rac_decorr', 45),
    ('rac_all', 81),
    ('combined_decorr', 50),
    ('combined_all', 87),
]
ALL_FEATURE_SETS = BASELINE_SETS + NON_BASELINE
SPLITS = ['topology', 'metal', 'random']

# helpers

def make_pipe(mlp_kw):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('mlp', MLPRegressor(**mlp_kw)),
    ])


def eval_metrics(y_true, y_pred):
    return dict(
        test_RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        test_MAE=float(mean_absolute_error(y_true, y_pred)),
        test_R2=float(r2_score(y_true, y_pred)),
    )


# fixed-arch baseline

def run_fixed(feat_set, n_feat, split):
    tag = f'{feat_set}__{split}'
    model_path = MODELS_DIR / f'{tag}__fixed.joblib'

    if model_path.exists():
        print(f'[SKIP fixed]  {tag}', flush=True)
        return None

    print(f'[START fixed] {tag}  feat={n_feat}  folds={CV_FOLDS}', flush=True)
    t0 = time.time()

    enc = 'target' if feat_set == 'baseline' else 'label'
    X_tr, X_te, y_tr, y_te = load_experiment(split, feat_set, encoding=enc)
    y_tr1 = y_tr[:, TARGET_IDX]
    y_te1 = y_te[:, TARGET_IDX]

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # CV RMSE with fixed arch
    scores = cross_val_score(make_pipe(FIXED_MLP), X_tr, y_tr1, cv=kf,
                               scoring='neg_mean_squared_error')
    cv_rmse = float(np.sqrt(-scores.mean()))

    # Refit on full train
    pipe = make_pipe(FIXED_MLP)
    pipe.fit(X_tr, y_tr1)
    m = eval_metrics(y_te1, pipe.predict(X_te))
    n_iter = pipe.named_steps['mlp'].n_iter_

    elapsed = (time.time() - t0) / 60
    print(f'[DONE fixed]  {tag}  cv_rmse={cv_rmse:.4f}  '
          f'test_rmse={m["test_RMSE"]:.4f}  R2={m["test_R2"]:.4f}  ({elapsed:.1f} min)',
          flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    return dict(feature_set=feat_set, split=split, method='fixed',
                n_features=n_feat, best_cv_RMSE=cv_rmse, n_iter=n_iter, **m)


# HPO

def run_hpo(feat_set, n_feat, split):
    tag = f'{feat_set}__{split}'
    model_path = MODELS_DIR / f'{tag}__optuna.joblib'

    if model_path.exists():
        print(f'[SKIP hpo]    {tag}', flush=True)
        return None

    print(f'[START hpo]   {tag}  feat={n_feat}  trials={N_TRIALS}  folds={CV_FOLDS}',
          flush=True)
    t0 = time.time()

    X_tr, X_te, y_tr, y_te = load_experiment(split, feat_set)
    y_tr1 = y_tr[:, TARGET_IDX]
    y_te1 = y_te[:, TARGET_IDX]

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        idx = trial.suggest_int('hidden_idx', 0, len(HIDDEN_CONFIGS) - 1)
        pipe = make_pipe(dict(
            hidden_layer_sizes=HIDDEN_CONFIGS[idx],
            activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
            alpha=trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            solver='adam', max_iter=200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ))
        scores = cross_val_score(pipe, X_tr, y_tr1, cv=kf,
                                 scoring='neg_mean_squared_error')
        return float(np.sqrt(-scores.mean()))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    bp = study.best_params
    hidden = HIDDEN_CONFIGS[bp['hidden_idx']]
    act, alph, lr, bs = bp['activation'], bp['alpha'], bp['lr'], bp['batch_size']

    best_kw = dict(hidden_layer_sizes=hidden, activation=act, alpha=alph,
                   learning_rate_init=lr, batch_size=bs,
                   solver='adam', max_iter=500, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20,
                   random_state=RANDOM_STATE)
    pipe = make_pipe(best_kw)
    pipe.fit(X_tr, y_tr1)
    m = eval_metrics(y_te1, pipe.predict(X_te))
    n_iter = pipe.named_steps['mlp'].n_iter_
    elapsed = (time.time() - t0) / 60

    print(f'[DONE hpo]    {tag}  cv_rmse={study.best_value:.4f}  '
          f'test_rmse={m["test_RMSE"]:.4f}  R2={m["test_R2"]:.4f}  ({elapsed:.1f} min)',
          flush=True)
    print(f'              arch={hidden}  act={act}  alpha={alph:.1e}  '
          f'lr={lr:.1e}  batch={bs}', flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    with open(MODELS_DIR / f'{tag}__optuna_params.json', 'w') as f:
        json.dump(dict(best_cv_RMSE=float(study.best_value), n_features=n_feat,
                       hidden_layer_sizes=str(hidden), activation=act,
                       alpha=float(alph), learning_rate_init=float(lr),
                       batch_size=int(bs)), f, indent=2)

    return dict(feature_set=feat_set, split=split, method='optuna',
                n_features=n_feat, best_cv_RMSE=study.best_value, n_iter=n_iter, **m)


# main

if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # baseline uses fixed-arch only (HPO not meaningful with 2 features)
    # non-baseline uses HPO only (the canonical result for the final CSV)
    fixed_jobs = [(fs, n, sp) for fs, n in BASELINE_SETS for sp in SPLITS]
    hpo_jobs = [(fs, n, sp) for fs, n in NON_BASELINE  for sp in SPLITS]

    n_fixed_done = sum(1 for fs, n, sp in fixed_jobs
                       if (MODELS_DIR / f'{fs}__{sp}__fixed.joblib').exists())
    n_hpo_done = sum(1 for fs, n, sp in hpo_jobs
                       if (MODELS_DIR / f'{fs}__{sp}__optuna.joblib').exists())

    print(f'Fixed-arch  : {len(fixed_jobs)} baseline experiments  ({n_fixed_done} done, '
          f'{len(fixed_jobs)-n_fixed_done} to run)')
    print(f'HPO         : {len(hpo_jobs)} experiments  ({n_hpo_done} done, '
          f'{len(hpo_jobs)-n_hpo_done} to run)')
    print(f'Parallelism : {N_PARALLEL} simultaneously')
    print()

    print('Fixed-arch baseline')
    fixed_results = Parallel(n_jobs=N_PARALLEL, prefer='processes', verbose=0)(
        delayed(run_fixed)(fs, n, sp) for fs, n, sp in fixed_jobs
    )

    print('\n HPO')
    hpo_results = Parallel(n_jobs=N_PARALLEL, prefer='processes', verbose=0)(
        delayed(run_hpo)(fs, n, sp) for fs, n, sp in hpo_jobs
    )

    new_rows = [r for r in fixed_results + hpo_results if r is not None]

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if SUMMARY_CSV.exists():
            existing = pd.read_csv(SUMMARY_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(SUMMARY_CSV, index=False)
        print(f'\nAdded {len(new_rows)} rows -> {SUMMARY_CSV}')
    else:
        print('\nAll experiments already done.')

    # build canonical CSV: one row per (feature_set, split)
    # baseline => fixed arch;  all others => optuna (HPO)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        baseline_rows = df[(df.feature_set == 'baseline') & (df.method == 'fixed')]
        optuna_rows = df[(df.feature_set != 'baseline') & (df.method == 'optuna')]
        canonical = pd.concat([baseline_rows, optuna_rows], ignore_index=True)

        FEAT_ORDER = ['baseline', 'geo_decorr', 'geo_all',
                      'rac_decorr', 'rac_all', 'combined_decorr', 'combined_all']
        SPLIT_ORDER = ['random', 'topology', 'metal']
        canonical['_fs'] = pd.Categorical(canonical.feature_set, categories=FEAT_ORDER, ordered=True)
        canonical['_sp'] = pd.Categorical(canonical.split,        categories=SPLIT_ORDER, ordered=True)
        canonical = (canonical.sort_values(['_fs', '_sp'])
                               .drop(columns=['_fs', '_sp', 'method', 'n_iter'], errors='ignore')
                               [['feature_set', 'split', 'n_features',
                                 'best_cv_RMSE', 'test_RMSE', 'test_MAE', 'test_R2']])

        canonical_path = RESULTS_DIR / 'mlp_results_final.csv'
        canonical.to_csv(canonical_path, index=False)
        print(f'Canonical CSV ({len(canonical)} rows) -> {canonical_path}')
        print(canonical.to_string(index=False))
