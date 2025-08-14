#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import time
import gc
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse, mae


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def load_and_preprocess_data(file_path: str,
                             min_ratings: int = 15,
                             use_time_split: bool = False,
                             time_split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users, items, ratings, times = [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            users.append(r['reviewerID'])
            items.append(r['asin'])
            ratings.append(float(r['overall']))
            times.append(int(r.get('unixReviewTime,', r.get('unixReviewTime', 0))))  # robust hơn

    df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings, 'ts': times})
    df = df.groupby(['user_id', 'item_id'], as_index=False).agg({'rating': 'mean', 'ts': 'max'})

    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= min_ratings].index)]
    item_counts = df['item_id'].value_counts()
    df = df[df['item_id'].isin(item_counts[item_counts >= min_ratings].index)]

    print(f"[INFO] Users: {df['user_id'].nunique():,} | Items: {df['item_id'].nunique():,} | Ratings: {len(df):,}")

    if use_time_split:
        df = df.sort_values('ts')
        cutoff_idx = int(len(df) * time_split_ratio)
        df_train = df.iloc[:cutoff_idx].copy()
        df_test = df.iloc[cutoff_idx:].copy()
        print(f"[INFO] Time-split -> train={len(df_train):,}, test={len(df_test):,}")
        return df_train[['user_id', 'item_id', 'rating']], df_test[['user_id', 'item_id', 'rating']]
    else:
        return df[['user_id', 'item_id', 'rating']], None


# -----------------------------
# Model / Grid
# -----------------------------
def grid_search_item(data: Dataset, fast: bool = True) -> Dict[str, Any]:
    if fast:
        param_grid = {
            'k': [30],
            'min_k': [5],
            'sim_options': {'name': ['msd'], 'user_based': [False]}
        }
    else:
        param_grid = {
            'k': [20, 30, 50],
            'min_k': [3, 5],
            'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline'], 'user_based': [False]}
        }
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=2, n_jobs=1, joblib_verbose=0)
    gs.fit(data)
    best = gs.best_params['rmse']
    print(f"[INFO] Best params (RMSE): {best}")
    return best


# -----------------------------
# Plotting
# -----------------------------
def plot_rmse_mae(rmse_val: float, mae_val: float, algo_name: str,
                  train_time: float, infer_time: float, best_params: Dict[str, Any],
                  fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    labels = ['RMSE', 'MAE']
    vals = [rmse_val, mae_val]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)

    plt.title(f"{algo_name} — Rating Prediction")
    plt.ylabel("Giá trị lỗi")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    t = f"Train: {train_time:.2f}s | Infer: {infer_time:.2f}s\nBest: {best_params}"
    plt.gcf().text(0.5, -0.05, t, ha='center', va='top', fontsize=9)

    out_png = os.path.join(fig_dir, f"{algo_name.lower().replace(' ', '_')}_rmse_mae.png")
    out_svg = os.path.join(fig_dir, f"{algo_name.lower().replace(' ', '_')}_rmse_mae.svg")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {out_png} & {out_svg}")


# -----------------------------
# Train/Eval
# -----------------------------
def run_item_based(df_train: pd.DataFrame, df_test: pd.DataFrame, random_state: int,
                   do_grid: bool, fig_dir: str):
    reader = Reader(rating_scale=(1, 5))

    if df_test is None:
        data = Dataset.load_from_df(df_train[['user_id', 'item_id', 'rating']], reader)
        best_params = grid_search_item(data, fast=True) if do_grid else \
            {'k': 30, 'min_k': 5, 'sim_options': {'name': 'msd', 'user_based': False}}

        trainset, testset = train_test_split(data, test_size=0.2, random_state=random_state)

        algo = KNNWithMeans(**best_params, verbose=False)
        t0 = time.time(); algo.fit(trainset); train_time = time.time() - t0
        t1 = time.time(); preds = algo.test(testset); infer_time = time.time() - t1

    else:
        data_train = Dataset.load_from_df(df_train[['user_id', 'item_id', 'rating']], reader)
        trainset = data_train.build_full_trainset()
        best_params = grid_search_item(data_train, fast=True) if do_grid else \
            {'k': 30, 'min_k': 5, 'sim_options': {'name': 'msd', 'user_based': False}}

        algo = KNNWithMeans(**best_params, verbose=False)
        t0 = time.time(); algo.fit(trainset); train_time = time.time() - t0

        testset = list(df_test[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
        t1 = time.time(); preds = algo.test(testset); infer_time = time.time() - t1

    r = rmse(preds, verbose=False); m = mae(preds, verbose=False)
    print(f"[TIME] Train: {train_time:.2f}s")
    print(f"[TIME] Infer: {infer_time:.2f}s")
    print(f"[RESULT] RMSE={r:.4f} | MAE={m:.4f}")

    plot_rmse_mae(r, m, "Item-Based CF", train_time, infer_time, best_params, fig_dir)
    del preds; gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Đường dẫn file JSONL reviews')
    ap.add_argument('--min_ratings', type=int, default=15)
    ap.add_argument('--time_split', action='store_true')
    ap.add_argument('--time_ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_grid', action='store_true', help='Bỏ GridSearch (nhanh hơn)')
    ap.add_argument('--fig_dir', type=str, default='figs_item_based')
    args = ap.parse_args()

    df_train, df_test = load_and_preprocess_data(
        args.file, min_ratings=args.min_ratings,
        use_time_split=args.time_split, time_split_ratio=args.time_ratio
    )
    run_item_based(df_train, df_test, random_state=args.seed,
                   do_grid=not args.no_grid, fig_dir=args.fig_dir)


if __name__ == '__main__':
    main()
