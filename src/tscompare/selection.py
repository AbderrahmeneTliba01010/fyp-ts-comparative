"""Stratified selection of a 'core' task subset so the compute budget stays bounded but domains/terms stay covered."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_core_tasks(task_df: pd.DataFrame, n=20, seed=0, strata=("domain", "term")) -> pd.DataFrame:
    """Round-robin over strata (domain x term), at most one task per dataset until every dataset is used.
    task_df columns: name, dataset, freq, term, domain, prediction_length (from TimeBenchSource.list_tasks)."""
    rng = np.random.default_rng(seed)
    df = task_df.copy()
    for c in strata:
        if c not in df or df[c].isna().all():
            df[c] = "NA"
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    chosen, used = [], set()
    groups = {k: g.index.tolist() for k, g in df.groupby(list(strata))}
    keys = sorted(groups, key=str)
    while len(chosen) < min(n, len(df)) and any(groups.values()):
        for k in keys:
            while groups[k]:
                i = groups[k].pop(0)
                if df.loc[i, "dataset"] not in used or len(used) >= df.dataset.nunique():
                    chosen.append(i); used.add(df.loc[i, "dataset"]); break
            if len(chosen) >= n:
                break
    return df.loc[chosen].sort_values(["domain", "dataset", "term"]).reset_index(drop=True)
