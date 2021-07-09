from warnings import warn

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def confidence_interval(
        statfunc, data, *,
        n_boot=1000, seed=None, confidence=0.95, pool=1, silent=False,
        ):
    stat = statfunc(data)
    bootstats = bootstrap_statistics(
        statfunc, data, n_boot, seed, pool=pool, silent=silent
        )
    jackstats = jackknife_statistics(statfunc, data, pool=pool, silent=silent)
    cis = np.column_stack([
        bca_ci(confidence)(s, b, j)
        for s, b, j in zip(np.atleast_1d(stat), bootstats.T, jackstats.T)
        ])
    return same_type_as(stat)(cis)


def bca_ci(confidence):
    def _(stat, bootstats, jackstats):
        if np.var(jackstats) == 0:
            warn("A stat had no variance", category=RuntimeWarning)
            return np.array([bootstats[0]] * 2)
        quantiles = bca_quantiles(
            bias=bias(stat, bootstats),
            acceleration=acceleration(jackstats),
            confidence=confidence,
            )
        return np.quantile(bootstats, quantiles)
    return _


def bootstrap_statistics(statfunc, data, n_boot, seed, pool=1, silent=False):
    seeds_ = seeds(n_boot, seed)
    stats = joblib.Parallel(n_jobs=pool)(
        joblib.delayed(bootstrap_statistic)(statfunc, data, seed)
        for seed in tqdm(seeds_, desc="Bootstrapping", disable=silent)
        )
    return np.vstack(stats)


def bootstrap_statistic(statfunc, data, seed):
    rng = np.random.RandomState(seed)
    bootidx = rng.randint(len(data), size=len(data))
    boot_data = indexable(data)[bootidx]
    return statfunc(boot_data)


def jackknife_statistics(statfunc, data, pool=1, silent=False):
    lo_idxs = range(len(data))
    stats = joblib.Parallel(n_jobs=pool)(
        joblib.delayed(jackknife_statistic)(statfunc, data, lo_idx)
        for lo_idx in tqdm(lo_idxs, desc="Jackknifing", disable=silent)
        )
    return np.vstack(stats)


def jackknife_statistic(statfunc, data, lo_idx):
    jack_idx = np.r_[0:lo_idx, (lo_idx + 1):len(data)]
    jack_data = indexable(data)[jack_idx]
    return statfunc(jack_data)


def bca_quantiles(bias, acceleration, confidence):
    alpha = 1 - confidence
    quantiles_orig = np.array([alpha / 2, 1 - alpha])
    z_orig = stats.norm.ppf(quantiles_orig)
    numer = z_orig[:, None] + bias
    denom = 1 - acceleration * numer
    z_new = bias + (numer / denom)
    return stats.norm.cdf(z_new)


def bias(stat, bootstats):
    return stats.norm.ppf(np.mean(bootstats < stat, axis=0))


def acceleration(jackstats):
    deviations = np.mean(jackstats) - jackstats
    numer = np.sum(deviations) ** 3
    denom = 6 * np.sum(deviations ** 2) ** 1.5
    return numer / denom


def seeds(n_boot, seed):
    rng = np.random.RandomState(seed)
    return rng.randint(np.iinfo(np.int32).max, size=n_boot)


def indexable(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc
    return data


def same_type_as(template):
    def _(x):
        if isinstance(template, pd.Series):
            return pd.DataFrame(x, columns=template.index)
        return x
    return _
