from warnings import warn

import joblib as jl
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


class Bootstrapper():
    """
    Object that contains all the bootstrap parameters.

    Parameters
    ----------
    n_boot : int, optional
        Number of bootstraps to do for each analysis. Must be > 1.
    n_data : int, optional
        Number of data elements to accept.
    seed : int or 1-d array_like, optional
        Seed for random number generation.
    strict : bool, optional
        Only analyze datasets with the same number of elements.
    pool : int, optional
        Number of CPUs to use.
    progress : bool, optional
        Show progress for bootstrapping and jack-knifing.
    """

    def __init__(
            self,
            n_boot=1000,
            n_data=None,
            pool=1,
            progress=False,
            seed=None,
            strict=False,
            ):
        self.strict = bool(strict)
        self.n_boot = n_boot
        self.n_data = n_data
        self.pool = pool
        self.progress = bool(progress)
        self.seed = seed

    def __repr__(self):
        parts = ", ".join([
            f"n_boot={self.n_boot}",
            f"n_data={self.n_data}",
            f"pool={self.pool}",
            f"progress={self.progress}",
            f"seed={self.seed}",
            f"strict={self.strict}",
            ])
        return f"{self.__class__.__name__}({parts})"

    @property
    def n_boot(self):
        return self._n_boot

    @n_boot.setter
    def n_boot(self, value):
        if value < 2:
            raise ValueError("N bootstraps was less than 2.")
        self._n_boot = value

    @property
    def n_data(self):
        try:
            return self._n_data
        except AttributeError:
            ...

    @n_data.setter
    def n_data(self, value):
        if self.strict and self.n_data and not (value == self.n_data):
            raise ValueError(
                f"Expected {self.n_data} but got {value}!"
                )
        self._n_data = value

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self, value):
        if value < 1:
            raise ValueError("Pool was less than 1!")
        elif value > jl.cpu_count():
            raise ValueError("Pool was larger than the number of CPUs!")
        self._pool = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        if self.strict and value is None:
            # TODO: Make sure this check on value is valid #
            raise ValueError("Bootstrapper is strict without a seed!")
        self._seed = value

    @property
    def boot_seeds(self):
        # TODO: Make seeds a generator property? <19-02-21> #
        rng = np.random.RandomState(self.seed)
        return rng.randint(np.iinfo(np.int32).max, size=self.n_boot)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not np.ndim(data) in [1, 2]:
            raise ValueError(f"Data were {np.ndim(data)}-dimensional!")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self._data = data
            self._sampler = self.sample_data_as_pandas
        else:
            self._data = np.asarray(data)
            self._sampler = self.sample_data_as_array

    @data.deleter
    def data(self):
        del self._data

    @property
    def statfunc(self):
        return self._statfunc

    @statfunc.setter
    def statfunc(self, func):
        # TODO Wrap this to be at least 2d?
        if not callable(func):
            raise ValueError("Function is not callable!")
        stat = func(self._data)
        if np.ndim(stat) > 1:
            raise ValueError(
                f"Function output was {np.ndim(stat)}-dimensional!"
                )
        self._statfunc = func

    @statfunc.deleter
    def statfunc(self):
        del self._statfunc

    @property
    def bias(self):
        stat = np.array(self.statfunc(self.data))
        return stats.norm.ppf(np.mean(self.bootstats < stat, axis=0))

    @property
    def acceleration(self):
        deviations = np.mean(self.jackstats, axis=0) - self.jackstats
        numer = np.sum(deviations, axis=0) ** 3
        denom = 6 * np.sum(deviations ** 2, axis=0) ** 1.5
        return numer / denom

    def conf_int(self, statfunc, data, confidence=0.95):
        self.data = data
        self.statfunc = statfunc
        self.n_data = len(data)
        self.bootstrap()
        self.jackknife()
        stat = self.statfunc(self.data)
        n_stats = len(np.atleast_1d(stat))
        ci = np.column_stack(
            [self.bca_ci(confidence, x) for x in range(n_stats)]
            )
        if isinstance(stat, pd.Series):
            ci = pd.DataFrame(ci, columns=stat.index)
        self.conf_int_cleanup()
        return ci

    def bootstrap(self):
        results = jl.Parallel(n_jobs=self.pool)(
            jl.delayed(self.bootstrap_instance)(seed)
            for seed in tqdm(
                self.boot_seeds,
                desc="Bootstrapping",
                disable=(not self.progress)
                )
            )
        self.bootstats = np.vstack(results)

    def bootstrap_instance(self, seed):
        rng = np.random.RandomState(seed)
        boot_idx = rng.randint(len(self.data), size=len(self.data))
        boot_data = self.sample_data(boot_idx)
        return self.statfunc(boot_data)

    def jackknife(self):
        lo_idxs = range(len(self.data))
        results = jl.Parallel(n_jobs=self.pool)(
            jl.delayed(self.jackknife_instance)(lo_idx)
            for lo_idx in tqdm(
                lo_idxs, desc="Jackkniffing", disable=(not self.progress)
                )
            )
        self.jackstats = np.vstack(results)

    def jackknife_instance(self, lo_idx):
        jack_idx = np.r_[0:lo_idx, (lo_idx + 1):len(self.data)]
        jack_data = self.sample_data(jack_idx)
        return self.statfunc(jack_data)

    def bca_ci(self, confidence, stat_idx):
        jackstats = self.jackstats[:, stat_idx]
        bootstats = self.bootstats[:, stat_idx]
        if (jackstats.var() == 0):
            warn("A stat had no variance", category=RuntimeWarning)
            return np.array([bootstats[0]] * 2)
        quantiles = self.bca_quantiles(confidence)[:, stat_idx]
        return np.quantile(bootstats, quantiles)

    def bca_quantiles(self, confidence):
        alpha = 1 - confidence
        quantiles_orig = np.array([alpha / 2, 1 - alpha / 2])
        z_orig = stats.norm.ppf(quantiles_orig)
        numer = z_orig[:, None] + self.bias
        denom = 1 - self.acceleration * numer
        z_new = self.bias + numer / denom
        return stats.norm.cdf(z_new)

    def conf_int_cleanup(self):
        del self.data
        del self.statfunc
        del self.bootstats
        del self.jackstats

    def sample_data(self, sample_idx):
        return self._sampler(sample_idx)

    def sample_data_as_array(self, sample_idx):
        return self.data[sample_idx]

    def sample_data_as_pandas(self, sample_idx):
        return self.data.iloc[sample_idx]
