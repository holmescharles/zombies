from hypothesis import assume, given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from zombies import Bootstrapper


@st.composite
def data_sets(draw):
    return draw(st.lists(
        st.floats(min_value=-1e20, max_value=1e20),
        unique=True,
        min_size=2,
        ))


@st.composite
def seeds(draw):
    return draw(st.integers(min_value=0, max_value=(2 ** 32 - 1)))


class TestBootstrapper:

    @settings(deadline=400)
    @given(data=data_sets(), seed=seeds())
    def test_conf_int_fuzz(self, data, seed):
        boot = Bootstrapper(strict=True, seed=seed)
        boot.conf_int(np.mean, data)

    def test_error_when_strict_and_no_seed(self):
        with pytest.raises(ValueError):
            Bootstrapper(strict=True, seed=None)

    @settings(deadline=400)
    @given(data1=data_sets(), data2=data_sets(), seed=seeds())
    def test_error_when_strict_and_different_n_data(self, data1, data2, seed):
        assume(len(data1) != len(data2))
        boot = Bootstrapper(strict=True, seed=seed)
        boot.conf_int(np.mean, data1)
        with pytest.raises(ValueError):
            boot.conf_int(np.mean, data2)

    @settings(deadline=400)
    @given(data=data_sets(), seed=seeds())
    def test_reproducibility_with_same_bootstrapper(self, data, seed):
        boot = Bootstrapper(strict=True, seed=seed)
        ci1 = boot.conf_int(np.mean, data)
        ci2 = boot.conf_int(np.mean, data)
        assert np.array_equal(ci1, ci2)

    @settings(deadline=400)
    @given(data=data_sets(), seed=seeds())
    def test_reproducibility_with_different_bootstrappers(self, data, seed):
        ci1 = Bootstrapper(strict=True, seed=seed).conf_int(np.mean, data)
        ci2 = Bootstrapper(strict=True, seed=seed).conf_int(np.mean, data)
        assert np.array_equal(ci1, ci2)
