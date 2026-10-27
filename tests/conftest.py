"""Tiny cumulative graphs for the unit tests. No HI-Small files."""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib
matplotlib.use('Agg', force=True)

import pytest

from tests.support import make_bundle


@pytest.fixture
def bundle():
    return make_bundle()


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')
