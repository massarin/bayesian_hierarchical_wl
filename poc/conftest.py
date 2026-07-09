import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update("jax_enable_x64", True)

figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


@pytest.fixture
def panel(request):
    """Yields an axis; saves it under the test's own name, pass or fail."""
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.), constrained_layout=True)
    yield ax
    fig.savefig(os.path.join(figdir, request.node.name.replace('/', '_') + '.png'), dpi=110)
    plt.close(fig)
