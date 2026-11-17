import numpy as np
import pytest
import torch

from mvmd_torch import mvmd

FS = 250                       # sampling rate (Hz)
FREQS = [5, 20, 40]            # frequencies of the three components (Hz)
PARAMS = dict(alpha=2000, tau=0, K=3, DC=False, init=1, tol=1e-7, max_N=500)


def make_signal(n_channels=4, n_samples=1000, seed=0):
    """Three tones shared by all channels; the 40 Hz one grows with the channel index."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    x = np.stack([np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
                  + 0.6 * (c + 1) / n_channels * np.sin(2 * np.pi * 40 * t)
                  + 0.05 * rng.standard_normal(n_samples) for c in range(n_channels)])
    return x


def test_output_shapes():
    x = make_signal(n_channels=4, n_samples=1000)
    u, u_hat, omega = mvmd(torch.tensor(x, dtype=torch.float32), **PARAMS)
    assert u.shape == (3, 1000, 4)
    assert u_hat.shape == (1000, 3, 4)
    assert omega.shape == (1, 3)


def test_recovers_the_frequencies():
    x = make_signal()
    _, _, omega = mvmd(torch.tensor(x, dtype=torch.float32), **PARAMS)
    found = np.sort(omega.real.cpu().numpy().ravel() * FS)
    assert np.allclose(found, FREQS, atol=0.5)


@pytest.mark.parametrize("n_channels", [1, 3, 4, 7])
def test_modes_add_up_to_the_signal_of_the_same_channel(n_channels):
    """Regression test: the modes of channel c must rebuild channel c (not another channel)."""
    x = make_signal(n_channels=n_channels)
    u, _, _ = mvmd(torch.tensor(x, dtype=torch.float32), **PARAMS)
    rebuilt = u.sum(dim=0).cpu().numpy()            # (T, C)
    for c in range(n_channels):
        assert np.abs(rebuilt[:, c] - x[c]).mean() < 0.06   # the noise level is 0.05


def test_mode_amplitude_follows_the_channel():
    x = make_signal(n_channels=4)
    u, _, omega = mvmd(torch.tensor(x, dtype=torch.float32), **PARAMS)
    k40 = int(np.argmin(np.abs(omega.real.cpu().numpy().ravel() * FS - 40)))
    rms = u[k40].pow(2).mean(dim=0).sqrt().cpu().numpy()   # one value per channel
    assert np.all(np.diff(rms) > 0)                          # grows with the channel index


@pytest.mark.parametrize("init", [0, 1, 2])
def test_every_initialisation_runs(init):
    x = make_signal(n_channels=2)
    u, _, _ = mvmd(torch.tensor(x, dtype=torch.float32), **{**PARAMS, "init": init, "max_N": 20})
    assert torch.isfinite(u).all()
