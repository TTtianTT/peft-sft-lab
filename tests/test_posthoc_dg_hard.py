import math
import torch
from finetune.spectral_edit.dg_hard import lambda_star, mp_median, dg_hard_spectrum


def test_paper_coefficients():
    assert abs(lambda_star(1) - 4 / math.sqrt(3)) < 1e-12
    assert abs(lambda_star(1) / math.sqrt(mp_median(1)) - 2.858) < 0.001
    # Table IV, numerical coefficients rounded in the original supplement.
    for beta, omega in [(0.1, 1.6089), (0.25, 1.8371), (0.5, 2.1711)]:
        assert abs(lambda_star(beta) / math.sqrt(mp_median(beta)) - omega) < 0.0005


def test_full_lowrank_spectrum_is_identity():
    s = torch.tensor([7., 2., .5])
    t, stats = dg_hard_spectrum(s, 20, 40)
    assert torch.equal(t, s)
    assert stats['full_spectrum_median'] == stats['threshold'] == 0
    assert stats['retained_rank'] == 3 and stats['implicit_zero_count'] == 17
    assert stats['unchanged'] and stats['identity_due_to_zero_median']
    _, transposed = dg_hard_spectrum(s, 40, 20)
    assert transposed['omega'] == stats['omega']


def test_even_full_median_and_actual_thresholding():
    s = torch.tensor([10., 2., 1., 0.])
    t, stats = dg_hard_spectrum(s, 4, 4)
    assert stats['full_spectrum_median'] == 1.5
    assert torch.equal(t, torch.tensor([10., 0., 0., 0.]))
    assert not stats['unchanged'] and stats['retained_rank'] == 1


def test_strict_greater_than_threshold():
    omega = lambda_star(1) / math.sqrt(mp_median(1))
    t, stats = dg_hard_spectrum(torch.tensor([omega, 1., 1.], dtype=torch.float64), 3, 3)
    assert stats['threshold'] == omega
    assert torch.equal(t, torch.zeros(3, dtype=torch.float64))
