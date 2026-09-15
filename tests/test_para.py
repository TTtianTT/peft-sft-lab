import torch

from finetune.spectral_edit.para import epsilon_para_masks


def test_epsilon_para_uses_one_global_energy_threshold_and_keeps_ties():
    spectra = {
        "a": torch.tensor([4.0, 2.0, 1.0]),
        "b": torch.tensor([3.0, 2.0, 0.5]),
    }
    masks, stats = epsilon_para_masks(spectra, epsilon=0.90)
    # Total energy 34.25; 4,3,2,2 retain 33 (> 90%). The shared threshold is 2.
    assert stats.threshold == 2.0
    assert masks["a"].tolist() == [True, True, False]
    assert masks["b"].tolist() == [True, True, False]
    assert stats.retained_components == 4
    assert abs(stats.retained_energy_ratio - 33.0 / 34.25) < 1e-12


def test_epsilon_para_accounts_for_module_scaling():
    spectra = {"a": torch.tensor([2.0, 1.0]), "b": torch.tensor([3.0, 1.0])}
    masks, stats = epsilon_para_masks(
        spectra, epsilon=0.80, scaling={"a": 2.0, "b": 0.5}
    )
    assert stats.threshold == 2.0
    assert masks["a"].tolist() == [True, True]
    assert masks["b"].tolist() == [False, False]


def test_epsilon_para_rejects_invalid_budget():
    try:
        epsilon_para_masks({"a": torch.ones(2)}, epsilon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid epsilon was accepted")
