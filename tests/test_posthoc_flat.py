import torch
from finetune.spectral_edit.flat import flat_spectrum
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def test_flat_budgets_and_balanced_reconstruction():
    torch.manual_seed(7)
    b, a = torch.randn(37, 16), torch.randn(16, 29)
    u, s, vh, _ = lowrank_svd_from_ba(b, a)
    dense_s = torch.linalg.svdvals(b @ a)[:16]
    torch.testing.assert_close(s, dense_s)
    for method in ('flat_fro', 'flat_nuclear'):
        t = flat_spectrum(s, method)
        bn, an = rebuild_ba_from_uv_sigma(u, vh, t)
        saved_s = torch.linalg.svdvals(bn @ an)[:16]
        torch.testing.assert_close(saved_s, t)
        if method == 'flat_fro':
            torch.testing.assert_close(t.norm(), s.norm())
        else:
            torch.testing.assert_close(t.sum(), s.sum())
        torch.testing.assert_close(bn.T @ bn, an @ an.T, atol=1e-4, rtol=1e-5)


def test_zero_and_rank_deficient():
    for s in (torch.zeros(16), torch.tensor([4., 3., 0., 0.])):
        for method in ('flat_fro', 'flat_nuclear'):
            t = flat_spectrum(s, method)
            assert t.isfinite().all()
            assert torch.equal(t, t[0].expand_as(t))
            torch.testing.assert_close(t.norm() if method == 'flat_fro' else t.sum(), s.norm() if method == 'flat_fro' else s.sum())
