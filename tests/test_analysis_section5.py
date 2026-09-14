"""Fixed definitions and paired 32-state sample bootstrap (no GPU/model)."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from complete_analysis_section5 import eight_counts,counts_rates,interval,defined_mean
from hns_energy_matched import fro_match
from analyze_hns_energy_matched import transition_metrics


def test_all_eight_counts_including_zeros():
    bits=np.array([[(i>>2)&1,(i>>1)&1,i&1] for i in range(8)])
    counts=eight_counts(*bits.T)
    assert len(counts)==8 and all(v==1 for v in counts.values())
    zero=eight_counts([1],[1],[1])
    assert zero['n_111']==1 and sum(zero.values())==1


def test_joint_rates_match_existing_recovery_definitions():
    rng=np.random.default_rng(20260914); bits=rng.integers(0,2,size=(111,5))
    counts=np.bincount(bits@np.array([16,8,4,2,1]),minlength=32)
    result=counts_rates(counts)
    for i in range(3):
        existing=transition_metrics(bits[:,0],bits[:,1],bits[:,i+2])
        assert result[i]==pytest.approx([existing[k] for k in
            ('recovery_rate','retention_rate','new_success_rate','new_damage_rate')])


def test_paired_bootstrap_identical_edits_have_zero_difference():
    bits=np.array([[1,0,1,1,1],[0,1,0,0,0],[0,0,1,1,1],[1,1,0,0,0]])
    counts=np.bincount(bits@np.array([16,8,4,2,1]),minlength=32)
    draws=np.random.default_rng(3).multinomial(4,counts/4,size=100)
    rates=counts_rates(draws)
    assert np.allclose(rates[:,0],rates[:,1],equal_nan=True)
    assert interval(rates[:,0,0]-rates[:,1,0])[:2]==(0.,0.)
    undefined=np.zeros(32); undefined[31]=1
    assert np.isnan(counts_rates(undefined)[:,0:2]).all()
    assert interval([float('nan')])==(None,None,0)


def test_scalar_f_matches_observed_hns_not_ideal_exactflat():
    sigma=np.array([9.,3.,1.]); hns=np.array([5.,4.,4.])
    target,gain=fro_match(sigma,hns)
    assert np.linalg.norm(target)==pytest.approx(np.linalg.norm(hns))
    assert target/sigma==pytest.approx(np.full(3,gain))
    exactflat=np.full(3,sigma.sum()/3)
    assert np.linalg.norm(target)!=pytest.approx(np.linalg.norm(exactflat))
    assert target.sum()!=pytest.approx(sigma.sum())


def test_defined_macro_keeps_zero_denominators_undefined():
    result=defined_mean([[np.nan,1.],[np.nan,0.]])
    assert np.isnan(result[0]) and result[1]==.5


def test_performance_pairs_keep_checkpoint_pairing_and_fixed_groups(tmp_path,monkeypatch):
    import complete_analysis_section5 as analysis
    monkeypatch.setattr(analysis,'DEST',tmp_path)
    monkeypatch.setattr(analysis,'NBOOT',100)
    rows=[]
    for base in analysis.hns.BASES:
        for task in analysis.hns.TASKS[:3]:
            for seed in (42,43,44):
                for i,method in enumerate(analysis.METHODS):
                    rows.append(dict(checkpoint=f'{base}/{task}/seed{seed}',base=base,task=task,
                        seed=seed,method=method,target=seed+i,off_score=seed-i,forgetting_gap=i))
    analysis.performance_pairs(rows,'paired_ci.tsv')
    paired=analysis.load_tsv(tmp_path/'paired_ci.tsv')
    differences=analysis.load_tsv(tmp_path/'paired_checkpoint_differences.tsv')
    assert len(paired)==12*3*3 and len(differences)==18*3
    for row in paired:
        expected=3 if row['group'].startswith('base_source/') else (
            9 if row['group'].startswith('base/') else 6 if row['group'].startswith('source/') else 18)
        assert row['n_checkpoints']==expected
        assert row['ci_low']==pytest.approx(row['mean_difference'])
        assert row['ci_high']==pytest.approx(row['mean_difference'])
