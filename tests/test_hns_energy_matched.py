"""Algebra tests for the fixed controls (no model/data dependencies)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hns_energy_matched import energy_match, stats, archived_probe_manifest
from analyze_hns_energy_matched import transition_metrics, macro, regression, within_rho
import analyze_hns_energy_matched as analysis


def test_archived_probe_keeps_original_ids_without_merging_or_appending():
    variants=[dict(label=f'arm{i}',path=f'/adapter/{i}') for i in range(45)]
    current=dict(base='test',variants=variants)
    archived=dict(variant_order=[v['label'] for v in variants[:30]],
                  configuration=dict(max_num_seqs=4096))
    restored=archived_probe_manifest(current,archived)
    assert restored['variants']==variants[:30]
    assert len(current['variants'])==45
    assert [(i,v['label']) for i,v in enumerate(restored['variants'],1)]==[
        (i+1,f'arm{i}') for i in range(30)]


def test_archived_probe_rejects_changed_configuration_or_missing_arms():
    current=dict(variants=[dict(label='a')])
    with pytest.raises(ValueError):
        archived_probe_manifest(current,dict(variant_order=['a'],configuration=dict(max_num_seqs=2048)))
    with pytest.raises(ValueError):
        archived_probe_manifest(current,dict(variant_order=['missing'],configuration=dict(max_num_seqs=4096)))


def test_modulewise_matching_and_shape():
    sigma=np.array([9.,3.,1.]); h=np.array([4.,4.,3.]); q=np.array([10.,1.,.01])
    scalar,gain=energy_match(sigma,h,q)
    flat,_=energy_match(np.ones(3),h,q)
    assert np.allclose(scalar/sigma,gain)
    assert np.allclose(flat,flat[0])
    assert np.allclose(np.dot(scalar**2,q),np.dot(h**2,q))
    assert np.allclose(np.dot(flat**2,q),np.dot(h**2,q))
    fro_gain=np.linalg.norm(h)/np.linalg.norm(sigma)
    assert not np.isclose(fro_gain,gain)
    assert not np.isclose(scalar.sum(),sigma.sum())


def test_no_global_or_floored_q_matching():
    sigma=np.array([5.,1.]); h=np.array([2.,2.])
    q1=np.array([1.,.0001]); q2=np.array([.0001,1.])
    s1,g1=energy_match(sigma,h,q1); s2,g2=energy_match(sigma,h,q2)
    assert not np.isclose(g1,g2)
    assert np.isclose(np.dot(s1*s1,q1),np.dot(h*h,q1))
    assert np.isclose(np.dot(s2*s2,q2),np.dot(h*h,q2))


def test_pr_scale_invariance_and_full_moment():
    s=np.array([4.,2.,1.]); m=np.array([[2.,.4,0],[.4,1.,.1],[0,.1,.5]])
    q=np.diag(m)
    a=stats(s,q,m,2); b=stats(3*s,q,m,2)
    assert np.isclose(a['raw_fpr'],b['raw_fpr'])
    assert np.isclose(a['full_moment_pr'],b['full_moment_pr'])
    assert np.isclose(9*a['functional_energy'],b['functional_energy'])
    assert a['full_moment_pr'] <= a['raw_fpr']


def test_invalid_and_zero_energy():
    with pytest.raises(ValueError): energy_match([0,0],[1,1],[1,1])
    with pytest.raises(ValueError): energy_match([1,-1],[1,1],[1,1])
    with pytest.raises(ValueError): energy_match([1,1],[1,1],[float('nan'),1])
    s,_=energy_match([0,0],[0,0],[1,1]); assert np.array_equal(s,[0,0])


def test_transition_definitions_and_zero_sets():
    r=transition_metrics([1,1,0,0,1,0],[0,0,1,1,1,0],[1,0,1,0,0,1])
    assert r['recovery_rate']==.5 and r['retention_rate']==.5
    assert r['new_success']==1 and r['new_damage']==1
    assert r['recovery_set_size']==2 and r['retention_set_size']==2
    assert r['new_success_set_size']==1 and r['new_damage_set_size']==1
    z=transition_metrics([1],[1],[1])
    assert z['recovery_rate'] is None and z['retention_rate'] is None


def test_fixed_macro_not_sample_weighted():
    a=transition_metrics([1],[0],[1])
    b=transition_metrics([1]*99,[0]*99,[0]*99)
    r=macro([a,b],{})
    assert r['recovery_rate']==.5
    assert r['recovered']==1 and r['recovery_set_size']==100
    assert r['recovery_rate_defined']==2


def test_energy_matched_regression_has_zero_energy_rank():
    rows=[]
    for cp in range(4):
        for fpr in (1.,2.,3.):
            rows.append(dict(checkpoint=str(cp),method=str(fpr),energy=cp+1.,fpr=fpr,y=fpr*2+cp))
    energy,_,_,_=regression(rows,('energy',),'y')
    fpr,_,_,_=regression(rows,('fpr',),'y')
    combined,_,_,_=regression(rows,('energy','fpr'),'y')
    assert energy['rank']==0 and energy['r2']==0
    assert fpr['r2']==pytest.approx(1.) and fpr['loco_r2']==pytest.approx(1.)
    assert combined['rank']==1 and combined['r2']==pytest.approx(1.)
    assert within_rho([1,1,1],[1,2,3]) is None


def test_fixed_analysis_all_layers_on_synthetic_checkpoints(monkeypatch):
    monkeypatch.setattr(analysis,'BOOTSTRAPS',20)
    rows=[]
    for cp in range(18):
        for method,pr,e in [('original_lora',1.,1.),('hns_f4_s1',2.,.5),('scalar_e',1.,.5),('flat_e',3.,.5)]:
            row=dict(checkpoint=str(cp),method=method,raw_fpr=pr,full_moment_pr=.8*pr,
                log_energy_ratio=np.log(e))
            row.update({outcome:pr+.01*cp for outcome in analysis.OUTCOMES})
            rows.append(row)
    stats,pred,cor,design=analysis.fixed_analysis(rows)
    assert {r['cohort'] for r in stats}=={'A_all_versions','B_edited_only','C_energy_matched'}
    assert all(r['n_checkpoints']==18 for r in stats)
    assert pred and cor and design
    inc=[r for r in stats if r['kind']=='incremental_fpr_over_energy' and r['cohort']=='C_energy_matched']
    assert all(r['delta_loco_r2']==pytest.approx(1.) for r in inc)
    assert all(r['delta_loco_r2_ci_low']==pytest.approx(1.) for r in inc)
