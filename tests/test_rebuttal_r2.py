import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import rebuttal_iclr27_r2 as q
from finetune.spectral_edit.svd import lowrank_svd_from_ba


class RoundTwoProtocolTest(unittest.TestCase):
    def test_reconstruction_and_factorization_preserve_dense_update(self):
        generator=torch.Generator().manual_seed(51)
        b=torch.randn(19,4,generator=generator);a=torch.randn(4,13,generator=generator)
        u,s,vh,_=lowrank_svd_from_ba(b,a)
        recon=q.spectrum(s,'recon_0')
        torch.testing.assert_close(recon,s,rtol=0,atol=0)
        for method in ('recon_0','recon_0_unbal'):
            bn,an=q.writeback(u,vh,recon,method)
            torch.testing.assert_close(bn@an,b@a,rtol=2e-5,atol=1e-5)
        balanced=q.writeback(u,vh,q.spectrum(s,'hns_f4_s1'),'hns_f4_s1')
        unbalanced=q.writeback(u,vh,q.spectrum(s,'hns_unbal'),'hns_unbal')
        torch.testing.assert_close(balanced[0]@balanced[1],unbalanced[0]@unbalanced[1],rtol=2e-5,atol=1e-5)

    def test_all_doses_preserve_nuclear_budget_and_half_is_interpolation(self):
        s=torch.tensor([8.,2.,.5,.0001])
        for method in q.CONFIGS:
            torch.testing.assert_close(q.spectrum(s,method).sum(),s.sum())
        torch.testing.assert_close(q.spectrum(s,'hns_f4_s1_l05'),(s+q.spectrum(s,'hns_f4_s1'))/2)

    def test_checkpoint_identity_and_familywise_fg_clipping(self):
        cps=[dict(condition=c,base='Qwen3-8B',task='metamath',seed=42,rank=rank) for c,rank in [('half',16),('full',64)]]
        cells=[]
        for cp in cps:
            for method,values in [('base',[50,50,50,50]),('original_lora',[40,70,60,30])]:
                cells += [dict(condition=cp['condition'],method=method,eval_task=t,score=v) for t,v in zip(q.r.TASKS,values)]
        result=q.points(cells,cps)
        self.assertEqual(len(result),4)
        for row in result:
            if row['method']=='original_lora':
                self.assertEqual(row['target'],70)
                self.assertAlmostEqual(row['off_score'],130/3)
                self.assertAlmostEqual(row['forgetting_gap'],10)
        self.assertEqual({x['rank'] for x in result},{16,64})


if __name__=='__main__':unittest.main()
