#!/usr/bin/env python3
"""CPU-only audit of balanced/unbalanced represented updates, after tier A."""
from pathlib import Path
import os
import numpy as np
import rebuttal_iclr27 as r
import rebuttal_iclr27_r2 as q
from rebuttal_iclr27_e2_e3 import operations


def main():
    r.require_slurm();assert os.getenv('CUDA_VISIBLE_DEVICES','') in ('','NoDevFiles','-1')
    q.require_complete('R2-O2');p=q.folder('R2-O2');lock=q.check('R2-O2')
    torch,collect,error,load,*_=operations();assert not torch.cuda.is_initialized()
    old=r.source_rows();records=[]
    original_artifacts={x['path']:x for x in r.read(r.OUT/'E1/artifact_integrity_audit.json')['artifacts']}
    for cp in lock['checkpoints']:
        source=next(x for x in old if x['source']==cp['source'])
        recon_manifest=r.read(r.OUT/'E1'/f'{cp["base"]}_variant_manifest.json')
        recon=next(x['path'] for x in recon_manifest['variants'] if x['train_task']==cp['task'] and int(x['seed'])==cp['seed'] and x['method']=='recon_0')
        for unbalanced,balanced in [('recon_0_unbal',recon),('hns_unbal',source['hns_path'])]:
            path=p/'runtime/adapters'/cp['condition']/unbalanced
            marker=r.read(path/'r2_integrity.json')
            assert r.sha(path/'adapter_model.safetensors')==marker['artifact']['weights_sha256']
            expected=(original_artifacts[balanced]['weights_sha256'] if unbalanced.startswith('recon') else source['hns_weights_sha256'])
            assert r.sha(Path(balanced)/'adapter_model.safetensors')==expected
            left,_=load(str(path));right,_=load(balanced);a,b=collect(left),collect(right);assert a.keys()==b.keys()
            for name in sorted(a):
                delta=error(a[name]['B'][1].double(),a[name]['A'][1].double(),b[name]['B'][1].double(),b[name]['A'][1].double())
                assert np.isfinite(delta) and delta<5e-5,(cp['condition'],unbalanced,name,delta)
                records.append(dict(condition=cp['condition'],base=cp['base'],task=cp['task'],seed=cp['seed'],
                    unbalanced=unbalanced,balanced_path=balanced,module=name,relative_update_difference=delta,
                    unbalanced_weights_sha256=marker['artifact']['weights_sha256'],balanced_weights_sha256=expected))
            del left,right,a,b
        print('[Factor audit]',cp['condition'],flush=True)
    assert not torch.cuda.is_initialized()
    r.tsv(p/'factorization_update_audit.tsv',records)
    r.write(p/'factorization_update_audit.json',dict(status='pass',conditions=len(lock['checkpoints']),
        compared_module_pairs=len(records),max_relative_update_difference=max(x['relative_update_difference'] for x in records),
        fixed_tolerance=5e-5,device='cpu; CUDA never initialized',job_id=os.getenv('SLURM_JOB_ID'),gpus=0,
        audit_sha256=r.sha(p/'factorization_update_audit.tsv')))
    print('Balanced/unbalanced update audit passed.',flush=True)


if __name__=='__main__':main()
