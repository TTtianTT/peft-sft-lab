#!/usr/bin/env python3
"""Collect all source-basis activation moments on the original frozen-base protocol."""
import argparse
import csv
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure_lora_modification import collect_factors, canonical_name, render
from finetune.spectral_edit.io import get_scaling_for_module
from finetune.spectral_edit.svd import lowrank_svd_from_ba

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/functional_activation_three_seed_20260914'
PRIOR=ROOT/'reports/promising_spectral_metrics_three_seed_20260914'
SOURCE=ROOT/'reports/posthoc_flat_dghard_three_seed_20260913/source_manifest.json'

def read(p):
    return json.loads(Path(p).read_text())

def write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    cps=read(SOURCE)['checkpoints']; assert len(cps)==18
    old=read(PRIOR/'functional_audit.json')['source_manifests']
    cases=[]
    for base,task in dict.fromkeys((c['base'],c['task']) for c in cps):
        ref=next(r for r in old if r['base']==base and r['train_task']==task)
        a=read(ref['activation_manifest'])
        group=sorted([c for c in cps if c['base']==base and c['task']==task],key=lambda c:c['seed'])
        assert [c['seed'] for c in group]==[42,43,44]
        assert a['lora_path']==group[0]['source'] and a['samples']==256 and a['seed']==42 and a['max_seq_len']==512
        for c in group:
            assert sha(Path(c['source'])/'adapter_model.safetensors')==c['source_sha256']
            assert sha(Path(c['source'])/'adapter_config.json')==c['config_sha256']
        cases.append({'base':base,'task':task,'base_model':group[0]['base_model'],'dataset_path':a['dataset_path'],
            'dataset_kind':a['dataset_kind'],'sample_indices':a['sample_indices'],'sampling_seed':42,
            'samples':256,'max_seq_len':512,'old_activation_manifest':ref['activation_manifest'],
            'old_activation_sha256':sha(ref['activation_manifest']),'dataset_sha256':sha(a['dataset_path']),
            'checkpoints':group})
    write(OUT/'collection_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared',
        'source_manifest':str(SOURCE),'source_manifest_sha256':sha(SOURCE),'cases':cases,
        'max_gpus':2,'sampling_seed':42,'training_seeds':[42,43,44],
        'trajectory':'frozen pretrained base; identical rendered inputs shared by all training seeds',
        'precision':'model and Vh projection bf16, coordinate products fp32, accumulated moments fp64',
        'stored':'full uncentered 16x16 coordinate second moments, per-example coordinate energies, Base output energies, source spectra and basis hashes',
        'collector_sha256':sha(__file__),'original_collector_sha256':sha(ROOT/'scripts/measure_lora_modification.py')})
    print('[Prepared] 18 sources, six identical-input cases, all source hashes verified',flush=True)

def padded(batch,pad_id):
    width=max(map(len,batch)); ids=torch.full((len(batch),width),pad_id,dtype=torch.long,device='cuda')
    mask=torch.zeros_like(ids)
    for i,row in enumerate(batch):
        ids[i,:len(row)]=torch.tensor(row,device='cuda'); mask[i,:len(row)]=1
    return ids,mask

def run(base):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # B300 cuDNN SDPA cannot build a plan for these padded masks.
    torch.backends.cuda.enable_cudnn_sdp(False)
    assert torch.cuda.device_count()==1,'Each worker must see exactly one GPU'
    print('[GPU]',torch.cuda.get_device_name(0),'visible',torch.cuda.device_count(),flush=True)
    plan=read(OUT/'collection_manifest.json'); cases=[r for r in plan['cases'] if r['base']==base]
    assert len(cases)==3
    model_path=cases[0]['base_model']; tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    started=time.time()
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True,
        attn_implementation='sdpa').to('cuda').eval()
    model.config.use_cache=False
    modules=dict(model.model.named_modules())
    for case in cases:
        case_out=OUT/'activations'/base/case['task']; done_path=case_out/'case_audit.json'
        if done_path.exists() and read(done_path).get('status')=='complete':
            print('[Resume]',base,case['task'],flush=True); continue
        table=pq.read_table(case['dataset_path'])
        selected=np.random.default_rng(42).permutation(table.num_rows)[:256].tolist()
        assert selected==case['sample_indices']
        encoded=[render(tok,r,case['dataset_kind'],512) for r in table.take(selected).to_pylist()]
        assert len(encoded)==256 and min(map(len,encoded))>0
        del table
        ctime=time.time(); specs={}; names=None
        for c in case['checkpoints']:
            factors,cfg=collect_factors(c['source']); ns=sorted(factors)
            if names is None: names=ns
            assert names==ns and len(ns)==c['module_count']
            sigma=[]; vh=[]; scales=[]; basis_hash=[]
            for prefix in names:
                _,s,v, _=lowrank_svd_from_ba(factors[prefix]['B'].to('cuda'),factors[prefix]['A'].to('cuda'))
                sigma.append(s.cpu().numpy()); vh.append(v.to(torch.bfloat16)); scales.append(get_scaling_for_module(cfg,prefix))
                basis_hash.append(hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest())
            specs[c['seed']]={'checkpoint':c,'sigma':np.stack(sigma),'vh':vh,'scales':np.array(scales),'basis_hash':basis_hash}
            del factors
        joined=[torch.cat([specs[s]['vh'][j] for s in (42,43,44)],0) for j in range(len(names))]
        for s in specs: del specs[s]['vh']
        coords=torch.zeros((3,len(names),16,16),device='cuda',dtype=torch.float64)
        sample=torch.zeros((3,len(names),256,16),device='cuda',dtype=torch.float32)
        den=torch.zeros((len(names),256),device='cuda',dtype=torch.float32)
        state={'mask':None,'start':0,'record':False}
        def hook(j):
            def callback(module,inputs,output):
                hidden=inputs[0]; z=F.linear(hidden,joined[j]).float()
                if not state['record']: return
                mask=state['mask'].float(); start=state['start']; end=start+hidden.shape[0]
                masked=z*mask.unsqueeze(-1)
                for si in range(3):
                    a=masked[...,si*16:(si+1)*16]
                    sample[si,j,start:end]=(a*a).sum(1)
                    flat=a.reshape(-1,16)
                    coords[si,j].add_((flat.T@flat).double())
                out=output[0] if isinstance(output,tuple) else output
                den[j,start:end]=(out.float().square()*mask.unsqueeze(-1)).sum((1,2))
            return callback
        handles=[modules[canonical_name(n)].register_forward_hook(hook(j)) for j,n in enumerate(names)]
        pilots=[]; batch_size=0
        try:
            # Probe full collection hooks on longest rows before committing any moments.
            longest=sorted(encoded,key=len,reverse=True)
            for bs in (64,128,256):
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                state['record']=False
                ids,mask=padded(longest[:bs],tok.pad_token_id); state['mask']=mask
                t=time.time()
                try:
                    with torch.inference_mode(): model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
                    torch.cuda.synchronize()
                    peak=torch.cuda.max_memory_allocated(); free,total=torch.cuda.mem_get_info()
                    pilots.append({'batch_size':bs,'status':'pass','seconds':time.time()-t,'peak_allocated_gib':peak/2**30,'total_gib':total/2**30})
                    batch_size=bs; print('[Pilot]',base,case['task'],pilots[-1],flush=True)
                    if peak>total*.75: break
                except torch.cuda.OutOfMemoryError:
                    pilots.append({'batch_size':bs,'status':'oom'}); print('[Pilot OOM]',bs,flush=True); break
                finally:
                    del ids,mask; state['mask']=None; gc.collect(); torch.cuda.empty_cache()
            if batch_size==0:
                batch_size=32
            start=0; state['record']=True
            while start<256:
                bs=min(batch_size,256-start); ids,mask=padded(encoded[start:start+bs],tok.pad_token_id)
                state.update(start=start,mask=mask)
                # Hooks may mutate moments before a later-layer OOM; snapshot transaction.
                backup=coords.clone()
                try:
                    with torch.inference_mode(): model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
                    torch.cuda.synchronize(); start+=bs
                    print('[Forward]',base,case['task'],start,'/256',flush=True)
                except torch.cuda.OutOfMemoryError:
                    coords.copy_(backup); sample[:,:,start:start+bs].zero_(); den[:,start:start+bs].zero_()
                    if batch_size<=1: raise
                    batch_size=max(1,batch_size//2); print('[OOM reduce]',batch_size,flush=True)
                finally:
                    del ids,mask,backup; state['mask']=None; gc.collect(); torch.cuda.empty_cache()
        finally:
            for h in handles: h.remove()
        case_out.mkdir(parents=True,exist_ok=True)
        cov=coords.cpu().numpy(); per=sample.cpu().numpy(); base_energy=den.cpu().numpy(); tokens=np.array(list(map(len,encoded)))
        for si,seed in enumerate((42,43,44)):
            spec=specs[seed]; p=case_out/f'seed{seed}.npz'
            assert np.isfinite(cov[si]).all() and (np.diagonal(cov[si],axis1=-2,axis2=-1)>0).all()
            diag=np.diagonal(cov[si],axis1=-2,axis2=-1); rel=np.max(np.abs(diag-per[si].sum(1))/np.maximum(diag,1e-24))
            assert rel<2e-5,rel
            np.savez_compressed(p,names=np.array(names),sigma=spec['sigma'],scales=spec['scales'],
                coordinate_second_moment_sum=cov[si],per_sample_coordinate_energy=per[si],base_output_energy=base_energy,
                token_counts=tokens,sample_indices=np.array(selected),basis_sha256=np.array(spec['basis_hash']))
            write(case_out/f'seed{seed}.json',{'status':'complete','base':base,'task':case['task'],'training_seed':seed,
                'source':spec['checkpoint']['source'],'source_sha256':spec['checkpoint']['source_sha256'],
                'npz':str(p),'npz_sha256':sha(p),'modules':len(names),'directions':16,'samples':256,'tokens':int(tokens.sum()),
                'sampling_seed':42,'dataset_path':case['dataset_path'],'dataset_sha256':case['dataset_sha256'],
                'sample_indices_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                'rendered_token_ids_sha256':hashlib.sha256(json.dumps(encoded).encode()).hexdigest(),
                'trajectory':plan['trajectory'],'moment_sample_sum_max_relative_error':float(rel),
                'precision':plan['precision'],'batch_size':batch_size,'gpu':torch.cuda.get_device_name(0)})
        write(done_path,{'status':'complete','base':base,'task':case['task'],'seeds':[42,43,44],'samples':256,
            'tokens':int(tokens.sum()),'modules_per_seed':len(names),'seconds':time.time()-ctime,'pilots':pilots,'batch_size':batch_size,
            'collector_sha256':sha(__file__),'completed_utc':datetime.now(timezone.utc).isoformat()})
        print('[Case complete]',base,case['task'],time.time()-ctime,'s',flush=True)
        del joined,coords,sample,den,specs,cov,per,base_energy
        gc.collect(); torch.cuda.empty_cache()
    write(OUT/f'worker_{base}.json',{'status':'complete','base':base,'seconds':time.time()-started,'gpu_count':1,
        'completed_utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--base')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.base: run(args.base)
