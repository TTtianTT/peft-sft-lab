#!/usr/bin/env python3
"""Single-GPU forgetting worker with concurrent unchanged scoring and live reports."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from prepare_posthoc_forgetting_three_seed import ROOT, OUT, TASKS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    args=parser.parse_args()
    vm=json.loads((OUT/f'{args.base}_variant_manifest.json').read_text())
    dest=OUT/'eval'/args.base
    dest.mkdir(parents=True,exist_ok=True)
    probe=OUT/'batch_probe'/args.base
    probe.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONPATH=f'{ROOT}/src:{ROOT}/scripts',
        HF_HOME='/dataset1/zailong/cache/peft-sft-lab/huggingface',
        HF_HUB_CACHE='/dataset1/zailong/cache/peft-sft-lab/huggingface/hub',
        HF_DATASETS_CACHE='/dataset1/zailong/cache/peft-sft-lab/datasets',
        HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_BATCH_INVARIANT='1',PYTHONHASHSEED='42',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',VLLM_DISABLE_COMPILE_CACHE='1')
    tmp=ROOT/f".pf{os.environ.get('SLURM_JOB_ID',os.getpid())}"
    tmp.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),VLLM_CACHE_ROOT=str(tmp/'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'),TRITON_CACHE_DIR=str(tmp/'triton'))
    commands=json.loads((dest/'commands.json').read_text()) if (dest/'commands.json').is_file() else []
    score_process=None
    score_handle=None
    score_snapshot=set()
    last_publish=0
    score_counter=sum(c['phase']=='incremental_score' for c in commands)
    def timestamp():
        return datetime.now(timezone.utc).isoformat()
    def write_json(path,data):
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(data,indent=2)+'\n')
        temporary.replace(path)
    def status(phase,**extra):
        write_json(OUT/f'{args.base}_worker_status.json',dict(base=args.base,phase=phase,
            updated_utc=timestamp(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            node=os.environ.get('SLURMD_NODENAME'),gpu_count=1,**extra))
    def journal():
        write_json(dest/'commands.json',commands)
    def begin(cmd,log,phase):
        entry=dict(phase=phase,argv=cmd,log=str(log),started_utc=timestamp(),status='running')
        commands.append(entry);journal()
        handle=log.open('w')
        process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
        print(f'[Start] {phase}: {cmd}; log={log}',flush=True)
        return process,handle,entry
    def finish(process,handle,entry):
        code=process.wait();handle.close()
        entry.update(returncode=code,finished_utc=timestamp(),status='complete' if code==0 else 'failed')
        journal()
        return code
    def publish(force=False):
        nonlocal last_publish
        if force or time.monotonic()-last_publish>=30:
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_posthoc_forgetting_three_seed.py')],check=True,cwd=ROOT)
            last_publish=time.monotonic()
    def tick_score():
        nonlocal score_process,score_handle,score_snapshot,score_counter
        if score_process is not None and score_process[0].poll() is not None:
            process,entry=score_process
            code=finish(process,score_handle,entry)
            score_process=None;score_handle=None
            if code:
                raise RuntimeError(f'Scoring failed: {entry["log"]}')
            publish(True)
        completed={str(p.parent) for p in dest.glob('*/*/predictions.jsonl')}
        if score_process is None and completed-score_snapshot:
            score_snapshot=completed
            score_counter+=1
            cmd=[sys.executable,str(ROOT/'scripts/score_forgetting_matrix.py'),'--matrix_dir',str(dest),'--workers','32']
            process,score_handle,entry=begin(cmd,dest/f'score{score_counter}.log','incremental_score')
            score_process=(process,entry)
        publish()
    def generate(phase,output,tasks,seqs,limit=None):
        manifest=OUT/f'{args.base}_{"probe_" if limit else "variant_"}manifest.json'
        cmd=[sys.executable,str(ROOT/'scripts/eval_forgetting_matrix_vllm.py'),
            '--base_model',vm['base_model'],'--variant_manifest',str(manifest),
            '--config',vm['task_config'],'--output_dir',str(output),'--tasks',*tasks,
            '--max_model_len','4096','--gpu_memory_utilization','0.94',
            '--max_num_seqs',str(seqs),'--max_num_batched_tokens','65536',
            '--adapter_block_size','5','--max_lora_rank','16',
            '--prompt_chunk_short','4096','--prompt_chunk_long','1024','--seed','42']
        if limit:
            cmd+=['--diagnostic_max_samples',str(limit)]
        log=output/(f'{phase}.seqs{seqs}.log')
        output.mkdir(parents=True,exist_ok=True)
        status(phase,max_num_seqs=seqs,log=str(log))
        process,handle,entry=begin(cmd,log,phase)
        try:
            while process.poll() is None:
                if not limit:
                    tick_score()
                time.sleep(10)
        except BaseException:
            process.terminate();process.wait()
            raise
        code=finish(process,handle,entry)
        if code:
            with log.open('rb') as handle:
                handle.seek(max(0,log.stat().st_size-24000))
                tail=handle.read().decode(errors='replace')
            if 'out of memory' in tail.lower() or 'outofmemory' in tail.lower():
                print(f'[OOM] {phase} seqs={seqs}; lowering concurrency',flush=True)
                return False
            raise RuntimeError(f'Generation failed: {log}\n{tail}')
        if limit:
            label=vm['variants'][0]['label']
            dg=label.replace('__original_lora','__dg_hard')
            left=[json.loads(l) for l in (output/'commonsense'/label/'predictions.jsonl').read_text().splitlines()]
            right=[json.loads(l) for l in (output/'commonsense'/dg/'predictions.jsonl').read_text().splitlines()]
            assert [r['id'] for r in left]==[r['id'] for r in right]
            changed=sum(a['token_ids']!=b['token_ids'] for a,b in zip(left,right))
            write_json(output/'identity_check.json',dict(samples=len(left),token_mismatches=changed,
                status='pass' if changed==0 else 'fail'))
            if changed:
                raise RuntimeError(f'Identical LoRA/DG weights generated different tokens in probe: {output}')
        return True
    try:
        # Long benchmarks reuse the successful 2048-sequence B300 probe from the diagonal run.
        long_marker=dest/'long_generation_manifest.json'
        if not long_marker.is_file():
            for seqs in (2048,1024,512,256):
                if generate('long',dest,list(TASKS[:3]),seqs):
                    write_json(long_marker,json.loads((dest/'generation_manifest.json').read_text()))
                    break
            else:
                raise RuntimeError('All long-benchmark batch sizes OOM')
        short_marker=dest/'commonsense_generation_manifest.json'
        if not short_marker.is_file():
            selected=None
            # 512 examples/subtask × five adapters fill a large short-generation batch.
            for seqs in (4096,2048,1024,512):
                if generate('commonsense_probe',probe/f'seqs{seqs}',['commonsense'],seqs,512):
                    selected=seqs;break
            if selected is None:
                raise RuntimeError('All Commonsense probes OOM')
            for seqs in (s for s in (4096,2048,1024,512) if s<=selected):
                if generate('commonsense',dest,['commonsense'],seqs):
                    write_json(short_marker,json.loads((dest/'generation_manifest.json').read_text()))
                    break
            else:
                raise RuntimeError('All full Commonsense batch sizes OOM')
        long=json.loads(long_marker.read_text());short=json.loads(short_marker.read_text())
        combined=dict(long,tasks=long['tasks']+short['tasks'],
            configuration_by_phase=dict(long=long['configuration'],commonsense=short['configuration']))
        write_json(dest/'generation_manifest.json',combined)
        status('final_scoring')
        while True:
            tick_score()
            if score_process is None and len(list(dest.glob('*/*/metrics.json')))==184:
                break
            time.sleep(10)
        records=json.loads((dest/'score_manifest.json').read_text())['records']
        assert len(records)==184 and len({(r['task'],r['variant']) for r in records})==184
        status('complete',cells=184)
        publish(True)
        print(f'[Done] {args.base}: 184 forgetting cells',flush=True)
    except BaseException as exc:
        if score_process is not None and score_process[0].poll() is None:
            score_process[0].terminate();score_process[0].wait()
        status('failed',error=str(exc))
        publish(True)
        raise


if __name__=='__main__':
    main()
