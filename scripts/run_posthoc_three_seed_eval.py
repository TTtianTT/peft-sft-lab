#!/usr/bin/env python3
"""One GPU worker, short batch probe then full diagonal evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from audit_posthoc_three_seed import OUT, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--phase', choices=('flat', 'dg', 'joint'), required=True)
    args = p.parse_args()
    if args.phase == 'flat':
        assert (OUT / 'flat_build_complete.json').is_file()
    else:
        assert (OUT / 'flat_summary.json').is_file(), 'DG requires all Flat evaluations and summary first'
    if args.phase == 'joint':
        assert (OUT / 'dg_initial_summary.json').is_file()
    manifest = OUT / f'{args.base}_{args.phase}_variant_manifest.json'
    vm = json.loads(manifest.read_text())
    dest = OUT / 'eval' / args.phase / args.base
    probe = OUT / 'batch_probe' / args.phase / args.base
    probe.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONPATH=f'{ROOT}/src:{ROOT}/scripts',
        HF_HOME='/dataset1/zailong/cache/peft-sft-lab/huggingface',
        HF_HUB_CACHE='/dataset1/zailong/cache/peft-sft-lab/huggingface/hub',
        HF_DATASETS_CACHE='/dataset1/zailong/cache/peft-sft-lab/datasets',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_BATCH_INVARIANT='1', PYTHONHASHSEED='42',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', VLLM_DISABLE_COMPILE_CACHE='1')
    # node01 /tmp is full; a short workspace path also fits ZMQ socket limits.
    tmp = ROOT / f".pf{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp), VLLM_CACHE_ROOT=str(tmp / 'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp / 'inductor'), TRITON_CACHE_DIR=str(tmp / 'triton'))
    commands = []
    def generate(output, seqs, limit=None):
        cmd = [sys.executable, str(ROOT / 'scripts/eval_forgetting_matrix_vllm.py'),
            '--base_model', vm['base_model'], '--variant_manifest', str(manifest),
            '--config', vm['task_config'], '--output_dir', str(output),
            '--tasks', 'magicoder', 'metamath', 'tulu', '--diagonal_only',
            '--max_model_len', '4096', '--gpu_memory_utilization', '0.94',
            '--max_num_seqs', str(seqs), '--max_num_batched_tokens', '65536',
            '--adapter_block_size', '5' if args.phase == 'joint' else '6', '--max_lora_rank', '16',
            '--prompt_chunk_short', '4096', '--prompt_chunk_long', '1024', '--seed', '42']
        if limit is not None:
            cmd += ['--diagnostic_max_samples', str(limit)]
        commands.append(cmd)
        log = output.parent / f"{output.name}.job{os.environ.get('SLURM_JOB_ID', os.getpid())}.log"
        print(f'[Run] {cmd}; log={log}', flush=True)
        with log.open('w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            tail = log.read_text()[-12000:]
            if 'out of memory' in tail.lower() or 'outofmemory' in tail.lower():
                print(f'[OOM] seqs={seqs}; reducing', flush=True)
                return False
            raise RuntimeError(f'Evaluation failed: {log}\n{tail}')
        return True
    if not (dest / 'generation_manifest.json').is_file():
        if args.phase == 'joint':
            # Five adapters/checkpoint fit within the successful six-adapter Flat probe.
            selected = 2048
        else:
            for seqs in (2048, 1024, 512, 256):
                if generate(probe / f'seqs{seqs}', seqs, 256):
                    selected = seqs
                    break
            else:
                raise RuntimeError('All batch probes OOM')
        for seqs in (s for s in (2048, 1024, 512, 256) if s <= selected):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if generate(dest, seqs):
                selected = seqs
                break
        else:
            raise RuntimeError('Full evaluation OOM at all sizes')
    cmd = [sys.executable, str(ROOT / 'scripts/score_forgetting_matrix.py'), '--matrix_dir', str(dest), '--workers', '32']
    commands.append(cmd)
    subprocess.run(cmd, check=True, cwd=ROOT)
    (dest / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(f'[Done] {args.base} {args.phase}', flush=True)


if __name__ == '__main__':
    main()
