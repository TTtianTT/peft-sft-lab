#!/usr/bin/env python3
"""Portable helper shipped with each published LoRA; matches effective zero warmup."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'build-hns', 'evaluate'])
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--grid-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--peer-root', type=Path)
    parser.add_argument('--token-budget', type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output_dir.resolve().is_relative_to(repo):
        parser.error('--output-dir must be outside the downloaded repository to avoid recursive grid copies')
    meta = json.loads((repo / 'publication.json').read_text())
    source = repo / 'source_snapshot'
    if not source.is_dir():
        raise RuntimeError('Extract code/source_snapshot.tar.gz in the repository directory first.')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(source / 'src') + os.pathsep + str(source / 'scripts')
    env.update(TOKENIZERS_PARALLELISM='false', VLLM_BATCH_INVARIANT='1', VLLM_USE_FLASHINFER_SAMPLER='0',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED=str(meta['training_seed'] if args.action == 'train' else 42),
               VLLM_DISABLE_COMPILE_CACHE='1')
    with tempfile.TemporaryDirectory(prefix='hr-') as temporary:
        env.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary, VLLM_CACHE_ROOT=temporary + '/vc')
        base = args.base_dir
        if args.action != 'build-hns' and base is None:
            from huggingface_hub import snapshot_download
            revision = meta['base_provenance']['revision']
            if revision is None:
                raise RuntimeError('A unique base revision is unavailable; provide --base-dir and verify publication.json hashes.')
            base = Path(snapshot_download(meta['base_provenance']['model_id'], revision=revision))
        if base is not None:
            for name, expected in meta['base_provenance']['files_sha256'].items():
                assert sha256(base / name) == expected, f'Base file mismatch: {name}'
        task = meta['train_task']
        if args.action == 'train':
            if args.data_dir is None:
                parser.error('train requires --data-dir')
            dataset = args.data_dir / meta['dataset']['filename']
            assert sha256(dataset) == meta['dataset']['file_sha256'], 'Training dataset hash mismatch'
            a = meta['requested_run_args']
            command = [sys.executable, '-m', 'finetune.train_sft_peft', '--base_model', str(base), '--task', 'if' if task == 'tulu' else task,
                       '--peft_method', 'lora', '--dataset_path', str(dataset), '--output_dir', str(args.output_dir), '--train_profile', 'none',
                       '--warmup_ratio', '0', '--seed', str(meta['training_seed']), '--dataset_seed', '42', '--bf16', '--gradient_checkpointing']
            fields = ['num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'global_train_batch_size', 'lr', 'weight_decay',
                      'min_lr_ratio', 'lr_scheduler_type', 'adam_beta1', 'adam_beta2', 'grad_clip', 'max_seq_len', 'max_train_samples', 'sft_format',
                      'chat_template_mode', 'target_modules', 'r', 'lora_alpha', 'lora_dropout', 'pad_to_multiple_of', 'save_strategy', 'save_steps',
                      'save_total_limit', 'logging_steps']
            for key in fields:
                if a.get(key) is not None:
                    command.extend(['--' + key, str(a[key])])
            subprocess.run(command, env=env, check=True)
        else:
            benchmark_meta = json.loads((repo / 'evaluation/benchmark_provenance.json').read_text())
            benchmark = repo / 'evaluation/benchmark_input' / benchmark_meta['filename']
            assert sha256(benchmark) == benchmark_meta['sha256']
            cfg = {'bases': {meta['base']: str(base)}, 'checkpoints': [{'base': meta['base'], 'train_task': task, 'lora': str(repo)}],
                   'evaluation_tasks': {task: {**meta['evaluation_task'], 'dataset': str(benchmark)}}}
            if args.action == 'build-hns':
                cfg['checkpoints'] = []
                for peer in meta['peer_source_adapters']:
                    if peer['task'] == task:
                        peer_dir = repo
                    elif args.peer_root is not None:
                        peer_dir = args.peer_root / peer['repo_id'].split('/')[-1]
                    else:
                        from huggingface_hub import snapshot_download
                        peer_dir = Path(snapshot_download(peer['repo_id'], allow_patterns=['adapter*', 'tokenizer*', 'chat_template*', 'special_tokens_map.json', 'publication.json']))
                    assert sha256(peer_dir / 'adapter_model.safetensors') == peer['weights_sha256'], f'Peer adapter mismatch: {peer["repo_id"]}'
                    cfg['checkpoints'].append({'base': meta['base'], 'train_task': peer['task'], 'lora': str(peer_dir)})
            config = Path(temporary) / 'task.json'
            config.write_text(json.dumps(cfg))
            if args.action == 'build-hns':
                command = [sys.executable, str(source / 'scripts/build_hns_step_grid_2x4.py'), '--task_config', str(config),
                           '--grid_config', str(repo / 'hns/grid_config.json'), '--base', meta['base'], '--output_dir', str(args.output_dir), '--device', 'cuda']
                subprocess.run(command, env=env, check=True)
            else:
                if args.grid_dir is None:
                    parser.error('evaluate requires --grid-dir')
                manifest = args.grid_dir / meta['base'] / 'variant_manifest.json'
                for script, flags in [
                    ('eval_forgetting_matrix_vllm.py', ['--base_model', str(base), '--variant_manifest', str(manifest), '--config', str(config),
                     '--output_dir', str(args.output_dir), '--tasks', task, '--diagonal_only', '--max_model_len', '4096', '--gpu_memory_utilization', '0.94',
                     '--max_num_seqs', '1024', '--max_num_batched_tokens', str(args.token_budget or meta['inference']['max_num_batched_tokens']),
                     '--adapter_block_size', '11', '--max_lora_rank', '16', '--prompt_chunk_short', '4096', '--prompt_chunk_long', '512', '--seed', '42']),
                    ('score_forgetting_matrix.py', ['--matrix_dir', str(args.output_dir), '--workers', '32']),
                ]:
                    subprocess.run([sys.executable, str(source / 'scripts' / script), *flags], env=env, check=True)


if __name__ == '__main__':
    main()
