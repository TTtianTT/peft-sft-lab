#!/usr/bin/env python3
"""CPU-only pinned official model downloads into ignored workspace assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rebuttal_iclr27/R2-assets'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-14B'])
    args = p.parse_args()
    os.environ.update(HF_HOME=str(OUT / 'runtime/hf'), HF_HUB_DISABLE_PROGRESS_BARS='1')
    from huggingface_hub import HfApi, snapshot_download
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        manifest = OUT / f'{name}.json'
        if manifest.exists():
            print('[Existing manifest]', name, flush=True)
            continue
        repo = 'Qwen/' + name
        revision = HfApi().model_info(repo).sha
        destination = OUT / 'runtime/models' / name
        snapshot_download(repo, revision=revision, local_dir=destination, max_workers=4,
            allow_patterns=['*.safetensors', '*.json', '*.jinja', 'merges.txt', 'vocab.*', 'LICENSE', 'README.md'])
        files = {}
        for path in sorted(destination.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as h:
                    for chunk in iter(lambda: h.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                files[path.name] = digest.hexdigest()
        assert 'config.json' in files and any(n.endswith('.safetensors') for n in files)
        manifest.write_text(json.dumps(dict(status='complete', repository=repo, revision=revision,
            path=str(destination), files_sha256=files), indent=2)+'\n')
        print('[Downloaded and hash-checked]', repo, revision, flush=True)


if __name__ == '__main__':
    main()
