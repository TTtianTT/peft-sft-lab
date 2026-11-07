#!/usr/bin/env python3
"""Fetch pinned official auxiliary test data, without model execution."""
import gzip
import json
import os
from pathlib import Path
import sys
import urllib.request
import rebuttal_iclr27 as r

P=r.OUT/'R2-O12'
os.environ['HF_HOME']=str(P/'runtime/hf')
os.environ['XDG_CACHE_HOME']=str(P/'runtime/cache')
sys.path.insert(0,str(P/'runtime/deps'))


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from evalplus.data import get_mbpp_plus
    data=P/'runtime/data';data.mkdir(parents=True,exist_ok=True)
    api=HfApi();repo='HuggingFaceH4/MATH-500';revision=api.dataset_info(repo).sha
    files=api.list_repo_files(repo,repo_type='dataset',revision=revision)
    selected=[x for x in files if x.endswith('test.jsonl')];assert len(selected)==1,files
    path=Path(hf_hub_download(repo,selected[0],repo_type='dataset',revision=revision,local_dir=data/'math500'))
    assert len(path.read_text().splitlines())==500
    url='https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz'
    mbpp=data/'MbppPlus-v0.2.0.jsonl'
    if not mbpp.exists():
        with urllib.request.urlopen(url,timeout=120) as response:raw=gzip.decompress(response.read())
        mbpp.write_bytes(raw)
    os.environ['MBPP_OVERRIDE_PATH']=str(mbpp)
    # Environment variable is read at import time; update the official data module.
    import evalplus.data.mbpp as module
    module.MBPP_OVERRIDE_PATH=str(mbpp)
    problems=get_mbpp_plus();assert len(problems)==378
    packages={}
    from importlib import metadata
    for name in ('evalplus','math-verify','latex2sympy2-extended','antlr4-python3-runtime','tree-sitter','tree-sitter-python','tempdir','wget','appdirs','stop-sequencer'):
        packages[name]=metadata.version(name)
    hashes={str(x.relative_to(r.ROOT)):r.sha(x) for x in (P/'runtime/deps').rglob('*')
            if x.is_file() and x.suffix in ('.py','.so')}
    r.write(P/'assets.json',dict(status='complete',math500=dict(repo=repo,revision=revision,path=str(path),sha256=r.sha(path),samples=500),
        mbppplus=dict(url=url,version='v0.2.0',path=str(mbpp),sha256=r.sha(mbpp),samples=378),
        package_versions=packages,scorer_files_sha256=hashes))
    print('Pinned MATH500 / MBPP+ data and scorer sources; no model run.',flush=True)


if __name__=='__main__':main()
