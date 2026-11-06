import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rebuttal_iclr27_finish as finish


def test_publication_refuses_to_absorb_existing_staged_work(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-b', 'refactor/sft-chat-template', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / 'user_work.txt').write_text('keep staged\n')
    subprocess.run(['git', 'add', 'user_work.txt'], cwd=tmp_path, check=True)
    monkeypatch.setattr(finish.r, 'ROOT', tmp_path)
    monkeypatch.setattr(finish.r, 'OUT', tmp_path / 'reports/rebuttal_iclr27')
    monkeypatch.setattr(finish, 'validate', lambda exp: None)
    with pytest.raises(AssertionError, match='already staged'):
        finish.publish(['E1'])
    assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=tmp_path, text=True).strip() == 'user_work.txt'
    assert (tmp_path / 'user_work.txt').read_text() == 'keep staged\n'


def test_incomplete_package_cannot_be_published(tmp_path, monkeypatch):
    monkeypatch.setattr(finish.r, 'OUT', tmp_path)
    folder = tmp_path / 'E1'; folder.mkdir()
    (folder / 'summary.json').write_text('{"status":"awaiting_E0"}')
    with pytest.raises(AssertionError):
        finish.validate('E1')
