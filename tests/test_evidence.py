"""Mutation checks for scientific publication failures, using isolated copies."""
import json
import shutil

import pytest

from check_manuscript import INPUTS, ROOT, verify
from check_results import check_tables


@pytest.fixture
def evidence_copy(tmp_path):
    for name in (*INPUTS, 'paper/build-report.json'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_current_paper_and_tables_are_bound_to_results():
    assert 1 <= verify() <= 5
    check_tables()


@pytest.mark.parametrize('name', ['integration.py', 'results/independent.json', 'paper/paper.pdf'])
def test_changed_code_results_or_pdf_cannot_reuse_old_manifest(evidence_copy, name):
    with (evidence_copy / name).open('ab') as stream:
        stream.write(b'\nchanged\n')
    with pytest.raises(ValueError, match='Stale manuscript evidence'):
        verify(evidence_copy)


@pytest.mark.parametrize('filename', ['table.tex', 'independent-table.tex'])
def test_stale_table_rejected_without_overwriting_it(evidence_copy, filename):
    path = evidence_copy / 'results' / filename
    path.write_text('stale scientific table', encoding='utf-8')
    with pytest.raises(ValueError, match='Stale manuscript table'):
        check_tables(evidence_copy)
    assert path.read_text(encoding='utf-8') == 'stale scientific table'


def test_missing_hash_cannot_bypass_integrity_check(evidence_copy):
    path = evidence_copy / 'paper/build-report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    del report['sha256']['integration.py']
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='required nine files'):
        verify(evidence_copy)


def test_page_count_is_read_from_pdf(evidence_copy):
    path = evidence_copy / 'paper/build-report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['pages'] = 1 if report['pages'] != 1 else 2
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='page count'):
        verify(evidence_copy)
