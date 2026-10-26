"""Recompute every recorded diagnostic without overwriting the evidence."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from integration import experiment
from verification import experiment as independent
from verification import write_table as independent_table
from paper_table import write_table

ROOT = Path(__file__).resolve().parent


def check_tables(root=ROOT):
    """Regenerate tables from versioned JSON in isolation; reject stale tables."""
    with TemporaryDirectory() as directory:
        for name, filename, writer in (
            ('integration', 'table.tex', write_table),
            ('independent', 'independent-table.tex', independent_table),
        ):
            data = json.loads((root / f'results/{name}.json').read_text(encoding='utf-8'))
            generated = Path(directory) / filename
            writer(data, generated)
            if generated.read_bytes() != (root / 'results' / filename).read_bytes():
                raise ValueError(f'Stale manuscript table: results/{filename}')


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-11), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


if __name__ == '__main__':
    check_tables()
    print('Both manuscript tables match the recorded numerical results.')
    for name, function in [('integration', experiment), ('independent', independent)]:
        compare(function(), json.loads((ROOT/f'results/{name}.json').read_text()), name)
        print(f'{name}: all recorded values reproduced (rtol=1e-6, atol=1e-11).')
