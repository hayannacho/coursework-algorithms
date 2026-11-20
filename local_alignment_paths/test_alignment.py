import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_alignment import alignment_summary, count_optimal_alignments


def exhaustive_paths(x, y, match):
    """Independent oracle: enumerate paths without a DP score/count table."""
    best, count = 0, 0
    def visit(i, j, score):
        nonlocal best, count
        if score <= 0:
            return
        if score > best:
            best, count = score, 1
        elif score == best:
            count += 1
        if i < len(x) and j < len(y):
            visit(i + 1, j + 1, score + (match if x[i] == y[j] else -1))
        if i < len(x):
            visit(i + 1, j, score - 1)
        if j < len(y):
            visit(i, j + 1, score - 1)
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                visit(i + 1, j + 1, match)
    return best, count


class AlignmentTests(unittest.TestCase):
    def test_exhaustive_short_inputs(self):
        sequences = [''.join(p) for n in range(5) for p in itertools.product('AC', repeat=n)]
        for match in (1, 2):
            for x in sequences:
                for y in sequences:
                    result = alignment_summary(x, y, match)
                    self.assertEqual((result['score'], result['count']), exhaustive_paths(x, y, match), (x, y, match))
                    self.assertEqual(result['count'], sum(e['count'] for e in result['endpoints']))

    def test_known_examples(self):
        self.assertEqual(alignment_summary('ACGT', 'ACGT')['score'], 8)
        self.assertEqual(count_optimal_alignments('A', 'ACGA'), 2)
        self.assertEqual(count_optimal_alignments('AAAA', 'AA'), 3)

    def test_no_positive_alignment(self):
        for x, y in [('', ''), ('A', ''), ('', 'A'), ('AAA', 'CCC')]:
            self.assertEqual(alignment_summary(x, y), {'score': 0, 'count': 0, 'endpoints': []})

    def test_invalid_api_input(self):
        for x, y, match in [('N', 'A', 2), ('a', 'A', 2), ('A-', 'A', 2), ('A', 'A', 3)]:
            with self.assertRaises(ValueError):
                alignment_summary(x, y, match)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'local_alignment.py'), *args], capture_output=True, text=True)

    def test_cli_json_and_normalization(self):
        run = self.run_cli('acgt', 'ACGT')
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual((result['score'], result['count'], result['match']), (8, 1, 2))
        self.assertEqual(result['endpoints'], [{'x_end': 4, 'y_end': 4, 'count': 1}])
        self.assertEqual(json.loads(self.run_cli('ACGT', 'ACGT', '--match', '1').stdout)['score'], 4)

    def test_cli_rejects_invalid_and_oversized_input(self):
        for args in [('N', 'A'), ('A'*1000, 'A'*1000)]:
            run = self.run_cli(*args)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(run.stdout, '')

    def test_repeatability(self):
        self.assertEqual(self.run_cli('AAAA', 'AA').stdout, self.run_cli('AAAA', 'AA').stdout)


if __name__ == '__main__':
    unittest.main()
