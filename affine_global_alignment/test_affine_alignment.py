import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from affine_alignment import *


class Tests(unittest.TestCase):
    def test_exhaustive_scores(self):
        from itertools import product

        def oracle(a, b, state="M"):
            if not a and not b:
                return 0
            choices = []
            if a and b:
                choices.append((2 if a[0] == b[0] else -1) + oracle(a[1:], b[1:], "M"))
            if a:
                choices.append((-1 if state == "X" else -3) + oracle(a[1:], b, "X"))
            if b:
                choices.append((-1 if state == "Y" else -3) + oracle(a, b[1:], "Y"))
            return max(choices)

        seq = ["".join(p) for n in range(4) for p in product("AC", repeat=n)]
        for a in seq:
            for b in seq:
                r = align_affine(a, b)
                self.assertEqual(r["score"], oracle(a, b))
                self.assertEqual(r["aligned_a"].replace("-", ""), a)
                self.assertEqual(r["aligned_b"].replace("-", ""), b)

    def test_known(self):
        self.assertEqual(align_affine("ACGT", "ACGT")["score"], 8)
        self.assertEqual(align_affine("AAA", "")["score"], -5)


if __name__ == "__main__":
    unittest.main()
