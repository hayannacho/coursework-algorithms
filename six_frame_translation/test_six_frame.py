import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from six_frame import *


class Tests(unittest.TestCase):
    def test_frames(self):
        self.assertEqual(six_frames("ATGAAATAG")["+1"], "MK*")
        self.assertEqual(len(six_frames("ATGAAATAG")), 6)
        self.assertEqual(six_frames("a t g"), six_frames("ATG"))

    def test_reverse(self):
        self.assertEqual(reverse_complement("ATGC"), "GCAT")
        self.assertEqual(six_frames("A")["+1"], "")
        with self.assertRaises(ValueError):
            six_frames("N")

    def test_all_codons(self):
        from itertools import product

        aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
        for codon, aa in zip(product("TCAG", repeat=3), aas):
            self.assertEqual(six_frames("".join(codon))["+1"], aa)


if __name__ == "__main__":
    unittest.main()
