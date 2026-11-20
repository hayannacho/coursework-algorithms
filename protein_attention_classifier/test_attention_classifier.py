import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_classifier import *


class Tests(unittest.TestCase):
    def test_padding_invariance(self):
        torch.manual_seed(3)
        m = AttentionClassifier(2, 16, 2, 1, 32).eval()
        x = one_hot_encode("ACDE").unsqueeze(0)
        n = torch.tensor([4])
        torch.testing.assert_close(m(x, n), m(torch.cat([x, torch.zeros(1, 5, 20)], dim=1), n))

    def test_training(self):
        r = demo()
        self.assertLess(r["final_loss"], r["initial_loss"])
        self.assertEqual(r["output_shape"], [4, 2])

    def test_encoding_and_invalid(self):
        x = one_hot_encode(amino_acids)
        torch.testing.assert_close(x, torch.eye(20))
        for s in ["", "AX"]:
            with self.assertRaises(ValueError):
                one_hot_encode(s)
        with self.assertRaises(ValueError):
            AttentionClassifier(2, 16, 2, 1, 32)(torch.zeros(1, 3, 20), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
