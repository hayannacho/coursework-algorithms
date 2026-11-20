import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_models import *


class Tests(unittest.TestCase):
    def test_synthetic_training(self):
        result = demo()
        for name, r in result.items():
            self.assertLess(r["final_loss"], r["initial_loss"], name)
        self.assertEqual(result["MLPNodeClassifier"]["output_shape"], [12, 2])
        self.assertEqual(result["GNNNodeClassifier"]["output_shape"], [12, 2])
        self.assertEqual(result["GCNGraphClassifier"]["output_shape"], [1, 2])

    def test_determinism(self):
        self.assertEqual(demo(), demo())

    def test_node_permutation_equivariance(self):
        torch.manual_seed(1)
        m = GNNNodeClassifier(4, 8, 2).eval()
        x = torch.randn(3, 4)
        e = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        perm = torch.tensor([2, 0, 1])
        inv = torch.argsort(perm)
        torch.testing.assert_close(m(x[perm], inv[e]), m(x, e)[perm])


if __name__ == "__main__":
    unittest.main()
