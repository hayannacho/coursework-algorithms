import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmm import prob, E_step, M_step, likelihood, compute_sigma, train_model
from demo import run, make_data


class GMMTests(unittest.TestCase):
    def test_density_against_analytic_standard_normal(self):
        x = np.array([[0., 0.], [1., 2.]])
        expected = np.exp(-np.sum(x*x, axis=1)/2)/(2*np.pi)
        np.testing.assert_allclose(prob(x, np.zeros(2), np.eye(2)), expected)
        self.assertAlmostEqual(prob(x[0], np.zeros(2), np.eye(2)), 1/(2*np.pi))

    def test_identical_components_have_prior_responsibilities(self):
        X = np.array([[0., 0.], [1., -1.]])
        r = E_step(X, np.zeros((2,2)), np.array([np.eye(2)]*2), np.array([.25,.75]), 2)
        np.testing.assert_allclose(r, [[.25,.25],[.75,.75]])

    def test_m_step_known_weighted_moments(self):
        X = np.array([[0.], [2.], [4.]])
        mu, sigma, pi = M_step(X, np.array([[1.,1.,0.],[0.,0.,1.]]), 2)
        np.testing.assert_allclose(mu, [[1.],[4.]])
        np.testing.assert_allclose(sigma[:,0,0], [1.,0.])
        np.testing.assert_allclose(pi, [2/3,1/3])

    def test_training_posteriors_match_final_parameters(self):
        X = make_data()
        mu = X[[0,-1]]
        params = (mu, compute_sigma(X, mu), np.array([.5,.5]))
        result = train_model(X, 2, lambda a,b,c:(c+1,True), params)
        np.testing.assert_allclose(result[3], E_step(X,*result[:3],2))
        self.assertGreaterEqual(likelihood(X,*result[:3],2), likelihood(X,*params,2))

    def test_demo_likelihood_and_known_centers(self):
        summary, X, labels, history = run()
        self.assertEqual(summary['observations'], 600)
        self.assertTrue(np.all(np.diff(history) >= -1e-8))
        np.testing.assert_allclose(summary['means'], [[-3,0],[3,1]], atol=.15)
        self.assertEqual(summary['assigned_counts'], [300,300])
        self.assertLess(summary['max_responsibility_sum_error'], 1e-12)

    def test_invalid_inputs_and_iteration_limit(self):
        for X,k in [(np.array([]),2),(np.array([[np.nan]]),1),(np.ones((3,2)),4)]:
            with self.assertRaises(ValueError):train_model(X,k)
        X=make_data()
        with self.assertRaises(RuntimeError):
            train_model(X,2,lambda a,b,c:(c,False),max_iter=1)

    def test_cli_outputs_and_repeatability(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as d:
            outputs=[]
            for i in range(2):
                out=Path(d)/str(i)
                p=subprocess.run([sys.executable,str(root/'demo.py'),'--outdir',str(out)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertEqual(json.loads(p.stdout)['components'],2)
                outputs.append({f.name:f.read_bytes() for f in out.iterdir()})
            self.assertEqual(outputs[0],outputs[1])
            self.assertEqual(set(outputs[0]), {'summary.json','assignments.csv','likelihood.csv'})


if __name__=='__main__':unittest.main()
