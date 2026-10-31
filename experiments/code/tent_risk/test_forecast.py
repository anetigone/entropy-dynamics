"""Synthetic checks of probability fitting, task separation and BN probe resets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from tent_forecast import FEATURES, fit_logistic, predict, probe


class ForecastChecks(unittest.TestCase):
    def test_probability_signal(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(400,3))
        y = 1/(1+np.exp(-3*x[:,2]))
        model = fit_logistic(x[:200],y[:200],.01)
        prediction = predict(model,x[200:])
        self.assertLess(np.mean((prediction-y[200:])**2),.01)
        constant = fit_logistic(x,np.zeros(len(x)))
        self.assertTrue(np.isfinite(predict(constant,x)).all())

    def test_batch_probes_reset_and_restore(self):
        # Keep Torch and the CPU fitting runtime in separate processes, as in
        # the actual collection/fitting entry points (Windows OpenMP runtimes).
        if os.environ.get('TENT_PROBE_TEST_CHILD') != '1':
            env = dict(os.environ, TENT_PROBE_TEST_CHILD='1')
            result = subprocess.run([sys.executable, __file__,
                'ForecastChecks.test_batch_probes_reset_and_restore'],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return
        import torch
        from types import SimpleNamespace
        torch.manual_seed(3)
        class Toy:
            def __init__(self):
                self.model = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(3,track_running_stats=False),torch.nn.Linear(3,2))
                self.args = SimpleNamespace(batch_size=8)
                self.states = []
            def inputs(self, x):
                return torch.tensor(x,dtype=torch.float32)
            def set_vector(self, x):
                with torch.no_grad():
                    self.model[0].weight.copy_(torch.tensor(x[:3]))
                    self.model[0].bias.copy_(torch.tensor(x[3:]))
            def vector(self):
                return np.concatenate([self.model[0].weight.detach().numpy(),
                                       self.model[0].bias.detach().numpy()])
            def update(self, images, rng):
                self.states.append(self.vector().copy())
                self.model.zero_grad(set_to_none=True)
                logits = self.model(self.inputs(images[rng.integers(len(images),size=8)]))
                loss = -(logits.softmax(1)*logits.log_softmax(1)).sum(1).mean()
                loss.backward()
                return np.concatenate([-.01*self.model[0].weight.grad.numpy(),
                                       -.01*self.model[0].bias.grad.numpy()])
        toy = Toy()
        state = np.array([1.,1.,1.,0.,0.,0.],dtype=np.float32)
        images = np.random.default_rng(2).normal(size=(40,3))
        features = probe(toy,images,state,np.random.default_rng(4),4,2)
        self.assertEqual(features.shape,(16,))
        for value in toy.states+[toy.vector()]:
            np.testing.assert_array_equal(value,state)
        self.assertTrue(np.isfinite(features).all())

    def test_frozen_cli_and_degenerate_data(self):
        here = Path(__file__).resolve().parent
        def run(script, *args, success=True):
            result = subprocess.run([sys.executable,str(here/script),*map(str,args)],
                                    capture_output=True,text=True)
            if success:
                self.assertEqual(result.returncode,0,result.stderr)
            else:
                self.assertNotEqual(result.returncode,0)
            return result
        temp_root = Path(__file__).resolve().parents[3]/'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            root = Path(tmp)
            def tasks(name, seed, zero=False):
                folder = root/name
                folder.mkdir()
                rng = np.random.default_rng(seed)
                protocol = dict(complete=True,seed=seed,features=FEATURES,parameter_names=['toy'],
                    mean=[0,0,0],std=[1,1,1],batch_size=8,eval_batch_size=10,horizon=20,
                    epsilon=.02,probe_batches=4,panel_batches=2)
                (folder/'protocol.json').write_text(json.dumps(protocol))
                for i in range(60):
                    x = rng.normal(size=len(FEATURES))
                    probability = 1/(1+np.exp(-3*x[-2]))
                    y = np.zeros(8,dtype=bool) if zero else rng.random(8)<probability
                    meta = dict(corruption=['a','b','c'][i%3],severity=5,lr=.001,start_step=0)
                    np.savez(folder/f'task_{i:04d}.npz',features=x,events=y,metadata=json.dumps(meta))
                return folder
            dev, confirmation, zero = tasks('dev',17), tasks('confirm',23), tasks('zero',31,True)
            fitted = root/'fitted'
            run('fit_forecast.py','--development',dev,'--output',fitted)
            score = root/'scores'
            run('score_forecast.py','--tasks',confirmation,'--predictor',fitted/'predictor.json',
                '--output',score,'--bootstrap',100)
            result = json.loads((score/'summary.json').read_text())
            self.assertLess(result['brier']['dynamics'],result['brier']['constant'])
            run('forecast_plot.py','--result',score)
            self.assertTrue((score/'forecast.png').exists())
            run('score_forecast.py','--tasks',dev,'--predictor',fitted/'predictor.json',
                '--output',root/'bad',success=False)
            run('fit_forecast.py','--development',zero,'--output',root/'zero_fit',success=False)
            self.assertTrue((root/'zero_fit'/'coverage.json').exists())
            self.assertFalse((root/'zero_fit'/'predictor.json').exists())


if __name__ == '__main__':
    unittest.main()
