"""Sanity checks: run with `python -m unittest discover -s tests -v`."""
import unittest
import torch
from src.data import load_offline_digits
from src.models import EdgeCNN,count_params,estimate_macs

class TestCybertron(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data, cls.labels, cls.meta=load_offline_digits()
    def test_split_nonoverlap(self):
        ids=self.meta['split_indices']
        tr,va,te=(set(ids[k]) for k in ('train','val','test'))
        self.assertEqual((len(tr),len(va),len(te)),(1257,270,270))
        self.assertFalse(tr&va or tr&te or va&te)
        self.assertEqual(len(tr|va|te),1797)
    def test_tensor_and_classes(self):
        x,y=self.data['train'];self.assertEqual(tuple(x.shape),(1257,3,32,32))
        self.assertGreaterEqual(float(x.min()),0.0);self.assertLessEqual(float(x.max()),1.0)
        self.assertEqual(set(int(z) for z in y),set(range(10)))
    def test_models(self):
        for lightweight,expected_params,expected_macs in [(False,26218,2804032),(True,5813,396608)]:
            m=EdgeCNN(10,lightweight).eval()
            self.assertEqual(count_params(m),expected_params)
            self.assertEqual(estimate_macs(m,32),expected_macs)
            with torch.no_grad():self.assertEqual(tuple(m(torch.randn(2,3,32,32)).shape),(2,10))
    def test_lightweight_constraint(self):
        self.assertLessEqual(count_params(EdgeCNN(10,True)),100000)
if __name__=='__main__':unittest.main()
