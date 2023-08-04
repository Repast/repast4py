import unittest
import pickle

from repast4py import random, checkpoint


class CheckpointTests(unittest.TestCase):

    def pickle(self, fname, checkpoint):
        with open(fname, 'wb') as fout:
            pickle.dump(checkpoint, fout)

    def unpickle(self, fname):
        with open(fname, 'rb') as fin:
            obj = pickle.load(fin)
        return obj

    def test_random(self):
        random.init(42)
        # generate 10 values
        random.default_rng.random((10,))
        ckp = checkpoint.Checkpoint()
        checkpoint.save_random(ckp)

        exp_vals = random.default_rng.random((10,))
        random.init(31)
        checkpoint.restore_random(ckp)
        self.assertEqual(42, random.seed)
        vals = random.default_rng.random((10,))
        self.assertEqual(list(exp_vals), list(vals))

        # test pickling
        fname = './test_data/checkpoint.pkl'
        self.pickle(fname, ckp)
        ckp = self.unpickle(fname)
        checkpoint.restore_random(ckp)
        self.assertEqual(42, random.seed)
        vals = random.default_rng.random((10,))
        self.assertEqual(list(exp_vals), list(vals))
