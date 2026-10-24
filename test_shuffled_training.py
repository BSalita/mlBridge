"""Shuffled resident batches and the one-cycle learning-rate schedule.

Run with the torch venv used by FFBridge stage 5:
    ..\\acbl-pipeline\\.venv\\Scripts\\python.exe -m unittest mlBridge.test_shuffled_training
(from src).
"""

import unittest

import torch

from mlBridge.mlBridgeAiLib import make_lr_scheduler, resident_shuffled_iter_fn


def _shards():
    # Two in-memory shards: rows 0-5 and 6-9, row value == row index.
    def shard(start, stop):
        rows = torch.arange(start, stop, dtype=torch.float32)
        return {
            "X_cont": rows.unsqueeze(1).repeat(1, 2),
            "X_cat": torch.arange(start, stop, dtype=torch.int64).unsqueeze(1),
            "y": torch.arange(start, stop, dtype=torch.int64),
        }
    return [shard(0, 6), shard(6, 10)]


def _epoch_rows(fn):
    return [int(v) for _, y in fn() for v in y.tolist()]


class ShuffledIteratorTests(unittest.TestCase):
    def test_every_row_once_per_epoch_in_a_new_order(self):
        fn, n = resident_shuffled_iter_fn(_shards(), batch_size=3, device="cpu", seed=7, combine_cat=True)
        first, second = _epoch_rows(fn), _epoch_rows(fn)
        self.assertEqual(n, 10)
        self.assertEqual(sorted(first), list(range(10)))
        self.assertEqual(sorted(second), list(range(10)))
        self.assertNotEqual(first, second)

    def test_rows_keep_their_features_and_labels(self):
        fn, _ = resident_shuffled_iter_fn(_shards(), batch_size=4, device="cpu", seed=1, combine_cat=True)
        for x, y in fn():
            # combine_cat appends X_cat as a float column; every column equals the label.
            self.assertEqual(x.shape[1], 3)
            self.assertTrue(torch.equal(x, y.float().unsqueeze(1).repeat(1, 3)))

    def test_same_seed_same_order(self):
        a, _ = resident_shuffled_iter_fn(_shards(), batch_size=4, device="cpu", seed=3, combine_cat=True)
        b, _ = resident_shuffled_iter_fn(_shards(), batch_size=4, device="cpu", seed=3, combine_cat=True)
        self.assertEqual(_epoch_rows(a), _epoch_rows(b))

    def test_regression_layout_keeps_x_cat_separate(self):
        fn, _ = resident_shuffled_iter_fn(_shards(), batch_size=10, device="cpu", seed=0, combine_cat=False)
        (x_cont, x_cat, y), = list(fn())
        self.assertEqual(tuple(x_cont.shape), (10, 2))
        self.assertEqual(tuple(x_cat.shape), (10, 1))
        self.assertTrue(torch.equal(x_cat.squeeze(1), y))


class LrScheduleTests(unittest.TestCase):
    def test_onecycle_warms_up_then_anneals(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = make_lr_scheduler(opt, "onecycle", 1e-3, epochs=4, steps_per_epoch=10)
        rates = []
        for _ in range(40):
            rates.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        peak = max(rates)
        self.assertLess(rates[0], peak / 10)
        self.assertAlmostEqual(peak, 1e-3, delta=5e-5)
        self.assertLess(rates[-1], peak / 100)

    def test_no_schedule_and_bad_names(self):
        opt = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=1e-3)
        self.assertIsNone(make_lr_scheduler(opt, None, 1e-3, 4, 10))
        with self.assertRaises(ValueError):
            make_lr_scheduler(opt, "cosine", 1e-3, 4, 10)
        with self.assertRaises(ValueError):
            make_lr_scheduler(opt, "onecycle", 1e-3, 4, None)


if __name__ == "__main__":
    unittest.main()
