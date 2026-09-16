import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_pipeline", ROOT / "training" / "kaggle_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class LearningRateArgsTests(unittest.TestCase):
    def test_no_lr_keeps_policy_preset(self):
        self.assertEqual(pipeline.lr_args(None, 4000), [])

    def test_lr_goes_to_policy_fields_not_optimizer(self):
        args = pipeline.lr_args(3e-5, 4000)
        self.assertIn("--policy.optimizer_lr=3e-05", args)
        self.assertIn("--policy.scheduler_decay_steps=4000", args)
        self.assertIn("--policy.scheduler_warmup_steps=400", args)
        self.assertFalse(any(a.startswith("--optimizer.") for a in args))

    def test_warmup_is_capped_and_overridable(self):
        self.assertIn("--policy.scheduler_warmup_steps=1000", pipeline.lr_args(1e-5, 30000))
        self.assertIn("--policy.scheduler_warmup_steps=50", pipeline.lr_args(1e-5, 3000, 50))

    def test_invalid_values_rejected(self):
        for lr in (0.0, -1e-4, float("nan"), float("inf")):
            with self.subTest(lr=lr), self.assertRaises(ValueError):
                pipeline.lr_args(lr, 1000)
        with self.assertRaises(ValueError):
            pipeline.lr_args(1e-5, 100, 100)


if __name__ == "__main__":
    unittest.main()
