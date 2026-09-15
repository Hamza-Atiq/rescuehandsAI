import unittest

from rescuehandsai.motion import MotionFollower, Move


class MotionTests(unittest.TestCase):
    def test_reaches_goal_and_respects_step_limit(self):
        f = MotionFollower({"a": 0.0, "b": 1.0}, max_delta=0.1)
        move = Move({"a": 1.0}, steps=3)
        f.begin(move)
        self.assertGreaterEqual(move.steps, 11)  # stretched to respect the limit
        previous = dict(f.targets)
        done = False
        while not done:
            done = f.advance(move)
            self.assertLessEqual(abs(f.targets["a"] - previous["a"]), 0.1)
            previous = dict(f.targets)
        self.assertAlmostEqual(f.targets["a"], 1.0, places=6)
        self.assertEqual(f.targets["b"], 1.0)

    def test_unknown_joint_rejected(self):
        f = MotionFollower({"a": 0.0}, max_delta=0.1)
        with self.assertRaises(KeyError):
            f.begin(Move({"z": 1.0}, steps=5))

    def test_zero_length_move_finishes(self):
        f = MotionFollower({"a": 0.5}, max_delta=0.1)
        move = Move({"a": 0.5}, steps=1)
        f.begin(move)
        self.assertTrue(f.advance(move))


if __name__ == "__main__":
    unittest.main()
