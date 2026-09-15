"""Catch arm swaps, unsafe commands and false placement success."""
import unittest

from rescuehandsai.control import validate_action
from rescuehandsai.contracts import BimanualAction
from rescuehandsai.evaluation import PlacementTracker


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.names = ('left_arm/pan', 'right_arm/pan')
        self.limits = {name: (-1.0, 1.0) for name in self.names}
        self.previous = dict.fromkeys(self.names, 0.0)

    def check(self, targets, timestamp=1.0):
        return validate_action(BimanualAction(timestamp, targets), self.names,
                               self.limits, self.previous, now=1.0,
                               max_age=0.1, max_delta=0.2)

    def test_name_mapping_ignores_input_dictionary_order(self):
        self.assertEqual(self.check({'right_arm/pan': -0.1, 'left_arm/pan': 0.2}),
                         (0.2, -0.1))

    def test_rejects_missing_or_unknown_joint(self):
        for targets in ({'left_arm/pan': 0.0}, {'wrong': 0.0, 'right_arm/pan': 0.0}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.check(targets)

    def test_rejects_bad_numbers_and_limits_without_changing_previous(self):
        for value in (float('nan'), float('inf'), -float('inf'), 1.1, -1.1, 0.3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check({'left_arm/pan': value, 'right_arm/pan': 0.0})
        self.assertEqual(self.previous, {'left_arm/pan': 0.0, 'right_arm/pan': 0.0})

    def test_rejects_stale_future_and_nonfinite_time(self):
        for timestamp in (0.8, 1.1, float('nan'), float('inf')):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.check(self.previous, timestamp)


class PlacementTests(unittest.TestCase):
    def test_requires_support_release_and_continuous_stability(self):
        tracker = PlacementTracker(required_seconds=0.5, max_speed=0.02)
        good = dict(inside=True, supported=True, released=True, both_arms_used=True,
                    speed=0.0)
        self.assertFalse(tracker.update(time=0.0, **good))
        self.assertFalse(tracker.update(time=0.4, **good))
        self.assertTrue(tracker.update(time=0.5, **good))
        self.assertFalse(tracker.update(time=0.6, **(good | {'supported': False})))
        self.assertFalse(tracker.update(time=0.7, **good))
        self.assertTrue(tracker.update(time=1.2, **good))

    def test_flying_held_or_one_arm_object_cannot_count_as_success(self):
        good = dict(inside=True, supported=True, released=True, both_arms_used=True,
                    speed=0.0)
        for bad in ({'inside': False}, {'supported': False}, {'released': False},
                    {'both_arms_used': False}, {'speed': 0.1}, {'speed': float('nan')}):
            tracker = PlacementTracker(required_seconds=0.5, max_speed=0.02)
            with self.subTest(bad=bad):
                self.assertFalse(tracker.update(time=0.0, **(good | bad)))
                self.assertFalse(tracker.update(time=1.0, **(good | bad)))

    def test_rejects_clock_reversal(self):
        tracker = PlacementTracker(required_seconds=0.5, max_speed=0.02)
        fields = dict(inside=True, supported=True, released=True, both_arms_used=True,
                      speed=0.0)
        tracker.update(time=1.0, **fields)
        with self.assertRaises(ValueError):
            tracker.update(time=0.0, **fields)


if __name__ == '__main__':
    unittest.main()
