"""Regression: a jump height mismatch must not censor the v2 rollout."""
import unittest

from dex_mimic.evaluation.criteria import assess, summarize


class JumpCriteriaTests(unittest.TestCase):
    def row(self, height, tilt=0., profile='jump_tracking_v2'):
        return dict(assess(height, tilt, .25, .8, profile), height_error_m=height)

    def test_recorded_dance_girl_jump_continues(self):
        row = self.row(abs(1.0862011752 - 1.3477020264))
        self.assertTrue(row['height_violation'])
        self.assertFalse(row['failed'])
        self.assertIsNone(row['termination_reason'])
        self.assertTrue(self.row(.2615008512, profile='strict_v1')['failed'])

    def test_height_mismatch_can_recover_and_complete_without_passing_tracking(self):
        result = summarize([self.row(.1), self.row(.3), self.row(.02)], 3, .02)
        self.assertTrue(result['completed'])
        self.assertFalse(result['failed'])
        self.assertFalse(result['tracking_criteria_passed'])
        self.assertEqual(result['outcome'], 'completed_with_tracking_deviation')
        self.assertEqual(result['height_violation_seconds'], .02)
        self.assertIsNone(result['physical_fall_detected'])

    def test_tilt_guard_remains_and_is_not_a_confirmed_fall(self):
        row = self.row(.01, .81)
        result = summarize([row], 100, .02)
        self.assertTrue(result['failed'])
        self.assertFalse(result['completed'])
        self.assertEqual(result['outcome'], 'tilt_tracking_termination')
        self.assertIsNone(result['physical_fall_detected'])

    def test_threshold_and_step_limit(self):
        self.assertFalse(self.row(.25, .8)['failed'])
        self.assertEqual(summarize([self.row(.1)], 2, .02)['outcome'], 'step_limit')
        with self.assertRaises(ValueError):
            self.row(.1, profile='typo')


if __name__ == '__main__':
    unittest.main()
