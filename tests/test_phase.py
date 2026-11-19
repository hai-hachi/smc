import math
import unittest
import numpy as np

from smc_lab.core import Config, CONTROLLERS, simulate
from smc_lab.phase import PHASE_PLANES, phase_portrait


class PhasePortraitTests(unittest.TestCase):
    def test_all_planes_work_for_all_controllers(self):
        c = Config(path='straight', duration=0.6, dt=0.1, dy0=0., dx0=0.)
        for method in CONTROLLERS:
            run = simulate(c, method)
            for plane in PHASE_PLANES:
                out = phase_portrait(run.history, method, plane)
                self.assertEqual(len(out['x']), len(run.history['t']))
                self.assertEqual(len(out['y']), len(run.history['t']))
                self.assertEqual(len(out['break_before']), len(run.history['t']))
                self.assertTrue(np.all(np.isfinite(out['x'])))
                self.assertTrue(np.all(np.isfinite(out['y'])))

    def test_angular_surface_variant_mapping(self):
        h = dict(t=np.array([0., 1., 2.]), ex=np.zeros(3), ey=np.zeros(3),
                 etheta=np.array([0.1, 0.2, 0.3]),
                 s1=np.array([1., 1.5, 2.]), s2=np.array([0.1, 0.2, 0.3]))
        one = phase_portrait(h, CONTROLLERS[2], PHASE_PLANES[3])
        two = phase_portrait(h, CONTROLLERS[3], PHASE_PLANES[3])
        self.assertTrue(np.array_equal(one['x'],h['s1']))
        self.assertTrue(np.array_equal(two['x'],h['s2']))
        self.assertTrue(np.allclose(one['y'],0.5))
        self.assertTrue(np.allclose(two['y'],0.1))

    def test_wrapped_heading_rate_not_false_2pi_jump(self):
        h = dict(t=np.array([0., 0.1, 0.2]), ex=np.zeros(3), ey=np.zeros(3),
                 etheta=np.array([3.10, -3.12, -3.06]))
        p = phase_portrait(h, CONTROLLERS[0], PHASE_PLANES[2])
        self.assertTrue(p['break_before'][1])
        self.assertLess(np.max(np.abs(p['y'])), 50.)

    def test_replan_breaks_line_and_masks_rate(self):
        h = dict(t=np.array([0., 1., 2., 3.]), ex=np.array([0., 1., 4., 5.]),
                 ey=np.zeros(4), etheta=np.zeros(4),
                 path_revision=np.array([0., 0., 1., 1.]))
        p = phase_portrait(h, CONTROLLERS[0], PHASE_PLANES[4])
        self.assertTrue(p['break_before'][2])
        self.assertTrue(np.isnan(p['y'][1]))
        self.assertTrue(np.isnan(p['y'][2]))
        self.assertTrue(np.isfinite(p['y'][0]))
        self.assertTrue(np.isfinite(p['y'][3]))

    def test_single_sample_does_not_break(self):
        h = dict(t=np.array([0.]), ex=np.array([.5]), ey=np.array([0.]),
                 etheta=np.array([0.]), s2=np.array([0.]))
        p = phase_portrait(h, CONTROLLERS[0], PHASE_PLANES[3])
        self.assertEqual(len(p['x']), 1)
        self.assertTrue(math.isnan(p['y'][0]))

    def test_invalid_plane_and_controller(self):
        h = dict(t=np.array([0.]), ex=np.array([0.]), ey=np.array([0.]), etheta=np.array([0.]))
        with self.assertRaises(ValueError):
            phase_portrait(h, CONTROLLERS[0], 'unknown')
        with self.assertRaises(ValueError):
            phase_portrait(h, 'unknown', PHASE_PLANES[0])

if __name__ == '__main__':
    unittest.main()