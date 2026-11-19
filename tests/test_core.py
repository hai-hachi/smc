import math
import unittest
from dataclasses import replace
import numpy as np
from smc_lab.core import (Config, CONTROLLERS, PATHS, CASES, make_path,
                          simulate, reference, errors, control, batch)

class SMCTests(unittest.TestCase):
    def test_four_paths_are_monotone_and_finite(self):
        for name in PATHS:
            p=make_path(Config(path=name))
            self.assertTrue(np.all(np.diff(p.s)>0),name)
            self.assertTrue(np.all(np.isfinite(p.x)))
            self.assertTrue(np.all(np.isfinite(p.kappa)))
    def test_straight_projection(self):
        p=make_path(Config(path='straight'))
        s,y=p.project(3.,2.)
        self.assertAlmostEqual(s,3.,places=5)
        self.assertAlmostEqual(y,2.,places=5)
    def test_circle_reference(self):
        c=Config(path='circle',circle_radius=2.,reference_mode='trajectory')
        p=make_path(c)
        r=reference(c,p,1.,0.)
        self.assertAlmostEqual(r[4],c.v_ref/2,places=5)
    def test_curvature_varies_on_smooth(self):
        p=make_path(Config(path='smooth'))
        self.assertGreater(np.ptp(p.kappa),0.5)
    def test_zigzag_turn_angle(self):
        c=Config(path='zigzag',zigzag_angle_deg=60.)
        p=make_path(c)
        self.assertAlmostEqual(max(p.theta),math.pi/3,places=4)
    def test_full_laws_reaching_dynamics(self):
        c=Config(reference_mode='trajectory')
        ref=(0.,0.,0.2,0.4,0.1,0.)
        e=(0.3,0.2,0.1,0.5,0.3)
        for method in CONTROLLERS[2:]:
            v,w,s1,s2,den=control(c,method,ref,e)
            ex,ey,eth=e[:3]
            exdot=ref[3]*math.cos(eth)-v+ey*w
            eydot=ref[3]*math.sin(eth)-ex*w
            ethdot=ref[4]-w
            self.assertAlmostEqual(exdot,-c.k_x*ex,places=9)
            if method==CONTROLLERS[2]:
                sd=ethdot+c.lambda_y*eydot+c.lambda_x*exdot
                self.assertAlmostEqual(sd,-c.k_s*s1,places=9)
            else:
                sd=ethdot+c.lambda_y*eydot
                self.assertAlmostEqual(sd,-c.k_s*s2,places=9)
    def test_stationary_aligned_case(self):
        c=Config(path='straight',reference_mode='trajectory',dx0=0,dy0=0,dtheta_deg=0)
        for method in CONTROLLERS:
            r=simulate(c,method)
            self.assertEqual(r.status,'completed')
            self.assertTrue(math.isfinite(r.metrics['path_rmse_m']))
            self.assertLess(r.metrics['path_rmse_m'],0.1)
    def test_singularity_report(self):
        c=Config(path='straight',reference_mode='trajectory',dx0=0.4,dy0=0,dtheta_deg=0)
        r=simulate(c,CONTROLLERS[3])
        self.assertIn('Singular',r.status)
    def test_batch_count(self):
        c=Config(duration=1.,dt=0.1)
        rows=batch(c)
        self.assertEqual(len(rows),64)
        self.assertEqual({row['path'] for row in rows},set(PATHS))
        self.assertEqual({row['case'] for row in rows},set(CASES))

if __name__ == '__main__': unittest.main()
