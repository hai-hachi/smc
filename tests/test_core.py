import math
import unittest
from dataclasses import replace
import numpy as np
from smc_lab.core import (Config, CONTROLLERS, PATHS, CASES, make_path,
                          simulate, reference, errors, control, batch, circle_carrot, sampled_curvature, make_replanned_path, sat, switching_terms)

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
        c=Config(path='circle',circle_radius=2.)
        p=make_path(c)
        r=reference(c,p,1.,0.,pose=np.array([0.,0.,0.]))
        self.assertAlmostEqual(r[4],c.v_ref/2,delta=0.01)
    def test_curvature_varies_on_smooth(self):
        p=make_path(Config(path='smooth'))
        self.assertGreater(np.ptp(p.kappa),0.5)
    def test_zigzag_turn_angle(self):
        c=Config(path='zigzag',zigzag_angle_deg=60.)
        p=make_path(c)
        self.assertAlmostEqual(max(p.theta),math.pi/3,places=4)
    def test_zigzag_is_sharp_polyline_with_no_rounding(self):
        angle=math.radians(90.)
        cfg=Config(path='zigzag',zigzag_angle_deg=90.,zigzag_leg=2.,zigzag_turns=3)
        p=make_path(cfg,ds=0.1)
        self.assertTrue(p.sharp_corners)
        self.assertAlmostEqual(p.length,8.,places=8)
        self.assertTrue(np.all(p.kappa==0.))  # undefined only at vertices
        # All consecutive edges must be straight in each leg.
        directions=np.arctan2(np.diff(p.y),np.diff(p.x))
        self.assertEqual(len(set(np.round(directions,8))),3)
        self.assertAlmostEqual(p.sample(1.99)[2],0.,places=8)
        self.assertAlmostEqual(p.sample(2.)[2],angle,places=8)
        self.assertAlmostEqual(p.sample(2.01)[2],angle,places=8)
        # Nonzero *estimated* curvature only when the window spans the corner.
        self.assertAlmostEqual(sampled_curvature(p,1.,0.1)[1],0.,places=7)
        self.assertGreater(abs(sampled_curvature(p,2.,0.1)[1]),1.)

    def test_legacy_reference_modes_are_removed(self):
        self.assertFalse(hasattr(Config(),'reference_mode'))
        self.assertFalse(hasattr(Config(),'zigzag_radius'))
        with self.assertRaises(TypeError):
            Config(reference_mode='trajectory')
        with self.assertRaises(TypeError):
            Config(reference_mode='closest_lookahead')

    def test_full_laws_reaching_dynamics(self):
        c=Config()
        ref=(0.,0.,0.2,0.4,0.1,0.)
        e=(0.3,0.2,0.1,0.5,0.3)
        for method in CONTROLLERS[2:]:
            v,w,s1,s2,den=control(c,method,ref,e)
            ex,ey,eth=e[:3]
            exdot=ref[3]*math.cos(eth)-v+ey*w
            eydot=ref[3]*math.sin(eth)-ex*w
            ethdot=ref[4]-w
            self.assertAlmostEqual(exdot,-c.k_x*ex-c.eta_x*sat(ex/c.phi_x),places=9)
            if method==CONTROLLERS[2]:
                sd=ethdot+c.lambda_y*eydot+c.lambda_x*exdot
                self.assertAlmostEqual(sd,-c.k_s*s1-c.eta*sat(s1/c.phi),places=9)
            else:
                sd=ethdot+c.lambda_y*eydot
                self.assertAlmostEqual(sd,-c.k_s*s2-c.eta*sat(s2/c.phi),places=9)
    def test_saturation_function_and_parameter_validation(self):
        self.assertEqual(sat(3.0), 1.0)
        self.assertEqual(sat(-3.0), -1.0)
        self.assertAlmostEqual(sat(0.25), 0.25)
        self.assertEqual(sat(0.), 0.)
        for kw in ({'phi':0.}, {'phi_x':0.}, {'eta':-0.1}, {'eta_x':-0.1}):
            with self.assertRaises(ValueError):
                Config(**kw).validate()

    def test_truncated_laws_switch_with_saturation(self):
        cfg = Config()
        ref = (0.,0.,0.,0.4,0.2,0.)
        e = (0.2,0.8,0.1,1.0,0.0)
        surface = e[2] + cfg.lambda_y*e[1]
        base = cfg.lambda_y*ref[3]*math.sin(e[2])+cfg.k_s*surface
        for method in CONTROLLERS[:2]:
            _,w,*_ = control(cfg,method,ref,e)
            self.assertAlmostEqual(w,base+cfg.eta,places=9)
        # When switched off, the previous law is reproduced exactly.
        cfg.use_saturation = False
        for method in CONTROLLERS[:2]:
            _,w,*_ = control(cfg,method,ref,e)
            self.assertAlmostEqual(w,base,places=9)

    def test_switching_disabled_matches_previous_full_laws(self):
        cfg=Config(use_saturation=False)
        ref=(0.,0.,0.2,0.4,0.1,0.)
        e=(0.3,0.2,0.1,0.5,0.3)
        ex,ey,eth=e[:3]
        s2=eth+cfg.lambda_y*ey
        den=1+cfg.lambda_y*ex
        for method in CONTROLLERS[2:]:
            v,w,s1,s2_out,_=control(cfg,method,ref,e)
            surface=s2+cfg.lambda_x*ex if method==CONTROLLERS[2] else s2
            subtract=cfg.lambda_x*cfg.k_x*ex if method==CONTROLLERS[2] else 0.
            expected_w=(ref[4]+cfg.lambda_y*ref[3]*math.sin(eth)-subtract+cfg.k_s*surface)/den
            self.assertAlmostEqual(w,expected_w,places=9)
            self.assertAlmostEqual(v,ref[3]*math.cos(eth)+ey*w+cfg.k_x*ex,places=9)
            self.assertEqual(switching_terms(cfg,method,ex,s1,s2_out),(0.,0.))

    def test_switching_terms_are_logged(self):
        cfg=Config(path='straight',duration=1.,dt=0.05,dy0=0.5)
        for method in CONTROLLERS:
            run=simulate(cfg,method)
            h=run.history
            self.assertIn('switch_omega_term',h)
            self.assertIn('switch_v_term',h)
            self.assertEqual(len(h['switch_omega_term']),len(h['t']))
            self.assertLessEqual(np.max(abs(h['switch_omega_term'])),cfg.eta+1e-9)
            self.assertLessEqual(np.max(abs(h['switch_v_term'])),cfg.eta_x+1e-9)

    def test_stationary_aligned_case(self):
        c=Config(path='straight',dx0=0,dy0=0,dtheta_deg=0)
        for method in CONTROLLERS:
            r=simulate(c,method)
            self.assertEqual(r.status,'completed')
            self.assertTrue(math.isfinite(r.metrics['path_rmse_m']))
            self.assertLess(r.metrics['path_rmse_m'],0.1)
    def test_singularity_report(self):
        c=Config(path='straight',lookahead=0.4,dx0=0,dy0=0,dtheta_deg=180.)
        r=simulate(c,CONTROLLERS[3])
        self.assertIn('Singular',r.status)
    def test_pp_circle_intersection_uses_radial_not_arc_distance(self):
        p = make_path(Config(path='straight'))
        # Robot 0.3 m off a straight path, radial radius 0.5 m.
        # Forward intersection is 0.4 m ahead, NOT 0.5 m along path.
        s, reason = circle_carrot(p, 0., 0.3, 0.5, 0.)
        self.assertEqual(reason, 'intersection')
        self.assertAlmostEqual(s, 0.4, places=6)
        self.assertAlmostEqual(math.hypot(p.sample(s)[0],p.sample(s)[1]-0.3),0.5,places=6)
        cfg = Config(path='straight', lookahead=0.5)
        ref = reference(cfg, p, 1., 0., pose=np.array([0.,0.3,0.]))
        self.assertAlmostEqual(ref[5],0.4,places=6)
        self.assertAlmostEqual(ref[3],0.4,places=6)
        self.assertAlmostEqual(ref[4],0.,places=6)

    def test_pp_circle_curvature_on_circle_and_closed_seam(self):
        cfg = Config(path='circle', circle_radius=2.,lookahead=0.6)
        p = make_path(cfg)
        s,reason = circle_carrot(p,0.,0.,0.6,0.)
        self.assertEqual(reason,'intersection')
        self.assertAlmostEqual(s,4*math.asin(0.6/4),delta=0.02)
        theta,kappa = sampled_curvature(p,s,cfg.curvature_window)
        self.assertAlmostEqual(kappa,0.5,delta=0.02)
        self.assertAlmostEqual(reference(cfg,p,0.,0.,pose=np.array([0.,0.,0.]))[4],0.2,delta=0.01)
        end = p.sample(p.length-0.05)
        behind_s,status = circle_carrot(p,end[0],end[1],0.4,p.length-0.05)
        self.assertEqual(status,'intersection')
        self.assertGreater(behind_s,p.length)
        theta2,kappa2 = sampled_curvature(p,behind_s,0.14)
        self.assertAlmostEqual(kappa2,0.5,delta=0.02)

    def test_pp_no_intersection_reports_fallback(self):
        p=make_path(Config(path='straight',length=4.))
        s,status=circle_carrot(p,0.,2.,0.5,0.)
        self.assertEqual(status,'nearest_fallback')
        self.assertAlmostEqual(s,0.)
        s,status=circle_carrot(p,3.8,0.,0.5,3.8)
        self.assertEqual(status,'end_fallback')
        self.assertAlmostEqual(s,4.)

    def test_pp_reference_recomputed_when_path_changes(self):
        cfg=Config(lookahead=1.0)
        robot=np.array([1.0,0.2,0.])
        pa=make_path(replace(cfg,path='straight'))
        pb=make_path(replace(cfg,path='smooth'))
        sa,_=pa.project(robot[0],robot[1])
        sb,_=pb.project(robot[0],robot[1])
        ra=reference(cfg,pa,12.,sa,robot)
        rb=reference(cfg,pb,12.,sb,robot)
        self.assertNotAlmostEqual(ra[1],rb[1],delta=0.03)
        self.assertTrue(all(math.isfinite(x) for x in rb))

    def test_pp_simulation_diagnostics(self):
        cfg=Config(path='circle',duration=1.,dt=0.05,
                   dx0=0.,dy0=0.,dtheta_deg=0.)
        run=simulate(cfg,CONTROLLERS[0])
        self.assertEqual(run.status,'completed')
        h=run.history
        for key in ('kappa_ref','carrot_radial_distance','carrot_fallback',
                    'actual_ref_longitudinal','actual_ref_lateral','actual_ref_omega'):
            self.assertEqual(len(h[key]),len(h['t']))
        self.assertLess(np.max(abs(h['kappa_ref']-0.5)),0.03)
        self.assertTrue(math.isfinite(run.metrics['carrot_fallback_pct']))

    def test_pp_rejects_missing_pose(self):
        cfg=Config()
        with self.assertRaises(ValueError):
            reference(cfg,make_path(cfg),0.,0.)

    def test_dynamic_replan_changes_reference_and_remains_finite(self):
        c=Config(path='straight',dx0=0.,dy0=0.,
                 dtheta_deg=0.,duration=4.,dt=0.05,replan_time=2.,replan_offset=1.2)
        original=make_path(c)
        updated=make_replanned_path(original,c.replan_offset)
        self.assertAlmostEqual(updated.x[0],original.x[0],places=9)
        self.assertAlmostEqual(updated.y[0],original.y[0],places=9)
        self.assertAlmostEqual(updated.x[-1],original.x[-1],places=9)
        self.assertAlmostEqual(updated.y[-1],original.y[-1],places=9)
        self.assertGreater(np.max(np.abs(updated.y-original.y)),1.0)
        self.assertTrue(np.all(np.diff(updated.s)>0))
        run=simulate(c,CONTROLLERS[0],original)
        self.assertEqual(run.status,'completed')
        h=run.history
        self.assertTrue(np.all(h['path_revision'][h['t']<2]==0))
        self.assertTrue(np.all(h['path_revision'][h['t']>=2]==1))
        self.assertTrue(np.all(np.isfinite(h['actual_ref_omega'])))
        self.assertGreater(np.max(abs(h['actual_ref_lateral'])),0.0)

    def test_batch_count(self):
        c=Config(duration=1.,dt=0.1)
        rows=batch(c)
        self.assertEqual(len(rows),64)
        self.assertEqual({row['path'] for row in rows},set(PATHS))
        self.assertEqual({row['case'] for row in rows},set(CASES))

if __name__ == '__main__': unittest.main()