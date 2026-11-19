"""SMC differential-drive kinematic comparison laboratory.

Physical actuator clamps are used; no sign() or sat() switching term appears
in any SMC control law. The PP-circle carrot is robot-dependent, so
nominal feedforward does not represent its actual derivatives; see README.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np

CONTROLLERS = (
    'Truncated: fixed speed',
    'Truncated: scaled speed',
    'Full: one surface (all 3)',
    'Full: two surfaces',
)
PATHS = ('straight', 'circle', 'zigzag', 'smooth')
CASES = {
    'Aligned': (0.0, 0.0, 0.0),
    'Lateral offset': (0.0, 2.0, 0.0),
    'Heading offset': (0.0, 0.0, 150.0),
    'Combined offsets': (-1.5, 2.0, 150.0),
}


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def clip(x, a, b):
    return max(a, min(b, x))


@dataclass
class Config:
    path: str = 'smooth'
    dx0: float = 0.0
    dy0: float = 2.0
    dtheta_deg: float = 0.0
    duration: float = 30.0
    dt: float = 0.05
    v_ref: float = 0.40
    lookahead: float = 0.60
    curvature_window: float = 0.14
    replan_time: float = 0.0  # 0 disables replanning demo
    replan_offset: float = 0.75  # metres, peak lateral path deformation
    length: float = 14.0
    circle_radius: float = 2.0
    zigzag_angle_deg: float = 90.0
    zigzag_leg: float = 2.5
    zigzag_turns: int = 5
    smooth_radius_min: float = 0.9
    smooth_radius_max: float = 4.0
    smooth_period: float = 6.0
    lambda_y: float = 2.5
    lambda_x: float = 0.6
    k_s: float = 1.2
    k_x: float = 0.8
    k_rho: float = 0.8
    heading_stop_deg: float = 70.0
    v_max: float = 0.60
    w_max: float = 1.50
    a_max: float = 1.0
    alpha_max: float = 2.0
    allow_reverse: bool = False
    slowdown_dist: float = 0.75
    settle_distance: float = 0.15
    settle_heading_deg: float = 10.0
    settle_hold: float = 1.5
    warmup: float = 3.0
    singular_guard: float = 0.06

    def validate(self):
        if self.path not in PATHS:
            raise ValueError('Invalid path or reference mode')
        for name in ('duration','dt','length','circle_radius','zigzag_leg',
                     'smooth_radius_min','smooth_radius_max','smooth_period','lambda_y',
                     'k_s','k_x','k_rho','curvature_window','v_max','w_max','a_max','alpha_max','singular_guard'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.smooth_radius_min > self.smooth_radius_max:
            raise ValueError('smooth_radius_min must not exceed smooth_radius_max')
        if not 5 <= self.zigzag_angle_deg <= 165 or self.zigzag_turns < 1:
            raise ValueError('Invalid zigzag geometry')
        if self.replan_time < 0 or self.replan_offset < -3 or self.replan_offset > 3:
            raise ValueError('Invalid replan configuration')
        if self.v_ref < 0 or self.lookahead < 0 or self.slowdown_dist < 0:
            raise ValueError('Negative reference or slowdown parameter')
        return self


@dataclass
class Path:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    closed: bool = False
    sharp_corners: bool = False

    @property
    def length(self):
        return float(self.s[-1])

    def sample(self, distance):
        if self.closed:
            laps = math.floor(distance / self.length)
            d = distance - laps * self.length
        else:
            laps = 0
            d = clip(distance, 0.0, self.length)
        # For a sharp polyline the path tangent jumps at each vertex.
        # Never interpolate the heading across an incoming straight leg.
        if self.sharp_corners:
            idx = min(int(np.searchsorted(self.s, d + 1e-10, side='right') - 1), len(self.s) - 1)
            heading = float(self.theta[max(idx, 0)])
        else:
            heading = float(np.interp(d, self.s, self.theta))
        return (float(np.interp(d,self.s,self.x)), float(np.interp(d,self.s,self.y)),
                heading + (2*math.pi*laps if self.closed else 0.0),
                float(np.interp(d,self.s,self.kappa)))

    def project(self, x, y):
        dx, dy = np.diff(self.x), np.diff(self.y)
        ax, ay = self.x[:-1], self.y[:-1]
        den = np.maximum(dx*dx + dy*dy, 1e-16)
        u = np.clip(((x-ax)*dx + (y-ay)*dy)/den,0.0,1.0)
        px, py = ax+u*dx, ay+u*dy
        d2 = (x-px)**2 + (y-py)**2
        i = int(np.argmin(d2))
        ss = float(self.s[i] + u[i]*(self.s[i+1]-self.s[i]))
        th = math.atan2(float(dy[i]),float(dx[i]))
        signed = -(x-float(px[i]))*math.sin(th)+(y-float(py[i]))*math.cos(th)
        return ss, signed


def make_path(c: Config, ds=0.04):
    c.validate()
    if c.path == 'straight':
        s=np.linspace(0,c.length,max(3,round(c.length/ds)+1))
        return Path(s,s.copy(),np.zeros_like(s),np.zeros_like(s),np.zeros_like(s))
    if c.path == 'circle':
        r=c.circle_radius
        s=np.linspace(0,2*math.pi*r,max(80,round(2*math.pi*r/ds)+1))
        return Path(s,r*np.sin(s/r),r*(1-np.cos(s/r)),s/r,np.full_like(s,1/r),True)
    if c.path == 'smooth':
        s=np.linspace(0,c.length,max(4,round(c.length/ds)+1))
        low,high=1/c.smooth_radius_max,1/c.smooth_radius_min
        modulation=0.5+0.5*np.sin(2*math.pi*s/(0.63*c.smooth_period))
        k=np.sin(2*math.pi*s/c.smooth_period)*(low+(high-low)*modulation)
        th=np.zeros_like(s); x=np.zeros_like(s); y=np.zeros_like(s)
        for j in range(1,len(s)):
            step=s[j]-s[j-1]
            th[j]=th[j-1]+0.5*(k[j-1]+k[j])*step
            mid=(th[j]+th[j-1])*0.5
            x[j]=x[j-1]+step*math.cos(mid)
            y[j]=y[j-1]+step*math.sin(mid)
        return Path(s,x,y,th,k)
    # Sharp-cornered polyline: ALL segments straight, no circular fillets.
    # The user-selected angle is a heading change, not a rounded radius.
    # Exact vertex curvature is undefined; the PP carrot estimates an
    # effective curvature over its finite three-point sampling window.
    points = [[0., 0., 0., 0., 0.]]
    beta = math.radians(c.zigzag_angle_deg)
    headings = (0., beta, 0., -beta)
    count = max(1, math.ceil(c.zigzag_leg / ds))
    for leg in range(c.zigzag_turns + 1):
        heading = headings[leg % 4]
        # Store the outgoing tangent at this vertex; no duplicate distance.
        points[-1][3] = heading
        for _ in range(count):
            ss, x, y, _, _ = points[-1]
            step = c.zigzag_leg / count
            points.append([ss + step,
                           x + step * math.cos(heading),
                           y + step * math.sin(heading), heading, 0.])
    arr = np.asarray(points)
    return Path(arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4],
                sharp_corners=True)


def make_replanned_path(original: Path, lateral_offset: float) -> Path:
    """Illustrative path update: a lateral detour that preserves start/end.

    The offset is sin^2(pi*s/total) along the normal and goes to zero at
    both endpoints. Recompute arc length, tangent and curvature after update.
    This is not a collision-aware planner; it is a repeatable replan stress test.
    """
    fraction = original.s / original.length
    lateral = lateral_offset * np.sin(math.pi * fraction) ** 2
    # Use the geometric secant tangent, including at sharp vertices.
    # This prevents a jump in the lateral offset when segment heading changes.
    tx = np.gradient(original.x, original.s)
    ty = np.gradient(original.y, original.s)
    tangent = np.arctan2(ty, tx)
    xx = original.x - np.sin(tangent) * lateral
    yy = original.y + np.cos(tangent) * lateral
    ds = np.hypot(np.diff(xx), np.diff(yy))
    if np.any(ds < 1e-9):
        raise ValueError('Replan produced coincident samples')
    ss = np.r_[0., np.cumsum(ds)]
    tangent = np.unwrap(np.arctan2(np.gradient(yy), np.gradient(xx)))
    # np.gradient above uses unit-index spacing; the ratio is valid for heading.
    curvature = np.gradient(tangent, ss)
    return Path(ss, xx, yy, tangent, curvature, original.closed)


def circle_carrot(path: Path, robot_x: float, robot_y: float, radius: float,
                  closest_s: float | None = None):
    """Find the first forward path/circle intersection, using exact segment roots.

    Returns (arc_length, reason). ``closest_s`` is the projection onto the
    path. Search starts there so earlier path intersections are not selected.
    A closed path is searched for up to one full circuit, including across
    its stored seam. This is *radial*, not a fixed distance along the path.

    No-intersection fallback is explicit: choose the endpoint for an open path
    whose endpoint is within the circle, otherwise choose closest projection.
    For a closed path entirely inside the circle, choose a point about one
    quarter lap ahead (labelled as a fallback, not an exact intersection).
    """
    if closest_s is None:
        closest_s, _ = path.project(robot_x, robot_y)
    if radius <= 0.0:
        return float(closest_s), 'nearest_fallback'
    end_s = closest_s + path.length if path.closed else path.length
    if end_s <= closest_s + 1e-12:
        return float(closest_s), 'end_fallback'

    grid = (np.r_[path.s, path.s[1:] + path.length]
            if path.closed else path.s)
    interior = grid[(grid > closest_s + 1e-10) & (grid < end_s - 1e-10)]
    distances = np.r_[closest_s, interior, end_s]
    if path.closed:
        d = np.mod(distances, path.length)
    else:
        d = distances
    px = np.interp(d, path.s, path.x)
    py = np.interp(d, path.s, path.y)
    vx, vy = np.diff(px), np.diff(py)
    ox, oy = px[:-1] - robot_x, py[:-1] - robot_y
    aa = vx * vx + vy * vy
    bb = ox * vx + oy * vy
    cc = ox * ox + oy * oy - radius * radius
    disc = bb * bb - aa * cc
    valid = (aa > 1e-14) & (disc >= -1e-12)
    if np.any(valid):
        indices = np.flatnonzero(valid)
        root = np.sqrt(np.maximum(disc[indices], 0.0))
        aa_i, bb_i = aa[indices], bb[indices]
        candidates = []
        for fraction in ((-bb_i - root) / aa_i, (-bb_i + root) / aa_i):
            possible = (fraction >= -1e-8) & (fraction <= 1.0 + 1e-8)
            if np.any(possible):
                ii = indices[possible]
                uu = np.clip(fraction[possible], 0., 1.)
                sc = distances[ii] + uu * (distances[ii+1] - distances[ii])
                candidates.extend(sc[sc > closest_s + 1e-7])
        if candidates:
            return float(min(candidates)), 'intersection'

    if not path.closed:
        distance_to_end = math.hypot(path.x[-1] - robot_x,
                                     path.y[-1] - robot_y)
        if distance_to_end < radius + 1e-7:
            return float(path.length), 'end_fallback'
        return float(closest_s), 'nearest_fallback'
    # A large circle may contain the entire closed path: no exact carrot.
    start_dist = math.hypot(path.sample(closest_s)[0] - robot_x,
                            path.sample(closest_s)[1] - robot_y)
    if start_dist < radius:
        return float(closest_s + path.length * 0.25), 'loop_fallback'
    return float(closest_s), 'nearest_fallback'


def sampled_curvature(path: Path, s: float, window: float):
    """Estimate signed kappa at s from three neighboring *positions*.

    Two positions only determine a chord/tangent, not curvature. We use
    points a,b,c along arc length and the signed circumcircle curvature
    2*cross(b-a,c-a)/(|b-a|*|c-b|*|c-a|).
    Also returns local tangent angle from a -> c, for e_theta.
    A window naturally smooths abrupt corner curvature at its scale.
    """
    half = min(window, path.length / 2.0)
    if path.closed:
        ds = (s - half, s, s + half)
    else:
        begin = clip(s - half, 0., max(0., path.length - 2 * half))
        ds = (begin, begin + half, begin + 2 * half)
    a, b, c = (path.sample(v)[:2] for v in ds)
    abx, aby = b[0] - a[0], b[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    cross = abx * acy - aby * acx
    divisor = math.hypot(abx, aby) * math.hypot(bcx, bcy) * math.hypot(acx, acy)
    kappa = (2 * cross / divisor) if divisor > 1e-12 else 0.
    heading = math.atan2(acy, acx) if math.hypot(acx, acy) > 1e-12 else path.sample(s)[2]
    return heading, kappa


def initial_pose(c,p):
    x,y,th,_=p.sample(0.)
    return np.array([x+c.dx0*math.cos(th)-c.dy0*math.sin(th),
                     y+c.dx0*math.sin(th)+c.dy0*math.cos(th),
                     wrap(th+math.radians(c.dtheta_deg))],dtype=float)


def reference(c, p, t, s_near, pose=None, carrot=None):
    """PP-circle carrot with nominal speed and three-point curvature.

    The time argument is retained for compatibility with callers, but does
    NOT advance the reference. A pose or a precomputed carrot is required.
    """
    if carrot is None:
        if pose is None:
            raise ValueError('PP circle reference requires robot pose or precomputed carrot')
        carrot = circle_carrot(p, float(pose[0]), float(pose[1]), c.lookahead, s_near)
    s = carrot[0]
    if not p.closed:
        s = min(p.length, s)
    x, y = p.sample(s)[:2]
    th, k = sampled_curvature(p, s, c.curvature_window)
    if p.closed:
        v = c.v_ref
    else:
        remaining = max(0., p.length - s_near)
        v = c.v_ref * (min(1., remaining / c.slowdown_dist) if c.slowdown_dist else 1.)
        if remaining <= 0.025:
            v = 0.
    return (x, y, th, v, v * k, s)


def errors(pose,ref):
    dx,dy=ref[0]-pose[0],ref[1]-pose[1]
    ct,st=math.cos(pose[2]),math.sin(pose[2])
    ex=ct*dx+st*dy
    ey=-st*dx+ct*dy
    eth=wrap(ref[2]-pose[2])
    rho=math.hypot(dx,dy)
    alpha=wrap(math.atan2(dy,dx)-pose[2]) if rho>1e-10 else 0.
    return ex,ey,eth,rho,alpha


def control(c,method,ref,e):
    """Returns (desired_v, desired_w, first_surface, second_surface, denominator).

    Truncated versions intentionally omit angular feedforward wR and the
    nonlinear denominator, matching the user's approximate baseline.
    """
    if method not in CONTROLLERS: raise ValueError('Unknown SMC version')
    ex,ey,eth,rho,alpha=e
    vr,wr=ref[3:5]
    s2=eth+c.lambda_y*ey
    if method==CONTROLLERS[0]:
        return vr,c.lambda_y*vr*math.sin(eth)+c.k_s*s2,0.,s2,1.
    if method==CONTROLLERS[1]:
        v=c.k_rho*rho*math.cos(alpha)
        if abs(alpha)>=math.radians(c.heading_stop_deg): v=0.
        return max(0.,v),c.lambda_y*vr*math.sin(eth)+c.k_s*s2,0.,s2,1.
    denom=1+c.lambda_y*ex
    if abs(denom)<=c.singular_guard:
        raise ArithmeticError(f'Singular full SMC: 1+lambda*ex={denom:.4f}')
    if method==CONTROLLERS[2]:
        s=s2+c.lambda_x*ex
        w=(wr+c.lambda_y*vr*math.sin(eth)-c.lambda_x*c.k_x*ex+c.k_s*s)/denom
        v=vr*math.cos(eth)+ey*w+c.k_x*ex
        return v,w,s,0.,denom
    w=(wr+c.lambda_y*vr*math.sin(eth)+c.k_s*s2)/denom
    v=vr*math.cos(eth)+ey*w+c.k_x*ex
    return v,w,ex,s2,denom


@dataclass
class Run:
    config: Config
    controller: str
    path: Path
    history: dict
    metrics: dict
    status: str


def _settle(t,condition,hold):
    began=None
    for ti,ok in zip(t,condition):
        if ok and began is None: began=float(ti)
        if not ok: began=None
        if began is not None and ti-began>=hold-1e-8:return began
    return math.nan


def simulate(c,method,p=None):
    c=replace(c).validate()
    p=make_path(c) if p is None else p
    pose=initial_pose(c,p)
    alternate=make_replanned_path(p,c.replan_offset) if c.replan_time>0 else None
    active_path=p
    v_old=w_old=0.
    names=('t','x','y','theta','rx','ry','rtheta','vR','omegaR','ex','ey',
           'etheta','rho','s1','s2','denom','v','omega','desired_v','desired_omega',
           'path_s','path_error','path_heading_error','reference_s','limited',
           'kappa_ref','carrot_radial_distance','carrot_fallback','path_revision')
    hist={name:[] for name in names}
    status='completed'
    previous_den=None
    for i in range(round(c.duration/c.dt)+1):
        t=i*c.dt
        if alternate is not None and t >= c.replan_time:
            active_path=alternate
        nearest,signed=active_path.project(float(pose[0]),float(pose[1]))
        carrot = circle_carrot(active_path, float(pose[0]), float(pose[1]),
                               c.lookahead, nearest)
        ref = reference(c, active_path, t, nearest, pose=pose, carrot=carrot)
        k_ref = sampled_curvature(active_path, ref[5], c.curvature_window)[1]
        carrot_radial = math.hypot(ref[0] - pose[0], ref[1] - pose[1])
        carrot_fallback = float(carrot[1] != 'intersection')
        e=errors(pose,ref)
        try:
            v_des,w_des,s1,s2,den=control(c,method,ref,e)
            if method in CONTROLLERS[2:] and previous_den is not None and den*previous_den<0:
                raise ArithmeticError('Full SMC denominator crossed zero')
            previous_den=den
        except ArithmeticError as exc:
            status=str(exc);break
        v_min=-c.v_max if c.allow_reverse and method in CONTROLLERS[2:] else 0.
        v_limit=clip(v_des,v_min,c.v_max)
        w_limit=clip(w_des,-c.w_max,c.w_max)
        limited=abs(v_limit-v_des)>1e-8 or abs(w_limit-w_des)>1e-8
        v=v_old+clip(v_limit-v_old,-c.a_max*c.dt,c.a_max*c.dt)
        w=w_old+clip(w_limit-w_old,-c.alpha_max*c.dt,c.alpha_max*c.dt)
        path_th=active_path.sample(nearest)[2]
        values=(t,*pose,*ref[:3],ref[3],ref[4],e[0],e[1],e[2],e[3],s1,s2,den,
                v,w,v_des,w_des,nearest,abs(signed),wrap(path_th-pose[2]),ref[5],float(limited),
                k_ref,carrot_radial,carrot_fallback,float(alternate is not None and t>=c.replan_time))
        if not all(math.isfinite(z) for z in values):
            status='Non-finite state or command';break
        for name,val in zip(names,values): hist[name].append(float(val))
        mid=pose[2]+w*c.dt/2
        pose[0]+=v*c.dt*math.cos(mid)
        pose[1]+=v*c.dt*math.sin(mid)
        pose[2]=wrap(pose[2]+w*c.dt)
        v_old,w_old=v,w
    h={name:np.asarray(vals) for name,vals in hist.items()}
    # Finite-difference geometric target motion for diagnostics only.
    # This is not fed into the four existing SMC control laws, which use
    # nominal feedforward. A carrot tied to robot projection can jump or
    # move laterally, violating the reference unicycle model.
    n_ref = len(h['t'])
    if n_ref > 1:
        interval = np.diff(h['t'])
        vx = np.diff(h['rx']) / interval
        vy = np.diff(h['ry']) / interval
        angle = h['rtheta'][:-1]
        vlong = vx*np.cos(angle) + vy*np.sin(angle)
        vlat = -vx*np.sin(angle) + vy*np.cos(angle)
        wactual = np.arctan2(np.sin(np.diff(h['rtheta'])),
                             np.cos(np.diff(h['rtheta']))) / interval
        h['actual_ref_longitudinal'] = np.r_[vlong, vlong[-1]]
        h['actual_ref_lateral'] = np.r_[vlat, vlat[-1]]
        h['actual_ref_omega'] = np.r_[wactual, wactual[-1]]
    else:
        for key in ('actual_ref_longitudinal','actual_ref_lateral','actual_ref_omega'):
            h[key] = np.zeros(n_ref)
    if not len(h['t']):
        return Run(c,method,p,h,{'status':status,'path_rmse_m':math.nan},status)
    mask=h['t']>=c.warmup
    if not np.any(mask):mask=np.ones_like(h['t'],dtype=bool)
    within=(h['path_error']<=c.settle_distance)&(np.abs(h['path_heading_error'])<=math.radians(c.settle_heading_deg))
    settle=_settle(h['t'],within,c.settle_hold)
    metrics=dict(status=status,
        path_rmse_m=float(np.sqrt(np.mean(h['path_error'][mask]**2))),
        max_path_m=float(np.max(h['path_error'])),
        final_path_m=float(h['path_error'][-1]),
        reference_rmse_m=float(np.sqrt(np.mean(h['ex'][mask]**2+h['ey'][mask]**2))),
        final_ex_m=float(h['ex'][-1]),final_ey_m=float(h['ey'][-1]),
        final_etheta_deg=float(math.degrees(h['etheta'][-1])),
        final_path_heading_deg=float(math.degrees(h['path_heading_error'][-1])),
        settle_s=settle,converged=bool(math.isfinite(settle)),
        travel_m=float(np.sum(np.abs(h['v']))*c.dt),
        angular_effort_rad=float(np.sum(np.abs(h['omega']))*c.dt),
        limited_pct=float(np.mean(h['limited'])*100),
        min_abs_den=float(np.min(np.abs(h['denom']))),
        elapsed_s=float(h['t'][-1]),
        replanned=bool(np.any(h['path_revision']>0)),
        carrot_fallback_pct=float(np.mean(h['carrot_fallback'])*100),
        carrot_radius_rmse_m=float(np.sqrt(np.mean((h['carrot_radial_distance']-c.lookahead)**2))))
    return Run(c,method,p,h,metrics,status)


def batch(c,methods=CONTROLLERS,paths=PATHS,cases=CASES):
    output=[]
    for name in paths:
        cfg=replace(c,path=name)
        p=make_path(cfg)
        for case in cases:
            offsets=cases[case]
            cc=replace(cfg,dx0=offsets[0],dy0=offsets[1],dtheta_deg=offsets[2])
            for method in methods:
                result=simulate(cc,method,p)
                output.append({'path':name,'case':case,'controller':method,
                               **result.metrics})
    return output