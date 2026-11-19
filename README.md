# SMC Differential Drive Lab (Python)

An interactive kinematic simulation to compare **4 SMC versions × 4 initial cases × 4 paths**, with a single PP circle-intersection reference and an optional dynamic path-update experiment.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, generally http://localhost:8501.

For the no-UI simulation tests (NumPy required):

```bash
python3 -m unittest discover -s tests -v
```

## Mathematical model

The robot's kinematic center is the controlled point (`c = 0`). The body-frame errors use `eθ = θR − θ`:

\[
\dot e_x=v_R\cos e_\theta-v+\omega e_y,\qquad
\dot e_y=v_R\sin e_\theta-\omega e_x,\qquad
\dot e_\theta=\omega_R-\omega.
\]

All four SMC laws **omit both `sign(s)` and `sat(s/phi)`**. We use only continuous proportional sliding-surface feedback `k_s s`. The separate physical velocity and acceleration limits are retained in simulation.

Let `s = eθ + λy ey`, `D = 1 + λy ex`, nominal reference velocity `vR` and feedforward `ωR = vR κ`.

| Version | Desired linear velocity | Desired angular velocity |
|---|---|---|
| **Truncated: fixed speed** | `v = vR` | `ω = λy vR sin(eθ) + ks s` |
| **Truncated: scaled speed** | `v = max(0, kρ ρ cos(α))`, zero when `abs(α) >= heading_stop` | same truncated ω |
| **Full: one surface (all three)** | `v = vR cos(eθ) + ey ω + kx ex` | `ω = [ωR + λy vR sin(eθ) − λx kx ex + ks s_all] / D`, `s_all = eθ + λy ey + λx ex` |
| **Full: two surfaces** | `v = vR cos(eθ) + ey ω + kx ex` | `ω = [ωR + λy vR sin(eθ) + ks s] / D`, `s1 = ex`, `s2 = s` |

**Why the full single-surface controller includes `−λx kx ex`:** With the chosen longitudinal law, `ex_dot = −kx ex`. Differentiating `s_all = s + λx ex` then gives `s_all_dot = ωR + λy vR sin(eθ) − D ω − λx kx ex`. The extra term is needed to impose `s_all_dot = −ks s_all`.

These are four **research variants**, not four proven globally stabilizing controllers. In particular, the full controllers have an unavoidable singularity at `D = 0`; the simulator terminates and reports this if `|D|` is below the configured threshold or if the sign of `D` flips between steps. Actuator saturation invalidates ideal reaching-law identities.

## PP circle-intersection reference: proposal and implementation

This is the only reference mode in the lab. It is an **experimental, PP-inspired geometry-based feedforward method** for a path that may be updated while the robot drives. It is not an exact time-parameterized reference trajectory.

At each control step:

1. Find the closest projection `s_near` onto the current path.
2. Intersect the current forward path polyline with the **Euclidean circle centered at the robot**, radius `lookahead`. Select the first valid intersection after `s_near` (including crossing the closed-loop seam). This matches the defining radial geometry of Pure Pursuit, although the exact pruning/fallback rules are not claimed to reproduce every Nav2 RPP implementation detail.
3. If there is **no intersection**, record an explicit fallback: path endpoint if the open-path endpoint is inside the circle; closest point otherwise; a forward quarter-loop fallback when a closed loop is entirely inside the circle. Fallback targets are not necessarily a distance `lookahead` from the robot.
4. Around the carrot at arc length `s_R`, sample the **three positions** `A=P(s_R-h)`, `B=P(s_R)`, `C=P(s_R+h)` (use a forward/backward window near open-path endpoints), where `h = curvature_window`.
5. Estimate the tangent direction from `C-A` and signed curvature from the three-point circumcircle:

   ```text
   kappa_R = 2 * cross2d(B-A, C-A) / (|B-A| * |C-B| * |C-A|)
   ```

   If the three samples are collinear, curvature is zero. A window averages sharp changes and reduces noise but can blur tight turns. **Two positions alone do not determine curvature**; two reliable path tangent headings at separated arc lengths would be an alternative method.
6. Assign `vR_nom = v_ref` (default **0.40 m/s**, with optional slowdown near the end of open paths). Assign `omegaR_nom = vR_nom * kappa_R`.
7. Feed `xR, yR, thetaR, vR_nom, omegaR_nom` into all four controllers; truncated controllers deliberately ignore the supplied omega feedforward.

The UI includes a pause-frame inspector that shows the robot-centered circle, current carrot, projection point, and curvature, with explicit fallback status. The Reference / geometry tab plots **nominal vs finite-difference actual carrot rates**, including lateral carrot motion. The metrics include no-intersection fallback percentage.

**Critical mathematical limitation:** a robot-dependent carrot does **not** generally obey `dot xR = vR_nom cos(thetaR)`, `dot yR = vR_nom sin(thetaR)`, `dot thetaR = omegaR_nom`. The true geometric carrot motion can have different longitudinal speed, lateral velocity, or jumps (especially on path replans and projection changes). Thus the full SMC's ideal reaching law and its zero-error conclusions do not directly follow from using this nominal feedforward. Nonzero robot-to-carrot distance is expected with positive lookahead. This mode is intended as a **practical controller experiment**, not a new stability proof.

### Dynamic replanning demonstration

Enable **Replace path during the run** in the UI. At a selected time, the synthetic path changes to a smooth lateral detour that keeps the same start and goal points. Every controller then projects onto the **new** path and recomputes its circle intersection and curvature immediately. The plots show both path versions, along with changes/jumps in carrot position and actual rates. This is a deterministic proxy for dynamic replanning, **not** a real collision-aware Nav2 planner or ROS subscription.

In an eventual ROS/Nav2 implementation, process the newest `nav_msgs/Path` on each update, keep track of the robot's progress branch to avoid projection jumping between nearby loops, and handle stop/rotate-in-place motion separately. No use of message timestamps as an arrival-time schedule is intended here. Curvature/speed references must also respect wheel-speed, angular-velocity, acceleration, and braking constraints before physical deployment.

## Reference model: PP circle intersection only

Every SMC variant uses the forward **robot-centered lookahead circle** intersection on
its current path, with three-point signed curvature near the carrot. The nominal
feedforward is `vR = v_ref` (default 0.4 m/s, slowed near open endpoints) and
`omegaR = vR * kappa_est`. All removed reference modes (time-based trajectory
and closest-point + arc-length lookahead) are no longer selectable or
supported by `Config`.

**Model limitation:** the carrot is robot-dependent, so the feedforward is not
the true carrot velocity and the ideal full-SMC convergence proofs do not
apply without rederiving the path-relative dynamics. The truncated SMC variants
still deliberately omit angular feedforward.

## Initial conditions

1. **Aligned** `(dx, dy, dθ)=(0, 0, 0°)`
2. **Lateral offset** `(0, 2 m, 0°)`
3. **Heading offset** `(0, 0, 150°)`
4. **Combined offsets** `(-1.5 m, 2 m, 150°)`

Start offsets use the local tangent/normal frame of the first path pose. The UI allows custom offsets and heading angles, including ±180°.

## Paths

* **Straight:** length-adjustable, zero curvature.
* **Circle:** radius-adjustable, closed loop, `κ=1/R`.
* **Zigzag (sharp rectilinear-style polyline):** only straight legs with instantaneous corners. Choose leg length, number of turns and heading-change angle (90° by default, with adjustable other angles). **No fillet/corner radius**. `zigzag_angle_deg` is a change in heading, not an interior polygon angle.
* **Smooth:** numerically integrated arc-length path using a sinusoidal curvature with separately variable minimum and maximum local radii, plus spatial curvature period. Curvature can change sign, so sign-changing bends are possible.

**Zigzag feasibility note:** geometric curvature is zero on straight legs and **undefined (impulsive) at the sharp corners**. Our PP three-point finite sampling window averages across a corner and yields a finite *effective* curvature and angular feedforward, which depends on the sampling window. This is not a kinematically feasible exact 90°/custom-angle instantaneous turn. For actual robots, smooth the corner or explicitly stop and rotate in place.

## Metrics and output

The app overlays 2D trajectories, allows PP circle inspection and a mid-run path-update experiment, plots `ex`, `ey`, `eθ`, geometric path error, commands and sliding variables, and displays a comparison table. The settling criterion always uses geometric path distance and path-tangent heading error. Other metrics include RMSE, max error, final error, distance, angular effort, velocity limiting percentage and minimum absolute denominator. Full CSV and configuration JSON exports are available (including sampled curvature, carrot radial distance, fallback flags, reference revision, and measured target rates). Batch mode executes 64 combinations, with failed/singular cases clearly labeled. You can save single comparisons within Streamlit session state and export them to JSON.

## Limitations and scope

This models ideal differential-drive **kinematics**, not Gazebo, ROS 2, Nav2, wheel-slip, drivetrain dynamics, motor PID or state-estimation error. All variants use the **same** reference and physical velocity/acceleration clamps for a fair comparison. If `allow_reverse` is false, velocity commands are clamped to forward-only. Low-level acceleration limiting starts from zero speed, which introduces transient tracking error. Path RMSE is based on closest geometric path distance, not reference pose distance.

The one-surface full controller guarantees only convergence of its **combined** sliding variable in the ideal nonsingular unsaturated model; it does not by itself establish that all three individual errors converge. Avoid comparing `s=0` with `ex=ey=eθ=0` without analyzing the zero dynamics.