# SMC Differential Drive Lab (Python)

An interactive kinematic simulation to compare **4 SMC versions × 4 initial cases × 4 fixed paths**.

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

## Reference modes: extremely important

* **Time-based trajectory**: The nominal reference moves along path arc length `sR(t)=vR*t` (clamped at a non-closed endpoint). On differentiable path segments with constant `vR`, `ωR=vR κ` is the true reference angular rate. This is the mathematically appropriate mode to study the SMC error equations. At abrupt curvature jumps (zigzag joins) and final stopping, the reference still has discontinuities / modeling caveats.
* **Closest point + lookahead**: At each step project robot onto a path segment and select target `sR = s_closest + L`. The displayed `vR` and `ωR` are only **nominal feedforward**; the actual target motion depends on the robot pose. Consequently the ideal full-SMC proof **does not apply** to this reference without rederivation, and `ex → 0` should not be expected for nonzero lookahead.

The **truncated** laws intentionally set angular feedforward to zero. They can have persistent tracking error on circles and variable-curvature paths. Their default `vR` does not automatically adapt to real reference speed.

## Initial conditions

1. **Aligned** `(dx, dy, dθ)=(0, 0, 0°)`
2. **Lateral offset** `(0, 2 m, 0°)`
3. **Heading offset** `(0, 0, 150°)`
4. **Combined offsets** `(-1.5 m, 2 m, 150°)`

Start offsets use the local tangent/normal frame of the first path pose. The UI allows custom offsets and heading angles, including ±180°.

## Paths

* **Straight:** length-adjustable, zero curvature.
* **Circle:** radius-adjustable, closed loop, `κ=1/R`.
* **Zigzag:** user-set heading-change angle, leg length, number of turns and *circular fillet radius*; finite curvature on fillets, but curvature jumps at line/arc junctions. `zigzag_angle_deg` is a change in heading, not an interior polygon angle.
* **Smooth:** numerically integrated arc-length path using a sinusoidal curvature with separately variable minimum and maximum local radii, plus spatial curvature period. Curvature can change sign, so sign-changing bends are possible.

**Zigzag feasibility note:** the path is composed of exact straight and circular segments; at each line-to-arc join curvature jumps. Therefore even though curvature is finite, the requested angular rate may change discontinuously. Finite angular acceleration limits make exact tracking at such joins impossible.

## Metrics and output

The app overlays 2D trajectories, plots `ex`, `ey`, `eθ`, geometric path error, commands and sliding variables, and displays a comparison table. The settling criterion depends on reference mode: pose error (trajectory mode), versus path-distance plus path-tangent heading error (lookahead mode). Other metrics include RMSE, max error, final error, distance, angular effort, velocity limiting percentage and minimum absolute denominator. Full CSV and configuration JSON exports are available. Batch mode executes 64 combinations, with failed/singular cases clearly labeled. You can save single comparisons within Streamlit session state and export them to JSON.

## Limitations and scope

This models ideal differential-drive **kinematics**, not Gazebo, ROS 2, Nav2, wheel-slip, drivetrain dynamics, motor PID or state-estimation error. All variants use the **same** reference and physical velocity/acceleration clamps for a fair comparison. If `allow_reverse` is false, velocity commands are clamped to forward-only. Low-level acceleration limiting starts from zero speed, which introduces transient tracking error. Path RMSE is based on closest geometric path distance, not reference pose distance.

The one-surface full controller guarantees only convergence of its **combined** sliding variable in the ideal nonsingular unsaturated model; it does not by itself establish that all three individual errors converge. Avoid comparing `s=0` with `ex=ey=eθ=0` without analyzing the zero dynamics.
