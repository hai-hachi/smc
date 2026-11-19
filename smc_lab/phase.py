"""Phase-plane diagnostics for SMC runs (not a stability proof).

The PP carrot is robot-dependent and may jump at a path update. Rates below
are finite differences of *observed* error/surface histories, not the ideal
reaching-law derivatives assumed by a time-parameterized reference model.
"""
from __future__ import annotations

import math
import numpy as np

from .core import CONTROLLERS

PHASE_PLANES = (
    'Lateral error vs heading error',
    'Longitudinal error vs lateral error',
    'Heading error vs error rate',
    'Angular sliding surface vs rate',
    'Longitudinal error vs error rate',
)


def _rate(t, values):
    """Sampled time derivative; undefined for a single sample."""
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(t) != len(values):
        raise ValueError('Time and values must have the same length')
    if len(t) < 2:
        return np.full(len(t), np.nan)
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time history must increase strictly')
    return np.gradient(values, t)


def phase_portrait(history, controller, plane):
    """Return phase axes, discontinuity flags, and readable labels.

    ``break_before[i]`` means the polyline must not join sample i-1 to i.
    At a replan, derivatives on both sides are marked undefined because the
    underlying target has jumped; no artificial infinite rate is asserted.
    """
    if plane not in PHASE_PLANES:
        raise ValueError(f'Unknown phase portrait: {plane}')
    if controller not in CONTROLLERS:
        raise ValueError(f'Unknown controller: {controller}')
    t = np.asarray(history['t'], dtype=float)
    ex = np.asarray(history['ex'], dtype=float)
    ey = np.asarray(history['ey'], dtype=float)
    eth = np.asarray(history['etheta'], dtype=float)
    n = len(t)
    if not all(len(a) == n for a in (ex, ey, eth)):
        raise ValueError('Error histories must have the same length')
    revisions = np.asarray(history.get('path_revision', np.zeros(n)), dtype=float)
    if len(revisions) != n:
        raise ValueError('Path revision history must match time')
    replan = np.r_[False, np.diff(revisions) != 0] if n else np.zeros(0, dtype=bool)
    # Heading is stored wrapped; do not draw spurious connections across +/-pi.
    wrap_jump = np.r_[False, np.abs(np.diff(eth)) > math.pi] if n else np.zeros(0, dtype=bool)
    breaks = replan | wrap_jump

    if plane == PHASE_PLANES[0]:
        x, y = ey.copy(), np.degrees(eth)
        x_label, y_label = 'Lateral error ey (m)', 'Heading error eθ (deg)'
    elif plane == PHASE_PLANES[1]:
        x, y = ex.copy(), ey.copy()
        x_label, y_label = 'Longitudinal error ex (m)', 'Lateral error ey (m)'
    elif plane == PHASE_PLANES[2]:
        x = np.degrees(eth)
        # Unwrap before differentiating: a +/-pi representation change
        # is not a physical 2*pi angular jump.
        y = np.degrees(_rate(t, np.unwrap(eth)))
        x_label, y_label = 'Heading error eθ (deg)', 'Observed deθ/dt (deg/s)'
    elif plane == PHASE_PLANES[3]:
        key = 's1' if controller == CONTROLLERS[2] else 's2'
        surface = np.asarray(history[key], dtype=float)
        if len(surface) != n:
            raise ValueError('Sliding surface history must match time')
        x = surface.copy()
        # Each angular surface contains e_theta with coefficient 1.
        continuous_surface = surface + np.unwrap(eth) - eth
        y = _rate(t, continuous_surface)
        x_label, y_label = 'Angular sliding surface s (rad-equivalent)', 'Observed ds/dt (rad-equivalent/s)'
    else:
        x, y = ex.copy(), _rate(t, ex)
        x_label, y_label = 'Longitudinal error ex (m)', 'Observed dex/dt (m/s)'

    if 'rate' in plane.lower():
        for idx in np.flatnonzero(replan):
            y[max(0,idx - 1):min(n, idx + 1)] = np.nan

    return {
        't': t, 'x': x, 'y': y, 'x_label': x_label, 'y_label': y_label,
        'break_before': breaks,
    }