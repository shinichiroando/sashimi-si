"""Picard integration of the native SASHIMI tidal stripping equation.

Adapted from gomeshun/sashimi-c at 2ffefc8f02ff1db2f634aae73c1b2a394914a98e.
The caller supplies the redshift-dependent host mass, stripping rate and exponent.
"""

# MIT License
#
# Copyright (c) 2026 Shin'ichiro Ando
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import hashlib
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp


class PicardFallbackWarning(UserWarning):
    """Report replacement of a Picard estimate by direct log-mass integration.

    Notes
    -----
    The warning records why the request exceeded the numerical domain or
    iteration tolerance. The solver also records the affected mass count.
    """


def _real(value, name):
    """Read mass or redshift coordinates as finite real numbers.

    Parameters
    ----------
    value : array_like
        Scalar or array in the caller's physical unit.
    name : str
        Coordinate name used to identify invalid input.

    Returns
    -------
    numpy.ndarray
        Floating values with the input shape and unit.

    Raises
    ------
    ValueError
        If any element is non-real or non-finite.
    """
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or np.any(~np.isfinite(a)):
        raise ValueError(f"{name} must contain finite real numbers.")
    return a.astype(float, copy=False)


def _integer(value, name, minimum):
    """Validate a grid size or iteration count.

    Parameters
    ----------
    value : scalar
        Requested dimensionless count.
    name : str
        Numerical control named in an error.
    minimum : int
        Smallest count supported by the operation.

    Returns
    -------
    int
        The validated count.

    Raises
    ------
    ValueError
        If the input is fractional, non-finite, boolean or below the minimum.
    """
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def physics_key(solver):
    """Identify the host and rate inputs that determine a stripped mass.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.

    Returns
    -------
    tuple
        Immutable key for reusing results on this solver.

    Notes
    -----
    Scalar settings, numerical arrays and rate callables enter the key.
    A provider with additional internal state can supply picard_physics_key().
    """
    result = []
    for name, value in sorted(vars(solver).items()):
        if name.startswith(("_picard", "_eps", "eps_")):
            continue
        if isinstance(value, (float, int, str, bool, np.number)):
            result.append((name, repr(value)))
        elif isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
            result.append((name, value.shape, hashlib.sha256(value.tobytes()).hexdigest()))
        elif callable(value):
            result.append((name, id(value)))
    if hasattr(solver, "picard_physics_key"):
        result.append(("provider", solver.picard_physics_key()))
    for name in ("Phi", "zetaMz", "Mzvir"):
        value = getattr(solver, name)
        result.append((name, id(getattr(value, "__func__", value))))
    return tuple(result)


def _state(solver):
    """Invalidate cached tidal solutions when their host or rates change.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.

    Returns
    -------
    tuple
        Current physical-state key for the solver-owned cache.
    """
    key = physics_key(solver)
    if getattr(solver, "_picard_physics_key", None) != key:
        solver._picard_coefficients = {}
        solver._picard_host_cache = {}
        solver._picard_physics_key = key
    if not hasattr(solver, "_picard_events"):
        solver._picard_events = []
    return key


def cached_host_mass(solver, z):
    """Reuse the native virial conversion across stripping coefficients.

    Parameters
    ----------
    solver : object
        Native solver with an uncached host history anchored at M200(z=0).
    z : array_like
        Dimensionless redshift grid.

    Returns
    -------
    float or numpy.ndarray
        Virial host mass in the native mass unit and input shape. A caller
        may modify its returned array without changing the cached history.
    """
    _state(solver)
    epoch = np.asarray(z, dtype=float)
    key = (epoch.shape, epoch.tobytes())
    cache = solver._picard_host_cache
    if key not in cache:
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[key] = np.asarray(solver._Mzvir_uncached(z)).copy()
    value = cache[key]
    return float(value) if value.ndim == 0 else value.copy()


def _coefficients(solver, z_acc, z_final, count):
    """Sample the tidal coefficients from accretion to the final epoch.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    z_acc, z_final : float
        Accretion and final redshifts, with z_acc >= z_final.
    count : int
        Number of samples along the path.

    Returns
    -------
    tuple of numpy.ndarray
        Redshift, rate per unit redshift, exponent and log host mass, each
        of length count.

    Raises
    ------
    ValueError
        If the rates are non-finite, the host mass is nonpositive or Phi < 0.
    """
    _state(solver)
    key = (float(z_acc), float(z_final), count)
    cached = solver._picard_coefficients.get(key)
    if cached is not None:
        return cached
    z = np.linspace(z_acc, z_final, count)
    phi = np.broadcast_to(np.asarray(solver.Phi(z), dtype=float), z.shape)
    zeta = np.broadcast_to(np.asarray(solver.zetaMz(z), dtype=float), z.shape)
    host = np.broadcast_to(np.asarray(solver.Mzvir(z), dtype=float), z.shape)
    if (
        not np.all(np.isfinite(phi))
        or not np.all(np.isfinite(zeta))
        or not np.all(np.isfinite(host))
        or np.any(host <= 0)
        or np.any(phi < 0)
    ):
        raise ValueError(
            "Tidal stripping requires finite, positive host mass and nonnegative rate."
        )
    result = z, phi, zeta, np.log(host)
    # Bound retained coefficient paths; SI catalogs may contain many slices.
    if len(solver._picard_coefficients) >= 512:
        solver._picard_coefficients.pop(next(iter(solver._picard_coefficients)))
    solver._picard_coefficients[key] = result
    return result


def _iterate(log_ma, coefficients, n_iterations):
    """Integrate the nonlinear log-mass equation on one descending redshift grid.

    Parameters
    ----------
    log_ma : 1-D array_like
        Natural log of each accreted mass in the unit of the host mass.
    coefficients : tuple of numpy.ndarray
        Redshift, Phi, zeta and log host mass on the same 1-D path.
    n_iterations : int
        Number of Picard updates before estimating convergence.

    Returns
    -------
    z : numpy.ndarray
        Redshifts from accretion toward observation.
    delta : numpy.ndarray
        Log bound-to-accreted mass ratio, shape (len(z), len(log_ma)).
    estimate : numpy.ndarray
        Estimated remaining iteration error for each mass. It is infinite
        where the local contraction estimate reaches unity.

    Notes
    -----
    Each update integrates Phi * exp(zeta * (ln(ma/Mhost) + delta))
    by the cumulative trapezoidal rule, with delta(z_acc) = 0. A further
    update estimates the iteration error on this fixed grid.
    """
    z, phi, zeta, log_host = coefficients
    log_ma = np.asarray(log_ma)
    initial = cumulative_trapezoid(phi, x=z, initial=0.0)
    delta = np.broadcast_to(initial[:, None], (len(z), log_ma.size)).copy()
    log_ratio = log_ma[None, :] - log_host[:, None]

    def update(current):
        """Evaluate one Picard update over the mass grid.

        Parameters
        ----------
        current : numpy.ndarray
            Previous log mass change, shape (n_redshift, n_mass).

        Returns
        -------
        numpy.ndarray
            Updated log mass change anchored at zero at accretion.
        """
        rhs = phi[:, None] * np.exp(zeta[:, None] * (log_ratio + current))
        return cumulative_trapezoid(rhs, x=z, axis=0, initial=0.0)

    for _ in range(n_iterations):
        delta = update(delta)
    next_delta = update(delta)
    residual = np.max(np.abs(next_delta - delta), axis=0)
    # A conservative local contraction estimate from the integrated rate.
    contraction = np.max(np.abs(zeta)) * np.max(np.abs(next_delta), axis=0)
    estimate = np.full(residual.shape, np.inf)
    valid = contraction < 1.0
    estimate[valid] = residual[valid] / (1 - contraction[valid])
    return z, delta, estimate


def direct_log_mass(solver, ma, za, z, *, rtol=1e-11, atol=1e-12):
    """Integrate the native stripping law directly for a guarded Picard fallback.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma : array_like
        Positive virial mass at accretion, in the same unit as Mzvir.
    za : float
        Accretion redshift.
    z : float or 1-D array_like
        Output redshift(s) in non-increasing order from za toward the present.
    rtol, atol : float, optional
        Local relative and absolute DOP853 tolerances in natural log mass.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit. A redshift vector prepends a time
        axis to the input mass shape.

    Raises
    ------
    ValueError
        If coordinates are non-finite, masses nonpositive, or time runs backward.
    RuntimeError
        If the integrator fails or returns non-finite log masses.
    """
    mass = _real(ma, "ma")
    redshifts = _real(z, "z")
    za = float(_real(za, "za"))
    if np.any(mass <= 0) or za <= -1 or np.any(redshifts <= -1) or np.any(redshifts > za):
        raise ValueError("Require ma>0 and -1<z<=za.")
    scalar_z = redshifts.ndim == 0
    times = np.atleast_1d(redshifts)
    if times.ndim != 1 or np.any(np.diff(times) > 0):
        raise ValueError("History redshifts must be one-dimensional and non-increasing.")
    if mass.size == 0 or times.size == 0:
        return np.empty(mass.shape if scalar_z else times.shape + mass.shape)
    if times[-1] == za:
        return mass.copy() if scalar_z else np.broadcast_to(mass, times.shape + mass.shape).copy()

    def rhs(redshift, log_mass):
        """Evaluate the derivative of log bound mass with respect to redshift.

        Parameters
        ----------
        redshift : float
            Dimensionless redshift at the ODE stage.
        log_mass : numpy.ndarray
            Log masses in the same unit as the native host mass.

        Returns
        -------
        numpy.ndarray
            Phi times (m/Mvir)**zeta for each mass.
        """
        return solver.Phi(redshift) * np.exp(
            solver.zetaMz(redshift) * (log_mass - np.log(solver.Mzvir(redshift)))
        )

    # Integrate from the declared accretion time even if it is not in t_eval.
    unique, reverse = np.unique(times, return_inverse=True)
    result = solve_ivp(
        rhs,
        (za, float(times[-1])),
        np.log(mass.ravel()),
        t_eval=unique[::-1],
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success or np.any(~np.isfinite(result.y)):
        raise RuntimeError("Log-mass ODE failed: " + result.message)
    answer = np.exp(result.y.T[::-1][reverse]).reshape(times.shape + mass.shape)
    answer = answer[0] if scalar_z else answer
    return float(answer) if answer.ndim == 0 else answer


def _fallback(solver, ma, za, z, reason):
    """Record why Picard requests a direct integration and return its mass.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma : array_like
        Positive virial mass at accretion, in the same unit as Mzvir.
    za : float
        Accretion redshift.
    z : float or 1-D array_like
        Output redshift(s) in non-increasing order from za toward the present.
    reason : str
        Domain or iteration condition triggering the alternative integration.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit. A redshift vector prepends a time
        axis to the input mass shape.
    """
    _state(solver)
    event = dict(
        reason=reason,
        mass_count=int(np.size(ma)),
        z_acc=float(za),
        z_final=float(np.min(np.atleast_1d(z))),
    )
    solver._picard_events.append(event)
    warnings.warn(
        f"Picard fallback to log-mass DOP853: {reason}; {event['mass_count']} masses.",
        PicardFallbackWarning,
        stacklevel=3,
    )
    return direct_log_mass(solver, ma, za, z)


def history_mass(
    solver,
    ma,
    za,
    z,
    *,
    n_integration=100,
    n_iterations=3,
    convergence_tolerance=8e-4,
    n_mass_nodes=32,
):
    """Trace the bound mass from accretion to each requested epoch.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma : array_like
        Positive virial mass at accretion, in the same unit as Mzvir.
    za : float
        Accretion redshift.
    z : float or 1-D array_like
        Output redshift(s) in non-increasing order from za toward the present.
    n_integration : int, optional
        Number of uniformly spaced redshift samples, including both ends.
    n_iterations : int, optional
        Number of nonlinear updates after the mass-independent first guess.
    convergence_tolerance : float, optional
        Threshold on the estimated remaining change in natural log mass.
        Quadrature and interpolation errors must be checked independently.
    n_mass_nodes : int, optional
        Target number of nodes for interpolating large log-mass arrays. The
        grid is refined to keep spacing at or below 0.25 dex.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit. A redshift vector prepends a time
        axis to the input mass shape.

    Raises
    ------
    ValueError
        If input coordinates, time ordering, or grid controls are invalid.

    Notes
    -----
    The numerical domain is 0 <= z <= za <= 7 with accretion ratios
    from 1e-24 to 1e3. Outside it, or on failure of the iteration check,
    DOP853 supplies the affected histories and emits PicardFallbackWarning.
    """
    mass = _real(ma, "ma")
    times = _real(z, "z")
    za = float(_real(za, "za"))
    scalar_z = times.ndim == 0
    times = np.atleast_1d(times)
    count = _integer(n_integration, "n_integration", 2)
    iterations = _integer(n_iterations, "n_iterations", 1)
    mass_nodes = _integer(n_mass_nodes, "n_mass_nodes", 2)
    if np.any(mass <= 0) or za <= -1 or np.any(times <= -1) or np.any(times > za):
        raise ValueError("Require ma>0 and -1<z<=za.")
    if times.ndim != 1 or np.any(np.diff(times) > 0):
        raise ValueError("History redshifts must be one-dimensional and non-increasing.")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be finite and positive.")
    _state(solver)
    if mass.size == 0 or times.size == 0:
        return np.empty(mass.shape if scalar_z else times.shape + mass.shape)
    if times[-1] == za:
        return mass.copy() if scalar_z else np.broadcast_to(mass, times.shape + mass.shape).copy()
    ratio = np.log10(mass / solver.Mzvir(za))
    if (
        za > 7.0 + 32 * np.finfo(float).eps
        or times[-1] < 0
        or np.any(ratio < -24)
        or np.any(ratio > 3.0 + 32 * np.finfo(float).eps)
    ):
        return _fallback(solver, mass, za, z, "outside validated redshift or mass-ratio domain")
    log_mass = np.log(mass.ravel())
    # Wider requests receive enough mass nodes to keep the interpolation
    # spacing within the validated quarter-dex bound.
    mass_nodes = max(mass_nodes, int(np.ceil(np.ptp(log_mass) / (np.log(10.0) * 0.25))) + 1)
    use_mass_grid = log_mass.size > mass_nodes and np.ptp(log_mass) > 0
    nodes = (
        np.linspace(np.min(log_mass), np.max(log_mass), mass_nodes) if use_mass_grid else log_mass
    )
    path = _coefficients(solver, za, float(times[-1]), count)
    grid, delta, error = _iterate(nodes, path, iterations)
    # Vectorized linear interpolation of the entire requested trajectory.
    if np.array_equal(grid, times):
        requested = delta
    else:
        positions = (za - times) / (za - times[-1]) * (count - 1)
        left = np.minimum(positions.astype(int), count - 2)
        fraction = positions - left
        requested = delta[left] * (1 - fraction[:, None]) + delta[left + 1] * fraction[:, None]
    if use_mass_grid:
        coordinate = (log_mass - nodes[0]) / (nodes[-1] - nodes[0]) * (mass_nodes - 1)
        left = np.minimum(coordinate.astype(int), mass_nodes - 2)
        fraction = coordinate - left
        requested = (
            requested[:, left] * (1 - fraction[None, :])
            + requested[:, left + 1] * fraction[None, :]
        )
        error = np.maximum(error[left], error[left + 1])
    output = (mass.ravel()[None, :] * np.exp(requested)).reshape(times.shape + mass.shape)
    bad = error > convergence_tolerance
    if np.any(bad):
        flattened = output.reshape(len(times), -1)
        flattened[:, bad] = _fallback(
            solver,
            mass.ravel()[bad],
            za,
            times,
            "Picard iteration convergence estimate exceeded tolerance",
        )
    output = output[0] if scalar_z else output
    return float(output) if output.ndim == 0 else output
