"""Physics unit tests for sashimi_si.

These tests pin the equations of Yang et al. (2023) [arXiv:2305.16176] and the
Carroll-Press-Turner growth factor directly, by comparing the implementation against
independently written references (finite differences, brute-force quadrature, exact NFW
constants). They deliberately do NOT compare against the output of an older commit: a
physics fix is *supposed* to change the numbers, so an equivalence-with-the-past test
can only ever stand in the way of correcting the code.

Run with:  pytest test_sashimi.py
"""
import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import brentq

import sashimi_si


SIGMA0_M = 147.1  # cm^2/g, Milky Way analog of Yang et al. (2023)
W = 24.33         # km/s


@pytest.fixture(scope="module")
def pm():
    return sashimi_si.SIDM_parametric_model(sigma0_m=SIGMA0_M, w=W)


@pytest.fixture(scope="module")
def cosmo():
    return sashimi_si.cosmology()


@pytest.fixture(scope="module")
def tau():
    """Normalized time tau = (t - t_f)/t_c, within the calibrated range."""
    return np.linspace(1.e-4, 1.1, 501)


# --------------------------------------------------------------------------------------
# Eq. (2.4) of Yang et al. (2023), written out independently of the implementation.
# --------------------------------------------------------------------------------------

def Vmax_fit(t):
    """Vmax(tau)/Vmax_0, Eq. (2.4)."""
    return 1. + 0.1777*t - 4.399*t**3 + 16.66*t**4 - 18.87*t**5 + 9.077*t**7 - 2.436*t**9


def rmax_fit(t):
    """rmax(tau)/rmax_0, Eq. (2.4)."""
    return 1. + 0.007623*t - 0.7200*t**2 + 0.3376*t**3 - 0.1375*t**4


def dVmax_fit(t):
    """d/dtau of Vmax_fit, i.e. the polynomial just below Eq. (3.3)."""
    return 0.1777 - 3*4.399*t**2 + 4*16.66*t**3 - 5*18.87*t**4 + 7*9.077*t**6 - 9*2.436*t**8


def drmax_fit(t):
    """d/dtau of rmax_fit, i.e. the polynomial just below Eq. (3.3)."""
    return 0.007623 - 2*0.7200*t + 3*0.3376*t**2 - 4*0.1375*t**3


class TestParametricModelDerivatives:
    """The equations just below Eq. (3.3) must be the exact derivative of Eq. (2.4)."""

    def test_dVmax_matches_finite_difference_of_eq24(self, pm, tau):
        h = 1.e-6
        expected = (Vmax_fit(tau + h) - Vmax_fit(tau - h)) / (2 * h)
        assert np.allclose(pm.dVmaxSIDMdtt(tau, np.ones_like(tau)), expected,
                           rtol=0., atol=1.e-6)

    def test_drmax_matches_finite_difference_of_eq24(self, pm, tau):
        h = 1.e-6
        expected = (rmax_fit(tau + h) - rmax_fit(tau - h)) / (2 * h)
        assert np.allclose(pm.drmaxSIDMdtt(tau, np.ones_like(tau)), expected,
                           rtol=0., atol=1.e-6)

    def test_numexpr_variants_agree_with_the_numpy_variants(self, pm, tau):
        ones = np.ones_like(tau)
        assert np.allclose(pm.dVmaxSIDMdtt_numexpr_optimized(tau, ones),
                           pm.dVmaxSIDMdtt(tau, ones), rtol=1.e-12, atol=0.)
        assert np.allclose(pm.drmaxSIDMdtt_numexpr_optimized(tau, ones),
                           pm.drmaxSIDMdtt(tau, ones), rtol=1.e-12, atol=0.)

    def test_derivatives_are_linear_in_the_cdm_normalization(self, pm, tau):
        scale = np.linspace(2., 50., tau.size)
        assert np.allclose(pm.dVmaxSIDMdtt(tau, scale),
                           scale * pm.dVmaxSIDMdtt(tau, np.ones_like(tau)))

    def test_evolution_is_frozen_beyond_the_threshold(self, pm):
        """The fit is calibrated only up to tau ~ 1, so evolution must stop above tt_th."""
        beyond = np.array([pm.tt_th + 1.e-6, 2., 10.])
        ones = np.ones_like(beyond)
        assert np.all(pm.dVmaxSIDMdtt(beyond, ones) == 0.)
        assert np.all(pm.drmaxSIDMdtt(beyond, ones) == 0.)
        assert np.all(pm.dVmaxSIDMdtt_numexpr_optimized(beyond, ones) == 0.)
        assert np.all(pm.drmaxSIDMdtt_numexpr_optimized(beyond, ones) == 0.)


class TestParametricModelInversion:

    def test_get_Vmax0_inverts_eq24(self, pm, tau):
        assert np.allclose(pm.get_Vmax0(Vmax_fit(tau), tau), 1., rtol=1.e-12, atol=0.)

    def test_get_rmax0_inverts_eq24(self, pm, tau):
        assert np.allclose(pm.get_rmax0(rmax_fit(tau), tau), 1., rtol=1.e-12, atol=0.)

    def test_eq23_reduces_to_the_initial_nfw_at_tau_zero(self, pm):
        """At tau = 0 the SIDM halo is the initial NFW halo, with no core yet."""
        assert pm.get_rhos(1., 0.) == pytest.approx(1., abs=1.e-12)
        assert pm.get_rs(1., 0.) == pytest.approx(1., abs=1.e-12)
        assert pm.get_rc(1., 0.) == pytest.approx(0., abs=1.e-12)


class TestIntegralApproach:
    """Eq. (3.3): the SIDM correction is normalized by the CDM track at the running time
    t of the integral, not by its value frozen at the accretion time t_f."""

    @staticmethod
    def _synthetic_cdm_history(pm, n_t=801):
        """A tidally declining CDM track, shaped as master_function expects:
        (N_herm, N_t, N_ma) for Vmax/rmax, (N_t, N_ma) for t, (N_ma,) for t_f."""
        Gyr, km, s, kpc = pm.Gyr, pm.km, pm.s, pm.kpc
        t_f = np.array([2.0, 2.0]) * Gyr
        t = np.linspace(2.0, 13.8, n_t)[:, None] * Gyr * np.ones((1, 2))
        frac = (t - t_f) / (13.8 * Gyr - t_f)          # 0 at accretion -> 1 today
        Vmax = (28. - 14. * frac) * km / s             # 28 -> 14 km/s
        rmax = (3.0 - 1.5 * frac) * kpc                # 3.0 -> 1.5 kpc
        return Vmax[None, ...], rmax[None, ...], t, t_f

    @staticmethod
    def _eq33_reference(pm, Vmax_CDM, rmax_CDM, t, t_f, freeze_at_tf=False):
        """Independent evaluation of Eq. (3.3), using the polynomials defined at the top
        of this file rather than the ones inside sashimi_si."""
        t_c = pm.t_collapse(pm.sigma_eff_m(Vmax_CDM), rmax_CDM, Vmax_CDM)
        tau = (t - t_f) / t_c
        norm_V = Vmax_CDM[:, :1] if freeze_at_tf else Vmax_CDM
        norm_r = rmax_CDM[:, :1] if freeze_at_tf else rmax_CDM
        dV = np.where(tau <= pm.tt_th, dVmax_fit(tau), 0.) * norm_V / t_c
        dr = np.where(tau <= pm.tt_th, drmax_fit(tau), 0.) * norm_r / t_c
        x = t * np.ones((len(Vmax_CDM), 1, 1))
        return (Vmax_CDM[:, -1] + integrate.simpson(dV, x=x, axis=1),
                rmax_CDM[:, -1] + integrate.simpson(dr, x=x, axis=1))

    def test_master_function_matches_an_independent_eq33_reference(self, pm):
        args = self._synthetic_cdm_history(pm)
        Vmax_SIDM, rmax_SIDM, _, _, _ = pm.master_function(*args)
        V_ref, r_ref = self._eq33_reference(pm, *args)
        assert np.allclose(Vmax_SIDM, V_ref, rtol=1.e-10, atol=0.)
        assert np.allclose(rmax_SIDM, r_ref, rtol=1.e-10, atol=0.)

    def test_normalizing_at_t_f_would_give_a_different_answer(self, pm):
        """Guards the test above. On a declining CDM track the two normalizations must
        genuinely disagree, otherwise that test would still pass after a regression."""
        args = self._synthetic_cdm_history(pm)
        V_running, r_running = self._eq33_reference(pm, *args, freeze_at_tf=False)
        V_frozen, r_frozen = self._eq33_reference(pm, *args, freeze_at_tf=True)
        assert np.all(np.abs(V_frozen / V_running - 1.) > 1.e-3)
        assert np.all(np.abs(r_frozen / r_running - 1.) > 1.e-3)


class TestCrossSection:

    def test_analytic_sigma_eff_matches_brute_force_quadrature(self, pm):
        """The analytic sigma_eff must reproduce a direct evaluation of Eq. (1.1)."""
        km, s, cm, gram = pm.km, pm.s, pm.cm, pm.gram
        sigma0_m, w = SIGMA0_M * cm**2 / gram, W * km / s
        f_ana = pm.sigma_eff_m_interpolate_analytical(sigma0_m, w)

        for V in [0.1, 1., 10., 24.33, 100., 300.]:
            Vmax = V * km / s
            veff = 0.64 * Vmax
            v = np.linspace(1.e-8 * veff, 40. * veff, 20000)
            costheta = np.linspace(-1., 1., 2000)
            integrand = (pm.dsigmadcostheta(sigma0_m, w, v[:, None], costheta[None, :])
                         * v[:, None]**7 * (1. - costheta[None, :]**2)
                         * np.exp(-v[:, None]**2 / (4. * veff**2)))
            inner = integrate.simpson(integrand, x=costheta, axis=-1)
            reference = integrate.simpson(inner, x=v) / (512. * veff**8)
            assert f_ana(Vmax) == pytest.approx(reference, rel=1.e-2)

    def test_sigma_eff_approaches_sigma0_in_the_contact_limit(self, pm):
        """For v << w the scattering is isotropic and contact-like, so sigma_eff -> sigma0."""
        km, s, cm, gram = pm.km, pm.s, pm.cm, pm.gram
        f_ana = pm.sigma_eff_m_interpolate_analytical(SIGMA0_M * cm**2 / gram, W * km / s)
        assert f_ana(1.e-3 * km / s) / (cm**2 / gram) == pytest.approx(SIGMA0_M, rel=1.e-3)

    def test_subhalo_properties_shares_sigma_eff_with_its_parametric_model(self):
        """The t_c behind the returned tt_ratio must be the t_c driving the evolution."""
        sh = sashimi_si.subhalo_properties(sigma0_m=SIGMA0_M, w=W)
        assert sh.sigma_eff_m is sh.param_model.sigma_eff_m


class TestCosmology:

    def test_growthD_is_normalized_to_unity_today(self, cosmo):
        assert cosmo.growthD(0.) == pytest.approx(1., abs=1.e-12)

    def test_dDdz_is_the_derivative_of_growthD(self, cosmo):
        """Pins the absence of a spurious h^-2 in Omega_L(z). With one, dDdz is off by
        57% at z = 0 and the Correa+15 accretion-history exponent flips sign."""
        h = 1.e-6
        for z in [0., 0.5, 1., 2., 4., 7., 10.]:
            expected = (cosmo.growthD(z + h) - cosmo.growthD(z - h)) / (2 * h)
            assert cosmo.dDdz(z) == pytest.approx(expected, rel=1.e-5)

    def test_dark_energy_density_parameter_is_recovered_today(self, cosmo):
        """Omega_L(z=0) must be Omega_L: a stray factor of h would break this."""
        assert cosmo.OmegaL / (cosmo.OmegaL + cosmo.OmegaM) == pytest.approx(cosmo.OmegaL,
                                                                             rel=1.e-12)


class TestNFWConstants:
    """The NFW structure constants hard-coded in the code, against their exact values."""

    @staticmethod
    def _c_max():
        """rmax/rs, i.e. the c maximizing [ln(1+c) - c/(1+c)]/c."""
        def stationarity(c):
            return c**2 / (1 + c)**2 - (np.log(1 + c) - c / (1 + c))
        return brentq(stationarity, 1., 5.)

    def test_rmax_over_rs(self):
        assert self._c_max() == pytest.approx(2.1626, abs=1.e-4)

    def test_vmax_and_rhos_normalizations(self):
        c = self._c_max()
        mu = (np.log(1 + c) - c / (1 + c)) / c
        # Vmax / (rs sqrt(G rho_s)), used in t_collapse
        assert np.sqrt(4 * np.pi * mu) == pytest.approx(1.648, abs=1.e-3)
        # 4 pi G rs^2 rho_s / Vmax^2, used to rebuild rho_s from Vmax and rs
        assert 1. / mu == pytest.approx(4.625, abs=1.e-3)


class TestEndToEnd:

    CALC_INPUT = dict(M0=1.e12, redshift=0., M0_at_redshift=True,
                      dz=0.2, N_herm=3, zmax=4., logmamin=9, N_ma=30)

    OUT_NAMES = ['ma200', 'z_acc', 'rsCDM_acc', 'rhosCDM_acc', 'rmaxCDM_acc', 'VmaxCDM_acc',
                 'rsSIDM_acc', 'rhosSIDM_acc', 'rcSIDM_acc', 'rmaxSIDM_acc', 'VmaxSIDM_acc',
                 'm_z0', 'rsCDM_z0', 'rhosCDM_z0', 'rmaxCDM_z0', 'VmaxCDM_z0', 'rsSIDM_z0',
                 'rhosSIDM_z0', 'rcSIDM_z0', 'rmaxSIDM_z0', 'VmaxSIDM_z0', 'ctCDM_z0',
                 'tt_ratio', 'weightCDM', 'weightSIDM', 'surviveCDM', 'surviveSIDM']

    def _run(self, **overrides):
        kwargs = dict(sigma0_m=SIGMA0_M, w=W)
        kwargs.update(overrides)
        out = sashimi_si.subhalo_properties(**kwargs).subhalo_properties_calc(**self.CALC_INPUT)
        assert len(out) == len(self.OUT_NAMES)
        return dict(zip(self.OUT_NAMES, out))

    def test_catalog_is_finite_and_positive_where_subhaloes_survive(self):
        cat = self._run()
        alive = cat['weightCDM'] > 0.
        assert alive.any()
        for key in ['m_z0', 'VmaxCDM_z0', 'rmaxCDM_z0', 'rsCDM_z0', 'rhosCDM_z0']:
            values = np.asarray(cat[key])[alive]
            assert np.all(np.isfinite(values))
            assert np.all(values > 0.)

    def test_sidm_reduces_to_cdm_when_the_cross_section_vanishes(self):
        """With sigma_0 -> 0 the collapse time diverges, so tau -> 0 and every SIDM
        quantity must fall back onto its CDM counterpart, with a vanishing core."""
        cat = self._run(sigma0_m=1.e-8)
        alive = cat['weightSIDM'] > 0.
        assert alive.any()
        for sidm, cdm in [('VmaxSIDM_z0', 'VmaxCDM_z0'), ('rmaxSIDM_z0', 'rmaxCDM_z0'),
                          ('rsSIDM_z0', 'rsCDM_z0'), ('rhosSIDM_z0', 'rhosCDM_z0')]:
            assert np.allclose(np.asarray(cat[sidm])[alive], np.asarray(cat[cdm])[alive],
                               rtol=1.e-3, atol=0.)
        rs = np.asarray(cat['rsSIDM_z0'])[alive]
        assert np.all(np.abs(np.asarray(cat['rcSIDM_z0'])[alive]) < 1.e-3 * rs)
