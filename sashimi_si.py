import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import special
from scipy.integrate import odeint, cumulative_trapezoid
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
import numexpr as ne
import warnings
import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)




class units_and_constants:

    
    def __init__(self):
        self.Mpc      = 1.
        self.kpc      = self.Mpc/1000.
        self.pc       = self.kpc/1000.
        self.cm       = self.pc/3.086e18
        self.km       = 1.e5*self.cm
        self.s        = 1.
        self.yr       = 3.15576e7*self.s
        self.Gyr      = 1.e9*self.yr
        self.Msun     = 1.
        self.gram     = self.Msun/1.988e33
        self.c        = 2.9979e10*self.cm/self.s
        self.G        = 6.6742e-8*self.cm**3/self.gram/self.s**2



        
class cosmology(units_and_constants):
    
    
    def __init__(self):
        units_and_constants.__init__(self)
        self.OmegaB        = 0.049
        self.OmegaM        = 0.315
        self.OmegaC        = self.OmegaM-self.OmegaB
        self.OmegaL        = 1.-self.OmegaM
        self.h             = 0.674
        self.H0            = self.h*100*self.km/self.s/self.Mpc 
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G)


    def g(self, z):
        return self.OmegaM*(1.+z)**3+self.OmegaL


    def Hubble(self, z):
        return self.H0*np.sqrt(self.OmegaM*(1.+z)**3+self.OmegaL)

    
    def rhocrit(self, z):
        return 3.*self.Hubble(z)**2/(np.pi*8.0*self.G)


    def growthD(self, z):
        Omega_Lz = self.OmegaL/(self.OmegaL+self.OmegaM*(1.+z)**3)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1.+Omega_Mz/2.0)*(1.+Omega_Lz/70.0)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1.+self.OmegaM/2.0)*(1.+self.OmegaL/70.0)
        return (Omega_Mz/self.OmegaM)*(phi0/phiz)/(1.+z)

        
    def dDdz(self, z):
        def dOdz(z):
            return -self.OmegaL*3*self.OmegaM*(1+z)**2*(self.OmegaL+self.OmegaM*(1+z)**3.)**-2
        Omega_Lz = self.OmegaL*pow(self.OmegaL+self.OmegaM*pow(self.h,-2)*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1+Omega_Mz/2.)*(1+Omega_Lz/70.)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1+self.OmegaM/2.)*(1+self.OmegaL/70.)
        dphidz   = dOdz(z)*(-4./7.*Omega_Mz**(-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.+1./70.-3./2.)
        return (phi0/self.OmegaM)*(-dOdz(z)/(phiz*(1+z))-Omega_Mz*(dphidz*(1+z)+phiz)/phiz**2/(1+z)**2)

        


class halo_model(cosmology):

    
    def __init__(self):
        cosmology.__init__(self)

        
    def xi(self, M):
        return (M/((1.e10*self.Msun)/self.h))**-1

    
    def sigmaMz(self, M, z):    
        """ Ludlow et al. (2016) """
        return self.growthD(z)*22.26*self.xi(M)**0.292/(1.+1.53*self.xi(M)**0.275+3.36*self.xi(M)**0.198)


    def deltac_func(self, z):
        return 1.686/self.growthD(z)

    
    def s_func(self, M):
        return self.sigmaMz(M,0)**2


    def fc(self, x):
        return np.log(1+x)-x*pow(1+x,-1)

    
    def Delc(self, x):
        return 18.*np.pi**2+82.*x-39.*x**2

    
    def conc200(self,M200,z): 
        """ Correa et al. (2015) """
        alpha_cMz_1 = 1.7543-0.2766*(1.+z)+0.02039*(1.+z)**2
        beta_cMz_1  = 0.2753+0.00351*(1.+z)-0.3038*(1.+z)**0.0269
        gamma_cMz_1 = -0.01537+0.02102*(1.+z)**-0.1475
        c_Mz_1      = np.power(10.,alpha_cMz_1+beta_cMz_1*np.log10(M200/self.Msun) \
                               *(1+gamma_cMz_1*np.log10(M200/self.Msun)**2))
        alpha_cMz_2 = 1.3081-0.1078*(1.+z)+0.00398*(1.+z)**2
        beta_cMz_2  = 0.0223-0.0944*(1.+z)**-0.3907
        c_Mz_2      = pow(10,alpha_cMz_2+beta_cMz_2*np.log10(M200/self.Msun))
        return np.where(z<=4.,c_Mz_1,c_Mz_2)

    
    def Mvir_from_M200(self, M200, z):
        gz    = self.g(z)
        c200  = self.conc200(M200,z)
        r200  = (3.0*M200/(4*np.pi*200*self.rhocrit0*gz))**(1./3.)
        rs    = r200/c200
        fc200 = self.fc(c200)
        rhos  = M200/(4*np.pi*rs**3*fc200)
        Dc    = self.Delc(self.OmegaM*(1.+z)**3/self.g(z)-1.)
        rvir  = optimize.fsolve(lambda r: 3.*(rs/r)**3*self.fc(r/rs)*rhos-Dc*self.rhocrit0*gz,r200)
        Mvir  = 4*np.pi*rs**3*rhos*self.fc(rvir/rs)
        return Mvir

    
    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM*(1.+z)**3/self.g(z)
        def ffunc(x):
            return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
        def xfunc(f):
            p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
            return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
        return self.Delc(Oz-1)/200.0*M200 \
            *np.power(self.conc200(M200,z) \
            *xfunc(self.Delc(Oz-1)/200.0*ffunc(1.0/self.conc200(M200,z))),-3.0)

    
    def Mzi(self, M0, z):
        a   = 1.686*np.sqrt(2./np.pi)*self.dDdz(0)+1.
        zf  = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q   = 4.137/zf**0.9476
        fM0 = (self.sigmaMz(M0/q,0)**2-self.sigmaMz(M0,0)**2)**-0.5
        return M0*np.power(1.+z,a*fM0)*np.exp(-fM0*z)

    
    def Mzzi(self, M0, z, zi):
        Mzi0  = self.Mzi(M0,zi)
        zf    = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q     = 4.137/zf**0.9476
        fMzi  = (self.sigmaMz(Mzi0/q,zi)**2-self.sigmaMz(Mzi0,zi)**2)**-0.5
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)/self.growthD(zi)**2*self.dDdz(zi)+1.)
        beta  = -fMzi
        return Mzi0*np.power(1.+z-zi,alpha)*np.exp(beta*(z-zi))

    
    def dMdz(self, M0, z, zi):
        Mzi0    = self.Mzi(M0,zi)
        zf      = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q       = 4.137/zf**0.9476
        fMzi    = (self.sigmaMz(Mzi0/q,zi)**2-self.sigmaMz(Mzi0,zi)**2)**-0.5
        alpha   = fMzi*(1.686*np.sqrt(2./np.pi)/self.growthD(zi)**2*self.dDdz(zi)+1)
        beta    = -fMzi
        Mzzidef = Mzi0*(1.+z-zi)**alpha*np.exp(beta*(z-zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef,z)
        return (beta+alpha/(1.+z-zi))*Mzzivir

    
    def dsdm(self,M,z):  
        """ Ludlow et al. (2016) """
        s         = self.sigmaMz(M,z)**2
        dsdsigma  = 2.*self.sigmaMz(M,z)
        dxidm     = -1.e10*self.Msun/self.h/M**2
        dsigmadxi = self.sigmaMz(M,z)*(0.292/self.xi(M)-(0.275*1.53*self.xi(M)**-0.725+0.198*3.36* \
            self.xi(M)**-0.802)/(1.+1.53*self.xi(M)**0.275+3.36*self.xi(M)**0.198))
        return dsdsigma*dsigmadxi*dxidm

    
    def dlogSdlogM(self,M,z):  
        """ Ludlow et al. (2016) """
        s         = self.sigmaMz(M,z)**2
        dsdsigma  = 2.*self.sigmaMz(M,z)
        dxidm     = -1.e10*self.Msun/self.h/M**2
        dsigmadxi = self.sigmaMz(M,z)*(0.292*pow(self.xi(M),-1)-(0.275*1.53*pow(self.xi(M),-0.725)+0.198*3.36* \
            pow(self.xi(M),-0.802))*pow(1.+1.53*pow(self.xi(M),0.275)+3.36*pow(self.xi(M),0.198),-1))
        return (M/s)*dsdsigma*dsigmadxi*dxidm
        
       

class SIDM_cross_section(units_and_constants):
    """ Calculate the effective cross section of SIDM.

    Notes
    ---
    - v is the relative velocity between the two initial particles.
    - w is defined as w = m_phi * c / m_chi, where m_phi is the mass of the mediator, c is the speed of light, and m_chi is the mass of the dark matter particle.

    References
    ---
        - Yang et al. (2023), https://arxiv.org/abs/2305.16176
        - Yang et al. (2022), https://arxiv.org/abs/2205.03392
    """

    def __init__(self):
        units_and_constants.__init__(self)


    def dsigmadcostheta(self, sigma0_m, w, v, costheta):
        """ Returns Eq. (1.2) of Yang et al. (2023) divided by m.
        Eq. (1.2) is given by

        $$
        \frac{d\sigma}{d\cos\theta} = \frac{\sigma_0 w^4}{2(w^2+v^2\sin^2(\theta/2))^2}
        $$

        Parameters
        ----------
        sigma0_m : float
            The value of sigma_0 / m.
        w : float
            The value of w in unit of km/s.
        v : float
            The value of v in unit of km/s.

        Returns
        -------
        dsigmadcostheta : float
            The value of d\sigma / d\cos\theta divided by m.

        """
        return sigma0_m*w**4/2./(w**2+v**2/2.*(1.-costheta))**2

    
    def sigma_total(self, sigma0_m, w, v):
        """ Returns the total cross section of SIDM divided by m. See Yang et al. (2022) [arXiv:2205.03392]
        
        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        v : float
            The value of v in unit of km/s.

        Returns
        ---
        sigma_total_m : float
            The value of the total cross section of SIDM divided by m.
        """
        return sigma0_m/(1.+v**2/w**2)**2
    
    
    def sigma_viscosity(self, sigma0_m, w, v):
        """ Returns the viscosity cross section of SIDM divided by m, given by Eq. (2.7) of Yang et al. (2022) [arXiv:2205.03392].

        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.

        Returns
        ---
        sigma_V_m : float
            The value of the viscosity cross section of SIDM divided by m.
        """
        sigma_V_m  = 6.*sigma0_m*w**4/v**4
        sigma_V_m *= (2.*w**2/v**2+1.)*np.log(1.+v**2/w**2)-2.
        v0 = 1.e-2*w
        sigma_V_m[v<v0]  = 6.*sigma0_m*w**4/v0**4
        sigma_V_m[v<v0] *= (2.*w**2/v0**2+1.)*np.log(1.+v0**2/w**2)-2.
        return sigma_V_m

    
    def sigma_eff_m_interpolate(self, sigma0_m, w):
        """ Returns the interpolation function of the effective cross section of SIDM divided by m.
        The effective cross section is defined by Eq. (1.1) of Yang et al. (2023) [arXiv:2305.16176]:

        $$
        \sigma_{eff} = \frac{1}{512 \nu_{eff}^8} \int v^2 dv \int d\cos\theta \frac{d\sigma}{d\cos\theta} v^5 \sin^2\theta \exp(-v^2/4\nu_{eff}^2)
        $$

        where $\nu_{eff} = 0.64 V_{max}$ is a characteristic velocity dispersion of dark matter particles in the halo.

        Parameters
        ----------
        sigma0_m : float
            The value of sigma_0 / m.
        w : float
            The value of w.
        
        Returns
        -------
        sigma_eff_m_interpolate : scipy.interpolate.interp1d
            The interpolation function of the effective cross section of SIDM divided by m.
        """
        Vmax_dummy = np.logspace(-5.,3.,1000)*self.km/self.s

        v = np.linspace(0.,30.*Vmax_dummy,1000)
        v2 = np.expand_dims(v,axis=-1)
        veff = 0.64*Vmax_dummy
        veff2 = np.expand_dims(veff,axis=-1)
        costheta = np.linspace(-1.,1.,100)
        integrand = self.dsigmadcostheta(sigma0_m,w,v2,costheta)*v2**7*(1.-costheta**2)*np.exp(-v2**2/(4.*veff2**2))
        integrand2 = integrate.simps(integrand,x=costheta,axis=-1)
        sigma_eff_m = integrate.simps(integrand2,x=v,axis=0)
        sigma_eff_m = sigma_eff_m/(512.*veff**8)

        f_int = interp1d(Vmax_dummy,sigma_eff_m)
        return f_int
    

    def sigma_eff_analytical(self,sigma0_m, w, Vmax, a_threshold=703):
        """
        Returns the analytic evaluation of the effective cross section of SIDM divided by m,
        simplified via the substitution
            a = w^2/(4*nu_eff^2)   with   nu_eff = 0.64 * Vmax.
        
        The expression is:
        
            sigma_eff = -sigma0_m * a^2 * [ exp(a) * (1+a) * Ei(-a) + 1 ]
        
        where Ei(-a) is the exponential integral function.
        
        For large a (i.e. when a > a_threshold), the asymptotic behavior gives
            sigma_eff -> sigma0_m.
        In that case, the function returns sigma0_m.
        
        Parameters
        ----------
        sigma0_m : float
            The value of sigma_0 / m.
        w : float
            The value of w (in the same units as used in the numerical methods).
        Vmax : float or np.ndarray
            The maximum circular velocity.
        a_threshold : float, optional
            The threshold for a above which sigma_eff is set to sigma0_m.
            (Default is 703.)
        
        Returns
        -------
        sigma_eff : float or np.ndarray
            The effective cross section of SIDM divided by m.
        """
        nu_eff = 0.64 * Vmax
        a = w**2 / (4 * nu_eff**2)
        # Compute the expression based on the analytic result:
        # sigma_eff = -sigma0_m * a^2 * [ exp(a)*(1+a)*Ei(-a) + 1 ]
        # special.expi(-a) returns Ei(-a)
        expr = - sigma0_m * a**2 * ( np.exp(a) * (1 + a) * special.expi(-a) + 1 )
        # For a > a_threshold, return sigma0_m (i.e., effective cross section converges to sigma0_m)
        sigma_eff = np.where(a > a_threshold, sigma0_m, expr)
        return sigma_eff





class SIDM_parametric_model(SIDM_cross_section):
    """ A class to calculate the SIDM parametric model proposed by Yang et al. (2023) [arXiv:2305.16176].

    Refereces
    ---
    - Yang et al. (2023), https://arxiv.org/abs/2305.16176
    """


    def __init__(self, sigma0_m, w, tt_th=1.1):
        """ Initialize the SIDM_parametric_model class.

        Parameters
        ---
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        """
        SIDM_cross_section.__init__(self)
        self.sigma0_m      = sigma0_m*self.cm**2/self.gram
        self.w             = w*self.km/self.s
        # self.sigma_eff_m   = self.sigma_eff_m_interpolate(self.sigma0_m,self.w)
        self.tt_th         = tt_th


    def sigma_eff_m(self, Vmax):
        """ Returns the effective cross section of SIDM divided by m.

        Parameters
        ---
        Vmax : float
            The maximum circular velocity.

        Returns
        ---
        sigma_eff_m : float
            The effective cross section of SIDM divided by m.
        """
        return self.sigma_eff_analytical(self, self.sigma0_m, self.w, Vmax)


    def t_collapse(self, sigma_eff_m, rmax, Vmax):
        """ Returns the collapse time of a subhalo according to Eq. (2.2) of Yang et al. (2023)

        Parameters
        ---
        sigma_eff_m : float
            The effective cross section of SIDM divided by m.
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        Vmax : float
            The maximum circular velocity.

        Returns
        ---
        t_c : float
            The collapse time of a subhalo.
        """
        C = 0.75
        reff   = rmax/2.1626  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        rhoeff = (Vmax/(1.648*reff))**2/self.G  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        t_c = 150/C/(sigma_eff_m*rhoeff*reff)/np.sqrt(4.*np.pi*self.G*rhoeff)
        return t_c


    def dVmaxSIDMdtt(self, tt, Vmax0):
        """ Returns differential of Vmax obtained by Eq. (2.4) of Yang et al. (2023)

        Parameters
        ---
        tt : float
            The time normalized by the collapse time.
        Vmax0 : float
            The maximum circular velocity of the initial NFW profile.

        Returns
        ---
        out : float
            The differential of Vmax.

        Notes
        ---
        - In the integral approach, Vmax0 and rmax0 in eq. (2.4) have been replaced by Vmax_{CDM}(t) and rmax_{CDM}(t), respectively.
        """
        out = 0.1777-4.399*(3.*tt**2)+16.66*(4.*tt**3)-18.87*(5.*tt**4)+9.044*(7.*tt**6)-2.436*(9.*tt**8)
        out = np.where(tt>self.tt_th,0.,out)
        out = Vmax0*out
        return out


    def dVmaxSIDMdtt_numexpr_optimized(self, tt, Vmax0):
        """
        Returns differential of Vmax obtained by Eq. (2.4) of Yang et al. (2023) using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.
        
        Parameters
        ---
        tt : np.ndarray
            The time normalized by the collapse time.
        Vmax0 : np.ndarray or float
            The maximum circular velocity of the initial NFW profile.
            
        Returns
        ---
        out : np.ndarray
            The differential of Vmax.
        """
        tt_th = self.tt_th
        out = ne.evaluate(
            "where(tt <= tt_th, (0.1777 - 13.197*tt*tt + 66.64*tt*tt*tt"
            " - 94.35*tt*tt*tt*tt + 63.308*tt*tt*tt*tt*tt*tt"
            " - 21.924*tt*tt*tt*tt*tt*tt*tt*tt) * Vmax0, 0)"
        )
        return out


    def drmaxSIDMdtt(self, tt, rmax0):
        """ Returns differential of rmax obtained by in Eq. (2.4) of Yang et al. (2023)

        Parameters
        ---
        tt : float
            The time normalized by the collapse time.
        rmax0 : float
            The radius at which the maximum circular velocity is obtained.

        Returns
        ---
        out : float
            The differential of rmax.

        Notes
        ---
        - In the integral approach, Vmax0 and rmax0 in eq. (2.4) have been replaced by Vmax_{CDM}(t) and rmax_{CDM}(t), respectively.
        """
        out = 0.007623-0.7200*(2.*tt)+0.3376*(3.*tt**2)-0.1375*(4.*tt**3)
        out = np.where(tt>self.tt_th,0.,out)
        out = rmax0*out
        return out
    

    def drmaxSIDMdtt_numexpr_optimized(self, tt, rmax0):
        """
        Returns differential of rmax obtained by Eq. (2.4) of Yang et al. (2023) using NumExpr for optimization.
        This optimized version performs the entire calculation in a single evaluation for maximum speed.
        
        Parameters
        ---
        tt : np.ndarray
            The time normalized by the collapse time.
        rmax0 : np.ndarray or float
            The radius at which the maximum circular velocity is obtained.
        tt_th : float
            Threshold time.
            
        Returns
        ---
        out : np.ndarray
            The differential of rmax.
        """
        tt_th = self.tt_th
        out = ne.evaluate(
            "where(tt <= tt_th, (0.007623 - 1.44*tt + 1.0128*tt*tt - 0.55*tt*tt*tt) * rmax0, 0)"
        )
        return out


    def get_Vmax0(self, Vmax, tt):
        """ Returns the maximum circular velocity of the initial NFW profile, given by Eq. (2.3) of Yang et al. (2023)
        
        Parameters
        ---
        Vmax : float
            The maximum circular velocity.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        Vmax0 : float
            The value of Vmax of the initial NFW profile.
        """
        Vmax0 = Vmax/(1.+0.1777*tt-4.399*tt**3+16.66*tt**4-18.87*tt**5+9.077*tt**7-2.436*tt**9)
        return Vmax0


    def get_rmax0(self, rmax, tt):
        """ Returns the radius at which the maximum circular velocity is obtained, given by Eq. (2.3) of Yang et al. (2023)
        
        Parameters
        ---
        rmax : float
            The radius at which the maximum circular velocity is obtained.
        tt : float
            The time normalized by the collapse time.
        
        Returns
        ---
        rmax0 : float
            The value of rmax of the initial NFW profile.
        """
        rmax0 = rmax/(1.+0.007623*tt-0.7200*tt**2+0.3376*tt**3-0.1375*tt**4)
        return rmax0


    def get_rhos(self, rhos0, tt):
        """ Returns the density parameter of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rhos0 : float
            The density parameter of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rhos : float
            The density parameter of a SIDM halo.
        """
        rhos = (2.033+0.7381*tt+7.264*tt**5-12.73*tt**7+9.915*tt**9+(1.-2.033)/np.log(0.001)*np.log(tt+0.001))*rhos0
        return rhos


    def get_rs(self, rs0, tt):
        """ Returns the scale radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)

        Parameters
        ---
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.
        
        Returns
        ---
        rs : float
            The scale radius of a SIDM halo.
        """
        rs = (0.7178-0.1026*tt+0.2474*tt**2-0.4079*tt**3+(1.-0.7178)/np.log(0.001)*np.log(tt+0.001))*rs0
        return rs


    def get_rc(self, rs0, tt):
        """ Returns the core radius of a SIDM halo, given by Eq. (2.3) of Yang et al. (2023)
        
        Parameters
        ---
        rs0 : float
            The scale radius of the initial NFW profile.
        tt : float
            The time normalized by the collapse time.

        Returns
        ---
        rc : float
            The core radius of a SIDM halo.
        """
        rc = (2.555*np.sqrt(tt)-3.632*tt+2.131*tt**2-1.415*tt**3+0.4683*tt**4)*rs0
        return rc
    

    def master_function(self, Vmax_CDM, rmax_CDM, t, t_f):
        """ Calculate the properties of a SIDM halo at a given time t according to the integral approach proposed by Yang et al. (2023) [arXiv:2305.16176].

        Parameters
        ---
        Vmax_CDM : float
            The maximum circular velocity of the CDM halo.
        rmax_CDM : float
            The radius at which the maximum circular velocity is obtained.
        t : float
            The time.

        Returns
        ---
        VmaxSIDM_z0 : float
            The maximum circular velocity of the SIDM halo at z=0.
        rmaxSIDM_z0 : float
            The radius at which the maximum circular velocity is obtained at z=0.
        rhosSIDM_z0 : float
            The density parameter of the SIDM halo at z=0.
        rsSIDM_z0 : float
            The scale radius of the SIDM halo at z=0.
        rcSIDM_z0 : float
            The core radius of the SIDM halo at z=0.
        """
        t_c         = self.t_collapse(self.sigma_eff_m(Vmax_CDM),rmax_CDM,Vmax_CDM)
        Vmax0       = np.expand_dims(Vmax_CDM[:,0],axis=1)
        integrand   = self.dVmaxSIDMdtt_numexpr_optimized((t-t_f)/t_c,Vmax0)/t_c
        VmaxSIDM_z0 = Vmax_CDM[:,-1]+integrate.simps(integrand,x=t*np.ones((len(Vmax_CDM),1,1)),axis=1)
        rmax0       = np.expand_dims(rmax_CDM[:,0],axis=1)
        integrand   = self.drmaxSIDMdtt_numexpr_optimized((t-t_f)/t_c,rmax0)/t_c
        rmaxSIDM_z0 = rmax_CDM[:,-1]+integrate.simps(integrand,x=t*np.ones((len(rmax_CDM),1,1)),axis=1)
        
        tt             = np.minimum(((t-t_f)/t_c)[:,-1],self.tt_th)
        Vmax0_CDM_fict = self.get_Vmax0(VmaxSIDM_z0,tt)
        rmax0_CDM_fict = self.get_rmax0(rmaxSIDM_z0,tt)
        rs0_CDM_fict   = rmax0_CDM_fict/2.1626
        rhos0_CDM_fict = (4.625/(4.*np.pi*self.G))*(Vmax0_CDM_fict/rs0_CDM_fict)**2

        rhosSIDM_z0 = self.get_rhos(rhos0_CDM_fict,tt)
        rsSIDM_z0   = self.get_rs(rs0_CDM_fict,tt)
        rcSIDM_z0   = self.get_rc(rs0_CDM_fict,tt)

        return VmaxSIDM_z0, rmaxSIDM_z0, rhosSIDM_z0, rsSIDM_z0, rcSIDM_z0



class TidalStrippingSolver(halo_model):
    """ Solve the tidal stripping equation for a given subhalo. """
    
    def __init__(self, M0, z_min=0.0, z_max=7.0, n_z_interp=64):
        """ Initial function of the class. 
        
        -----
        Input
        -----
        M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
            Note that this is *not* the host mass at any given redshift! It can be obtained
            via Mzi(M0,redshift). 
        (Optional) z_min:          Minimum redshift to end the calculation of evolution to. (default: 0.)
        (Optional) z_max:          Maximum redshift to start the calculation of evolution from. (default: 7.)
        (Optional) n_z_interp:     Number of redshifts to calculate epsilon functions. (default: 64)
        """
        halo_model.__init__(self)
        self.z_min       = z_min
        self.z_max       = z_max
        self.n_z_interp  = n_z_interp
        self.M0          = M0


    @property
    def M0(self):
        return self._M0


    @M0.setter
    def M0(self, value):
        self._M0 = value
        self.reset_interpolation(
            z_max=self.z_max, 
            z_min=self.z_min,
            n_z=self.n_z_interp)

    
    def reset_interpolation(self, z_max, z_min, n_z):
        """ Reset interpolation for epsilon functions. 
        
        This function is called when the mass of the host
        halo is changed.

        -----
        Input
        -----
        za_max: float
            Maximum redshift to start the calculation of evolution from.
        z_min: float
            Minimum redshift to end the calculation of evolution to.
        n_z: int
            Number of redshifts to calculate epsilon functions.
        """
        _z, _eps_0 = self._eps_0(z_max, z_min, n_z)
        _, _eps_10, _eps_11 = self._eps_1(z_max, z_min, n_z)
        _, _eps_20, _eps_21, _eps_22 = self._eps_2(z_max, z_min, n_z)
        _, _eps_30, _eps_31, _eps_32, _eps_33 = self._eps_3(z_max, z_min, n_z)
        # get the interpolation functions as indefinite integrals
        self._eps_0_interp = lambda z: np.interp(z, _z[::-1], _eps_0[::-1])
        self._eps_10_interp = lambda z: np.interp(z, _z[::-1], _eps_10[::-1])
        self._eps_11_interp = lambda z: np.interp(z, _z[::-1], _eps_11[::-1])
        self._eps_20_interp = lambda z: np.interp(z, _z[::-1], _eps_20[::-1])
        self._eps_21_interp = lambda z: np.interp(z, _z[::-1], _eps_21[::-1])
        self._eps_22_interp = lambda z: np.interp(z, _z[::-1], _eps_22[::-1])
        self._eps_30_interp = lambda z: np.interp(z, _z[::-1], _eps_30[::-1])
        self._eps_31_interp = lambda z: np.interp(z, _z[::-1], _eps_31[::-1])
        self._eps_32_interp = lambda z: np.interp(z, _z[::-1], _eps_32[::-1])
        self._eps_33_interp = lambda z: np.interp(z, _z[::-1], _eps_33[::-1])
        # define the epsilon functions as definite integrals from za to z
        self.eps_0 = lambda _za, _z: self._eps_0_interp(_z) - self._eps_0_interp(_za)
        self.eps_10 = lambda _za, _z: self._eps_10_interp(_z) - self._eps_10_interp(_za)
        self.eps_11 = lambda _za, _z: self._eps_11_interp(_z) - self._eps_11_interp(_za)
        self.eps_20 = lambda _za, _z: self._eps_20_interp(_z) - self._eps_20_interp(_za)
        self.eps_21 = lambda _za, _z: self._eps_21_interp(_z) - self._eps_21_interp(_za)
        self.eps_22 = lambda _za, _z: self._eps_22_interp(_z) - self._eps_22_interp(_za)
        self.eps_30 = lambda _za, _z: self._eps_30_interp(_z) - self._eps_30_interp(_za)
        self.eps_31 = lambda _za, _z: self._eps_31_interp(_z) - self._eps_31_interp(_za)
        self.eps_32 = lambda _za, _z: self._eps_32_interp(_z) - self._eps_32_interp(_za)
        self.eps_33 = lambda _za, _z: self._eps_33_interp(_z) - self._eps_33_interp(_za)


    def Mzvir(self,z):
        Mz200 = self.Mzzi(self.M0,z,0.)
        Mvir = self.Mvir_from_M200_fit(Mz200,z)
        return Mvir


    def AMz(self,z):
        log10a = (-0.0003*np.log10(self.Mzvir(z)/self.Msun)+0.02)*z \
                        +(0.011*np.log10(self.Mzvir(z)/self.Msun)-0.354)
        return 10.**log10a


    def zetaMz(self,z):
        return (0.00012*np.log10(self.Mzvir(z)/self.Msun)-0.0033)*z \
                    +(-0.0011*np.log10(self.Mzvir(z)/self.Msun)+0.026)


    def tdynz(self,z):
        Oz_z = self.OmegaM*(1.+z)**3/self.g(z)
        return 1.628/self.h*(self.Delc(Oz_z-1.)/178.0)**-0.5/(self.Hubble(z)/self.H0)*1.e9*self.yr


    def msolve(self,m, z):
        return self.AMz(z)*(m/self.tdynz(z))*(m/self.Mzvir(z))**self.zetaMz(z)/(self.Hubble(z)*(1+z))


    def subhalo_mass_stripped_odeint(self, ma, za, z0, **kwargs):
        """ Solve the subhalo mass stripping equation using the odeint function. 

        If z0 is scalar, then the function returns the final mass of the subhalo at z0.
        If z0 is an array, then the function returns the mass of the subhalo at all redshifts in za.
        """
        if np.isscalar(z0):
            zcalc = np.linspace(za,z0,100)
            sol = odeint(self.msolve,ma,zcalc,**kwargs)
            return sol[-1]
        else:
            sol = odeint(self.msolve,ma,z0,**kwargs)
            return sol

    
    # Functions to calculate perturbative corrections to the subhalo mass function
    def Phi(self,z):
        """ subhalo stripping factor assuming zetaMz(z) = 0.
        The stripping rate dm/dt is given by
          dm/dt(z) = m(z) * Phi(z) * (m(z)/Mzvir(z))**zetaMz(z)
        """
        return self.AMz(z)/self.tdynz(z)/self.Hubble(z)/(1+z)


    # @memoize_with_pickle()
    def _eps_0(self,za,z,n_z=64):
        """ calculate epsilon0.
        
        Returns
        -------
        _z : array
            redshift array
        eps0 : array
            epsilon0 array.
        """
        _z = np.linspace(za,z,n_z)
        Phi_z = self.Phi(_z)
        return _z, cumulative_trapezoid(Phi_z,x=_z,initial=0)
    

    # @memoize_with_pickle()
    def _eps_1(self,za,z,n_z=64):
        """ calculate the first order correction.

        The first order correction epsilon_1 is given by the following equation:
            epsilon_1 = epsilon_10 + epsilon_11 * ln_ma
        
        Returns
        -------
        _z : array
            redshift array
        eps10 : array
            epsilon10 array.
        eps11 : array
            epsilon11 array.
        """
        _z, eps_0 = self._eps_0(za,z,n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_10 = Phi_z * (eps_0 - ln_Mvir_z) * zeta_z
        integrand_11 = Phi_z * zeta_z
        integral_10 = cumulative_trapezoid(integrand_10,x=_z,initial=0)
        integral_11 = cumulative_trapezoid(integrand_11,x=_z,initial=0)
        return _z, integral_10, integral_11
    

    # @memoize_with_pickle()
    def _eps_2(self,za,z,n_z=64):
        """ calculate the second order correction.

        The second order correction epsilon_2 is given by the following equation:
            epsilon_2 = epsilon_20 + epsilon_21 * ln_ma + epsilon_22 * ln_ma^2
        
        Returns
        -------
        _z : array
            redshift array
        eps20 : array
            epsilon20 array.
        eps21 : array
            epsilon21 array.
        eps22 : array
            epsilon22 array.
        """
        _z, eps_0 = self._eps_0(za,z,n_z)
        _, eps_10, eps_11 = self._eps_1(za,z,n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_20 = Phi_z * zeta_z **2 * (eps_0 - ln_Mvir_z)**2 /2 + Phi_z * zeta_z * eps_10
        integrand_21 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) + Phi_z * zeta_z * eps_11
        integrand_22 = Phi_z * zeta_z**2 / 2
        integral_20 = cumulative_trapezoid(integrand_20,x=_z,initial=0)
        integral_21 = cumulative_trapezoid(integrand_21,x=_z,initial=0)
        integral_22 = cumulative_trapezoid(integrand_22,x=_z,initial=0)
        return _z, integral_20, integral_21, integral_22


    # @memoize_with_pickle()
    def _eps_3(self, za, z, n_z=64):
        """ calculate the third order correction.

        The third order correction epsilon_3 is given by the following equation:
            epsilon_3 = epsilon_30 + epsilon_31 * ln_ma + epsilon_32 * ln_ma^2 + epsilon_33 * ln_ma^3
        
        Returns
        -------
        _z : array
            redshift array
        eps30 : array
            epsilon30 array.
        eps31 : array
            epsilon31 array.
        eps32 : array
            epsilon32 array.
        eps33 : array
            epsilon33 array.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        _, eps_20, eps_21, eps_22 = self._eps_2(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_30 = (Phi_z * (eps_0 - ln_Mvir_z)**3 * zeta_z**3 / 6.0
                        + Phi_z * (eps_0 - ln_Mvir_z) * eps_10 * (zeta_z**2)
                        + Phi_z * eps_20 * zeta_z)
        integrand_31 = (Phi_z * (eps_0 - ln_Mvir_z)**2 * (zeta_z**3) / 2.0
                        + Phi_z * eps_10 * (zeta_z**2)
                        + Phi_z * eps_21 * zeta_z
                        + Phi_z * (eps_0 - ln_Mvir_z) * eps_11 * (zeta_z**2))
        integrand_32 = (Phi_z * (eps_0 - ln_Mvir_z) * (zeta_z**3) / 2.0
                        + Phi_z * eps_11 * (zeta_z**2)
                        + Phi_z * eps_22 * zeta_z)
        integrand_33 = Phi_z * (zeta_z**3) / 6.0
        integral_30 = cumulative_trapezoid(integrand_30, x=_z, initial=0)
        integral_31 = cumulative_trapezoid(integrand_31, x=_z, initial=0)
        integral_32 = cumulative_trapezoid(integrand_32, x=_z, initial=0)
        integral_33 = cumulative_trapezoid(integrand_33, x=_z, initial=0)
        return _z, integral_30, integral_31, integral_32, integral_33
    

    def subhalo_mass_stripped_pert0(self, ma, za, z):
        """Calculate subhalo mass stripping using zeroth-order perturbation."""
        eps_0 = self.eps_0(za, z)
        return ma * np.exp(eps_0)
    

    def subhalo_mass_stripped_pert1(self, ma, za, z):
        """Calculate subhalo mass stripping using first-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11)
        return ma * np.exp(eps)
    

    def subhalo_mass_stripped_pert2(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11 
               + eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22)
        return ma * np.exp(eps)
    

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation with Shanks transformation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        # NOTE: Shanks transformation
        # For A_n = \sum_{i=0}^{n} a_i, the Shanks transformation is given by
        #  S_n = A_{n+1} - (A_{n+1} - A_n)^2 / (A_{n+1} - 2A_n + A_{n-1})
        #      = A_{n+1} - (A_{n+1} - A_n)^2 / ((A_{n+1} - A_n) - (A_n - A_{n-1}))
        # Since A_n = \sum_{i=0}^{n} a_i, we can write
        #  S_n = A_{n+1} - a_{n+1}^2 / (a_{n+1} - a_n)
        # In our case, we calculate the epsilon as eps = \sum_{i=0}^{n} eps_i.
        # Therefore, we can apply Shanks transformation to eps up to the second order as follows:
        # S_2 = (eps_0 + eps_1 + eps_2) - eps_2^2 / (eps_2 - eps_1)
        eps_1 = eps_10 + ln_ma * eps_11
        eps_2 = eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22
        eps_2m1 = (eps_20 - eps_10) + ln_ma * (eps_21 - eps_11) + ln_ma**2 * eps_22
        eps_shanks = - eps_2**2 / eps_2m1
        eps_shanks = np.where(np.isnan(eps_shanks), 0, eps_shanks)
        # When the correction is too small, the Shanks transformation may not be stable.
        eps_shanks = np.where(np.abs((eps_1+eps_2)/eps_0) < 0.02, 0, eps_shanks)
        eps = eps_0 + eps_1 + eps_2 + eps_shanks
        # import pandas as pd
        # df = pd.DataFrame({'z': _z, 'eps_0': eps_0, 'eps_1': eps_1, 'eps_2': eps_2, 'eps_shanks': eps_shanks, 'eps': eps})
        # display(df[(df['z'] > 4) & (df['z'] < 6)])
        return ma * np.exp(eps)
    
    
    def subhalo_mass_stripped_pert3(self, ma, za, z):
        """Calculate subhalo mass stripping using third-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        eps_30 = self.eps_30(za, z)
        eps_31 = self.eps_31(za, z)
        eps_32 = self.eps_32(za, z)
        eps_33 = self.eps_33(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11 
               + eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22
               + eps_30 + ln_ma * eps_31 + ln_ma**2 * eps_32 + ln_ma**3 * eps_33)
        return ma * np.exp(eps)
    
    
    def subhalo_mass_stripped(self,ma,za,z,method="pert2_shanks",**kwargs):
        """ A wrapper function to calculate subhalo mass stripping.
        
        Parameters
        ----------
        ma : float
            initial subhalo mass.
        za : float
            initial redshift.
        z : float
            final redshift.
        method : str, optional
            method to calculate the subhalo mass stripping.
            - "odeint" : use odeint to solve the differential equation.
            - "pert0" : use perturbative method with zeroth-order correction.
            - "pert1" : use perturbative method with first-order correction.
            - "pert2" : use perturbative method with second-order correction.
            - "pert2_shanks" : use perturbative method with second-order correction and Shanks transformation.
            - "pert3" : use perturbative method with third-order correction.
        kwargs : dict, optional
            additional arguments for the odeint function.

        Returns
        -------
        float or np.ndarray
            the mass of the subhalo
        """
        # match method:
        #     case "odeint":
        #         return self.subhalo_mass_stripped_odeint(ma,za,z,**kwargs)
        #     # NOTE: odeint returns (len(z),len(ma)) array for array input.
        #     case "pert0":
        #         return self.subhalo_mass_stripped_pert0(ma,za,z)
        #     case "pert1":
        #         return self.subhalo_mass_stripped_pert1(ma,za,z)
        #     case "pert2":
        #         return self.subhalo_mass_stripped_pert2(ma,za,z)
        #     case "pert2_shanks":
        #         return self.subhalo_mass_stripped_pert2_shanks(ma,za,z)
        #     case "pert3":
        #         return self.subhalo_mass_stripped_pert3(ma,za,z)
        #     case _:
        #         raise ValueError(f"Invalid method: {method}")
        if method == "odeint":
            # NOTE: odeint returns (len(z),len(ma)) array for array input.
            return self.subhalo_mass_stripped_odeint(ma,za,z,**kwargs)
        elif method[:4] == "pert":
            # When z and ma are given as 1d arrays, perturbative methods raise error.
            # To return the similar output as odeint, we broadcast the input arrays.
            if np.isscalar(z) and np.isscalar(ma):
                return getattr(self, f"subhalo_mass_stripped_{method}")(ma,za,z)
            else:
                ma = np.atleast_1d(ma)
                z = np.atleast_1d(z)
                ma = ma[np.newaxis,:]  # (1, len(ma))
                z = z[:,np.newaxis]  # (len(z), 1)
                return getattr(self, f"subhalo_mass_stripped_{method}")(ma,za,z)
        else:
            raise ValueError(f"Invalid method: {method}")



            

    
class subhalo_properties(halo_model, SIDM_parametric_model, SIDM_cross_section):
    

    def __init__(self, sigma0_m=147.1, w=24.33, beta=4, tt_th=1.1):
        """ Initialize the subhalo_properties class.
        Default values of sigma0_m and w are taken from Yang et al. (2023) [arXiv:2305.16176]:

        > The cosmological zoom-in simulation in ref. [5] contains a Milky Way analog and the differential cross section is given in eq. (1.2), 
        > with sigma_0/m = 147.1 cm^2/g and w = 24.33 km/s.
        
        Parameters
        ----------
        sigma0_m : float
            The value of sigma_0 / m in units of cm^2/g.
        w : float
            The value of w in unit of km/s.
        z_f : float
            The formation redshift (default: 10)
        beta : float
            The power index of the density profile.
        """
        halo_model.__init__(self)
        self.param_model   = SIDM_parametric_model(sigma0_m, w, tt_th)
        sidm_cs            = SIDM_cross_section()

        self.sigma0_m      = sigma0_m*self.cm**2/self.gram
        self.w             = w*self.km/self.s
        self.sigma_eff_m   = sidm_cs.sigma_eff_m_interpolate(self.sigma0_m,self.w)

        self.beta          = beta

        ctemp              = np.linspace(0,100,1000)
        self.ct_func       = interp1d(self.fc(ctemp),ctemp,fill_value='extrapolate')



    def Ffunc(self, dela, s1, s2):
        """ Returns Eq. (12) of Yang et al. (2011) 

        Refereces
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        return 1/np.sqrt(2.*np.pi)*dela/(s2-s1)**1.5

    
    def Gfunc(self, dela, s1, s2):
        """ Returns the G function used in Eq. (13) of Yang et al. (2011) 

        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        G0     = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1   = np.sqrt(s1)
        sig2   = np.sqrt(s2)
        return G0*pow(sig2/sig1,gamma1)*pow(dela/sig1,gamma2)

    
    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        """ Returns Eq. (14) of Yang et al. (2011) 
        
        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        return 1./np.sqrt(2.*np.pi)*(delc2-delc1)/(s2-s1)**1.5 \
            *np.exp(-(delc2-delc1)**2/(2.*(s2-s1)))

    
    def Na_calc(self, ma, zacc, Mhost, z0=0., N_herm=200, Nrand=1000, Na_model=3):
        """ Returns Na, Eq. (3) of Yang et al. (2011) 

        Parameters
        ---
        ma : float
            The mass of the subhalo.
        zacc : float
            The accretion redshift of the subhalo.
        Mhost : float
            The mass of the host halo.
        z0 : float
            The redshift.
        N_herm : int
            The number of Hermite-Gauss quadrature points.
        Nrand : int
            The number of random numbers.
        Na_model : int
            The model to calculate Na. Default is 3.

        Returns
        ---
        Na : float
            The value of Na.
        
        References
        ---
            - Yang et al. (2011), https://arxiv.org/abs/1104.1757
        """
        zacc_2d   = zacc.reshape(-1,1)
        M200_0    = self.Mzzi(Mhost,zacc_2d,z0)
        logM200_0 = np.log10(M200_0)

        xxi,wwi = hermgauss(N_herm)
        xxi = xxi.reshape(-1,1,1)
        wwi = wwi.reshape(-1,1,1)
        # Eq. (21) in Yang et al. (2011) 
        sigmalogM200 = 0.12-0.15*np.log10(M200_0/Mhost)
        logM200 = np.sqrt(2.)*sigmalogM200*xxi+logM200_0
        M200 = 10.**logM200
            
        mmax = np.minimum(M200,Mhost/2.)
        Mmax = np.minimum(M200_0+mmax,Mhost)
        
        if Na_model==3:
            zlist    = zacc_2d*np.linspace(1,0,Nrand)
            iMmax    = np.argmin(np.abs(self.Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
            z_Max    = zlist[np.arange(len(zlist)),iMmax]
            z_Max_3d = z_Max.reshape(N_herm,len(zlist),1)
            delcM    = self.deltac_func(z_Max_3d)
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(Mmax)
            sa       = self.s_func(ma)
            xmax     = (delca-delcM)**2/(2.*(self.s_func(mmax)-sM))
            normB    = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)
            # those reside in the exponential part of Eq. (14) 
            Phi      = self.Ffunc_Yang(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        elif Na_model==1:
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(M200)
            sa       = self.s_func(ma)
            xmin     = self.s_func(mmax)-self.s_func(M200)
            normB    = 1./np.sqrt(2*np.pi)*delca*2./xmin**0.5*special.hyp2f1(0.5,0.,1.5,-sM/xmin)
            Phi      = self.Ffunc(delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        elif Na_model==2:
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(M200)
            sa       = self.s_func(ma)
            xmin     = self.s_func(mmax)-self.s_func(M200)
            normB    = 1./np.sqrt(2.*np.pi)*delca*0.57 \
                           *(delca/np.sqrt(sM))**-0.01*(2./(1.-0.38))*sM**(-0.38/2.) \
                           *xmin**(0.5*(0.38-1.)) \
                           *special.hyp2f1(0.5*(1-0.38),-0.38/2.,0.5*(3.-0.38),-sM/xmin)
            Phi      = self.Ffunc(delca,sM,sa)*self.Gfunc(delca,sM,sa)/normB \
                           *np.heaviside(mmax-ma,0)
        # calculate Na
        if N_herm==1:
            F2t = np.nan_to_num(Phi)
            F2  =F2t.reshape((len(zacc_2d),len(ma)))
        else:
            F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
        Na = F2*self.dsdm(ma,0.)*self.dMdz(Mhost,zacc_2d,z0)*(1.+zacc_2d)
        return Na

    
    def subhalo_properties_calc(self, M0, redshift=0.0, dz=0.01, zmax=5.0, N_ma=500, sigmalogc=0.128,
                                N_herm=20, logmamin=6, logmamax=None, N_hermNa=200, Na_model=3, 
                                ct_th=0., M0_at_redshift=False,
                                method="pert2_shanks", **kwargs):
        """
        This is the main function of SASHIMI-C, which makes a semi-analytical subhalo catalog.
        
        -----
        Input
        -----
        M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
            Note that this is *not* the host mass at the given redshift! It can be obtained
            via Mzi(M0,redshift). If you want to give this parameter as the mass at the given
            redshift, then turn 'M0_at_redshift' parameter on (see below).
        
        (Optional) redshift:       Redshift of interest. (default: 0)
        (Optional) dz:             Grid of redshift of halo accretion. (default 0.01)
        (Optional) zmax:           Maximum redshift to start the calculation of evolution from. (default: 5.)
        (Optional) N_ma:           Number of logarithmic grid of subhalo mass at accretion defined as M_{200}.
                                   (default: 500)
        (Optional) sigmalogc:      rms scatter of concentration parameter defined for log_{10}(c).
                                   (default: 0.128)
        (Optional) N_herm:         Number of grid in Gauss-Hermite quadrature for integral over concentration.
                                   (default: 20)
        (Optional) logmamin:       Minimum value of subhalo mass at accretion defined as log_{10}(m_{min}/Msun). 
                                   (default: 6)
        (Optional) logmamax:       Maximum value of subhalo mass at accretion defined as log_{10}(m_{max}/Msun).
                                   If None, m_{max}=0.1*M0. (default: None)
        (Optional) N_hermNa:       Number of grid in Gauss-Hermite quadrature for integral over host evoluation, 
                                   used in Na_calc. (default: 200)
        (Optional) Na_model:       Model number of EPS defined in Yang et al. (2011). (default: 3)
        (Optional) ct_th:          Threshold value for c_t(=r_t/r_s) parameter, below which a subhalo is assumed to
                                   be completely desrupted. Suggested values: 0.77 or 0 (no desruption; default).
        (Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.
        (Optional) method:         Method to calculate the subhalo mass stripping. (default: "pert2_shanks")
        (Optional) kwargs:         Additional arguments for the odeint function.
        
        ------
        Output
        ------
        List of subhalos that are characterized by the following parameters.
        ma200:        Mass m_{200} at accretion.
        z_acc:        Redshift at accretion.
        rsCDM_acc:    Scale radius r_s at accretion for CDM.
        rhosCDM_acc:  Characteristic density \rho_s at accretion for CDM.
        rmaxCDM_acc:  Radius at which the maximum circular velocity is obtained at accretion for CDM.
        VmaxCDM_acc:  Maximum circular velocity at accretion for CDM.
        rsSIDM_acc:   Scale radius r_s at accretion for SIDM.
        rhosSIDM_acc: Characteristic density \rho_s at accretion for SIDM.
        rcSIDM_acc:   Core radius r_c at accretion for SIDM.
        rmaxSIDM_acc: Radius at which the maximum circular velocity is obtained at accretion for SIDM.
        VmaxSIDM_acc: Maximum circular velocity at accretion for SIDM.
        m_z0:         Mass up to tidal truncation radius at a given redshift.
        rsCDM_z0:     Scale radius r_s at a given redshift for CDM.
        rhosCM_z0:    Characteristic density \rho_s at a given redshift for CDM.
        rmaxCDM_z0:   Radius at which the maximum circular velocity is obtained at a given redshift for CDM.
        VmaxCDM_z0:   Maximum circular velocity at a given redshift for CDM.
        rsSIDM_z0:    Scale radius r_s at a given redshift for SIDM.
        rhosSIDM_z0:  Characteristic density \rho_s at a given redshift for SIDM.
        rcSIDM_z0:    Core radius r_c at a given redshift for SIDM.
        rmaxSIDM_z0:  Radius at which the maximum circular velocity is obtained at a given redshift for SIDM.
        VmaxSIDM_z0:  Maximum circular velocity at a given redshift for SIDM.
        ctCDM_z0:     Tidal truncation radius in units of r_s at a given redshift for CDM.
        tt_ratio:     The age of SIDM subhalos normalized by the collapse time scale (t-t_f)/t_c. 
                      tt_ratio>1 means that the subhalo core is gravo-thermally collapsed.
        weightCDM:    Effective number of subhalos for CDM that are characterized by the same set of the parameters above.
        weightSIDM:   Effective number of subhalos for SIDM that are characterized by the same set of the parameters above.
        surviveCDM:   If that subhalo survive against tidal disruption or not for CDM.
        surviveSIDM:  If that subhalo survive against tidal disruption or not for SIDM.
        """
        
        if M0_at_redshift:
            Mz        = M0
            M0_list   = np.logspace(0.,3.,1000)*Mz
            fint      = interp1d(self.Mzi(M0_list,redshift),M0_list)
            M0        = fint(Mz)
        self.M0       = M0
        self.redshift = redshift

        _zmax              = np.linspace(redshift,10.,1001)
        z_dummy            = np.linspace(redshift,_zmax,1000)
        t_L                = integrate.simps(1./(self.Hubble(z_dummy)*(1.+z_dummy)),x=z_dummy,axis=0)
        self.lookback_time = interp1d(_zmax,t_L)
        z_dummy            = np.linspace(redshift,1000,10000)
        self.t_U           = integrate.simps(1./(self.Hubble(z_dummy)*(1.+z_dummy)),x=z_dummy)
        del z_dummy, _zmax, t_L

        zdist         = np.arange(redshift+dz,zmax+dz,dz)  # zdist.shape = (n_z,)
        if logmamax==None:
            logmamax  = np.log10(0.1*M0/self.Msun)
        ma200_z0      = np.logspace(logmamin,logmamax,N_ma)*self.Msun  # ma200_z0.shape = (N_ma,)
        ma_z0         = self.Mvir_from_M200_fit(ma200_z0,redshift)  # ma_z0.shape = (N_ma,)

        ma200_0            = np.empty_like(ma200_z0)  # ma200_0.shape = (N_ma,)
        ma_0               = np.empty_like(ma_z0)   # ma_0.shape = (N_ma,)
        if redshift==0.:
            ma200_0        = ma200_z0
            ma_0           = ma_z0
        else:
            ma200_0_list   = np.logspace(0.,2.,200)*(ma200_z0.reshape(-1,1))  # ma200_0_list.shape = (N_ma,200)
            ma_0_list      = np.logspace(0.,2.,200)*(ma_z0.reshape(-1,1))  # ma_0_list.shape = (N_ma,200)
            for i in np.arange(len(ma200_z0)):
                fint_ma200 = interp1d(self.Mzi(ma200_0_list[i],redshift),ma200_0_list[i])
                fint_ma    = interp1d(self.Mzi(ma_0_list[i],redshift),ma_0_list[i])
                ma200_0[i] = fint_ma200(ma200_z0[i])
                ma_0[i]    = fint_ma(ma_z0[i])
            
        z_f           = -0.0064*(np.log10(ma_0/self.Msun))**2+0.0237*np.log10(ma_0/self.Msun)+1.8837  # z_f.shape = (N_ma,)
        t_f           = self.t_U-self.lookback_time(z_f) #/2.  # t_f.shape = (N_ma,)
        # NOTE: For the integral approach, we take the formation time tf to be half of t_U-t_f.
        #       Ref: Sec. 5 of Yang et al. (2023)
        # NOTE: For a more careful examination, this extra factor of 1/2 has been removed.
        #       t-z relation must be made consistently throughout the paper.

        # Initialize the arrays
        ma200_matrix     = self.Mzi(ma200_0,zdist[:,np.newaxis])  # ma200_matrix.shape = (n_z,N_ma)
        ma_matrix        = self.Mvir_from_M200_fit(ma200_matrix,zdist[:,np.newaxis])  # ma_matrix.shape = (n_z,N_ma)
        accretion        = z_f>zdist[:,np.newaxis]  # accretion.shape = (n_z,N_ma)
        # Arrays for only the redshifts where at least one subhalo accretes
        zdist_accreted = zdist[accretion.any(axis=1)]  # consider only the redshifts where at least one subhalo accretes
        ma200_matrix_accreted = ma200_matrix[accretion.any(axis=1)]  # consider only the redshifts where at least one subhalo accretes
        ma_matrix_accreted    = ma_matrix[accretion.any(axis=1)]  # consider only the redshifts where at least one subhalo accretes

        # Initialize the arrays
        # NOTE: shape = (N_z, N_conc, N_ma)
        rsCDM_acc     = np.zeros((len(zdist_accreted),N_herm,len(ma200_z0)))
        rhosCDM_acc   = np.zeros_like(rsCDM_acc)
        rmaxCDM_acc   = np.zeros_like(rsCDM_acc)
        VmaxCDM_acc   = np.zeros_like(rsCDM_acc)
        rsCDM_z0      = np.zeros_like(rsCDM_acc)
        rhosCDM_z0    = np.zeros_like(rsCDM_acc)
        rmaxCDM_z0    = np.zeros_like(rsCDM_acc)
        VmaxCDM_z0    = np.zeros_like(rsCDM_acc)
        rsSIDM_acc    = np.zeros_like(rsCDM_acc)
        rhosSIDM_acc  = np.zeros_like(rsCDM_acc)
        rcSIDM_acc    = np.zeros_like(rsCDM_acc)
        rmaxSIDM_acc  = np.zeros_like(rsCDM_acc)
        VmaxSIDM_acc  = np.zeros_like(rsCDM_acc)
        rsSIDM_z0     = np.zeros_like(rsCDM_acc)
        rhosSIDM_z0   = np.zeros_like(rsCDM_acc)
        rcSIDM_z0     = np.zeros_like(rsCDM_acc)
        rmaxSIDM_z0   = np.zeros_like(rsCDM_acc)
        VmaxSIDM_z0   = np.zeros_like(rsCDM_acc)
        ctCDM_z0      = np.zeros_like(rsCDM_acc)
        tt_ratio      = np.zeros_like(rsCDM_acc)
        surviveCDM    = np.zeros_like(rsCDM_acc)
        surviveSIDM   = np.zeros_like(rsCDM_acc)
        m0CDM_matrix  = np.zeros_like(rsCDM_acc)


        # def Mzvir(z):
        #     Mz200 = self.Mzzi(M0,z,0.)
        #     Mvir = self.Mvir_from_M200_fit(Mz200,z)
        #     return Mvir

        # def AMz(z):
        #     log10a = (-0.0003*np.log10(Mzvir(z)/self.Msun)+0.02)*z \
        #                  +(0.011*np.log10(Mzvir(z)/self.Msun)-0.354)
        #     return 10.**log10a

        # def zetaMz(z):
        #     return (0.00012*np.log10(Mzvir(z)/self.Msun)-0.0033)*z \
        #                +(-0.0011*np.log10(Mzvir(z)/self.Msun)+0.026)

        # def tdynz(z):
        #     Oz_z = self.OmegaM*(1.+z)**3/self.g(z)
        #     return 1.628/self.h*(self.Delc(Oz_z-1.)/178.0)**-0.5/(self.Hubble(z)/self.H0)*1.e9*self.yr

        # def msolve(m, z):
        #     return AMz(z)*(m/tdynz(z))*(m/Mzvir(z))**zetaMz(z)/(self.Hubble(z)*(1+z))
        solver = TidalStrippingSolver(
            M0=M0,
            z_min=redshift,
            z_max=zmax,
            n_z_interp=64
        )
        
        # def t_collapse(sigma_eff_m, rmax, Vmax):
        #     """ Returns the collapse time of a subhalo according to Eq. (2.2) of Yang et al. (2023)
        #     """
        #     C = 0.75
        #     reff   = rmax/2.1626  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        #     rhoeff = (Vmax/(1.648*reff))**2/self.G  # NOTE: from the lines just after Eq. (1.1) of Yang et al. (2023)
        #     t_c = 150/C/(sigma_eff_m*rhoeff*reff)/np.sqrt(4.*np.pi*self.G*rhoeff)
        #     return t_c

        # For the concentration parameter
        x1,w1        = hermgauss(N_herm)
        x1           = x1.reshape(-1,1,1)
        w1           = w1.reshape(-1,1,1)
        for iz, za in tqdm.tqdm(enumerate(zdist_accreted),total=len(zdist_accreted),desc='Calculating subhalo properties'):
            # Before accretion (ba) onto the host
            z_ba    = np.linspace(z_f,za,100)
            m200_ba = self.Mzi(ma200_0,z_ba)  # This is valid for redshift = 0. case for now
            # NOTE: m200_ba.shape = (len(z_ba),len(ma200_z0)) = (100,N_ma) = (100,500) (for default values)
            ma               = ma_matrix_accreted[iz]  # shape = (N_ma,)
            
            c200_med_ba = self.conc200(m200_ba,z_ba)  # shape = (len(z_ba),len(ma200_z0)) = (100,N_ma) = (100,500) (for default values)
            r200_ba      = (3.*m200_ba/(4.*np.pi*self.rhocrit0*self.g(z_ba)*200.))**(1./3.)
            log10c200_ba = np.sqrt(2.)*sigmalogc*x1+np.log10(c200_med_ba)
            # log10c200_ba = np.random.normal(np.log10(c200_med_ba),sigmalogc,size=(1,len(z_ba),len(ma200_z0)))  # TODO
            # w1 = np.ones_like(log10c200_ba) / len(log10c200_ba)  # TODO
            c200_ba      = 10.**log10c200_ba
            rs_ba        = r200_ba/c200_ba
            rhos_ba      = m200_ba/(4.*np.pi*rs_ba**3*self.fc(c200_ba))
            rmax_ba      = 2.1626*rs_ba  # shape = (N_herm,len(z_ba),len(ma200_z0)) = (N_herm,100,N_ma) = (100,500) (for default values)
            Vmax_ba      = np.sqrt(rhos_ba*4.*np.pi*self.G/4.625)*rs_ba
            
            # After accretion (aa): take the last value of the above along the redshift axis
            rsCDM_acc[iz]   = rs_ba[:,-1]  # shape = (N_herm,N_ma) = (5,500) (for default values)
            rhosCDM_acc[iz] = rhos_ba[:,-1]
            rmaxCDM_acc[iz] = rmax_ba[:,-1]
            VmaxCDM_acc[iz] = Vmax_ba[:,-1]

            # After accretion (aa)
            zcalc       = np.linspace(za,redshift,100)
            # m_aa        = odeint(msolve,ma,zcalc)
            m_aa        = solver.subhalo_mass_stripped(ma, za, zcalc, method=method, **kwargs)
            rmax_acc    = np.expand_dims(rmax_ba[:,-1],axis=1)  # shape = (N_herm,1) = (5,1) (for default values)
            Vmax_acc    = np.expand_dims(Vmax_ba[:,-1],axis=1)
            # Penarrubia et al. (2010)
            Vmax_aa     = Vmax_acc*(2.**0.4*(m_aa/ma)**0.3*(1.+m_aa/ma)**-0.4)
            rmax_aa     = rmax_acc*(2.**-0.3*(m_aa/ma)**0.4*(1.+m_aa/ma)**0.3)
            # Green and van den Bosch (2019)
            #fb          = m_aa/ma
            #cs          = np.expand_dims(c200_ba[:,-1],axis=1)
            #Vmax_aa     = Vmax_acc*X_func(fb,cs,'Vmax')
            #rmax_aa     = rmax_acc*X_func(fb,cs,'rmax')

            rmaxCDM_z0[iz] = rmax_aa[:,-1]
            VmaxCDM_z0[iz] = Vmax_aa[:,-1]
            rsCDM_z0[iz]   = rmax_aa[:,-1]/2.1626
            rhosCDM_z0[iz] = (4.625/(4.*np.pi*self.G))*(Vmax_aa[:,-1]/rsCDM_z0[iz])**2

            ctCDM_z0[iz]     = self.ct_func(m_aa[-1]/(4.*np.pi*rhosCDM_z0[iz]*rsCDM_z0[iz]**3))
            surviveCDM[iz]   = np.where(ctCDM_z0[iz]>ct_th,1,0)
            m0CDM_matrix[iz] = m_aa[-1]*np.ones((N_herm,1))

            z                = np.concatenate((z_ba,(zcalc[1:].reshape(-1,1))*np.ones_like(ma200_z0)),axis=0)
            t                = self.t_U-self.lookback_time(z)
            # concatenate Vmax and rmax to obtain entire history of the subhalo
            # NOTE: shape = (len(z_ba)+len(zcalc)-1,len(ma200_z0)) = (199,N_ma) = (199,500) (for default values)
            Vmax_CDM         = np.concatenate((Vmax_ba,Vmax_aa[:,1:]),axis=1)
            rmax_CDM         = np.concatenate((rmax_ba,rmax_aa[:,1:]),axis=1)
            t_c              = self.t_collapse(self.sigma_eff_m(Vmax_CDM),rmax_CDM,Vmax_CDM)

            tt_ratio[iz]     = ((self.t_U-t_f)/t_c)[:,-1]

            VmaxSIDM_z0[iz], rmaxSIDM_z0[iz], rhosSIDM_z0[iz], rsSIDM_z0[iz], rcSIDM_z0[iz] \
                = self.param_model.master_function(Vmax_CDM,rmax_CDM,t,t_f)

            t2               = self.t_U-self.lookback_time(z_ba)

            VmaxSIDM_acc[iz], rmaxSIDM_acc[iz], rhosSIDM_acc[iz], rsSIDM_acc[iz], rcSIDM_acc[iz] \
                = self.param_model.master_function(Vmax_ba,rmax_ba,t2,t_f)


            surviveSIDM[iz]  = np.where(((VmaxSIDM_z0[iz]<0.)+(rmaxSIDM_z0[iz]<0.)\
                                          +(VmaxSIDM_acc[iz]<0.)+(rmaxSIDM_acc[iz]<0.)\
                                          +(rcSIDM_z0[iz]<0.)+(rcSIDM_acc[iz]<0.))==1,False,True)


        Na           = self.Na_calc(ma_matrix,zdist,M0,z0=0.,N_herm=N_hermNa,Nrand=1000,
                                    Na_model=Na_model)
        Na_total     = integrate.simps(integrate.simps(Na,x=np.log(ma_matrix)),x=np.log(1+zdist))
        weightCDM    = Na/(1.+zdist.reshape(-1,1))
        weightCDM    = weightCDM/np.sum(weightCDM)*Na_total
        weightCDM    = np.expand_dims(weightCDM,axis=1)*(w1.reshape(-1,1))/np.sqrt(np.pi)
        weightCDM    = weightCDM[accretion.any(axis=1)]*surviveCDM*np.expand_dims(accretion[accretion.any(axis=1)],axis=1)  # consider only the redshifts where at least one subhalo accretes
        weightSIDM   = weightCDM*surviveSIDM
        z_acc        = (zdist_accreted.reshape(-1,1,1))*np.ones((1,N_herm,N_ma))
        z_acc        = z_acc.reshape(-1)

        ma200         = np.expand_dims(ma200_matrix_accreted,axis=1)
        ma200         = ma200*np.ones((N_herm,1))
        ma200         = ma200.reshape(-1)
        m_z0          = m0CDM_matrix.reshape(-1)
        rsCDM_acc     = rsCDM_acc.reshape(-1)
        rhosCDM_acc   = rhosCDM_acc.reshape(-1)
        rmaxCDM_acc   = rmaxCDM_acc.reshape(-1)
        VmaxCDM_acc   = VmaxCDM_acc.reshape(-1)
        rsCDM_z0      = rsCDM_z0.reshape(-1)
        rhosCDM_z0    = rhosCDM_z0.reshape(-1)
        rmaxCDM_z0    = rmaxCDM_z0.reshape(-1)
        VmaxCDM_z0    = VmaxCDM_z0.reshape(-1)
        rsSIDM_acc    = rsSIDM_acc.reshape(-1)
        rcSIDM_acc    = rcSIDM_acc.reshape(-1)
        rhosSIDM_acc  = rhosSIDM_acc.reshape(-1)
        rmaxSIDM_acc  = rmaxSIDM_acc.reshape(-1)
        VmaxSIDM_acc  = VmaxSIDM_acc.reshape(-1)
        rsSIDM_z0     = rsSIDM_z0.reshape(-1)
        rhosSIDM_z0   = rhosSIDM_z0.reshape(-1)
        rcSIDM_z0     = rcSIDM_z0.reshape(-1)
        rmaxSIDM_z0   = rmaxSIDM_z0.reshape(-1)
        VmaxSIDM_z0   = VmaxSIDM_z0.reshape(-1)
        ctCDM_z0      = ctCDM_z0.reshape(-1)
        tt_ratio      = tt_ratio.reshape(-1)
        weightCDM     = weightCDM.reshape(-1)
        weightSIDM    = weightSIDM.reshape(-1)
        surviveCDM    = surviveCDM.reshape(-1)
        surviveSIDM   = surviveSIDM.reshape(-1)

        return ma200, z_acc, rsCDM_acc, rhosCDM_acc, rmaxCDM_acc, VmaxCDM_acc, rsSIDM_acc, rhosSIDM_acc, rcSIDM_acc, rmaxSIDM_acc, VmaxSIDM_acc, m_z0, rsCDM_z0, rhosCDM_z0, rmaxCDM_z0, VmaxCDM_z0, rsSIDM_z0, rhosSIDM_z0, rcSIDM_z0, rmaxSIDM_z0, VmaxSIDM_z0, ctCDM_z0, tt_ratio, weightCDM, weightSIDM, surviveCDM, surviveSIDM, 
