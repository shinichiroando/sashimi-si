# Semi-Analytical SubHalo Inference ModelIng for Self-Interacting Dark Matter (SASHIMI-SIDM)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for self-interacting dark matter (SIDM). The results are well in agreement with those from numerical N-body simulations.

## Authors

- Shin'ichiro Ando
- Shunichi Horigome
- Ethan Nadler
- Daneng Yang
- Hai-Bo Yu

Please send enquiries to Shin'ichiro Ando (s.ando@uva.nl). We have checked that the codes work with python 3.10 but cannot guarantee for other versions of python. In any case, we cannot help with any technical issues not directly related to the content of SASHIMI (such as installation, sub-packages required, etc.)

## What can we do with SASHIMI-SIDM?

- SASHIMI provides a full catalog of dark matter subhalos in a host halo with arbitrary mass and redshift, which is calculated with semi-analytical models.
- Each subhalo in this catalog is characterized by its mass and density profile both at accretion and at the redshift of interest, accretion redshift, and effective number (or weight) corresponding to that particular subhalo.
- It can be used to quickly compute the subhalo mass function without making any assumptions such as power-law functional forms, etc. Only power law that we assume here is the one for primordial power spectrum predicted by inflation. Everything else is calculated theoretically.
- SASHIMI is not limited to numerical resolution which is often the most crucial limiting factor for the numerical simulation. One can easily set the minimum halo mass to be a micro solar mass or even lighter.
- SASHIMI is not limited to Poisson shot noise that inevitably accompanies when one has to count subhalos like in the case of numerical simulations.
- One can adopt both velocity-independent and velocity-dependent scattering cross sections of SIDM. For the latter, so far, the Rutherford-like scattering is used but this can be easily extended for a different type of interaction.

## References

When you use the outcome of this package for your scientific output, please cite the following publications.

- S. Ando, S. Horigome, E. O. Nadler, D. Yang, and H.-B. Yu, [https://arxiv.org/abs/xxxx.xxxx]
- D. Yang, E. O. Nadler, H.-B. Yu, and Y.-M. Zhong [https://arxiv.org/abs/2305.16176]


Note this is one of the variants of SASHIMI, which is based on its original version for CDM [https://github.com/shinichiroando/sashimi-c]


## Examples

The file 'sashimi_si.py' contains all the variables and functions that are used to compute various subhalo properties. Please read 'sample.ipynb' for more extensive examples.

Here, as a minimal example, is how you generate a semi-analytical catalog of subhalos:

```
from sashimi_si import *

sh = subhalo_properties()  # call the relevant class
M0 = 1.e12*sh.Msun         # input of host halo mass; here 10^{12} solar masses

ma200, z_acc, rsCDM_acc, rhosCDM_acc, rmaxCDM_acc, VmaxCDM_acc, rsSIDM_acc, rhosSIDM_acc, rcSIDM_acc, rmaxSIDM_acc, VmaxSIDM_acc, m_z0, rsCDM_z0, rhosCDM_z0, rmaxCDM_z0, VmaxCDM_z0, rsSIDM_z0, rhosSIDM_z0, rcSIDM_z0, rmaxSIDM_z0, VmaxSIDM_z0, ctCDM_z0, tt_ratio, weightCDM, weightSIDM, surviveCDM, surviveSIDM = sh.subhalo_properties_calc(M0)
```

For inputs and outputs of this function, see its documentation. For reference, it is:

```
-----
Input
-----
M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
    Note that this is *not* the host mass at the given redshift! It can be obtained
    via Mzi(M0,redshift). If you want to give this parameter as the mass at the given
    redshift, then turn 'M0_at_redshift' parameter on (see below).
        
(Optional) redshift:       Redshift of interest. (default: 0)
(Optional) dz:             Grid of redshift of halo accretion. (default 0.01)
(Optional) zmax:           Maximum redshift to start the calculation of evolution from. (default: 5.0)
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
```
