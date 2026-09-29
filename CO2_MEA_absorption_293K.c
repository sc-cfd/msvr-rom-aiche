/*============================================================
 * File   : CO2_MEA_absorption_293K.c
 * Brief  : UDF for CO2 absorption into MEA solution
 *          in a gas-liquid multiphase flow (Euler-Euler)
 *          Isothermal, 293.15 K
 *
 * Species index:
 *   Liquid phase  -- MEA (1)
 *   Gas    phase  -- CO2 (0), N2  (1)
 *
 * UDMI storage:
 *   UDMI[0] -- turbulent energy dissipation  [m²/s³]
 *   UDMI[1] -- MEA mass source term          [kg/(m³·s)]
 *============================================================*/

#include "udf.h"
#include "mem.h"
#include "sg_mphase.h"
#include "metric.h"
#include "global.h"
#define Henry   0.000685    /* Henry constant of CO2 in MEA solution    [mol/(L·kPa)] */
#define density 1012        /* liquid phase density                      [kg/m³]       */
#define K       0.167       /* coefficient in small eddy model                         */

/*------------------------------------------------------------
 * DEFINE_SOURCE : CO2_src
 * Returns : CO2 mass source term in the GAS phase  [kg/(m³·s)]
 *           Negative -> gas loses CO2 (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(CO2_src, cell, thread_gas, dS, eqn)
{
    Thread *thread_mix, *thread_liq;
    thread_mix = THREAD_SUPER_THREAD(thread_gas);        /* mixture thread              */
    thread_liq = THREAD_SUB_THREAD(thread_mix, 0);       /* liquid sub-thread (index 0) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_co2;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- CO2 gas-phase mass source term --- */
        /* -1 * E * kL [m/s] * ae [m²/m³] * M_CO2 [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_co2 = - E * kL * ae * 44.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        m_dot_co2 = 0.0;
        dS[eqn]   = 0.0;
    }

    return m_dot_co2;
}


/*------------------------------------------------------------
 * DEFINE_SOURCE : gas_src
 * Returns : total mass source term of the GAS phase  [kg/(m³·s)]
 *           Only CO2 leaves the gas phase, M_CO2 = 44 kg/kmol
 *           Negative -> gas phase loses mass (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(gas_src, cell, thread_gas, dS, eqn)
{
    Thread *thread_mix, *thread_liq;
    thread_mix = THREAD_SUPER_THREAD(thread_gas);        /* mixture thread              */
    thread_liq = THREAD_SUB_THREAD(thread_mix, 0);       /* liquid sub-thread (index 0) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_gas;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- total gas-phase mass source term, only CO2 leaves the gas phase --- */
        /* -1 * E * kL [m/s] * ae [m²/m³] * M_CO2 [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_gas = - E * kL * ae * 44.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        m_dot_gas = 0.0;
        dS[eqn]   = 0.0;
    }

    return m_dot_gas;
}


/*------------------------------------------------------------
 * DEFINE_SOURCE : liqmea_src
 * Returns : MEA mass source term in the LIQUID phase  [kg/(m³·s)]
 *           Negative -> liquid consumes MEA (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(liqmea_src, cell, thread_liq, dS, eqn)
{
    Thread *thread_mix, *thread_gas;
    thread_mix = THREAD_SUPER_THREAD(thread_liq);        /* mixture thread           */
    thread_gas = THREAD_SUB_THREAD(thread_mix, 1);       /* gas sub-thread (index 1) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_mea;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- MEA liq-phase mass source term --- */
        /* -2 * E * kL [m/s] * ae [m²/m³] * M_MEA [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_mea = - 2.0 * E * kL * ae * 61.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        c_d       = 0.0;
        m_dot_mea = 0.0;
        dS[eqn]   = 0.0;
    }
    C_UDMI(cell, thread_liq, 0) = c_d;
    C_UDMI(cell, thread_liq, 1) = m_dot_mea;
    return m_dot_mea;
}


/*------------------------------------------------------------
 * DEFINE_SOURCE : liqmeah_src
 * Returns : MEAH mass source term in the LIQUID phase  [kg/(m³·s)]
 *           Positive -> liquid generates MEAH (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(liqmeah_src, cell, thread_liq, dS, eqn)
{
    Thread *thread_mix, *thread_gas;
    thread_mix = THREAD_SUPER_THREAD(thread_liq);        /* mixture thread           */
    thread_gas = THREAD_SUB_THREAD(thread_mix, 1);       /* gas sub-thread (index 1) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_meah;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- MEAH liq-phase mass source term --- */
        /* E * kL [m/s] * ae [m²/m³] * M_MEAH [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_meah = E * kL * ae * 62.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        m_dot_meah = 0.0;
        dS[eqn]    = 0.0;
    }

    return m_dot_meah;
}


/*------------------------------------------------------------
 * DEFINE_SOURCE : liqmeacoo_src
 * Returns : MEACOO mass source term in the LIQUID phase  [kg/(m³·s)]
 *           Positive -> liquid generates MEACOO (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(liqmeacoo_src, cell, thread_liq, dS, eqn)
{
    Thread *thread_mix, *thread_gas;
    thread_mix = THREAD_SUPER_THREAD(thread_liq);        /* mixture thread           */
    thread_gas = THREAD_SUB_THREAD(thread_mix, 1);       /* gas sub-thread (index 1) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_meacoo;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- MEACOO liq-phase mass source term --- */
        /* E * kL [m/s] * ae [m²/m³] * M_MEACOO [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_meacoo = E * kL * ae * 104.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        m_dot_meacoo = 0.0;
        dS[eqn]      = 0.0;
    }

    return m_dot_meacoo;
}


/*------------------------------------------------------------
 * DEFINE_SOURCE : liq_src
 * Returns : total mass source term of the LIQUID phase  [kg/(m³·s)]
 *           The absorbed CO2 enters the liquid, M_CO2 = 44 kg/kmol
 *           Positive -> liquid phase gains mass (absorption)
 *------------------------------------------------------------*/
DEFINE_SOURCE(liq_src, cell, thread_liq, dS, eqn)
{
    Thread *thread_mix, *thread_gas;
    thread_mix = THREAD_SUPER_THREAD(thread_liq);        /* mixture thread           */
    thread_gas = THREAD_SUB_THREAD(thread_mix, 1);       /* gas sub-thread (index 1) */

    real ae, c_mea, d_n2o, d_co2, c_d, kL, k_r, Ha, E;
    real yL, m_dot_liq;

    if (C_VOF(cell, thread_gas) < 0.99 && C_VOF(cell, thread_gas) > 0.01)
    {
        /* --- interfacial area density (spherical bubble assumption) --- */
        /* ae = 6 * alpha_g / d_b  [m²/m³] */
        ae = 6.0 * C_VOF(cell, thread_gas) / C_PHASE_DIAMETER(cell, thread_gas);

        /* --- molar concentration --- */
        c_mea = C_YI(cell, thread_liq, 1) * density / 61.0;         /* MEA     [mol/L] */

        /* --- diffusivity via N2O analogy --- */
        d_n2o = (5.07 + 0.865 * c_mea + 0.278 * c_mea * c_mea)
                * exp((-2371.0 - 93.4 * c_mea) / 293.15) * 1e-6;    /* D_N2O  [m²/s] */
        d_co2 = d_n2o * 1.09;                                        /* D_CO2  [m²/s] */

        /* --- turbulent energy dissipation rate --- */
        /* epsilon = C_mu * k * omega  [m²/s³] */
        c_d = 0.09 * C_K(cell, thread_liq) * C_O(cell, thread_liq);

        /* --- liquid-side mass transfer coefficient (small eddy model) --- */
        kL = K * pow(d_co2, 0.5) * pow((c_d / 0.00000189), 0.25);   /* [m/s] */

        /* --- second-order reaction rate constant --- */
        k_r = 4450.0;                                               /* [L/(mol·s)] */

        /* --- Hatta number and enhancement factor --- */
        Ha = pow((k_r * d_co2 * c_mea), 0.5) / kL;       /* Hatta number          [-] */
        E  = pow((1.0 + pow(Ha, 2)), 0.5);               /* enhancement factor (fast reaction approximation) */

        /* --- interfacial CO2 concentration, bulk liquid CO2 ~ 0  [mol/L] --- */
        yL = (C_P(cell, thread_gas) / 1000.0 + 101.325) * Henry
             * (1.138 * C_YI(cell, thread_gas, 0)
             / (1.7878 - 0.6498 * C_YI(cell, thread_gas, 0)));

        /* --- total liquid-phase mass source term, only CO2 enters the liquid phase --- */
        /* E * kL [m/s] * ae [m²/m³] * M_CO2 [kg/kmol] * yL [mol/L] = [kg/(m³·s)] */
        m_dot_liq = E * kL * ae * 44.0 * yL;

        dS[eqn] = 0.0;
    }
    else
    {
        /* source set to zero when VOF is outside the valid interface range */
        m_dot_liq = 0.0;
        dS[eqn]   = 0.0;
    }

    return m_dot_liq;
}
