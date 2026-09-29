"""
Reduced-order model of the cyclone-integrated multistage vortex reactor (MSVR)

S. Chen et al., Scale-up of Vortex Reactors for CO2 Capture: from Reactive CFD
to Design Models (AIChE Journal). The equation numbers refer to the Supporting
Information (Eqs. S2-S18).

The script contains the CFD data set of the 23 calibration cases, the model
equations and the nonlinear least-squares calibration. Running it

    python msvr_rom.py

prints the fitted constants (Table 2) and the deviations from the CFD results,
and writes the model and CFD values of every case to parity.csv.

Requires numpy and scipy.

Swirl coefficient
    The swirl coefficient SF_c of Eq. (S3) enters the model only through the
    cyclone swirl velocity v_th,c = SF_c * u_c with

        u_c = Q_G / (2 pi R_in H) * (R_C1 / R_c).

    In Eqs. (S4), (S15) and (S16) it therefore appears only in the combinations
    Eu*SF_c**2, v_a/SF_c and v_crit/SF_c. These are fitted as single constants.

Units
    G in Nm3/h, L in kg/h, Q_G = G/3600 in m3/s, lengths in m,
    velocities in m/s, phi = L/G in kg/Nm3, dP in Pa.
"""
import csv

import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------
# CFD data set
# ----------------------------------------------------------------------------
# label, G [Nm3/h], L [kg/h], dP [kPa], eta_G,sep, eta_L,sep, eta_abs
CFD_DATA = [
    # geometry series
    ('S1-S2-C1-H1',  40, 260, 38.5, 0.958, 0.927, 0.672),
    ('S1-S3-C1-H1',  40, 260, 57.5, 0.925, 0.859, 0.682),
    ('S1-S5-C2-H1',  40, 260, 66.5, 0.947, 0.999, 0.688),
    ('S1-S5-C1-H1',  40, 260, 92.7, 0.627, 0.605, 0.689),
    ('S1-S5-C3-H1',  40, 260, 91.5, 0.247, 0.996, 0.688),
    ('SE2-S3-C1-H1', 40, 260, 87.2, 0.941, 0.865, 0.836),
    ('SE2-S1-C1-H1', 40, 260, 56.1, 0.950, 0.886, 0.812),
    ('SE1-S3-C1-H1', 40, 260, 69.3, 0.888, 0.733, 0.754),
    ('SE2-S3-C1-H2', 40, 260, 23.7, 0.924, 0.692, 0.886),
    ('SE2-S3-C1-H3', 40, 260,  4.5, 0.821, 0.484, 0.782),
    ('SE2-S1-C1-H3', 40, 260,  3.1, 0.821, 0.484, 0.831),
    ('S1-S5-C2-H2',  40, 260, 18.9, 0.947, 0.968, 0.759),
    ('S1-S5-C2-H3',  40, 260,  4.8, 0.964, 0.987, 0.823),
    ('S1-S3-C1-H2',  40, 260, 16.4, 0.946, 0.858, 0.855),
    ('S1-S3-C1-H3',  40, 260,  4.6, 0.918, 0.581, 0.863),
    # operating conditions, S1-S3-C1-H1
    ('S1-S3-C1-H1',  20, 260, 23.7, 0.972, 0.941, 0.803),
    ('S1-S3-C1-H1',  20, 180, 16.5, 0.937, 0.958, 0.759),
    ('S1-S3-C1-H1',  20, 100, 11.9, 0.915, 0.975, 0.744),
    ('S1-S3-C1-H1',  30, 260, 38.2, 0.956, 0.925, 0.734),
    # operating conditions, SE2-S3-C1-H1
    ('SE2-S3-C1-H1', 20, 260, 33.8, 0.957, 0.921, 0.913),
    ('SE2-S3-C1-H1', 20, 180, 25.9, 0.934, 0.897, 0.903),
    ('SE2-S3-C1-H1', 20, 100, 17.9, 0.902, 0.906, 0.893),
    ('SE2-S3-C1-H1', 30, 260, 59.1, 0.937, 0.886, 0.865),
]

# ----------------------------------------------------------------------------
# Geometry (Table 1). Label: [outermost stage]-[innermost stage]-[cyclone]-[height]
# ----------------------------------------------------------------------------
STAGE_RADIUS = {'SE2': 0.150, 'SE1': 0.135, 'S1': 0.110, 'S2': 0.095,
                'S3': 0.075, 'S4': 0.055, 'S5': 0.035}
STAGE_ORDER = ['SE2', 'SE1', 'S1', 'S2', 'S3', 'S4', 'S5']
CHAMBER_HEIGHT = {'H1': 0.0075, 'H2': 0.015, 'H3': 0.030}
CYCLONE = {'C1': (0.020, True),   # radius [m], conical bottom
           'C2': (0.035, True),
           'C3': (0.020, False)}  # straight bottom


def geometry(label):
    """Return R_out, R_in, N_stage, R_c, H and the bottom type of a configuration."""
    outer, inner, cyclone, height = label.split('-')
    i, j = STAGE_ORDER.index(outer), STAGE_ORDER.index(inner)
    R_c, conical = CYCLONE[cyclone]
    return dict(Rout=STAGE_RADIUS[outer], Rin=STAGE_RADIUS[inner], N=j - i + 1,
                Rc=R_c, H=CHAMBER_HEIGHT[height], conical=conical)


# ----------------------------------------------------------------------------
# Fixed values
# ----------------------------------------------------------------------------
RHO_G = 1.2                              # gas density, kg/m3
RHO_L = 1012.0                           # liquid density, kg/m3
A_W = 0.024                              # open slot width of every stage wall, m
R_C1 = 0.020                             # radius of the reference cyclone C1, m
PHI_0 = 260.0 / 40.0                     # reference liquid load, kg/Nm3
EPS_L0 = 0.09                            # liquid holdup of the reference case (CFD)
V_0 = (40.0 / 3600.0) / (A_W * 0.0075)   # reference injection velocity, 61.7 m/s
q = 2.0                                  # exponent q of the radius ratio in Eq. (S15)
F_B_C3 = 0.25                            # gas separation of the straight-bottomed C3 (CFD)

# Fitted constants
#   K_ch  = C_dP/2          chamber loss                     Eq. (S4)
#   K_cy  = Eu*SF_c**2/2    cyclone loss                     Eq. (S4)
#   b_L                     holdup exponent                  Eq. (S6)
#   C_NTU                   absorption coefficient, 1/s      Eq. (S14)
#   p, s_L                  NTU exponents                    Eq. (S14)
#   V_a   = v_a/SF_c        collection velocity, m/s         Eq. (S15)
#   V_cr  = v_crit/SF_c     re-entrainment onset, m/s        Eq. (S16)
#   k_ret                   retention coefficient            Eq. (S16)
#   k_cu                    carry-under coefficient          Eq. (S18)
NAMES = ['K_ch', 'K_cy', 'b_L', 'C_NTU', 'p', 's_L', 'V_a', 'V_cr', 'k_ret', 'k_cu']
LOWER = [1e-4, 1.0, 0.0, 1.0, -1.0, -1.0, 0.01, 0.1, 1e-3, 0.0]
UPPER = [10.0, 1e4, 3.0, 1e4, 3.0, 2.0, 50.0, 50.0, 10.0, 3.0]
# rounding used in Table 2: ('sig', n) significant digits, ('dec', n) decimals
ROUNDING = [('sig', 2), ('sig', 3), ('dec', 2), ('sig', 3), ('dec', 2), ('dec', 2),
            ('sig', 2), ('sig', 3), ('dec', 2), ('dec', 2)]


# ----------------------------------------------------------------------------
# Model, SI Eqs. (S2)-(S18)
# ----------------------------------------------------------------------------
def rom(c, Rout, Rin, N, Rc, H, G, L, conical=True):
    """Evaluate the reduced-order model.

    c        dict of the fitted constants (keys in NAMES)
    Rout, Rin, Rc, H in m, N number of stages, G in Nm3/h, L in kg/h,
    conical  True for the conical-bottomed cyclones C1 and C2.
    Returns a dict with dP [Pa], eta_abs, eta_L, eta_G and intermediate values.
    """
    Rout, Rin, N, Rc, H, G, L = (np.asarray(a, float) for a in (Rout, Rin, N, Rc, H, G, L))
    conical = np.asarray(conical, bool)
    Q_G = G / 3600.0
    phi = L / G

    v_H = Q_G / (A_W * H)                                               # (S2)
    u_c = Q_G / (2.0 * np.pi * Rin * H) * (R_C1 / Rc)                   # (S3) = v_th,c / SF_c

    eps_L = EPS_L0 * (phi / PHI_0) ** c['b_L']                          # (S6)
    rho_m = (1.0 - eps_L) * RHO_G + eps_L * RHO_L                       # (S5)
    dP = c['K_ch'] * rho_m * N * v_H ** 2 + c['K_cy'] * RHO_G * u_c ** 2   # (S4)

    NTU = (c['C_NTU'] * (Rout ** 2 - Rc ** 2) * H / Q_G
           * (v_H / V_0) ** c['p'] * (phi / PHI_0) ** c['s_L'])        # (S14)
    eta_abs = 1.0 - np.exp(-NTU)                                        # (S12)

    eta_coll = 1.0 - np.exp(-(u_c / c['V_a']) * (Rc / R_C1) ** q)      # (S15), q = 2
    eta_ret = np.where(u_c <= c['V_cr'], 1.0,
                       np.exp(-c['k_ret'] * (u_c / c['V_cr'] - 1.0)))  # (S16)
    # the straight-bottomed C3 drains its liquid through the open bottom
    eta_L = np.where(conical, eta_coll * eta_ret, 1.0)                  # (S17)
    f_b = np.where(conical, 1.0, F_B_C3)
    eta_G = f_b * (1.0 - c['k_cu'] * (1.0 - eta_L))                    # (S18)

    return dict(dP=dP, eta_abs=eta_abs, eta_L=eta_L, eta_G=eta_G,
                v_H=v_H, u_c=u_c, NTU=NTU, eta_coll=eta_coll, eta_ret=eta_ret)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def load_data():
    d = {k: [] for k in ('label', 'G', 'L', 'dP', 'eta_G', 'eta_L', 'eta_abs',
                         'Rout', 'Rin', 'N', 'Rc', 'H', 'conical')}
    for label, G, L, dP, eG, eL, eA in CFD_DATA:
        g = geometry(label)
        for k, v in (('label', label), ('G', G), ('L', L), ('dP', dP * 1e3),
                     ('eta_G', eG), ('eta_L', eL), ('eta_abs', eA)):
            d[k].append(v)
        for k in ('Rout', 'Rin', 'N', 'Rc', 'H', 'conical'):
            d[k].append(g[k])
    return {k: (v if k == 'label' else np.array(v)) for k, v in d.items()}


def predict(c, d):
    return rom(c, d['Rout'], d['Rin'], d['N'], d['Rc'], d['H'], d['G'], d['L'], d['conical'])


def residuals(x, d):
    """Relative error of dP, absolute errors of the efficiencies.

    The two separation efficiencies of C3 are not fitted, since f_b and
    eta_L,sep = 1 are taken from its simulation. 90 residuals in total.
    """
    m = predict(dict(zip(NAMES, x)), d)
    con = d['conical']
    return np.concatenate([np.log(m['dP'] / d['dP']),
                           m['eta_abs'] - d['eta_abs'],
                           (m['eta_L'] - d['eta_L'])[con],
                           (m['eta_G'] - d['eta_G'])[con]])


def fit(d, n_start=200, seed=0):
    """Multi-start nonlinear least squares."""
    lo, hi = np.array(LOWER), np.array(UPPER)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        x0 = np.where(lo > 0, np.exp(rng.uniform(np.log(np.maximum(lo, 1e-4)), np.log(hi))),
                      rng.uniform(lo, hi))
        r = least_squares(residuals, x0, args=(d,), bounds=(lo, hi), x_scale='jac',
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return dict(zip(NAMES, best.x))


def round_constants(c):
    out = {}
    for (k, v), (kind, n) in zip(c.items(), ROUNDING):
        out[k] = float(round(v, n)) if kind == 'dec' else float(f'{v:.{n}g}')
    return out


def main(csv_path='parity.csv'):
    d = load_data()
    c_fit = fit(d)
    c = round_constants(c_fit)             # Table 2 values
    m = predict(c, d)

    print('Fitted constants (Table 2)')
    labels = {'K_ch': 'C_dP/2', 'K_cy': 'Eu*SF_c^2/2', 'V_a': 'v_a/SF_c',
              'V_cr': 'v_crit/SF_c'}
    for k in NAMES:
        print(f'  {labels.get(k, k):12s} {c_fit[k]:12.5g}  ->  {c[k]:g}')

    con = d['conical']
    rel = (m['dP'] - d['dP']) / d['dP']
    print('\nDeviation from CFD (Table 2 constants, 23 cases)')
    print(f'  pressure drop      mean {100 * np.mean(np.abs(rel)):.1f} %, '
          f'max {100 * np.max(np.abs(rel)):.1f} %')
    for key, name, mask in (('eta_abs', 'CO2 absorption', np.ones_like(con)),
                            ('eta_L', 'liquid separation', con),
                            ('eta_G', 'gas separation', con)):
        err = 100 * np.abs(m[key] - d[key])[mask]
        print(f'  {name:18s} mean {err.mean():.1f} pp, max {err.max():.1f} pp')

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'G [Nm3/h]', 'L [kg/h]',
                    'dP CFD [kPa]', 'dP ROM [kPa]',
                    'eta_G,sep CFD [%]', 'eta_G,sep ROM [%]',
                    'eta_L,sep CFD [%]', 'eta_L,sep ROM [%]',
                    'eta_abs CFD [%]', 'eta_abs ROM [%]'])
        for i, label in enumerate(d['label']):
            w.writerow([label, int(d['G'][i]), int(d['L'][i]),
                        f"{d['dP'][i] / 1e3:.1f}", f"{m['dP'][i] / 1e3:.1f}",
                        f"{100 * d['eta_G'][i]:.1f}", f"{100 * m['eta_G'][i]:.1f}",
                        f"{100 * d['eta_L'][i]:.1f}", f"{100 * m['eta_L'][i]:.1f}",
                        f"{100 * d['eta_abs'][i]:.1f}", f"{100 * m['eta_abs'][i]:.1f}"])
    print(f'\nModel and CFD values written to {csv_path}')
    return c, m


if __name__ == '__main__':
    main()
