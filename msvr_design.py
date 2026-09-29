"""
Model-based design of a new MSVR and design-space map (Section 4.4, Figure 8)

S. Chen et al., Scale-up of Vortex Reactors for CO2 Capture: from Reactive CFD
to Design Models (AIChE Journal).

The model equations are imported from msvr_rom.py, which has to be in the same
folder. The constants are those of Table 2. Running

    python msvr_design.py

prints the selected design at G = 60 Nm3/h and L = 216 kg/h and writes
    design_space.csv  design-space map with the Pareto-optimal set
    Figure8.png       design-space map (needs matplotlib)
    Figure8_3D.png    static three-dimensional view (needs matplotlib)
    Figure8_3D.html   interactive, rotatable three-dimensional view (needs plotly)

Requires numpy (matplotlib and plotly for the figures).
"""
import csv

import numpy as np

from msvr_rom import A_W, geometry, rom

# Table 2 constants, fitted with msvr_rom.py
TABLE2 = {'K_ch': 0.048, 'K_cy': 205.0, 'b_L': 0.73, 'C_NTU': 141.0, 'p': 0.69,
          's_L': 0.13, 'V_a': 0.78, 'V_cr': 1.82, 'k_ret': 0.19, 'k_cu': 0.49}

# ----------------------------------------------------------------------------
# Design duty and targets
# ----------------------------------------------------------------------------
G_DESIGN = 60.0                          # design gas flow rate, Nm3/h
L_DESIGN = 216.0                         # design liquid flow rate, kg/h (L/G = 3 kg/kg)
PITCH = 0.020                            # uniform radial pitch of the stages, m
TURNDOWN = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)   # G and L scaled together
TARGETS = dict(eta_abs=0.90, eta_L=0.95, eta_G=0.95, dP=25e3)

# ----------------------------------------------------------------------------
# Design space
#   R_in and R_c stay within the simulated ranges (35-110 mm and 20-35 mm).
#
#   R_out may exceed the largest simulated radius (150 mm), up to 180 mm.
#   With R_out <= 150 mm and the height limit below, no geometry meets all four
#   targets over the turndown. The selected design has R_out = 153 mm.
#
#   H is limited so that the injection velocity at the design duty is not lower
#   than that of the H2 simulations (G = 40 Nm3/h, H = 15 mm, v_H = 30.9 m/s).
#   This gives H <= 22.5 mm. At lower injection velocities the model
#   over-predicts the absorption of the radially expanded reactor, since it does
#   not capture the loss of turbulence at large chamber heights (Section 4.2).
#   SE2-S3-C1-H3 (v_H = 15.4 m/s) is predicted at 96 % against 78 % in the CFD.
#   Without this limit the search would select H = 30 mm (dP = 9.6 kPa), in the
#   region where the predicted absorption is not reliable.
# ----------------------------------------------------------------------------
V_H_MIN = (40.0 / 3600.0) / (A_W * 0.015)          # 30.9 m/s
R_OUT_RANGE = (0.110, 0.180)
R_IN_RANGE = (0.035, 0.110)
R_C_RANGE = (0.020, 0.035)
H_RANGE = (0.0075, round((G_DESIGN / 3600.0) / (A_W * V_H_MIN), 6))   # 7.5-22.5 mm
N_RANGE = (2, 8)
DP_TOL = 100.0                           # Pa, pressure drops within this margin count as equal
REFERENCE = 'SE2-S3-C1-H2'               # simulated case with the highest absorption


def design_grid(step_rout=0.001, step_rc=0.0005, step_h=0.00025):
    """All combinations of R_out, N_stage, R_c and H, with R_in = R_out - (N - 1) * PITCH."""
    triples = []
    for N in range(N_RANGE[0], N_RANGE[1] + 1):
        for Rout in np.arange(R_OUT_RANGE[0], R_OUT_RANGE[1] + 1e-9, step_rout):
            Rin = Rout - (N - 1) * PITCH
            if R_IN_RANGE[0] - 1e-9 <= Rin <= R_IN_RANGE[1] + 1e-9:
                triples.append((Rout, Rin, N))
    t = np.round(np.array(triples), 6)
    Rc = np.round(np.arange(R_C_RANGE[0], R_C_RANGE[1] + 1e-9, step_rc), 6)
    H = np.round(np.arange(H_RANGE[0], H_RANGE[1] + 1e-9, step_h), 6)
    n1, n2, n3 = len(t), len(Rc), len(H)
    return dict(Rout=np.repeat(t[:, 0], n2 * n3), Rin=np.repeat(t[:, 1], n2 * n3),
                N=np.repeat(t[:, 2], n2 * n3), Rc=np.tile(np.repeat(Rc, n3), n1),
                H=np.tile(H, n1 * n2))


def evaluate(c, g, factor=1.0):
    """Model at factor * (G_DESIGN, L_DESIGN) for conical-bottomed cyclones."""
    return rom(c, g['Rout'], g['Rin'], g['N'], g['Rc'], g['H'],
               factor * G_DESIGN, factor * L_DESIGN, True)


def meets_targets(c, g):
    """True where all four targets hold over the whole turndown range."""
    ok = np.ones(np.shape(g['H']), bool)
    for f in TURNDOWN:
        m = evaluate(c, g, f)
        ok &= ((m['eta_abs'] >= TARGETS['eta_abs']) & (m['eta_L'] >= TARGETS['eta_L'])
               & (m['eta_G'] >= TARGETS['eta_G']) & (m['dP'] <= TARGETS['dP']))
    return ok


def design(c):
    """Minimum pressure drop, then smallest outer radius, then highest eta_L,sep."""
    g = design_grid()
    m = evaluate(c, g)
    idx = np.flatnonzero(meets_targets(c, g))
    near = idx[m['dP'][idx] <= m['dP'][idx].min() + DP_TOL]
    best = near[np.lexsort((-m['eta_L'][near], g['Rout'][near]))[0]]
    return {k: float(g[k][best]) for k in g}


def pareto(dP, eta_abs, eta_L, mask):
    """Non-dominated set for low dP, high eta_abs and high eta_L,sep within mask."""
    X = np.column_stack([-dP, eta_abs, eta_L])
    idx = np.flatnonzero(mask)
    front = np.zeros(len(dP), bool)
    for i in idx:
        dominated = np.all(X[idx] >= X[i], axis=1) & np.any(X[idx] > X[i], axis=1)
        front[i] = not dominated.any()
    return front


def design_space(c, n_points=6000, seed=1):
    """Map of the design space (Figure 8).

    A random sample of n_points geometries is drawn from the fine design grid
    (R_out step 1 mm, R_c step 0.5 mm, H step 0.25 mm). The Pareto-optimal set
    is built for dP, eta_abs and eta_L,sep among the sampled geometries that
    meet the gas separation target at the design duty.
    """
    g = design_grid()
    pick = np.sort(np.random.default_rng(seed).choice(len(g['H']), n_points, replace=False))
    g = {k: v[pick] for k, v in g.items()}
    m = evaluate(c, g)
    front = pareto(m['dP'], m['eta_abs'], m['eta_L'], m['eta_G'] >= TARGETS['eta_G'])
    return g, m, front, meets_targets(c, g)


def plot_figure8(m, front, best, ref, path='Figure8.png'):
    """Design-space map: dP against eta_abs, coloured by eta_L,sep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    with plt.rc_context({'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'font.family': 'DejaVu Sans',
                         'axes.linewidth': 0.9}):
        dP, eA, eL = m['dP'] / 1e3, 100 * m['eta_abs'], 100 * m['eta_L']
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sc = ax.scatter(dP, eA, c=eL, cmap='RdYlGn', vmin=70, vmax=100, s=7, alpha=0.6, lw=0,
                        zorder=2)
        ax.scatter(dP[front], eA[front], facecolors='none', edgecolors='k', s=26, lw=0.8, zorder=3,
                   label='Pareto-optimal set')
        ax.scatter([float(best['dP']) / 1e3], [100 * float(best['eta_abs'])], marker='*', s=300,
                   c='crimson', ec='k', lw=0.8, zorder=6, label='selected design')
        ax.scatter([float(ref['dP']) / 1e3], [100 * float(ref['eta_abs'])], marker='D', s=65,
                   c='royalblue', ec='k', lw=0.8, zorder=6,
                   label=f'{REFERENCE} (re-evaluated at this duty)')
        ax.axhline(100 * TARGETS['eta_abs'], color='k', ls=':', lw=1.1, zorder=1,
                   label=r'$\eta_{abs}$ = 90% target')
        ax.set_xscale('log')
        ax.set_xlim(4, 300)
        ax.set_ylim(50, 100)
        ax.set_xticks([5, 10, 20, 50, 100, 200])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(r'Pressure drop $\Delta P$ [kPa]')
        ax.set_ylabel(r'CO$_2$ absorption efficiency $\eta_{abs}$ [%]')
        ax.set_title(r'(colour: $\eta_{L,sep}$; Pareto: $\eta_{abs}$, $\eta_{L,sep}$, $\Delta P$)',
                     fontsize=11.5, pad=8)
        cb = plt.colorbar(sc, ax=ax, pad=0.015)
        cb.set_label(r'Liquid separation efficiency $\eta_{L,sep}$ [%]')
        ax.legend(loc='lower left', fontsize=8.5, markerscale=0.85, borderpad=0.45,
                  labelspacing=0.35, handletextpad=0.45, framealpha=0.95).set_zorder(7)
        plt.tight_layout()
        plt.savefig(path, dpi=400)
        plt.close(fig)


def plot_figure8_3d(m, front, best, ref, path='Figure8_3D.png'):
    """Static three-dimensional view of the design space in (dP, eta_L,sep, eta_abs).

    The Pareto-optimal set forms the boundary surface of the point cloud toward
    low dP, high eta_abs and high eta_L,sep. matplotlib has no logarithmic 3D
    axis, so log10(dP) is plotted with labelled ticks.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dP, eA, eL = m['dP'] / 1e3, 100 * m['eta_abs'], 100 * m['eta_L']
    x = np.log10(dP)
    with plt.rc_context({'font.size': 10, 'axes.labelsize': 10.5, 'xtick.labelsize': 9.5,
                         'ytick.labelsize': 9.5, 'font.family': 'DejaVu Sans'}):
        fig = plt.figure(figsize=(8.0, 6.6))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        ax.scatter(x[~front], eL[~front], eA[~front], c=eL[~front], cmap='RdYlGn', vmin=70,
                   vmax=100, s=3, alpha=0.18, lw=0, depthshade=False, zorder=1)
        ax.scatter(x[front], eL[front], eA[front], facecolors='none', edgecolors='k', s=18,
                   lw=0.7, depthshade=False, zorder=2)
        ax.scatter([np.log10(float(best['dP']) / 1e3)], [100 * float(best['eta_L'])],
                   [100 * float(best['eta_abs'])], marker='*', s=320, c='crimson', ec='k',
                   lw=0.8, depthshade=False, zorder=5)
        ax.scatter([np.log10(float(ref['dP']) / 1e3)], [100 * float(ref['eta_L'])],
                   [100 * float(ref['eta_abs'])], marker='D', s=60, c='royalblue', ec='k',
                   lw=0.8, depthshade=False, zorder=5)
        ticks = [5, 10, 20, 50, 100, 200]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(np.log10(4), np.log10(300))
        ax.set_ylim(40, 100)
        ax.set_zlim(50, 100)
        ax.set_xlabel(r'Pressure drop $\Delta P$ [kPa]', labelpad=8)
        ax.set_ylabel(r'Liquid separation efficiency $\eta_{L,sep}$ [%]', labelpad=8)
        ax.set_zlabel(r'CO$_2$ absorption efficiency $\eta_{abs}$ [%]', labelpad=8)
        ax.view_init(elev=18, azim=-52)
        ax.set_box_aspect(None, zoom=0.86)
        handles = [
            Line2D([], [], marker='o', ls='', mfc='none', mec='k', ms=6, label='Pareto-optimal set'),
            Line2D([], [], marker='*', ls='', mfc='crimson', mec='k', ms=13, label='selected design'),
            Line2D([], [], marker='D', ls='', mfc='royalblue', mec='k', ms=7,
                   label=f'{REFERENCE} (re-evaluated at this duty)')]
        ax.legend(handles=handles, loc='upper left', fontsize=8.5, framealpha=0.95)
        fig.subplots_adjust(left=0.0, right=0.94, bottom=0.04, top=0.99)
        plt.savefig(path, dpi=300)
        plt.close(fig)


def write_figure8_3d_html(m, front, best, ref, path='Figure8_3D.html'):
    """Interactive, rotatable three-dimensional view of the design space.

    The file contains plotly.js and opens offline in any browser. plotly has no
    star symbol for 3D markers, so the selected design is drawn as a star glyph.
    """
    import plotly.graph_objects as go

    dP, eA, eL = m['dP'] / 1e3, 100 * m['eta_abs'], 100 * m['eta_L']
    hov = 'ΔP = %{x:.1f} kPa<br>η_L,sep = %{y:.1f} %<br>η_abs = %{z:.1f} %<extra></extra>'
    b = (float(best['dP']) / 1e3, 100 * float(best['eta_L']), 100 * float(best['eta_abs']))
    r = (float(ref['dP']) / 1e3, 100 * float(ref['eta_L']), 100 * float(ref['eta_abs']))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=dP[~front], y=eL[~front], z=eA[~front], mode='markers', showlegend=False,
        marker=dict(size=2, color=eL[~front], colorscale='RdYlGn', cmin=70, cmax=100,
                    opacity=0.35, colorbar=dict(title='η_L,sep [%]', x=1.02)),
        hovertemplate=hov))
    fig.add_trace(go.Scatter3d(
        x=dP[front], y=eL[front], z=eA[front], mode='markers', name='Pareto-optimal set',
        marker=dict(size=4, color='black', symbol='circle-open', line=dict(width=1.5)),
        hovertemplate=hov))
    fig.add_trace(go.Scatter3d(
        x=[b[0]], y=[b[1]], z=[b[2]], mode='text', text=['★'], showlegend=False,
        textposition='middle center', textfont=dict(size=38, color='crimson'),
        hovertemplate=hov))
    fig.add_trace(go.Scatter3d(                     # legend entry of the star
        x=[b[0]], y=[b[1]], z=[b[2]], mode='markers', hoverinfo='skip',
        name='<span style="color:crimson">★</span> selected design',
        marker=dict(size=1, opacity=0)))
    fig.add_trace(go.Scatter3d(
        x=[r[0]], y=[r[1]], z=[r[2]], mode='markers',
        name=f'{REFERENCE} (re-evaluated at this duty)',
        marker=dict(size=7, color='royalblue', symbol='diamond',
                    line=dict(color='black', width=2)),
        hovertemplate=hov))
    fig.update_layout(
        scene=dict(xaxis=dict(title=dict(text='ΔP [kPa]'), type='log',
                              range=[np.log10(4), np.log10(300)],
                              tickvals=[5, 10, 20, 50, 100, 200],
                              ticktext=['5', '10', '20', '50', '100', '200']),
                   yaxis=dict(title=dict(text='η_L,sep [%]'), range=[40, 100]),
                   zaxis=dict(title=dict(text='η_abs [%]'), range=[50, 100],
                              tickvals=[60, 70, 80, 90, 100]),
                   camera=dict(eye=dict(x=-1.5, y=-1.7, z=0.7))),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)'),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f'Design space and Pareto-optimal set at the design duty '
                        f'(L = {L_DESIGN:.0f} kg/h, G = {G_DESIGN:.0f} Nm³/h). Drag to rotate.',
                   font=dict(size=13)))
    fig.write_html(path, include_plotlyjs=True)


def main(c=TABLE2, space_path='design_space.csv', figure_path='Figure8.png'):
    b = design(c)
    radii = ', '.join(f'{r * 1e3:.0f}' for r in b['Rout'] - PITCH * np.arange(int(b['N'])))
    print(f"Selected design: {int(b['N'])} stages, radii {radii} mm, "
          f"H = {b['H'] * 1e3:.1f} mm, R_c = {b['Rc'] * 1e3:.1f} mm (conical)")
    for f in (0.85, 1.0, 1.15):
        r = evaluate(c, b, f)
        print(f"  G = {f * G_DESIGN:4.1f} Nm3/h, L = {f * L_DESIGN:5.1f} kg/h: "
              f"dP = {float(r['dP']) / 1e3:5.1f} kPa, eta_abs = {100 * float(r['eta_abs']):.1f} %, "
              f"eta_L,sep = {100 * float(r['eta_L']):.1f} %, eta_G,sep = {100 * float(r['eta_G']):.1f} %")
    best = evaluate(c, b)
    ref = evaluate(c, geometry(REFERENCE))
    print(f"  {REFERENCE} at the design duty: dP = {float(ref['dP']) / 1e3:.1f} kPa, "
          f"eta_abs = {100 * float(ref['eta_abs']):.1f} %, eta_L,sep = {100 * float(ref['eta_L']):.1f} %, "
          f"eta_G,sep = {100 * float(ref['eta_G']):.1f} %")
    print(f"  pressure drop reduction: {100 * (1 - float(best['dP']) / float(ref['dP'])):.0f} %")

    g, m, front, ok = design_space(c)
    with open(space_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['R_out [mm]', 'R_in [mm]', 'N_stage', 'R_c [mm]', 'H [mm]',
                    'dP [kPa]', 'eta_abs [%]', 'eta_L,sep [%]', 'eta_G,sep [%]',
                    'Pareto-optimal', 'meets targets over turndown'])
        for i in range(len(g['H'])):
            w.writerow([f"{g['Rout'][i] * 1e3:.0f}", f"{g['Rin'][i] * 1e3:.0f}", int(g['N'][i]),
                        f"{g['Rc'][i] * 1e3:.1f}", f"{g['H'][i] * 1e3:.2f}",
                        f"{m['dP'][i] / 1e3:.2f}", f"{100 * m['eta_abs'][i]:.2f}",
                        f"{100 * m['eta_L'][i]:.2f}", f"{100 * m['eta_G'][i]:.2f}",
                        int(front[i]), int(ok[i])])
    print(f'\nDesign space ({len(g["H"])} geometries, {front.sum()} Pareto-optimal) '
          f'written to {space_path}')
    try:
        plot_figure8(m, front, best, ref, figure_path)
        plot_figure8_3d(m, front, best, ref, figure_path.replace('.png', '_3D.png'))
        print(f"Figure 8 written to {figure_path} and {figure_path.replace('.png', '_3D.png')}")
    except ImportError:
        print('matplotlib not available, Figure 8 not drawn')
    try:
        write_figure8_3d_html(m, front, best, ref, figure_path.replace('.png', '_3D.html'))
        print(f"Interactive 3D view written to {figure_path.replace('.png', '_3D.html')}")
    except ImportError:
        print('plotly not available, interactive 3D view not written')
    return b


if __name__ == '__main__':
    main()
