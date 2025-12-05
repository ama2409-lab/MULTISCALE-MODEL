import time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import warnings
import os
import sys

# Ensure CODE is importable
CODE_DIR = os.path.join(os.getcwd(), 'CODE')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from CODE.NETWORK_GENERATION_FINAL import build_voronoi_network
import openpnm as op


def assemble_advec_diff(pn, v0=0.0, D=1e-5):
    """Assemble sparse linear system A x = b for steady-state advection-diffusion
    using an upwind finite-volume discretization on the pore network graph.

    Parameters
    ----------
    pn : OpenPNM network
    v0 : float
        Characteristic velocity magnitude (m/s) along +x direction. Positive v0
        biases advection from low x -> high x.
    D : float
        Molecular diffusivity (m2/s) used to compute diffusive conductances.

    Returns
    -------
    A : scipy.sparse.csr_matrix
    b : numpy array
    inlet_pores, outlet_pores : arrays of pore indices for Dirichlet BCs
    """
    # Ensure geometry models exist
    pn.regenerate_models()

    conns = pn['throat.conns']
    Nt = pn.Nt
    Np = pn.Np

    # Geometric quantities
    length = pn['throat.length']
    dia = pn['throat.diameter']
    area = np.pi * (dia / 2.0) ** 2

    # Diffusive conductance g = D * A / L
    # Clip small lengths
    length = np.maximum(length, 1e-12)
    g = D * area / length

    # Advection: assign velocity along throat from projection of v0 along x-axis
    coords = pn['pore.coords']
    p0 = coords[conns[:, 0]]
    p1 = coords[conns[:, 1]]
    tvec = p1 - p0
    tlen = np.linalg.norm(tvec, axis=1)
    # avoid div by zero
    tlen = np.maximum(tlen, 1e-12)
    tdir = tvec / tlen[:, None]
    # velocity along throat: v0 * (direction . x_hat)
    v_along = v0 * tdir[:, 0]
    # Volumetric advective coefficient U = v_along * area
    U = v_along * area

    # Build sparse matrix entries
    row = []
    col = []
    data = []
    b = np.zeros(Np)

    # Identify inlet/outlet as pores with min/max x
    xs = coords[:, 0]
    xmin = xs.min(); xmax = xs.max()
    inlet = np.where(np.isclose(xs, xmin))[0]
    outlet = np.where(np.isclose(xs, xmax))[0]

    bc = np.zeros(Np, dtype=bool)
    bc[inlet] = True
    bc[outlet] = True

    C_in = 1.0
    C_out = 0.0

    # For each throat, distribute coefficients to connected pores
    for idx in range(Nt):
        i = conns[idx, 0]
        j = conns[idx, 1]
        gij = g[idx]
        uij = U[idx]
        # Upwind discretization: flux_ij = gij*(Ci - Cj) + uij*Ci if uij>0 else uij*Cj
        # This leads to coefficients depending on sign(uij).
        if uij >= 0:
            # contribution to i diagonal: gij + uij
            row.append(i); col.append(i); data.append(gij + uij)
            # off-diagonal i<-j: -gij
            row.append(i); col.append(j); data.append(-gij)
            # j diagonal: gij
            row.append(j); col.append(j); data.append(gij)
            # j off-diagonal j<-i: -gij - uij
            row.append(j); col.append(i); data.append(-gij - uij)
        else:
            # uij negative: flow from j->i
            # i diagonal: gij
            row.append(i); col.append(i); data.append(gij)
            # i off-diagonal i<-j: -gij - uij
            row.append(i); col.append(j); data.append(-gij - uij)
            # j diagonal: gij - uij (since -uij positive)
            row.append(j); col.append(j); data.append(gij - uij)
            # j off-diagonal j<-i: -gij
            row.append(j); col.append(i); data.append(-gij)

    A = sps.coo_matrix((data, (row, col)), shape=(Np, Np)).tocsr()

    # Apply Dirichlet BCs (overwrite rows)
    for pore in inlet:
        A[pore, :] = 0
        A[pore, pore] = 1
        b[pore] = C_in
    for pore in outlet:
        A[pore, :] = 0
        A[pore, pore] = 1
        b[pore] = C_out

    return A.tocsr(), b, inlet, outlet


def solve_with_openpnm(pn):
    # Ensure phase and diffusive conductances are present on this network
    air = op.phase.Air(network=pn)
    air.add_model_collection(op.models.collections.physics.basic)
    # Regenerate network and phase models so diffusive conductances exist
    pn.regenerate_models()
    air.regenerate_models()

    fd = op.algorithms.FickianDiffusion(network=pn, phase=air)
    # Identify inlet/outlet along x
    coords = pn['pore.coords']
    xs = coords[:, 0]
    inlet = np.where(np.isclose(xs, xs.min()))[0]
    outlet = np.where(np.isclose(xs, xs.max()))[0]
    fd.set_value_BC(pores=inlet, values=1.0)
    fd.set_value_BC(pores=outlet, values=0.0)
    t0 = time.perf_counter()
    fd.run()
    t1 = time.perf_counter()
    return t1 - t0, fd


def run_benchmark(sizes, vels):
    results = []
    for pts in sizes:
        print(f"\n== Size: {pts} points ==")
        # Build network with pore_density derived from pts and domain fixed
        domain = (250e-6, 250e-6, 250e-6)
        vol = np.prod(domain)
        pore_density = pts / vol
        pn = build_voronoi_network(domain_size=domain, pore_density=pore_density, points=None,
                                   pore_size_dist={'name': 'lognorm', 's': 0.25, 'scale': 20e-6},
                                   throat_size_dist={'name': 'weibull_min', 'c': 2.5, 'scale': 8e-6},
                                   correlate_pore_throat=False, seed=42, save_files=False)
        pn.regenerate_models()

        # Baseline: OpenPNM diffusion
        try:
            t_openpnm, fd = solve_with_openpnm(pn)
        except Exception as e:
            print('OpenPNM diffusion failed:', e)
            t_openpnm = None
        if t_openpnm is None:
            print("OpenPNM FickianDiffusion solve time: FAILED")
        else:
            print(f"OpenPNM FickianDiffusion solve time: {t_openpnm:.4f} s")

        for v0 in vels:
            print(f"-- v0 = {v0:.2e} m/s --")
            t_asm0 = time.perf_counter()
            A, b, inlet, outlet = assemble_advec_diff(pn, v0=v0, D=1e-5)
            t_asm1 = time.perf_counter()
            try:
                t_sol0 = time.perf_counter()
                x = spla.spsolve(A, b)
                t_sol1 = time.perf_counter()
                t_asm = t_asm1 - t_asm0
                t_sol = t_sol1 - t_sol0
                print(f"Assembly time: {t_asm:.4f} s, Solve time: {t_sol:.4f} s")
                # compute flux at inlet
                # approximate flux = sum over throats connected to inlet of gij*(C_in - Cj) + uij*Ci_upwind
            except Exception as e:
                print('Linear solve failed:', e)
                t_asm = None; t_sol = None
            results.append({'points': pts, 'v0': v0, 'openpnm_time': t_openpnm,
                            'asm_time': t_asm, 'sol_time': t_sol})
    return results


if __name__ == '__main__':
    sizes = [500, 2000]  # reduced for quick runtime
    vels = [0.0, 1e-4, 1e-2]
    res = run_benchmark(sizes, vels)
    print('\nBenchmark results:')
    for r in res:
        print(r)
    # Save CSV
    import csv
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['points', 'v0', 'openpnm_time', 'asm_time', 'sol_time'])
        writer.writeheader()
        for r in res:
            writer.writerow(r)
    print('Saved benchmark_results.csv')
