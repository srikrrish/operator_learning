#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp
from energy import kinetic
from initial_conditions import findsource
import torch
from utilities import torch_to_cp, cp_to_torch
from operator_learning.data.pic_dataset import normalize_per_sample, normalize_per_sample_torch
from cupyx.scipy.sparse.linalg import gmres, LinearOperator
from torch.func import jvp
from torch.autograd.functional import jacobian
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def accelerate(M: cp.ndarray, 
               E: cp.ndarray,
               wp: float, 
               QM: float,
               it: int,
               dim: int):
    """
    Compute particle acceleration from grid electric field and store E at the current timestep.

    Args:
        M (cp.ndarray): Projection/interpolation matrix from grid to particle positions.
        E (cp.ndarray): Electric field on the grid.
        Eout (cp.ndarray): Array to store electric field at each timestep.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).
        it (int): Current timestep index.
        dim (int): dimension (1D/2D)

    Returns:
        a (cp.ndarray): Particle accelerations.
        Eout (cp.ndarray): Updated electric field history.
    """
    if dim == 1:
        Etemp = M * E
        a = cp.transpose(Etemp) * QM / wp
        Eout = Etemp.astype(cp.float32)
    else:
        Extemp = M * E[0].flatten()
        Eytemp = M * E[1].flatten()
        a1 = cp.transpose(Extemp) * QM / wp
        a2 = cp.transpose(Eytemp) * QM / wp
        #Eout[it,:,0] = Extemp.astype(cp.float32)
        #Eout[it,:,1] = Eytemp.astype(cp.float32)
        a = cp.array([a1, a2])
        Eout = cp.zeros([2, a.shape[1]])
        Eout[0,:] = Extemp.astype(cp.float32) 
        Eout[1,:] = Eytemp.astype(cp.float32)
    
    return a, Eout


def accelerateML(E: cp.ndarray, xp: cp.ndarray, L: float, testCase: str, wp: float, QM: float):
    """
    Compute particle acceleration for ML-predicted electric fields.

    Args:
        E (cp.ndarray): Electric field at particle positions.
        xp (cp.ndarray): Particle positions
        L (float): Length array of the domain
        testCase (str): Testcase label
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).

    Returns:
        cp.ndarray: Particle accelerations.
    """
    
    if(testCase == 'penning'):
        #Add quadrupole external electric field
        E_ext = cp.zeros_like(E)
        E_ext[0] = -((15 / L[0]) * (xp[0] - (L[0]/2)))
        E_ext[1] = -((15 / L[1]) * (xp[1] - (L[1]/2)))
        E_ext[2] =  ((30 / L[2]) * (xp[2] - (L[2]/2)))
        
        a = (E + E_ext) * QM / wp
    else:
        a = E * QM / wp

    
    return a

def _box_length(L):
    """Domain length as a Python float (L may be a scalar or a 1-element array)."""
    return float(cp.asarray(L).reshape(-1)[0])


def _field_acc_torch(model, x, Lf, QM, Q, N, std, mean):
    """
    Acceleration a = (q/m) E_theta at the particles x (torch, physical units).
    Uses the FIXED box normalization x / L (the same in the residual and in the JVP,
    so that the Jacobian is the exact derivative of the residual).
    """
    x_norm = x / Lf
    E = model.field(x_norm)
    E = E * std + mean
    E = E * (Q * N)
    E = E - (torch.sum(E) / N)
    return QM * E


def make_Jv_xv(model, y_cp, y_old_cp, DT, QM, Q, N, std, mean, L, theta=0.5):
    """
    Matrix-free Jacobian of residual_xv (theta scheme) at the Newton iterate y.

        R_x = dx/DT - ((1-theta) v_old + theta v)
        R_v = (v - v_old)/DT - a(x_theta),   x_theta = x_old + theta*dx

    dR_x = ddx/DT - theta*ddv
    dR_v = ddv/DT - theta * (da/dx)(x_theta) ddx
    """
    Lf = _box_length(L)
    x_cp = y_cp[:N]
    x_old_cp = y_old_cp[:N]
    dx = x_cp - x_old_cp                      # unwrapped iterate: true displacement
    x_th = cp.mod(x_old_cp + theta * dx, Lf)
    x_t = cp_to_torch(x_th)

    def acc(x):
        return _field_acc_torch(model, x, Lf, QM, Q, N, std, mean)

    def Jv(dy_cp):
        ddx = dy_cp[:N]
        ddv = dy_cp[N:]
        _, da = jvp(acc, (x_t,), (cp_to_torch(theta * ddx),))
        Jx = ddx / DT - theta * ddv
        Jv_eq = ddv / DT - torch_to_cp(da)
        return cp.concatenate([Jx, Jv_eq]).reshape(-1)

    return Jv


def make_precond_xv(N, DT, theta, wp2):
    """
    Physics-based preconditioner for the theta scheme.

    For a (nearly) uniform plasma the 1D field response to particle displacements is
    da/dx ~ -wp^2 (I - mean), so the Jacobian is approximated by the 2x2 block
        [[1/DT, -theta], [theta*wp^2, 1/DT]]   on mean-free vectors,
        [[1/DT, -theta], [0,          1/DT]]   on the mean (uniform) component,
    which is inverted exactly. This removes the (wp*DT)^2 growth of the condition number.
    """
    a = 1.0 / DT
    b = -theta
    c = theta * wp2
    det_f = a * a - b * c
    det_m = a * a

    def M(r):
        rx = r[:N]
        rv = r[N:]
        mx = cp.mean(rx)
        mv = cp.mean(rv)
        fx = rx - mx
        fv = rv - mv
        xf = (a * fx - b * fv) / det_f
        vf = (a * fv - c * fx) / det_f
        xm = (a * mx - b * mv) / det_m
        vm = (a * mv) / det_m
        return cp.concatenate([xf + xm, vf + vm]).reshape(-1)

    return M


def residual_xv(model, y_cp, y_old_cp, DT, QM, Q, N, std, mean, L, theta=0.5):
    """
    Residual of the implicit theta scheme for (x^{n+1}, v^{n+1}) in a periodic box:

        R_x = dx/DT - ((1-theta) v_old + theta v)
        R_v = (v - v_old)/DT - a(x_theta),   x_theta = x_old + theta*dx (wrapped)

    with dx = x - x_old the true displacement: x is kept unwrapped during the Newton
    iteration and wrapped only after convergence.
    theta = 0.5: implicit midpoint (Crank-Nicolson), second order, no numerical damping.
    theta = 1.0: backward Euler (first order, damps oscillations).
    """
    Lf = _box_length(L)
    x_cp = y_cp[:N]
    v_cp = y_cp[N:]
    x_old_cp = y_old_cp[:N]
    v_old_cp = y_old_cp[N:]

    # x_cp is kept UNWRAPPED during the Newton iteration, so x - x_old is the true
    # displacement (a minimum-image displacement is wrong once |v| DT > L/2).
    dx = x_cp - x_old_cp
    x_th = cp.mod(x_old_cp + theta * dx, Lf)

    with torch.no_grad():
        a = torch_to_cp(_field_acc_torch(model, cp_to_torch(x_th), Lf, QM, Q, N, std, mean))

    Rx = dx / DT - ((1.0 - theta) * v_old_cp + theta * v_cp)
    Rv = (v_cp - v_old_cp) / DT - a

    return cp.concatenate([Rx, Rv]).reshape(-1)

def to_3d(x):
    if x.shape[0] == 3:
        return x
    elif x.shape[0] == 2:
        return cp.concatenate(
            (x, cp.zeros((1, x.shape[1]), dtype=x.dtype)),
            axis=0
        )
    else:
        raise ValueError(f"Expected 2D or 3D array, got {x.shape[0]}D")


def push1(vp, a, DT, QM, B0):

    # --- Kick 1 (eq. 4a): explicit half-step, uses old v in B force ---
    alpha = QM * DT / 2 
    vhx = vp[0] + ((DT / 2) * a[0]) + (alpha * vp[1] * B0[2])
    vhy = vp[1] + ((DT / 2) * a[1]) - (alpha * vp[0] * B0[2])
    vhz = vp[2] + ((DT / 2) * a[2])

    return cp.array([vhx, vhy, vhz])
    

def push2(vp, a, DT, QM, B0):

    alpha = QM * DT / 2
    beta  = alpha * B0[2]
    DrInv = 1.0 / (1.0 + beta**2)

    # --- Kick 2 (eq. 4c): Boris trick, analytical solution of implicit system ---
    vx_new = DrInv * (vp[0] + (DT / 2)*a[0] + alpha * (vp[1]*B0[2] + (DT / 2)*B0[2]*a[1]))
    vy_new = DrInv * (vp[1] + (DT / 2)*a[1] - alpha * (vp[0]*B0[2] + (DT / 2)*B0[2]*a[0]))
    vz_new = vp[2] + (DT / 2) * a[2]


    return cp.array([vx_new, vy_new, vz_new])


def push(vp: cp.ndarray, a: cp.ndarray, 
         DT: float, Q: float, 
         QM: float, wp: float,
         it: int, testCase: str,
         B0: cp.array, v0_at_minus_half: bool = False):
    """
    Update particle velocities using leapfrog integration and compute kinetic energy.

    Args:
        vp (cp.ndarray): Particle velocities.
        a (cp.ndarray): Particle accelerations.
        DT (float): Timestep size.
        Q (float): Particle charge.
        QM (float): Charge-to-mass ratio (q/m).
        wp (float): Particle weights.
        it (int): Current timestep index.
        testCase (str): Testcase label
        B0 (cp.array): External magnetic field for cyclotron test case
        v0_at_minus_half (bool): the initial velocity is v^{-1/2} (full first step) instead of v^0 (half first step)

    Returns:
        vp_new (cp.ndarray): Updated particle velocities.
        kinetic_energy (float): Kinetic energy after update.
    """
    #PEPC BH uses v^{-1/2} formulation for Boris; v0_at_minus_half matches it
    if ((it == 0) and not v0_at_minus_half):
        DT_eff = DT / 2
    else:
        DT_eff = DT

    if((testCase == 'cyclotron') or (testCase == 'penning')):
        v = to_3d(vp)
        a = to_3d(a)

        Vm = v + a * DT_eff / 2
        t = B0 * QM * DT_eff / 2
        Vprime = Vm + cp.cross(Vm, t, axisa=0, axisb=0, axisc=0)
        s = 2 * t / (1 + cp.dot(t, t))
        Vp = Vm + cp.cross(Vprime, s, axisa=0, axisb=0, axisc=0)

        new_vp = Vp + a * DT_eff / 2
        new_vp = new_vp[:vp.shape[0]]
    else:
        new_vp = vp + a * DT_eff

    if((testCase == 'cyclotron')):
        #For cases with external magnetic field average the kinetic energies instead of velocities as it is better
        return new_vp, (kinetic(vp, Q, QM, wp) + kinetic(new_vp, Q, QM, wp))/2
    else:
        if it == 0:
            return new_vp, kinetic(vp, Q, QM, wp)
        else:
            if((testCase == 'penning')):
                #For cases with external magnetic field average the kinetic energies instead of velocities as it is better
                return new_vp, (kinetic(vp, Q, QM, wp) + kinetic(new_vp, Q, QM, wp))/2
            else:
                return new_vp, kinetic((vp + new_vp)/2, Q, QM, wp)
            
def move(xp: cp.ndarray, vp: cp.ndarray,
        wp: float, DT: float, 
        L: float, it: int = None):
    """
    Update particle positions based on velocities with optional source term.

    Args:
        xp (cp.ndarray): Particle positions.
        vp (cp.ndarray): Particle velocities.
        wp (float): Particle weights.
        DT (float): Timestep size.
        L (float): Domain length.
        it (int, optional): Current timestep index, used for source term.

    Returns:
        xp_new (cp.ndarray): Updated particle positions.
        wp_new (float): Updated particle weights if source term applied.
    """
    if wp == 1:
        return xp + vp * DT, 1
    else:
        return xp + vp * DT, wp + DT * findsource(xp + vp * DT / 2, vp, L, it + 0.5, DT)

def newton_push_move_xv(model, xp, vp, DT,
                       QM, Q, N, L, dim, std, mean,
                       tol_newton=1e-5,
                       tol_gmres=1e-6,
                       max_newton=10,
                       theta=0.5,
                       precond=True,
                       wp2=None,
                       verbose=True,
                       rtol_res=1e-4,
                       rtol_dv=1e-6,
                       restart=10,
                       gmres_tol=None,
                       gmres_max_cycles=10):
    """
    One implicit step (1D) with Newton-Krylov (GMRES, matrix-free JVP via autograd).

    theta    : 0.5 implicit midpoint / Crank-Nicolson (default), 1.0 backward Euler.
    precond  : use the physics-based block preconditioner (make_precond_xv).
    wp2      : plasma frequency^2 for the preconditioner; None -> estimated each step
               from the network's own response to a smooth displacement (one JVP).
    restart  : GMRES restart length. CuPy's GMRES runs a full cycle of `restart`
               iterations (JVPs) before checking convergence, so keep it short.
    gmres_tol: relative tolerance of each linear solve. None (default) -> adaptive
               (Eisenstat-Walker type): eta_k = max(1e-6, min(1e-2, ||F_k||/||F_0||)),
               loose far from the solution and tightening as Newton converges.
    gmres_max_cycles : maximum number of GMRES restart cycles per linear solve.
    verbose  : True  -> per-iteration output (adds one extra JVP per Newton iteration
                        for the true linear residual);
               False -> one summary line per step, no extra JVPs (use for timings).
    Stopping (float32-aware; absolute tolerances do not scale with N):
      - ||F|| <= rtol_res * ||F_0||                (residual reduced enough)
      - ||dv|| <= rtol_dv * ||v||  or  ||dv|| < tol_newton
      - stagnation: once ||F|| < 1e-2 ||F_0||, stop if an iteration reduces ||F||
        by less than a factor 2 (round-off floor of the float32 network reached).
    """
    Lf = _box_length(L)

    # ------------------------------------------------------------
    # flatten state
    # ------------------------------------------------------------
    x_old = xp.reshape(-1)
    v_old = vp.reshape(-1)
    y_old = cp.concatenate([x_old, v_old])

    # ------------------------------------------------------------
    # initial guess: free-streaming predictor x_old + DT v_old
    # ------------------------------------------------------------
    x0 = x_old + DT * v_old          # unwrapped; wrapped only after convergence
    v0 = v_old.copy()
    y = cp.concatenate([x0, v0])

    r0_norm = None
    r_prev = None
    r_norm = None
    n_res = 0          # residual evaluations (one network forward pass each)
    n_jvp = 0          # JVPs inside GMRES (one network forward + tangent pass each)
    n_cycles = 0       # GMRES callbacks (CuPy: restart cycles)
    n_newton = 0       # Newton updates applied
    stop = "max_newton"
    w2_used = None
    etas = []
    for k in range(max_newton):

        r = residual_xv(model, y, y_old, DT, QM, Q, N, std, mean, L, theta=theta)
        n_res += 1
        r_norm = float(cp.linalg.norm(r))
        if r0_norm is None:
            r0_norm = r_norm
        if verbose:
            print(f"Newton iter {k}: ||F|| = {r_norm:.6e}")
        if k > 0 and r_norm <= rtol_res * r0_norm:
            stop = "residual"
            if verbose:
                print(f"    converged: ||F||/||F0|| = {r_norm / r0_norm:.2e}")
            break
        if k > 0 and r_norm < 1e-2 * r0_norm and r_norm > 0.5 * r_prev:
            stop = "stagnation"
            if verbose:
                print(f"    stopped: stagnation at ||F||/||F0|| = {r_norm / r0_norm:.2e} (round-off floor)")
            break
        r_prev = r_norm

        Jv = make_Jv_xv(model, y, y_old, DT, QM, Q, N, std, mean, L, theta=theta)
        Ntot = y.shape[0]

        def Jv_counted(dy_cp):
            nonlocal n_jvp
            n_jvp += 1
            return Jv(dy_cp)

        A = LinearOperator((Ntot, Ntot), matvec=Jv_counted, dtype=y.dtype)

        M = None
        if precond:
            # estimate wp^2 once per time step (first Newton iteration), then reuse it
            w2 = wp2 if wp2 is not None else w2_used
            if w2 is None:
                # Rayleigh quotient of -da/dx on a smooth, mean-free displacement
                dx_k = y[:N] - x_old
                x_th = cp.mod(x_old + theta * dx_k, Lf)
                u = cp.cos(2.0 * cp.pi * x_th / Lf)
                u = u - cp.mean(u)
                Ju = Jv(cp.concatenate([u, cp.zeros_like(u)]))
                # v-part of J [u,0] is -theta * (da/dx) u
                w2 = float(cp.dot(u, Ju[N:]) / (theta * cp.dot(u, u)))
                w2 = max(w2, 0.0)
                w2_used = w2
                if verbose and k == 0:
                    print(f"    preconditioner: estimated wp^2 = {w2:.4f}")
            M = LinearOperator((Ntot, Ntot), matvec=make_precond_xv(N, DT, theta, w2), dtype=y.dtype)

        gmres_history = []

        def gmres_callback(residual_norm):
            gmres_history.append(float(residual_norm))

        if gmres_tol is None:
            eta = max(1e-6, min(1e-2, r_norm / r0_norm))
        else:
            eta = float(gmres_tol)
        etas.append(eta)

        # CuPy: maxiter counts inner iterations in total (restart per cycle)
        dy, info = gmres(
            A,
            -r,
            x0=cp.zeros_like(y),
            tol=eta,
            restart=restart,
            maxiter=restart * gmres_max_cycles,
            M=M,
            callback=gmres_callback,
            callback_type='pr_norm'
        )

        n_cycles += len(gmres_history)
        n_newton += 1
        y = y + dy
        dy_norm = float(cp.linalg.norm(dy[N:]))
        v_norm = float(cp.linalg.norm(y[N:]))
        if verbose:
            lin_res = float(cp.linalg.norm(Jv(dy) + r))
            print(f"    GMRES (tol {eta:.1e}): {len(gmres_history)} callback(s), info = {info}, "
                  f"true linear residual = {lin_res:.3e}, ||dv|| = {dy_norm:.6e}")
        if dy_norm < tol_newton or dy_norm <= rtol_dv * v_norm:
            stop = "dv"
            if verbose:
                print(f"    converged: ||dv||/||v|| = {dy_norm / max(v_norm, 1e-30):.2e}")
            break

    # ------------------------------------------------------------
    # recover x, v
    # ------------------------------------------------------------
    x = cp.mod(y[:N], Lf)
    v = y[N:]

    if not verbose:
        rel = (r_norm / r0_norm) if (r0_norm and r_norm is not None) else float('nan')
        w2s = f", wp^2 = {w2_used:.4f}" if w2_used is not None else ""
        etas_s = ("/".join(f"{e:.0e}" for e in etas)) if etas else "-"
        print(f"implicit step: Newton {n_newton}, residual evals {n_res}, GMRES JVPs {n_jvp}, "
              f"GMRES cycles {n_cycles}, gmres tol {etas_s}, ||F||/||F0|| = {rel:.2e}, stop = {stop}{w2s}")

    return x[None, :], v[None, :]

def picard_push_move_xv(model, xp, vp, DT,
                         QM, Q, N, L, dim,
                         std, mean,
                         tol=1e-8, max_iter=20,
                         omega=1.0):

    x = xp.copy()
    v = vp.copy()

    for k in range(max_iter):

        # --- compute acceleration at current x ---
        x = toPeriodicND(x, L, dim)
        inputs = x[None, :, :].copy()
        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])


        E = model(inputs).squeeze()
        E = E * std + mean
        E = E * (Q * N)
        E = E - (cp.sum(E) / N)

        a = QM * E

        # --- Picard updates (x-v system) ---
        x_new = xp + DT * v
        v_new = vp + DT * a

        iterx_norm = cp.linalg.norm(x_new - x)
        iterv_norm = cp.linalg.norm(v_new - v)
        print(f"Picard iter {k}: ||vnew-v|| = {float(iterv_norm):.6e}")
        # convergence check
        if iterv_norm < tol:
            break
        
        # damping
        x = (1 - omega) * x + omega * x_new
        v = (1 - omega) * v + omega * v_new


    return x, v


def toPeriodic(x: cp.ndarray, L: float, discrete: bool=False):
    """
    Apply periodic boundary conditions to particle positions.

    Args:
        x (cp.ndarray): Particle positions (or indices).
        L (float): Domain length.
        discrete (bool, optional): Treat positions as discrete indices if True.

    Returns:
        cp.ndarray: Particle positions wrapped into [0, L).
    """
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x

def toPeriodicND(x: cp.ndarray, L: float, dim :int=2):
    x = cp.mod(x, cp.asarray(L)[:, None])
    return x.reshape(-1)
def toPeriodicND_old(x: cp.ndarray, L: float, dim :int=2):
    x = cp.mod(x, cp.asarray(L)[:, None])
    return x
