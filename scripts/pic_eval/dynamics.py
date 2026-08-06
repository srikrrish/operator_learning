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


def accelerateML(E: cp.ndarray, wp: float, QM: float):
    """
    Compute particle acceleration for ML-predicted electric fields.

    Args:
        E (cp.ndarray): Electric field at particle positions.
        wp (float): Particle weights.
        QM (float): Charge-to-mass ratio (q/m).

    Returns:
        cp.ndarray: Particle accelerations.
    """
    a = E * QM / wp
    
    return a

def make_Jv(model, x_cp, DT, QM, Q, N, std, mean):

    def Jv(dx_cp):
        inputs = x_cp[None, None, :].copy() # [batch=1, channel=dim, particles]
        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
        inputs = inputs.squeeze()
        x_t = cp_to_torch(inputs)
        dx_t = cp_to_torch(dx_cp)

        def acc(x):
            #E = model(x[None,None,:]).squeeze()
            Efieldparticle = model.field(x)
            Efieldparticle = Efieldparticle * std + mean
            Efieldparticle = Efieldparticle * ((Q * N))
            Efieldparticle = Efieldparticle - ((1/N) * torch.sum(Efieldparticle))
            return QM * Efieldparticle

        _, da_v = jvp(acc, (x_t,), (dx_t,))

        return torch_to_cp(dx_t - DT**2 * da_v)

    return Jv

def make_Jv_xv(model, y_cp, DT, QM, Q, N, std, mean):

    x_cp = y_cp[:N]   # freeze x at Newton iterate

    def Jv(dy_cp):

        dx_cp = dy_cp[:N]
        dv_cp = dy_cp[N:]

        # ---- x equation ----
        #Jx = dx_cp - DT * dv_cp
        Jx = (dx_cp / DT) - dv_cp

        # ---- field JVP ----
        #inputs = x_cp[None, None, :].copy()
        #inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
        #inputs = inputs.squeeze()

        #inputs_dx = dx_cp[None, None, :].copy()
        #inputs_dx[:, 0, :] = normalize_per_sample(inputs_dx[:, 0, :])
        #inputs_dx = inputs_dx.squeeze()
        #x_t = cp_to_torch(inputs)
        #dx_t = cp_to_torch(inputs_dx)
        x_t = cp_to_torch(x_cp)
        dx_t = cp_to_torch(dx_cp)

        def acc(x):
            #xmin_global = 0.0
            #xmax_global = 4.0 * torch.pi
            #x_norm = (x - xmin_global) / (xmax_global - xmin_global)
            inputs = x[None, None, :].clone()
            inputs[:, 0, :] = normalize_per_sample_torch(inputs[:, 0, :])
            inputs = inputs.squeeze()
            E = model.field(inputs)
            E = E * std + mean
            E = E * (Q * N)
            E = E - (torch.sum(E) / N)
            return QM * E
            #return E

        _, da_dx = jvp(acc, (x_t,), (dx_t,))

        da_dx_cp = torch_to_cp(da_dx)
        #eps = 1e-5

        ## random unit direction
        #dx_cp_rand = cp.random.randn(N)
        #dx_cp_rand = dx_cp_rand / cp.linalg.norm(dx_cp_rand)
        #
        ##x_t = cp_to_torch(x_cp)
        #dx_t_rand = cp_to_torch(dx_cp_rand)
        #
        ## JVP
        #_, jvp_val = jvp(acc, (x_t,), (dx_t_rand,))
        #jvp_val_cp = torch_to_cp(jvp_val)
        #
        ## finite difference
        #a0 = torch_to_cp(acc(x_t))
        ##eps_dx_cp_rand = cp_to_torch(eps*dx_cp_rand)
        ##a1 = torch_to_cp(acc(cp_to_torch(x_cp + eps*dx_cp_rand)))
        #a1 = torch_to_cp(acc(x_t + eps*dx_t_rand))
        #print("||a0|| =", cp.linalg.norm(a0))
        #print("||a1|| =", cp.linalg.norm(a1))
        #
        #fd = (a1 - a0)/eps
        #
        #print("||JVP|| =", cp.linalg.norm(jvp_val_cp))
        #print("||FD||  =", cp.linalg.norm(fd))
        #print("relative error =",
        #      cp.linalg.norm(jvp_val_cp-fd)/cp.linalg.norm(fd))
        #breakpoint()
        # ---- v equation ----
        #Jv_eq = dv_cp - DT * da_dx_cp
        Jv_eq = (dv_cp / DT) - da_dx_cp

        return cp.concatenate([Jx, Jv_eq]).reshape(-1)

    return Jv

def residual(model, x_cp, x_old_cp, v_old_cp, DT, QM, Q, N, std, mean):
    #breakpoint()
    inputs = x_cp[None, None, :].copy() # [batch=1, channel=dim, particles]
    inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    inputs = inputs.squeeze()
    x_t = cp_to_torch(inputs)

    #E = model(x_t[None,None,:]).squeeze()
    Efieldparticle = torch_to_cp(model.field(x_t))
    Efieldparticle = Efieldparticle * std + mean
    Efieldparticle = Efieldparticle * ((Q * N))
    Efieldparticle = Efieldparticle - ((1/N) * cp.sum(Efieldparticle))
    a = QM * Efieldparticle

    return (x_cp - x_old_cp - DT*v_old_cp - DT**2 * a).reshape(-1)

def residual_xv(model, y_cp, y_old_cp, DT, QM, Q, N, std, mean):

    x_cp = y_cp[:N]
    v_cp = y_cp[N:]

    x_old_cp = y_old_cp[:N]
    v_old_cp = y_old_cp[N:]

    # --- Torch input for field ---
    #inputs = x_cp[None, None, :].copy()
    #inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    #inputs = inputs.squeeze()

    #x_t = cp_to_torch(inputs)
    x_t = cp_to_torch(x_cp)
    x_norm = (x_t - 0.0) / (4 * torch.pi)
    #x_norm = x_t

    # --- field ---
    E = torch_to_cp(model.field(x_norm))
    E = E * std + mean
    E = E * (Q * N)
    E = E - (cp.sum(E) / N)

    a = QM * E

    # --- coupled residual ---
    #Rx = x_cp - x_old_cp - DT * v_cp
    #Rv = v_cp - v_old_cp - DT * a
    Rx = (x_cp - x_old_cp) / DT - v_cp
    Rv = (v_cp - v_old_cp) / DT - a

    return cp.concatenate([Rx, Rv]).reshape(-1)

def residual_xv_torch(model, y_torch, y_old_torch, DT, QM, Q, N, std, mean):

    x_torch = y_torch[:N]
    v_torch = y_torch[N:]

    x_old_torch = y_old_torch[:N]
    v_old_torch = y_old_torch[N:]

    # --- Torch input for field ---
    #inputs = x_cp[None, None, :].copy()
    #inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    #inputs = inputs.squeeze()

    #x_t = cp_to_torch(inputs)
    #x_t = cp_to_torch(x_cp)
    x_norm = (x_torch - 0.0) / (4 * torch.pi)
    #x_norm = x_t

    # --- field ---
    E = model.field(x_norm)
    E = E * std + mean
    E = E * (Q * N)
    E = E - torch.mean(E)

    a = QM * E

    # --- coupled residual ---
    Rx = x_torch - x_old_torch - DT * v_torch
    Rv = v_torch - v_old_torch - DT * a

    return torch.cat([Rx, Rv])

def push(vp: cp.ndarray, a: cp.ndarray, 
         DT: float, Q: float, 
         QM: float, wp: float,
         it: int, testCase: str,
         B0: cp.array):
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

    Returns:
        vp_new (cp.ndarray): Updated particle velocities.
        kinetic_energy (float): Kinetic energy after update.
    """
    if it == 0:
        if(testCase == 'cyclotron'):
            Ek = kinetic(vp, Q, QM, wp)
            vp = vp + DT * (a + QM * cp.cross(vp, B0, axisa=0)[:, 0:2].T) / 2
            return vp, Ek
        else:
            #return vp + a * DT / 2, kinetic(vp + a * DT / 2, Q, QM, wp)
            return vp + a * DT / 2, kinetic(vp, Q, QM, wp)

    else:
        if(testCase == 'cyclotron'):
            Vm = vp + a * DT / 2
            Vprime = Vm + cp.cross(Vm, B0, axisa=0)[:, 0:2].T * QM * DT / 2
            Vp = Vm + cp.cross(Vprime, B0, axisa=0)[:, 0:2].T * QM * DT / (1 + (cp.linalg.norm(B0)*QM*DT/2) ** 2)
            new_vp = Vp + a * DT / 2
            Ek = kinetic((vp + new_vp) / 2, Q, QM, wp)
            vp = new_vp
            return vp, Ek
        else:
            #return vp + a * DT, kinetic(vp + a * DT, Q, QM, wp)
            return vp + a * DT, kinetic((vp + (vp + a * DT))/2, Q, QM, wp)


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

def implicit_push_move(model, xp, vp, DT,
                       QM, Q, N, L, dim, std, mean, tol_newton=1e-8,
                       tol_gmres=1e-6, max_newton=8):

    x_old = xp.reshape(-1)
    v_old = vp.reshape(-1)

    # initial guess (explicit Euler)
    x = x_old + DT * v_old
    #x = x_old

    for k in range(max_newton):
        #Apply periodic BCs 
        x = toPeriodicND(x, L, dim)

        r = residual(model, x, x_old, v_old, DT, QM, Q, N, std, mean)

        r_norm = cp.linalg.norm(r) 

        print(f"Newton iter {k}: ||F|| = {float(r_norm):.6e}")
        if r_norm < tol_newton:
            break
        #r = r / cp.linalg.norm(r)
        b = -r

        r = r.reshape(-1)
        b = b.reshape(-1)

        # build matrix-free operator around CURRENT x
        Jv = make_Jv(model, x, DT, QM, Q, N, std, mean)

        
        def matvec(dx):
            y = Jv(dx)

            print("dx:", dx.shape,"y:", y.shape,
                  "dx nan:", cp.isnan(dx).any(),
                  "dx inf:", cp.isinf(dx).any(),
                  "y nan:", cp.isnan(y).any(),
                  "y inf:", cp.isinf(y).any())
            return y

        N = x.shape[0]
        #def precond(v):
        #    alpha = 1.0
        #    return v / (1.0 + DT**2 * alpha)

        gmres_history = []

        def gmres_callback(residual_norm):
            gmres_history.append(float(residual_norm))
            print(f"    GMRES iter {len(gmres_history)}: residual = {float(residual_norm):.6e}")

        A = LinearOperator(
            (N, N),
            matvec=matvec,
            dtype=x.dtype
        )
        #M = LinearOperator((N, N), matvec=precond, dtype=x.dtype)
        #breakpoint()
        dx, info = gmres(
            A,
            b,
            x0=x,
            tol=tol_gmres,
            restart=30,
            maxiter=10,
            callback = gmres_callback
        )

        print(f"    GMRES info = {info}")
        print(f"    ||dx|| = {float(cp.linalg.norm(dx)):.6e}")

        x = x + dx

    # recover velocity
    inputs = x[None, None, :].copy() # [batch=1, channel=dim, particles]
    inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    inputs = inputs.squeeze()
    x_t = cp_to_torch(inputs)
    #E = model(x_t[None,None,:]).squeeze()
    Efieldparticle = torch_to_cp(model.field(x_t))
    Efieldparticle = Efieldparticle * std + mean
    Efieldparticle = Efieldparticle * ((Q * N))
    Efieldparticle = Efieldparticle - ((1/N) * cp.sum(Efieldparticle))
    a = QM * Efieldparticle

    v_new = v_old + DT * a

    return x[None, :], v_new[None, :]

def newton_push_move_xv(model, xp, vp, DT,
                       QM, Q, N, L, dim, std, mean,
                       tol_newton=1e-8,
                       tol_gmres=1e-6,
                       max_newton=3):

    # ------------------------------------------------------------
    # flatten state
    # ------------------------------------------------------------
    x_old = xp.reshape(-1)
    v_old = vp.reshape(-1)

    y_old = cp.concatenate([x_old, v_old])

    # ------------------------------------------------------------
    # initial guess (IMPORTANT: consistent with x-v system)
    # ------------------------------------------------------------
    #x0 = x_old + DT * v_old
    x0 = x_old.copy()
    v0 = v_old.copy()
    y = cp.concatenate([x0, v0])
    
    # ------------------------------------------------------------
    # Newton loop
    # ------------------------------------------------------------
    for k in range(max_newton):

        # apply BC only to x-part
        x = y[:N]
        x = toPeriodicND(x, L, dim)
        y[:N] = x

        # residual (2N vector)
        r = residual_xv(model, y, y_old, DT, QM, Q, N, std, mean)
        y_torch = cp_to_torch(y)
        y_old_torch = cp_to_torch(y_old)
        y_torch.requires_grad_(True)
        J = jacobian(lambda y_torch:residual_xv_torch(model, y_torch, y_old_torch, DT, QM, Q, N, std, mean),y_torch)
        J_np = J.detach().cpu().numpy()
        plt.figure(figsize=(8,8))
        plt.imshow(np.log10(np.abs(J_np)+1e-15))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("jacobian_log.png", dpi=300)
        plt.close()
        Jxx = J_np[:N,:N]
        Jxv = J_np[:N,N:]
        Jvx = J_np[N:,:N]
        Jvv = J_np[N:,N:]
        fig, axs = plt.subplots(2,2, figsize=(10,10))

        axs[0,0].imshow(Jxx)
        axs[0,0].set_title("Jxx")
        
        axs[0,1].imshow(Jxv)
        axs[0,1].set_title("Jxv")
        
        axs[1,0].imshow(Jvx)
        axs[1,0].set_title("Jvx")
        
        axs[1,1].imshow(Jvv)
        axs[1,1].set_title("Jvv")
        
        plt.tight_layout()
        plt.savefig("jacobian_blocks.png", dpi=300)
        plt.close()
        u,s,vh = np.linalg.svd(J_np)
        print(s.max()/s.min())
        plt.semilogy(s)
        plt.savefig("singular_values_full_J.png", dpi=300)
        plt.close()
        eigvals = np.linalg.eigvals(J_np)
        plt.figure()
        plt.scatter(eigvals.real,eigvals.imag)
        plt.axvline(1)
        plt.axis('equal')
        plt.savefig("eigen_values.png", dpi=300)
        plt.close()

        breakpoint()
        #r_norm = cp.linalg.norm(r[N:])
        #dy_norm = cp.linalg.norm(dy[N:])
        #print(f"Newton iter {k}: ||F|| = {float(dy_norm):.6e}")
        #if r_norm < tol_newton:
        #    break

        b = -r.reshape(-1)

        # --------------------------------------------------------
        # Jacobian-vector product at current Newton state
        # --------------------------------------------------------
        Jv = make_Jv_xv(model, y, DT, QM, Q, N, std, mean)

        #vr = cp.random.randn(2*N)
        #vr /= cp.linalg.norm(vr)

        #Jv_vr = Jv(vr)

        #print("||vr||      =", cp.linalg.norm(vr))
        #print("||Jv(vr)||  =", cp.linalg.norm(Jv_vr))
        #print("ratio      =", cp.linalg.norm(Jv_vr) / cp.linalg.norm(vr))
        #print("diff norm  =", cp.linalg.norm(Jv_vr - vr))
        def matvec(dy):
            out = Jv(dy)

            #print(
            #    "dy:", dy.shape,
            #    "out:", out.shape,
            #    "dy nan:", cp.isnan(dy).any(),
            #    "out nan:", cp.isnan(out).any()
            #)
            return out

        #def precond(r):

        #    rx = r[:N]
        #    rv = r[N:]

        #    # block solve (very cheap!)
        #    vx = rx
        #    vv = rv + DT * rx

        #    return cp.concatenate([vx, vv])

        # IMPORTANT: 2N system now
        Ntot = y.shape[0]

        A = LinearOperator(
            (Ntot, Ntot),
            matvec=matvec,
            dtype=y.dtype
        )

        #M = LinearOperator(
        #    (Ntot, Ntot),
        #    matvec=precond,
        #    dtype=y.dtype
        #)

        gmres_history = []

        def gmres_callback(residual_norm):
            gmres_history.append(float(residual_norm))
            print(f"    GMRES iter {len(gmres_history)}: residual = {float(residual_norm):.6e}")
        # --------------------------------------------------------
        # GMRES solve (IMPORTANT: x0 must match shape)
        # --------------------------------------------------------
        dy, info = gmres(
            A,
            b,
            x0=cp.zeros_like(y),
            tol=tol_gmres,
            restart=30,
            maxiter=10,
            callback=gmres_callback,
            callback_type='pr_norm'
        )
        lin_res = cp.linalg.norm(matvec(dy) - b)
        print("true linear residual =", float(lin_res))

        print(f"    GMRES info = {info}")
        print(f"    ||dy|| = {float(cp.linalg.norm(dy)):.6e}")
        #breakpoint()
        #alpha = 1.0
        #y_trial = y + alpha * dy
        #r_trial = residual_xv(model, y_trial, y_old, DT, QM, Q, N, std, mean)

        #while cp.linalg.norm(r_trial) > cp.linalg.norm(r):

        #    alpha *= 0.5
        #    y_trial = y + alpha * dy
        #    r_trial = residual_xv(model, y_trial, y_old, DT, QM, Q, N, std, mean)

        #y = y_trial
        #print("alpha =", alpha)

        omega = 1.0
        y = y + omega * dy
        dy_norm = cp.linalg.norm(dy[N:])
        print(f"Newton iter {k}: ||dy|| = {float(dy_norm):.6e}")
        if dy_norm < tol_newton:
            break

    # ------------------------------------------------------------
    # recover x, v
    # ------------------------------------------------------------
    x = y[:N]
    v = y[N:]

    return x[None, :], v[None, :]

def picard_push_move_xv(model, xp, vp, DT,
                         QM, Q, N, L, dim,
                         std, mean,
                         tol=1e-8, max_iter=20,
                         omega=1.0):

    #x = xp.reshape(-1)
    #v = vp.reshape(-1)
    x = xp.copy()
    v = vp.copy()

    for k in range(max_iter):

        # --- compute acceleration at current x ---
        x = toPeriodicND(x, L, dim)
        inputs = x[None, :, :].copy()
        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
        #inputs = inputs.squeeze()

        #x_t = cp_to_torch(inputs)

        #E = torch_to_cp(model.field(x_t))
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
        #print(f"Picard iter {k}: ||xnew-x|| = {float(iterx_norm):.6e}")
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

def toPeriodicNDOld(x: cp.ndarray, L: float, dim :int=2):
    for i in range(dim):
        x[i] = toPeriodic(x[i], L[i])
    return x

def toPeriodicND(x: cp.ndarray, L: float, dim :int=2):
    x = cp.mod(x, cp.asarray(L)[:, None])
    #return x
    return x.reshape(-1)
def toPeriodicND_old(x: cp.ndarray, L: float, dim :int=2):
    x = cp.mod(x, cp.asarray(L)[:, None])
    return x
