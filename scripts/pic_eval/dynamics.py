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
from torch.func import jvp

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
        inputs = x_cp[None, :, :].copy() # [batch=1, channel=dim, particles]
        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
        x_t = cp_to_torch(inputs)
        dx_t = cp_to_torch(dx_cp)

        def acc(x):
            #E = model(x[None,None,:]).squeeze()
            Efieldparticle = model.field(x)
            Efieldparticle = Efieldparticle * std + mean
            Efieldparticle = Efieldparticle * ((Q * N))
            Efieldparticle = Efieldparticle - ((1/N) * cp.sum(Efieldparticle))
            return QM * Efieldparticle

        _, da_v = jvp(acc, (x_t,), (dx_t,))

        return torch_to_cp(dx_t - DT**2 * da_v)

    return Jv

def residual(model, x_cp, x_old_cp, v_old_cp, DT, QM, Q, N, std, mean):
    inputs = x_cp[None, :, :].copy() # [batch=1, channel=dim, particles]
    inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    x_t = cp_to_torch(inputs)

    #E = model(x_t[None,None,:]).squeeze()
    Efieldparticle = torch_to_cp(model.field(x_t))
    Efieldparticle = Efieldparticle * std + mean
    Efieldparticle = Efieldparticle * ((Q * N))
    Efieldparticle = Efieldparticle - ((1/N) * cp.sum(Efieldparticle))
    a = QM * Efieldparticle

    return x_cp - x_old_cp - DT*v_old_cp - DT**2 * a


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

    x_old = xp
    v_old = vp

    # initial guess (explicit Euler)
    x = x_old + DT * v_old

    for k in range(max_newton):
        #Apply periodic BCs 
        x = toPeriodicND(x, L, dim)

        r = residual(model, x, x_old, v_old, DT, QM, Q, N, std, mean)

        if cp.linalg.norm(r) < tol_newton:
            break

        b = -r

        # build matrix-free operator around CURRENT x
        Jv = make_Jv(model, x, DT, QM, Q, N, std, mean)

        def matvec(dx):
            return Jv(dx)

        N = x.shape[0]

        A = LinearOperator(
            (N, N),
            matvec=matvec,
            dtype=x.dtype
        )

        dx, info = gmres(
            A,
            b,
            tol=tol_gmres,
            restart=30,
            maxiter=5
        )

        x = x + dx

    # recover velocity
    inputs = x[None, :, :].copy() # [batch=1, channel=dim, particles]
    inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    x_t = cp_to_torch(inputs)
    #E = model(x_t[None,None,:]).squeeze()
    Efieldparticle = torch_to_cp(model.field(x_t))
    Efieldparticle = Efieldparticle * std + mean
    Efieldparticle = Efieldparticle * ((Q * N))
    Efieldparticle = Efieldparticle - ((1/N) * cp.sum(Efieldparticle))
    a = QM * Efieldparticle

    v_new = v_old + DT * a

    return x, v_new


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
    return x
