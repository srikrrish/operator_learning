import os
import sys
import contextlib
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp
import time
import mpl_toolkits.mplot3d
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
import h5py
from initial_conditions import InvTransSampling, inv_trans_sampling_gpu
from dynamics import toPeriodic, accelerate, accelerateML, move, push, push1, push2, newton_push_move_xv, picard_push_move_xv, toPeriodicND, toPeriodicND_old
from field import field, fieldInFourier
from interpolation import interpMatrix, interpolate, p2g_g2p_nostencil_arrays, scatterFourier, gatherFourier
from landau_decay import period, decayRate
from energy import potential, kinetic
from operator_learning.data.pic_dataset import normalize_per_sample
from FourierKernels import specKernel, freeSpaceKernelsPIF, freeSpaceKernelsPIC
import plot_diagnostics as PD

class PICVisualizer:
    def __init__(self, args):
        """
        Visualization utilities for PIC simulations.

        """
        self.args = args
        self.base_dir = Path(__file__).resolve().parent
        self.eval_dir = (self.base_dir / args.evalDir).resolve()
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.QM = args.Qm
        self.N = args.nParticle
        self.NG = args.NG
        self.DT = args.dt
        self.T = args.T
        self.VT = args.Vt
        self.NT = int(round(self.T/self.DT))
        self.times = np.linspace(0, self.NT * self.DT, self.NT)   # number of time steps
        self.dim = args.dim
        self.ref = args.ref
        if self.dim == 1:
            self.k = np.array([args.kc])
        elif self.dim == 2:
            self.k = np.array([args.kc, args.kc])
        else:
            self.k = np.array([args.kc, args.kc, args.kc])
        self.testCase = args.testCase
        self.v0AtMinusHalf = bool(getattr(args, "v0AtMinusHalf", False))
        self.ml_time_int = args.ml_time_int
        if(self.testCase == 'cyclotron'):
            self.L = np.array([1, 1])
            self.B0 = cp.array([0,0,300]) # Constant external magnetic field
        elif(self.testCase == 'penning'):
            self.L = np.array([25, 25, 25])
            self.B0 = cp.array([0,0,5]) # Constant external magnetic field
        else:
            sigma = 1.0
            self.L = (2*np.pi)/(self.k * sigma) # Length of the container
            self.B0 = None

        self.dx = np.array(self.L / self.NG) # cell length
        self.Ln = cp.asarray(self.L)
        self.dxn = cp.asarray(self.dx)
        self.kn = cp.asarray(self.k)
        self.alpha = args.alpha

        if self.dim == 1:
            self.Q = self.L[0]/ (self.QM * self.N)                                 # Charge of a particle
            self.rho_back = - self.Q * self.N / self.L[0]                          # background rho
        elif self.dim == 2:
            if(self.testCase != 'cyclotron'):
                self.Q = self.L[0] * self.L[1] / (self.QM * self.N)
            else:
                self.Q = (self.L[0] * self.L[1] * 10) / (self.QM * self.N)
            self.rho_back = - self.Q * self.N / (self.L[0] * self.L[1])
        else:
            if(self.testCase != 'penning'):
                self.Q = self.L[0] * self.L[1] * self.L[2] / (self.QM * self.N)
            else:
                self.Q = 1562.5 / (self.QM * self.N)
            self.rho_back = - self.Q * self.N / (self.L[0] * self.L[1] * self.L[2])


        # Barnes-Hut reference run (PEPC output, particles in the same order at every step).
        # It provides the initial particles for the cyclotron case and whenever ref == 'bh',
        # and the reference fields and trajectories for ref == 'bh'.
        self.bh_file = getattr(args, "bhFile", None)
        self.use_bh_file = (self.testCase == 'cyclotron') or (self.ref == 'bh')
        if self.use_bh_file and not self.bh_file:
            raise ValueError("testCase 'cyclotron' and ref 'bh' need the Barnes-Hut reference file: set bhFile")

        if not self.use_bh_file:
            self.xp0,self.vp0 = inv_trans_sampling_gpu(alpha=self.alpha, k=self.kn, L=self.Ln, N=self.N, dim=self.dim, label=self.testCase, ref=self.ref)
        else:
            with h5py.File(self.bh_file, "r") as f:
                self.xp0 = cp.array(f["pos_bh"][0], dtype=cp.float32)
                self.vp0 = cp.array(f["vel_bh"][0], dtype=cp.float32)

            self.xp0 = self.xp0.swapaxes(-1,-2)
            self.vp0 = self.vp0.swapaxes(-1,-2)

        print(f"Initial conditions done")
        # Set matplotlib defaults (better figures)
        plt.rcParams.update({
            "figure.figsize": (6, 4),
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "grid.alpha": 0.4,
            "grid.linestyle": "--"
        })

    def _img_path(self, name: str) -> Path:
        filename = f"{name}_run{self.args.runId}.{self.args.imgExt}"
        return filename


    def grid_field_energy(self, xp):
        """
        Self-field energy (1/2) int |E|^2 dV of a particle configuration, computed on
        the PIC grid: deposit the charge (with the neutralizing background), solve the
        periodic Poisson equation and integrate |E_grid|^2 over the cells. Returns the
        three components int E_i^2 dV. Unlike the particle average V/N sum_p |E(x_p)|^2,
        this is correct for non-uniform particle distributions such as the Penning cloud,
        and it depends only on the positions, so it is the same diagnostic for NEOPIC
        and for the reference.
        """
        rho, _, _ = p2g_g2p_nostencil_arrays(XP=xp, DX=self.dxn, NG=self.NG, L=self.Ln, dim=self.dim,
                                             testCase=self.testCase, Q=self.Q, rho_back=self.rho_back)
        _, Eg = field(rho=rho, L=self.Ln, dim=self.dim, J=None, T1=None, testCase=self.testCase, NG=self.NG)
        dV = self.dxn[0] * self.dxn[1] * self.dxn[2]
        return [cp.sum(Eg[i]**2) * dV for i in range(3)]

    def cyclotron_field_energy(self, xp, return_field=False):
        """
        Electrostatic self-energy W = (1/2) sum_p q phi(x_p) of the 2D cyclotron
        cloud with open boundaries, phi being the free-space potential
        (G = -ln r / 2pi) from the Vico-Greengard solver of the PIC reference.
        In 2D, (1/2) int |E|^2 dA diverges logarithmically for a net charge, so W is
        the energy that enters the conserved total; it is the same position-only
        diagnostic for NEOPIC, PIC and BH. W is translation invariant, so the
        positions are shifted (internally, by a multiple of dx fixed per run) to
        put the cloud in the middle of the solver box: the BH and NEOPIC frames have
        mean 0, the PIC frame mean L/2 (shift 0). Particles whose TSC stencil leaves
        the box are skipped (reported once).
        With return_field=True it also returns the free-space PIC field at the
        particles, shape (2, N) (zero for skipped particles), and the mask of the
        particles that were inside the box: (W, E_pic, inside).
        """
        NG = self.NG
        if self._cyc is None:
            J, T1 = freeSpaceKernelsPIC(NG=NG, L=self.Ln)
            mean = cp.mean(xp, axis=1)
            shift = cp.rint((self.Ln / 2 - mean) / self.dxn) * self.dxn
            self._cyc = dict(J=J, T1=T1, shift=shift[:, None], warned=False)
        xs = xp + self._cyc["shift"]
        g = cp.floor(xs / self.dxn[:, None]).astype(cp.int32)
        inside = cp.all((g >= 1) & (g <= NG - 2), axis=0)
        n_out = int(xs.shape[1] - int(cp.count_nonzero(inside)))
        if n_out > 0:
            xs = xs[:, inside]
            if not self._cyc["warned"]:
                print(f"cyclotron_field_energy: {n_out} particles outside the solver box are skipped")
                self._cyc["warned"] = True
        rho, _, _ = p2g_g2p_nostencil_arrays(XP=xs, DX=self.dxn, NG=NG, L=self.Ln, dim=2,
                                             testCase='cyclotron', Q=self.Q)
        phi, Eg = field(rho=rho, L=self.Ln, dim=2, J=self._cyc["J"], T1=self._cyc["T1"],
                        testCase='cyclotron', NG=NG)
        # physical grid nodes sit at [NG:2NG] of the extended grid (as for E in field())
        phic = phi[NG:2 * NG, NG:2 * NG]
        _, phip, _ = p2g_g2p_nostencil_arrays(XP=xs, DX=self.dxn, NG=NG, L=self.Ln, dim=2,
                                              testCase='cyclotron', E=cp.stack([phic, phic]))
        W = 0.5 * self.Q * cp.sum(phip[0].astype(cp.float64))
        if not return_field:
            return W
        _, Ein, _ = p2g_g2p_nostencil_arrays(XP=xs, DX=self.dxn, NG=NG, L=self.Ln, dim=2,
                                             testCase='cyclotron', E=Eg)
        Epic = cp.zeros((2, xp.shape[1]), dtype=cp.float64)
        Epic[:, inside] = Ein
        return W, Epic, inside

    def cyclotron_frame(self, xp):
        """Cloud-centred coordinates (origin at the centre of the solver box after the
        per-run shift of cyclotron_field_energy, i.e. at the initial centre of the
        cloud to within dx/2). The same for PIC (mean L/2) and BH/NEOPIC (mean 0)."""
        return xp + self._cyc["shift"] - self.Ln[:, None] / 2

    def canonical_angular_momentum(self, xc, v):
        """
        Canonical angular momentum about the fixed origin of xc for the uniform
        field B = B_z e_z (vector potential A = B_z/2 (-y, x)):
            L = sum_p [ m (x v_y - y v_x) + (q B_z / 2) (x^2 + y^2) ],
        m = Q/QM, q = Q. It is conserved by any self-field made of pairwise central
        forces (free-space PIC up to discretization, Barnes-Hut up to the tree
        error), but not built into a learned field. Also returns the rms radius
        sqrt(<x^2 + y^2>) of the cloud, which carries the magnetic part.
        v is the mean of v^{n-1/2} and v^{n+1/2} at the positions xc = x^n. For the
        Boris pusher the mechanical term is scaled by 1 + (omega_c dt/2)^2
        = 1/cos^2(theta/2): with this factor L is conserved by Boris to round-off for
        any central self-field (2e-6 in a free-space PIC test at omega_c dt = 3,
        against 2.6e-2 without it), while a torque from a non-central field error
        (e.g. Ex 1% too large) shows up directly (2e-3).
        """
        m = self.Q / self.QM
        Bz = self.B0[2]
        alpha = 1.0 + float(self.QM * Bz * self.DT / 2) ** 2
        r2 = xc[0].astype(cp.float64)**2 + xc[1].astype(cp.float64)**2
        mech = alpha * m * cp.sum(xc[0] * v[1] - xc[1] * v[0])
        mag = 0.5 * self.Q * Bz * cp.sum(r2)
        return mech + mag, cp.sqrt(cp.mean(r2))

    def _bin2d(self, xc, ext, nb, w=None):
        """Sum of w (or counts) on an nb x nb grid over [-ext, ext]^2 (index [ix, iy])."""
        ix = cp.floor((xc[0] + ext) / (2 * ext) * nb).astype(cp.int64)
        iy = cp.floor((xc[1] + ext) / (2 * ext) * nb).astype(cp.int64)
        ok = (ix >= 0) & (ix < nb) & (iy >= 0) & (iy < nb)
        flat = ix[ok] * nb + iy[ok]
        ww = None if w is None else w[ok].astype(cp.float64)
        return cp.bincount(flat, weights=ww, minlength=nb * nb).reshape(nb, nb)

    def cyclotron_snapshot(self, xc, E, Epic, it, label):
        """
        Binned snapshot of the cyclotron cloud on a fixed grid (the same for every
        method and step): particle counts, the mean field per bin of the method
        (E, shape (2, N)) and, if Epic is given, the mean of |E - E_pic| / max|E_pic|
        per bin. Saved to <evalDir>/snapshots/snap_<label>_<it>.npz and plotted with
        plot_diagnostics.plot_cyclotron_snapshot (fixed axes and colour scales).
        """
        a = self.args
        ext = float(getattr(a, "snapExtent", 0.3))
        nb = int(getattr(a, "snapBins", 512))
        cnt = self._bin2d(xc, ext, nb)
        safe = cp.maximum(cnt, 1)
        rec = dict(counts=cp.asnumpy(cnt.astype(cp.int32)),
                   Ex=cp.asnumpy((self._bin2d(xc, ext, nb, E[0]) / safe).astype(cp.float32)),
                   Ey=cp.asnumpy((self._bin2d(xc, ext, nb, E[1]) / safe).astype(cp.float32)),
                   extent=ext, nParticle=int(xc.shape[1]), it=int(it), t=float(it * self.DT),
                   label=label, testCase=self.testCase, ref=self.ref)
        if Epic is not None:
            dE = cp.sqrt(cp.sum((E - Epic)**2, axis=0)) / cp.max(cp.sqrt(cp.sum(Epic**2, axis=0)))
            rec["dE"] = cp.asnumpy((self._bin2d(xc, ext, nb, dE) / safe).astype(cp.float32))
        out = self.eval_dir / "snapshots"
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"snap_{label}_{it}_run{a.runId}.npz"
        np.savez_compressed(fname, **rec)
        PD.plot_cyclotron_snapshot(rec, str(out), tag=f"run{a.runId}", ext=a.imgExt,
                                   rho_range=getattr(a, "snapRhoRange", None),
                                   emax=getattr(a, "snapEmax", None))

    def cyclotron_series(self):
        """Cyclotron time series of the last picND run, for save_diagnostics."""
        if self.testCase != 'cyclotron':
            return {}
        return dict(Lcan=list(self.Lcan), Rrms=list(self.Rrms), EerrPIC=list(self.EerrPIC))

    def penning_external_energy(self, xp):
        """
        Potential energy of the particles in the external quadrupole field of the
        Penning trap, sum_p q phi_ext(x_p), with
            E_ext = (-15/L (x-L/2), -15/L (y-L/2), 30/L (z-L/2)) = -grad(phi_ext),
            phi_ext = 7.5/L [(x-L/2)^2 + (y-L/2)^2] - 15/L (z-L/2)^2.
        xp has shape (3, N) and holds the positions at the same time level as the
        kinetic and field energies.
        """
        L = self.Ln
        dx = xp[0] - L[0] / 2
        dy = xp[1] - L[1] / 2
        dz = xp[2] - L[2] / 2
        phi_ext = (7.5 / L[0]) * dx**2 + (7.5 / L[1]) * dy**2 - (15.0 / L[2]) * dz**2
        return self.Q * cp.sum(phi_ext)

    def picND(self, ml_acc: bool = False, model = None, data_file = None):
        """
        Run a 1D/2D/3D Particle-In-Cell (PIC), Particle-In-Fourier (PIF) or Particle-In-Neural Operator (PINOP) simulation.

        Args:
            ml_acc (bool, optional): If True, use machine-learning-based acceleration
                                    instead of the standard PIC acceleration. Default is False.
            model: FNO model Class

        Returns:
            tuple: (xp, vp, wp, E, Ek, Ep, momentum, Exp) where
                xp (cp.ndarray): Final particle positions (shape: [dim,N]).
                vp (cp.ndarray): Final particle velocities (shape: [dim,N]).
                wp (float or cp.ndarray): Particle weights.
                E (list[float]): Total energy per time step.
                Ek (list[float]): Kinetic energy per time step.
                Ep (list[float]): Potential energy per time step.
                momentum (list[float]): Total momentum per time step.
                Exp (list[float]): Electric field energy per time step.

        Notes:
            - Initializes particle positions using inverse transform sampling.
            - Uses quadratic (pic) or spectral (pif) interpolation to project particle charges to the grid.
            - Solves d-dimensional Poisson equation to compute electric potential and field.
            - Updates particle velocities and positions using standard or ML acceleration.
            - Computes kinetic, potential, and field energies, as well as momentum conservation.
        """
        xp, vp = self.xp0.copy(), self.vp0.copy()
        wp = 1.0

        # Mean and Std of training output data
        if data_file is not None:
            data = h5py.File(data_file, 'r')
            data_output_mean = data['infos']['output_mean'][()]
            data_output_std = data['infos']['output_std'][()]

        # Energy and momentum tracking
        Ek, Ep, Exp, E, momentum = [], [], [], [], []
        # External potential energy (Penning trap only); read by eval.py after each run
        self.Eext = []
        # free-space kernels and frame shift for the cyclotron field energy (set per run)
        self._cyc = None
        # cyclotron: canonical angular momentum, rms radius, relative field error vs PIC
        self.Lcan, self.Rrms, self.EerrPIC = [], [], []
        snap_on = bool(getattr(self.args, "snapshots", False)) and self.testCase == 'cyclotron'
        snap_every = int(getattr(self.args, "snapEvery", 100))
        snap_label = 'neopic' if ml_acc else self.ref
        # Particle output (x, v, E) every save_every steps, i.e. at t = 0, save_every*dt, ...
        # The state is written after the push, so that the stored velocity is the
        # time-centred average (v^{n-1/2} + v^{n+1/2})/2 at the same time t_n as x^n and E^n
        # (at n = 0 the initial velocity v^0). The raw half-step velocity v^{n-1/2}
        # (v^0 at n = 0) is stored as well.
        # saveEvery <= 0 switches the particle output off
        save_every_arg = int(getattr(self.args, "saveEvery", 0))
        save_on = save_every_arg > 0
        save_every = max(save_every_arg, 1)
        pending_save = None
        # start fresh output files (write_hdf5_step appends); never touch the Barnes-Hut input data
        out_files = [f"Inference_{self.N}_{self.testCase}.h5"] if ml_acc else \
                    ([f"Reference_{self.ref}_{self.N}_{self.testCase}.h5"] if self.ref in ('pic', 'pif') else [])
        for fn in (out_files if save_on else []):
            if os.path.exists(f'{self.eval_dir}/{fn}'):
                os.remove(f'{self.eval_dir}/{fn}')

        if(self.dim == 1):
            Eyp = None
            Ezp = None
        elif(self.dim == 2):
            Eyp = []
            Ezp = None
        else:
            Eyp = []
            Ezp = []

        # Time tracking
        times_acc = []

        _ev_start, _ev_end = [], []

        if((self.ref == 'pif') and (self.testCase != 'cyclotron')):
            SHat = specKernel(NG=self.NG, L=self.Ln, dx=self.dxn, dim=self.dim)

        if(self.testCase == 'cyclotron'):
            if(self.ref == 'pif'):
                SHat, J, T1, T2 = freeSpaceKernelsPIF(NG=self.NG, L=self.Ln)
            else:
                J, T1 = freeSpaceKernelsPIC(NG=self.NG, L=self.Ln)
        else:
            J = None
            T1 = None
            T2 = None

        output_keys_E = "Eout_bh"
        output_keys_pos = "pos_bh"
        output_keys_vel = "vel_bh"

        if(self.testCase == 'penning'):
            if ml_acc and model is not None:
                inputs = xp[None, :, :].copy() # [batch=1, channel=dim, particles]
                inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
                inputs[:, 1, :] = normalize_per_sample(inputs[:, 1, :])
                inputs[:, 2, :] = normalize_per_sample(inputs[:, 2, :])

                prediction = model(inputs) # [1, channel=dim, particles]
                Efieldparticle = prediction.squeeze()
                Efieldparticle[0] = Efieldparticle[0] * data_output_std[0] + data_output_mean[0]
                Efieldparticle[1] = Efieldparticle[1] * data_output_std[1] + data_output_mean[1]
                Efieldparticle[2] = Efieldparticle[2] * data_output_std[2] + data_output_mean[2]
                #Scale by normalization factor \alpha = Q_tot / (L_x * L_y * L_z)^(2/3) in 3D for the current problem
                Efieldparticle[:,:] = Efieldparticle[:,:] * ((self.Q * self.N)/((self.Ln[0] * self.Ln[1] * self.Ln[2])**(2/3)))
                #Subtract volume average of electric field for periodic compatibility
                Efieldparticle[0] = Efieldparticle[0] - ((1/self.N) * cp.sum(Efieldparticle[0]))
                Efieldparticle[1] = Efieldparticle[1] - ((1/self.N) * cp.sum(Efieldparticle[1]))
                Efieldparticle[2] = Efieldparticle[2] - ((1/self.N) * cp.sum(Efieldparticle[2]))
                a = accelerateML(E=Efieldparticle, xp=xp, L=self.Ln, testCase=self.testCase, wp=wp, QM=self.QM)
            else:
                if(self.ref == 'pic'):
                    # Interpolation: particle -> grid
                    rho, _, _ = p2g_g2p_nostencil_arrays(XP=xp, DX=self.dxn, NG=self.NG, L=self.Ln, dim=self.dim, testCase=self.testCase, Q=self.Q, rho_back=self.rho_back)
                    # Compute fields
                    phi, Eg = field(rho=rho, L=self.Ln, dim=self.dim, J=J, T1=T1, testCase=self.testCase, NG=self.NG)
                    # Interpolation: grid -> particle
                    _, Efieldparticle, a = p2g_g2p_nostencil_arrays(XP=xp, DX=self.dxn, NG=self.NG, L=self.Ln, dim=self.dim, testCase=self.testCase, E=Eg, QM=self.QM)
                elif(self.ref == 'pif'):
                    # Interpolation: particle -> Fourier space
                    rhoHat = scatterFourier(XP=xp, SHat=SHat, NG=self.NG, N=self.N, Q=self.Q, L=self.Ln, dim=self.dim, testCase=self.testCase)
                    # Compute fields in Fourier space
                    phiHat, EHat = fieldInFourier(rhoHat=rhoHat, L=self.Ln, dim=self.dim, testCase=self.testCase, ref=self.ref, J=J, T1=T1, Q=self.Q, T2=T2)
                    # Interpolation fields (in Fourier space) -> particles
                    Efieldparticle, a = gatherFourier(XP=xp, EHat=EHat, SHat=SHat, QM=self.QM, L=self.Ln, dim=self.dim, testCase=self.testCase)

        # the Barnes-Hut file is read step by step inside the loop
        with (h5py.File(self.bh_file, "r") if self.use_bh_file else contextlib.nullcontext()) as f:
            for it in range(self.NT):

                print(it)
                if(self.testCase == 'penning'):
                    vp = push1(vp, a, self.DT, self.QM, self.B0)
                    xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.Ln, it=it)

                if(self.testCase != 'cyclotron'):
                    #Apply periodic BCs
                    xp = toPeriodicND_old(x=xp, L=self.Ln, dim=self.dim)

                # Acceleration
                if ml_acc and model is not None:
                    if(self.ml_time_int == 'explicit'):
                        e_start = cp.cuda.Event(); e_end = cp.cuda.Event()
                        e_start.record()
                        inputs = xp[None, :, :].copy() # [batch=1, channel=dim, particles]
                        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])

                        if(self.dim > 1):
                            inputs[:, 1, :] = normalize_per_sample(inputs[:, 1, :])
                        if(self.dim > 2):
                            inputs[:, 2, :] = normalize_per_sample(inputs[:, 2, :])

                        prediction = model(inputs) # [1, channel=dim, particles]
                        Efieldparticle = prediction.squeeze()
                        if(self.dim == 1):
                            Efieldparticle = Efieldparticle * data_output_std + data_output_mean
                            #Scale by normalization factor \alpha = Q_tot in 1D for the current problem
                            Efieldparticle = Efieldparticle * ((self.Q * self.N))
                            #Subtract particle average of electric field for zero net self force
                            Efieldparticle = Efieldparticle - ((1/self.N) * cp.sum(Efieldparticle))
                        elif(self.dim == 2):
                            Efieldparticle[0] = Efieldparticle[0] * data_output_std[0] + data_output_mean[0]
                            Efieldparticle[1] = Efieldparticle[1] * data_output_std[1] + data_output_mean[1]
                            ###Scale by normalization factor \alpha = Q_tot / sqrt(L_x * L_y) in 2D for the current problem
                            Efieldparticle[:,:] = Efieldparticle[:,:] * ((self.Q * self.N)/cp.sqrt(self.Ln[0] * self.Ln[1]))
                            if(self.testCase != 'cyclotron'):
                                #Subtract particle average of electric field for zero net self force
                                Efieldparticle[0] = Efieldparticle[0] - ((1/self.N) * cp.sum(Efieldparticle[0]))
                                Efieldparticle[1] = Efieldparticle[1] - ((1/self.N) * cp.sum(Efieldparticle[1]))
                        else:
                            Efieldparticle[0] = Efieldparticle[0] * data_output_std[0] + data_output_mean[0]
                            Efieldparticle[1] = Efieldparticle[1] * data_output_std[1] + data_output_mean[1]
                            Efieldparticle[2] = Efieldparticle[2] * data_output_std[2] + data_output_mean[2]
                            #Scale by normalization factor \alpha = Q_tot / (L_x * L_y * L_z)^(2/3) in 3D for the current problem
                            Efieldparticle[:,:] = Efieldparticle[:,:] * ((self.Q * self.N)/((self.Ln[0] * self.Ln[1] * self.Ln[2])**(2/3)))
                            #Subtract particle average of electric field for zero net self force
                            Efieldparticle[0] = Efieldparticle[0] - ((1/self.N) * cp.sum(Efieldparticle[0]))
                            Efieldparticle[1] = Efieldparticle[1] - ((1/self.N) * cp.sum(Efieldparticle[1]))
                            Efieldparticle[2] = Efieldparticle[2] - ((1/self.N) * cp.sum(Efieldparticle[2]))
                        #
                        if save_on and it % save_every == 0:
                            pending_save = dict(filename=f"Inference_{self.N}_{self.testCase}.h5", xp=xp.copy(), vp=vp.copy(),
                                                Efieldparticle=Efieldparticle.copy(), pos_key="pos", vel_key="vel", E_key="Eout")
                        a = accelerateML(E=Efieldparticle, xp=xp, L=self.Ln, testCase=self.testCase, wp=wp, QM=self.QM)
                        e_end.record()
                        _ev_start.append(e_start); _ev_end.append(e_end)
                    else:
                        e_start = cp.cuda.Event(); e_end = cp.cuda.Event()
                        e_start.record()
                        xp, vp = newton_push_move_xv(model, xp, vp, self.DT, self.QM, self.Q, self.N, self.Ln, self.dim, data_output_std, data_output_mean,
                                                     theta=float(getattr(self.args, "implicitTheta", 0.5)),
                                                     precond=bool(getattr(self.args, "implicitPrecond", True)),
                                                     wp2=getattr(self.args, "implicitWp2", None),
                                                     verbose=bool(getattr(self.args, "implicitVerbose", True)),
                                                     restart=int(getattr(self.args, "implicitRestart", 10)),
                                                     gmres_tol=(None if getattr(self.args, "implicitGmresTol", None) is None
                                                                else float(getattr(self.args, "implicitGmresTol"))))
                        kinetic_energy = kinetic(vp, self.Q, self.QM, wp)
                        inputs = xp[None, :, :].copy() # [batch=1, channel=dim, particles]
                        inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
                        prediction = model(inputs) # [1, channel=dim, particles]
                        Efieldparticle = prediction.squeeze()
                        Efieldparticle = Efieldparticle * data_output_std + data_output_mean
                        #Scale by normalization factor \alpha = Q_tot in 1D for the current problem
                        Efieldparticle = Efieldparticle * ((self.Q * self.N))
                        #Subtract volume average of electric field for periodic compatibility
                        Efieldparticle = Efieldparticle - ((1/self.N) * cp.sum(Efieldparticle))
                        e_end.record()
                        _ev_start.append(e_start); _ev_end.append(e_end)

                else:
                    e_start = cp.cuda.Event(); e_end = cp.cuda.Event()
                    e_start.record()
                    if(self.testCase == "cyclotron" and self.ref == "bh"):
                        data_E = f[output_keys_E]
                        data_pos = f[output_keys_pos]
                        data_vel = f[output_keys_vel]
                        Efield_pepc = cp.array(data_E[it], dtype=cp.float32)
                        pos_pepc = cp.array(data_pos[it+1], dtype=cp.float32)
                        vel_pepc = cp.array(data_vel[it+1], dtype=cp.float32)
                        pos_pepc = pos_pepc.swapaxes(-1,-2)
                        vel_pepc = vel_pepc.swapaxes(-1,-2)
                        Efield_pepc = Efield_pepc.swapaxes(-1,-2)
                        vp_old = vp.copy()
                        Efieldparticle = Efield_pepc
                    else:
                        if(self.ref == 'pic'):
                            # Interpolation: particle -> grid
                            rho, _, _ = p2g_g2p_nostencil_arrays(XP=xp, DX=self.dxn, NG=self.NG, L=self.Ln, dim=self.dim, testCase=self.testCase, Q=self.Q, rho_back=self.rho_back)
                            # Compute fields
                            phi, Eg = field(rho=rho, L=self.Ln, dim=self.dim, J=J, T1=T1, testCase=self.testCase, NG=self.NG)
                            # Interpolation: grid -> particle
                            _, Efieldparticle, a = p2g_g2p_nostencil_arrays(XP=xp, DX=self.dxn, NG=self.NG, L=self.Ln, dim=self.dim, testCase=self.testCase, E=Eg, QM=self.QM)
                        elif(self.ref == 'pif'):
                            # Interpolation: particle -> Fourier space
                            rhoHat = scatterFourier(XP=xp, SHat=SHat, NG=self.NG, N=self.N, Q=self.Q, L=self.Ln, dim=self.dim, testCase=self.testCase)
                            # Compute fields in Fourier space
                            phiHat, EHat = fieldInFourier(rhoHat=rhoHat, L=self.Ln, dim=self.dim, testCase=self.testCase, ref=self.ref, J=J, T1=T1, Q=self.Q, T2=T2)
                            # Interpolation fields (in Fourier space) -> particles
                            Efieldparticle, a = gatherFourier(XP=xp, EHat=EHat, SHat=SHat, QM=self.QM, L=self.Ln, dim=self.dim, testCase=self.testCase)
                            if save_on and it % save_every == 0:
                                pending_save = dict(filename=f"Reference_pif_{self.N}_{self.testCase}.h5", xp=xp.copy(), vp=vp.copy(),
                                                    Efieldparticle=Efieldparticle.copy(), pos_key="pos_pif", vel_key="vel_pif", E_key="Eout_pif")
                        elif(self.ref == 'bh'):
                            Ept = cp.array(f[output_keys_E][it], dtype=cp.float32)
                            Efieldparticle = Ept.swapaxes(-1,-2)
                    e_end.record()
                    _ev_start.append(e_start); _ev_end.append(e_end)

                # External potential energy of the Penning trap (added to the total energy).
                # Evaluated here, at the positions where the field was just computed and
                # before any position update, so it belongs to the same time level as the
                # field energy for both the KDK and the staggered (push, then move) pushers.
                Eext = 0.0
                Egrid = None
                if(self.testCase == 'penning'):
                    Eext = self.penning_external_energy(xp)
                    # field energy on the PIC grid at the same positions (see grid_field_energy)
                    Egrid = self.grid_field_energy(xp)
                Wcyc = None
                if(self.testCase == 'cyclotron'):
                    # free-space electrostatic energy at the same positions (see cyclotron_field_energy),
                    # and the free-space PIC field at these positions
                    Wcyc, Epic, inside = self.cyclotron_field_energy(xp, return_field=True)
                    xc = self.cyclotron_frame(xp)
                    vp_prev = vp.copy()
                    # field check: the method's field against the PIC field at the same positions
                    # (skipped for the PIC reference itself, where it is zero)
                    check = ml_acc or (self.ref != 'pic')
                    Emeth = cp.asarray(Efieldparticle, dtype=cp.float64).reshape(2, -1)
                    if check:
                        dE = (Emeth - Epic)[:, inside]
                        self.EerrPIC.append(float(cp.sqrt(cp.sum(dE**2) / cp.sum(Epic[:, inside]**2))))
                    if snap_on and ((it % snap_every == 0) or (it == self.NT - 1)):
                        self.cyclotron_snapshot(xc, Emeth, Epic if check else None, it, snap_label)

                if(not ml_acc or (self.ml_time_int != 'implicit')):
                    if(self.testCase == 'cyclotron'):
                        if ml_acc:
                            vp, kinetic_energy = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it, testCase=self.testCase, B0=self.B0, v0_at_minus_half=self.v0AtMinusHalf)
                            xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.Ln, it=it)
                        else:
                            if(self.ref == "bh"):
                                vp = vel_pepc
                                xp = pos_pepc
                                if(it == 0):
                                    kinetic_energy = kinetic(vp_old, self.Q, self.QM)
                                else:
                                    #For cases with external magnetic field average the kinetic energies instead of velocities as it is better
                                    kinetic_energy = (kinetic(vp, self.Q, self.QM) + kinetic(vp_old, self.Q, self.QM))/2
                            else:
                                vp, kinetic_energy = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it, testCase=self.testCase, B0=self.B0, v0_at_minus_half=self.v0AtMinusHalf)
                                xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.Ln, it=it)
                    elif(self.testCase == 'penning'):
                        vp = push2(vp, a, self.DT, self.QM, self.B0)
                        kinetic_energy = kinetic(vp, self.Q, self.QM)
                    else:
                        if ml_acc or (self.ref != 'bh'):
                            vp, kinetic_energy = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it, testCase=self.testCase, B0=self.B0, v0_at_minus_half=self.v0AtMinusHalf)
                            xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.Ln, it=it)
                        else:
                            kinetic_energy = kinetic(vp, self.Q, self.QM)
                            vpt = cp.array(f[output_keys_vel][it+1], dtype=cp.float32)
                            vp = vpt.swapaxes(-1,-2)

                # deferred particle output: x^n, E^n and the time-centred velocity at t_n
                if pending_save is not None:
                    v_half = pending_save.pop("vp")          # v^{n-1/2} (v^0 at n = 0)
                    if self.testCase == 'penning':
                        v_cent = vp                          # KDK: vp is already v^n
                    else:
                        v_cent = v_half if it == 0 else 0.5 * (v_half + vp)
                    self.write_hdf5_step(vp=v_cent,
                                         extra={pending_save["vel_key"] + "_half": v_half},
                                         attrs=dict(dt=float(self.DT), save_every=int(save_every),
                                                    vel_note="vel: (v^{n-1/2}+v^{n+1/2})/2 at t_n (v^0 at n=0); vel_half: v^{n-1/2}"),
                                         **pending_save)
                    pending_save = None


                if(self.dim == 1):
                    # Mometum: Note since vp is at half time steps the momentum is calculated at these indices rather than integer time steps
                    mom = cp.abs(cp.sum(self.Q * vp / self.QM))
                    # Electric field energy
                    Egpx = cp.sum(Efieldparticle[:] ** 2) * self.Ln[0] / self.N
                    # Compute potential energy
                    Epotential = 0.5 * Egpx
                elif(self.dim == 2):
                    if(self.testCase != 'cyclotron'):
                        momx = cp.sum(self.Q * vp[0] / self.QM)
                        momy = cp.sum(self.Q * vp[1] / self.QM)
                        mom = cp.sqrt(momx**2 + momy**2)
                    # Electric field energy
                    Egpx = cp.sum(Efieldparticle[0,:]**2) * (self.Ln[0] * self.Ln[1]) / self.N
                    Egpy = cp.sum(Efieldparticle[1,:]**2) * (self.Ln[0] * self.Ln[1]) / self.N
                    # Compute potential energy
                    Epotential = 0.5 * (Egpx + Egpy)
                    if(Wcyc is not None):
                        # Cyclotron: free-space potential energy (Exp/Eyp keep the particle estimate)
                        Epotential = Wcyc
                        Lc, rrms = self.canonical_angular_momentum(xc, 0.5 * (vp_prev + vp))
                        self.Lcan.append(float(Lc))
                        self.Rrms.append(float(rrms))
                else:
                    momx = cp.sum(self.Q * vp[0] / self.QM)
                    momy = cp.sum(self.Q * vp[1] / self.QM)
                    momz = cp.sum(self.Q * vp[2] / self.QM)
                    mom = cp.sqrt(momx**2 + momy**2 + momz**2)
                    # Electric field energy
                    if(Egrid is not None):
                        # Penning trap: grid integral, since the cloud does not fill the box
                        Egpx, Egpy, Egpz = Egrid
                    else:
                        Egpx = cp.sum(Efieldparticle[0,:]**2) * (self.Ln[0] * self.Ln[1] * self.Ln[2]) / self.N
                        Egpy = cp.sum(Efieldparticle[1,:]**2) * (self.Ln[0] * self.Ln[1] * self.Ln[2]) / self.N
                        Egpz = cp.sum(Efieldparticle[2,:]**2) * (self.Ln[0] * self.Ln[1] * self.Ln[2]) / self.N
                    # Compute potential energy
                    Epotential = 0.5 * (Egpx + Egpy + Egpz)



                if(self.testCase == 'penning'):
                    self.Eext.append(Eext.get())

                # Append energies and momentum
                Ek.append(kinetic_energy.get())
                Ep.append(Epotential.get())
                E.append((kinetic_energy + Epotential + Eext).get())
                Exp.append(Egpx.get())
                if(self.dim > 1):
                    Eyp.append(Egpy.get())
                if(self.dim > 2):
                    Ezp.append(Egpz.get())
                if(self.testCase != 'cyclotron'):
                    momentum.append(mom.get())
                else:
                    momentum = None

            cp.cuda.Stream.null.synchronize()   # here we do a single sync for the whole run to ensure all events happened

            times_acc = [cp.cuda.get_elapsed_time(s, e) for s, e in zip(_ev_start, _ev_end)]
            #Starting from 1 and not 0 to exclude the startup time
            time_acc_mean = np.round(np.mean(times_acc[1:]), 3)   # already in ms
            print(f"Average acceleration time per iteration: {time_acc_mean:.3f} millisec")


        return xp, vp, wp, E, Ek, Ep, momentum, Exp, Eyp, Ezp, time_acc_mean

    # ---------------------------
    # Plotting methods
    # ---------------------------
    def phase_space(self, xp, vp, wp, ml_acc=False):
        # Filter: only keep particles within velocity cutoff
        mask = np.abs(vp) < 10 * self.VT
        xp = xp[mask]
        vp = vp[mask]

        g1 = np.floor(xp / self.dx).astype(int)
        g = np.array([g1 - 1, g1, g1 + 1])
        g = toPeriodic(g, self.NG, True)

        delta = xp % self.dx
        fraz = np.array([
            (1 - delta) ** 2 / 2,
            1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2),
            delta ** 2 / 2
        ]) * wp

        col = ((vp + 10 * self.VT) // (20 * self.VT / 128)).astype(int)

        n_rows = 128       # velocity bins
        n_cols = self.NG   # grid points

        mat = (
            sparse.csr_matrix((-fraz[0] * self.Q, (col, g[0])), shape=(n_rows, n_cols)) +
            sparse.csr_matrix((-fraz[1] * self.Q, (col, g[1])), shape=(n_rows, n_cols)) +
            sparse.csr_matrix((-fraz[2] * self.Q, (col, g[2])), shape=(n_rows, n_cols))
        ).todense()

        plt.figure(figsize=(7, 5))
        if ml_acc:
            filename = self._img_path("phase_spacePred")
            plt.title("Phase Space Distribution (Pred)")
        else:
            filename = self._img_path("phase_spaceRef")
            plt.title("Phase Space Distribution (Ref)")
        plt.imshow(mat, vmin=0, vmax=np.max(mat), cmap='plasma', interpolation="nearest", aspect='auto')
        plt.colorbar(label="Phase Space Density")
        plt.xlabel("Grid Index")
        plt.ylabel("Velocity Bin")
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/{filename}", dpi=200)
        plt.clf()
        return filename

    # ------------------------------------------------------------------
    # Energy, conservation and field-energy plots now live in
    # plot_diagnostics.py; these wrappers keep the old interface.
    # ------------------------------------------------------------------
    def _records(self, ref_kw, pred_kw):
        meta = dict(ref=self.ref, nParticle=self.N, dim=self.dim, testCase=self.testCase,
                    dt=self.DT, T=self.T, kc=float(self.k[0]), alpha=self.alpha,
                    runId=self.args.runId)

        def rec(kind, kw):
            d = {k: v for k, v in kw.items() if v is not None and len(v) > 0}
            if not d:
                return None
            d.update(meta, kind=kind)
            return d
        return rec("pred", pred_kw), rec("ref", ref_kw)

    def energy(self, ERef=None, EPred=None, EkRef=None, EpRef=None, EkPred=None, EpPred=None):
        pred, ref = self._records(dict(energy=ERef, Ek=EkRef, Ep=EpRef),
                                  dict(energy=EPred, Ek=EkPred, Ep=EpPred))
        return PD.plot_energy(pred, ref, f"{self.eval_dir}/{self._img_path('Energy')}")

    def conservation_errors(self, ERef=None, EPred=None, pRef=None, pPred=None):
        pred, ref = self._records(dict(energy=ERef, momentum=pRef),
                                  dict(energy=EPred, momentum=pPred))
        return PD.plot_conservation(pred, ref, f"{self.eval_dir}/{self._img_path('conservation_errors')}")

    def landau_decay(self, Ex=None, ExPred=None, Ey=None, EyPred=None, Ez=None, EzPred=None, label='weakLandau'):
        pred, ref = self._records(dict(Ex=Ex, Ey=Ey, Ez=Ez), dict(Ex=ExPred, Ey=EyPred, Ez=EzPred))
        for d in (pred, ref):
            if d is not None:
                d["testCase"] = label
        return PD.plot_field_energy(pred, ref, f"{self.eval_dir}/{self._img_path('landau_decay_rateRef')}")

    def visualize_spatial_comparison(self, true_positions, predicted_positions, true_efield, predicted_efield, timestep, output_filename):
        """
        Creates a 2x2 grid of plots showing particle positions colored by E-field values.
        Top row compares Ex (Predicted vs. True), Bottom row compares Ey.
        Crucially, it uses a shared color scale for each row to allow direct comparison.
        """
        # Use a diverging colormap, which is great for fields (positive/negative values)
        cmap = 'coolwarm'
        marker_size = 1 # Use a small marker size for 100k points

        # --- 1. Determine the shared color range for Ex and Ey ---
        # For Ex, find the min/max across both predicted and true values
        vmin_x = min(predicted_efield[0].min(), true_efield[0].min())
        vmax_x = max(predicted_efield[0].max(), true_efield[0].max())

        # For Ey, do the same
        vmin_y = min(predicted_efield[1].min(), true_efield[1].min())
        vmax_y = max(predicted_efield[1].max(), true_efield[1].max())

        # --- 2. Create the 2x2 plot grid ---
        fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=150)
        fig.suptitle(f'Spatial E-Field Comparison for Timestep {timestep}', fontsize=20)

        # --- 3. Plot the Ex comparison (Top Row) ---
        # Predicted Ex
        sc00 = axes[0, 0].scatter(predicted_positions[0], predicted_positions[1], c=predicted_efield[0],
                                cmap=cmap, vmin=vmin_x, vmax=vmax_x, s=marker_size, rasterized=True)
        axes[0, 0].set_title('Predicted $E_x$', fontsize=14)

        # True Ex
        sc01 = axes[0, 1].scatter(true_positions[0], true_positions[1], c=true_efield[0],
                                cmap=cmap, vmin=vmin_x, vmax=vmax_x, s=marker_size, rasterized=True)
        axes[0, 1].set_title('Ground Truth $E_x$', fontsize=14)

        # Add a single colorbar for the Ex row
        fig.colorbar(sc01, ax=axes[0, :], label='$E_x$ Value', fraction=0.046, pad=0.04)

        # --- 4. Plot the Ey comparison (Bottom Row) ---
        # Predicted Ey
        sc10 = axes[1, 0].scatter(predicted_positions[0], predicted_positions[1], c=predicted_efield[1],
                                cmap=cmap, vmin=vmin_y, vmax=vmax_y, s=marker_size, rasterized=True)
        axes[1, 0].set_title('Predicted $E_y$', fontsize=14)

        # True Ey
        sc11 = axes[1, 1].scatter(true_positions[0], true_positions[1], c=true_efield[1],
                                cmap=cmap, vmin=vmin_y, vmax=vmax_y, s=marker_size, rasterized=True)
        axes[1, 1].set_title('Ground Truth $E_y$', fontsize=14)

        # Add a single colorbar for the Ey row
        fig.colorbar(sc11, ax=axes[1, :], label='$E_y$ Value', fraction=0.046, pad=0.04)

        # --- 5. Final plot adjustments ---
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel('Particle X Position')
                ax.set_ylabel('Particle Y Position')
                ax.set_aspect('equal', adjustable='box') # Ensure correct spatial aspect ratio
                ax.grid(True, linestyle='--', alpha=0.5)
        filename = self._img_path(output_filename)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.close()


    def write_hdf5_step(
        self,
        filename,
        xp,
        vp,
        Efieldparticle,
        pos_key="pos",
        vel_key="vel",
        E_key="Eout",
        extra=None,
        attrs=None,
    ):
        """
        Append one inference time step to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output HDF5 file.

        xp : cupy.ndarray
            Particle positions of shape [dim, Np].

        vp : cupy.ndarray
            Particle velocities of shape [dim, Np].

        Efieldparticle : cupy.ndarray
            Electric field at particle positions of shape [dim, Np].

        pos_key : str
            Dataset name for particle positions.

        vel_key : str
            Dataset name for particle velocities.

        E_key : str
            Dataset name for electric field.
        """

        # Convert to NumPy and transpose to [Np, dim]
        pos = cp.asnumpy(xp.T.astype(cp.float32))
        vel = cp.asnumpy(vp.T.astype(cp.float32))
        E = cp.asnumpy(Efieldparticle.T.astype(cp.float32))
        extra_np = {k: cp.asnumpy(v.T.astype(cp.float32)) for k, v in (extra or {}).items()}


        first_time = not os.path.exists(f'{self.eval_dir}/{filename}')

        with h5py.File(f'{self.eval_dir}/{filename}', "a") as f:

            if first_time:

                # [1, Np, dim]; maxshape and chunks follow the array shape, so 1D, 2D and 3D work
                for key, arr in ((pos_key, pos), (vel_key, vel), (E_key, E)):
                    f.create_dataset(
                        key,
                        data=arr[np.newaxis, :, :],
                        maxshape=(None,) + arr.shape,
                        chunks=(1,) + arr.shape,
                        compression="gzip",
                    )

                for k, arr in extra_np.items():
                    f.create_dataset(
                        k,
                        data=arr[np.newaxis, :, :],
                        maxshape=(None,) + arr.shape,
                        chunks=(1,) + arr.shape,
                        compression="gzip",
                    )
                for k, val in (attrs or {}).items():
                    f.attrs[k] = val

            else:

                dset_pos = f[pos_key]
                dset_vel = f[vel_key]
                dset_E = f[E_key]

                t = dset_pos.shape[0]

                dset_pos.resize(t + 1, axis=0)
                dset_vel.resize(t + 1, axis=0)
                dset_E.resize(t + 1, axis=0)

                dset_pos[t] = pos
                dset_vel[t] = vel
                dset_E[t] = E

                for k, arr in extra_np.items():
                    f[k].resize(t + 1, axis=0)
                    f[k][t] = arr

    def visualize_Efield(self, positions, efield, timestep, output_filename):
        """
        Creates a 2x2 grid of plots showing particle positions colored by E-field values.
        Top row compares Ex (Predicted vs. True), Bottom row compares Ey.
        Crucially, it uses a shared color scale for each row to allow direct comparison.
        """
        # Use a diverging colormap, which is great for fields (positive/negative values)
        cmap = 'coolwarm'
        marker_size = 1 # Use a small marker size for 100k points

        # --- 1. Determine the shared color range for Ex and Ey ---
        # For Ex, find the min/max across both predicted and true values
        vmin_x = efield[0].min()
        vmax_x = efield[0].max()

        # For Ey, do the same
        vmin_y = efield[1].min()
        vmax_y = efield[1].max()

        # --- 2. Create the 2x2 plot grid ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

        # --- 3. Plot the Ex comparison (Top Row) ---
        # Predicted Ex
        sc00 = axes[0].scatter(positions[0], positions[1], c=efield[0],
                                   cmap=cmap, vmin=vmin_x, vmax=vmax_x, s=marker_size, rasterized=True)
        axes[0].set_title('$E_x$', fontsize=14)

        # Add a single colorbar for the Ex row
        fig.colorbar(sc00, ax=axes[0], label='$E_x$ Value', fraction=0.046, pad=0.04)

        # --- 4. Plot the Ey comparison (Bottom Row) ---
        # Predicted Ey
        sc01 = axes[1].scatter(positions[0], positions[1], c=efield[1],
                                   cmap=cmap, vmin=vmin_y, vmax=vmax_y, s=marker_size, rasterized=True)
        axes[1].set_title('$E_y$', fontsize=14)

        # Add a single colorbar for the Ey row
        fig.colorbar(sc01, ax=axes[1], label='$E_y$ Value', fraction=0.046, pad=0.04)

        # --- 5. Final plot adjustments ---
        for ax in axes:
            ax.set_xlabel('Particle X Position')
            ax.set_ylabel('Particle Y Position')
            ax.set_aspect('equal', adjustable='box') # Ensure correct spatial aspect ratio
            ax.grid(True, linestyle='--', alpha=0.5)

        filename = self._img_path(output_filename)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.close()

    def field2D(self, field, NG, output_filename):
        if isinstance(NG, int):
            NG=[NG,NG]
        x = np.linspace(0, NG[0]-1, NG[0]).astype(int).tolist()
        y = np.linspace(0, NG[0]-1, NG[0]).astype(int).tolist()
        x, y = np.meshgrid(x, y)
        fig, ax = plt.subplots(figsize=(6,5))
        pcm = ax.pcolormesh(x, y, field, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='Charge density')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        filename = self._img_path(output_filename)
        plt.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.close()

    def field2D_histogram(self, xp, NG, output_filename):

        xlim_val = float(max(np.abs(xp[0,:]).max(), np.abs(xp[1,:]).max())) * 1.3

        NBINS_D  = NG
        rng_xy   = [[-xlim_val, xlim_val], [-xlim_val, xlim_val]]
        hist0, _, _ = np.histogram2d(xp[0,:], xp[1,:], bins=NBINS_D, range=rng_xy)
        rho_vmax = max(float(hist0.max()), 1.0)
        rho_norm = matplotlib.colors.LogNorm(vmin=1, vmax=rho_vmax)
        fig, ax = plt.subplots(figsize=(6,5))
        im_rho = ax.imshow(
        hist0.T, origin='lower',
        extent=[-xlim_val, xlim_val, -xlim_val, xlim_val],
        cmap='inferno', norm=rho_norm,
        aspect='equal', interpolation='nearest',
        )
        ax.set_title(r'$\rho(x,\,y)$  density', fontsize=11)
        ax.set_xlabel(r'$x\;[c/\omega_{pe}]$', fontsize=9)
        ax.set_ylabel(r'$y\;[c/\omega_{pe}]$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15, color='white')
        cbar_rho = fig.colorbar(im_rho, ax=ax, fraction=0.046, pad=0.04)
        cbar_rho.set_label('counts / bin  (log)', fontsize=8)
        cbar_rho.ax.tick_params(labelsize=7)
        filename = self._img_path(output_filename)
        fig.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.close()

    def instability(self, Ex = None, ExPred = None, Ey = None, EyPred = None, Ez = None, EzPred = None, label='tsi'):
        pred, ref = self._records(dict(Ex=Ex, Ey=Ey, Ez=Ez), dict(Ex=ExPred, Ey=EyPred, Ez=EzPred))
        for d in (pred, ref):
            if d is not None:
                d["testCase"] = label
        return PD.plot_field_energy(pred, ref, f"{self.eval_dir}/{self._img_path('growth_rate')}")
