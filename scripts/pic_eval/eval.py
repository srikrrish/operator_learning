#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

import argparse
import torch
import numpy as np
import cupy as cp

from operator_learning.utils.misc import readConfig
from training.train_fno import FourierNeuralOperator
from pic_plotter import PICVisualizer
import plot_diagnostics as PD


def _to_float(v):
    """Scalar from a float, numpy/cupy 0-d array or torch tensor."""
    if hasattr(v, "detach"):                                   # torch
        v = v.detach().cpu().numpy()
    elif hasattr(v, "get") and not isinstance(v, (np.ndarray, dict)):   # cupy
        v = v.get()
    return float(np.asarray(v))


def save_diagnostics(fname, kind, args, time_acc_ms=None, **series):
    """
    Save the diagnostic time series of one run (NEOPIC prediction or reference)
    to an npz file and return them as a dict that the plotting functions of
    plot_diagnostics.py accept. Series that are None or empty (e.g. Ey and Ez
    in 1D, momentum for the cyclotron case) are skipped.
    """
    rec = {}
    for key, val in series.items():
        if val is None or len(val) == 0:
            continue
        rec[key] = np.array([_to_float(v) for v in val], dtype=np.float64)
    rec.update(kind=kind, ref=args.ref, nParticle=int(args.nParticle), dim=int(args.dim),
               testCase=args.testCase, dt=float(args.dt), T=float(args.T),
               kc=float(args.kc), alpha=float(args.alpha), runId=int(args.runId))
    if time_acc_ms is not None:
        rec["time_acc_ms"] = float(time_acc_ms)
    if getattr(args, "tTrain", None) is not None:
        rec["t_train"] = float(args.tTrain)
    os.makedirs(os.path.dirname(str(fname)), exist_ok=True)
    np.savez(fname, **{k: np.asarray(v) for k, v in rec.items()})
    print(f"saved {fname}")
    return rec

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Evaluate a PIC FNO model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--kc", default="0.5", type=float, help="wave vector")
parser.add_argument(
    "--NG", default="32", type=int, help="number of grid points")
parser.add_argument(
    "--T", default="20", type=float, help="Time")
parser.add_argument(
    "--dt", default="0.05", type=float, help="timestep")
parser.add_argument(
    "--alpha", default="0.05", type=float, help="pertubation")
parser.add_argument(
    "--Vt", default=1, type=float, help="thermal velocity")
parser.add_argument(
    "--nParticle", default="50", type=int, help="number of simulation particles")
parser.add_argument(
    "--Qm", default="-1", type=float, help="charge per mass")
parser.add_argument(
    "--checkpoint", default=None, help="model checkpoint")
parser.add_argument(
    "--runId", default="1",type=int,  help="run index")
parser.add_argument(
    "--imgExt", default="png", help="extension for figure files")
parser.add_argument(
    "--evalDir", default="eval", help="directory to store the evaluation results")
parser.add_argument(
    "--dim", default="1", type=int, help="dimension")
parser.add_argument(
    "--predOnly", action="store_true", help="Perform only ML predictions without reference results")
parser.add_argument(
    "--testCase", default="strongLandau", help="Choose the test case among weakLandau, strongLandau, tsi or bti")
parser.add_argument(
    "--ref", default="pic", help="Choose the reference numerical scheme pic or pif")
parser.add_argument(
    "--ml_time_int", default="explicit", help="Choose explicit or implicit time integration for NEOPIC")
parser.add_argument(
    "--implicitTheta", default=0.5, type=float,
    help="implicit NEOPIC: theta scheme, 0.5 = implicit midpoint (Crank-Nicolson), 1.0 = backward Euler")
parser.add_argument(
    "--implicitPrecond", default=True, type=lambda s: str(s).lower() not in ("0", "false", "no"),
    help="implicit NEOPIC: physics-based block preconditioner for GMRES")
parser.add_argument(
    "--implicitVerbose", default=True, type=lambda s: str(s).lower() not in ("0", "false", "no"),
    help="implicit NEOPIC: per-iteration output (True) or one summary line per step without extra JVPs (False)")
parser.add_argument(
    "--implicitRestart", default=10, type=int,
    help="implicit NEOPIC: GMRES restart length (CuPy runs a full cycle before checking convergence)")
parser.add_argument(
    "--implicitGmresTol", default=None, type=float,
    help="implicit NEOPIC: fixed relative GMRES tolerance; default: adaptive max(1e-6, min(1e-2, ||F_k||/||F_0||))")
parser.add_argument(
    "--implicitWp2", default=None, type=float,
    help="implicit NEOPIC: plasma frequency^2 for the preconditioner (default: estimated each step)")
parser.add_argument(
    "--tTrain", default=None, type=float,
    help="end of the training time horizon (t omega_p), shown in the diagnostic plots")
parser.add_argument(
    "--snapshots", action="store_true",
    help="cyclotron: save binned density/field snapshots (fixed axes and colour scales)")
parser.add_argument(
    "--saveEvery", default=0, type=int,
    help="write x, v (time-centred) and E of every saveEvery-th step to the HDF5 output; <= 0 disables it")
parser.add_argument(
    "--snapEvery", default=100, type=int, help="snapshot interval in steps (the last step is always saved)")
parser.add_argument(
    "--snapExtent", default=0.3, type=float, help="snapshot half-width around the cloud centre")
parser.add_argument(
    "--snapBins", default=512, type=int, help="snapshot bins per direction")
parser.add_argument(
    "--snapRhoRange", default=None, type=float, nargs=2, help="colour range of the density n/N")
parser.add_argument(
    "--snapEmax", default=None, type=float, help="colour range +-Emax of the field snapshots")
parser.add_argument(
    "--config", default=None, help="configuration file")
args = parser.parse_args()


if args.config is not None:
    config = readConfig(args.config)
    if "eval" in config:
        args.__dict__.update(**config["eval"])
    if "train" in config and "checkpoint" in config["train"]:
        args.checkpoint = config.train.checkpoint
        if "trainDir" in config.train:
            FourierNeuralOperator.TRAIN_DIR = config.train.trainDir

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
checkpoint = args.checkpoint
dim = args.dim
predOnly = args.predOnly
testCase = args.testCase

seed = 152
#seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
cp.random.seed(seed)
torch.cuda.manual_seed_all(seed)
vis = PICVisualizer(args)

EextPred = EextRef = None   # external potential energy (Penning trap only)
cycPred, cycRef = {}, {}    # cyclotron: Lcan, Rrms, EerrPIC
if checkpoint is not None:
    fno_model = FourierNeuralOperator(checkpoint=checkpoint, eval_only=True, device=device, data_class='pic')
    posPred, velPred, wPred, EnergyPred, EkPred, EpPred, pPred, ExpPred, EypPred, EzpPred, timePred = vis.picND(ml_acc=True, model=fno_model, data_file=config.data.dataFile)
    EextPred = list(vis.Eext) if testCase == "penning" else None
    cycPred = vis.cyclotron_series()
    phase_spacePred = None

else:
    EnergyPred = None
    EkPred = None
    EpPred = None
    EPred = None
    pPred = None
    ExpPred = None
    EypPred = None
    EzpPred = None
    timePred = None
    phase_spacePred = None
    growth_ratePred = None
    speedup = 1

if predOnly is False:
    posRef, velRef, wRef, EnergyRef, EkRef, EpRef, pRef, ExpRef, EypRef, EzpRef, timeRef = vis.picND(ml_acc=False)
    EextRef = list(vis.Eext) if testCase == "penning" else None
    cycRef = vis.cyclotron_series()
    phase_spaceRef = None
else:
    EnergyRef = None
    EkRef = None
    EpRef = None
    ERef = None
    pRef = None
    ExpRef = None
    EypRef = None
    EzpRef = None
    timeRef = None
    phase_spaceRef = None
    growth_rateRef = None
    speedup = 1

# -----------------------------------------------------------------------------
# Save diagnostics (prediction and reference in separate files), then plot them.
# The figures can be remade later without rerunning, with plot_diagnostics.py.
# -----------------------------------------------------------------------------
diag_dir = vis.eval_dir / "diagnostics"
tag = f"{testCase}_{args.ref}_N{args.nParticle}_run{args.runId}"
recPred = recRef = None
if checkpoint is not None:
    recPred = save_diagnostics(diag_dir / f"pred_{tag}.npz", "pred", args, time_acc_ms=timePred,
                               energy=EnergyPred, Ek=EkPred, Ep=EpPred, momentum=pPred,
                               Ex=ExpPred, Ey=EypPred, Ez=EzpPred, Eext=EextPred, **cycPred)
if predOnly is False:
    recRef = save_diagnostics(diag_dir / f"ref_{tag}.npz", "ref", args, time_acc_ms=timeRef,
                              energy=EnergyRef, Ek=EkRef, Ep=EpRef, momentum=pRef,
                              Ex=ExpRef, Ey=EypRef, Ez=EzpRef, Eext=EextRef, **cycRef)

figs = PD.plot_all(recPred, recRef, str(vis.eval_dir), tag=f"run{args.runId}", ext=args.imgExt)
energy = figs["energy"]
conserv_error = figs["conservation"]

HEADER = """
# FNO evaluation for PIC in {dim}D on {device}
## Simulation Configuration

| Parameter    | Value   |
|--------------|---------|
{rows}
"""

# Convert dict into Markdown table rows
rows = "\n".join([f"| {k:<12} | {v} |" for k, v in args.__dict__.items()])


op = os.path
with open(op.dirname(op.abspath(op.realpath(__file__)))+"/eval_template.md") as f:
    TEMPLATE = f.read()

summary = open(f"{args.evalDir}/eval_run{args.runId}.md", "w")
summary.write(HEADER.format(dim=dim, device=device_name, rows=rows))

if phase_spaceRef is not None:
    TEMPLATE += f"- [Phase space Ref]({phase_spaceRef})\n"
if phase_spacePred is not None:
    TEMPLATE += f"- [Phase space Pred]({phase_spacePred})\n"

TEMPLATE  += f"\nAverage time for Accleration per timestep in PIC (millisec): {timeRef}\n"

if timePred is not None and timeRef is not None:
    speedup = round(timeRef/timePred,3)
    TEMPLATE += f"Average Inference time for Accleration using FNO (millisec): {timePred}\n"
    TEMPLATE += f"Speed up PIC/FNO: {speedup}\n"
                
summary.write(TEMPLATE.format(
        dim=dim,
        device=device,
        energy=energy,
        conserv_errors=conserv_error,
        landau_decay=None,
        phase_spaceRef=phase_spaceRef,
        phase_spacePred=phase_spacePred,
        growth_rate=None,
        timeRef=timeRef,
        timePred=timePred,
        speedup=speedup
        ))
summary.close()
