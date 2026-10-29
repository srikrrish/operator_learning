# NEOPIC inference: `scripts/pic_eval`

Inference code for the NEOPIC paper (S. Muralikrishnan et al., "NEOPIC: A Neural Operator
Framework for Particle-based Kinetic Plasma Simulations"). A trained nonuniform Fourier neural
operator (NUFNO) replaces the field solve of a particle method; the same script also runs the
reference PIC, PIF or Barnes-Hut simulation and compares the two.

## Running

```bash
python eval.py --config config.yaml
```

The `eval` section of the YAML file overrides the command-line defaults (`python eval.py --help`
lists all options). The model is taken from `train.checkpoint`, the output normalization
(mean and std of the training labels) from `data.dataFile`. Without a checkpoint only the
reference is run; with `predOnly: true` only NEOPIC is run.

Main `eval` options:

| key | meaning |
|---|---|
| `dim` | 1, 2 or 3 |
| `testCase` | `weakLandau`, `strongLandau`, `tsi`, `bti`, `cyclotron` (2D), `penning` (3D) |
| `ref` | reference method: `pic`, `pif` or `bh` (Barnes-Hut, read from `bhFile`) |
| `nParticle`, `NG`, `T`, `dt`, `kc`, `alpha`, `Qm` | particles, grid points/modes per dimension, end time, time step, wave number, perturbation, charge-to-mass ratio |
| `ml_time_int` | `explicit` (leapfrog/Boris) or `implicit` (implicit midpoint, Newton-Krylov; 1D) |
| `bhFile` | HDF5 output of the Barnes-Hut code PEPC (`pos_bh`, `vel_bh`, `Eout_bh`, particles in the same order at every step); required for `testCase: cyclotron` (initial particles) and for `ref: bh` |
| `saveEvery` | write x, v and E every `saveEvery` steps to `Inference_*.h5` / `Reference_pif_*.h5` (explicit integration; not in 3D) |
| `snapshots` | cyclotron: binned density and field snapshots |
| `evalDir` | output directory |

The random seed is fixed in `eval.py`, so runs are reproducible.

## Output (in `evalDir`)

- `diagnostics/pred_*.npz`, `diagnostics/ref_*.npz`: time series of the total, kinetic and
  potential energy, momentum, field energy per component and, for the cyclotron case, the
  canonical angular momentum, rms radius and field error against PIC;
- figures made from these by `plot_diagnostics.py` (they can be remade later without rerunning);
- `eval_run<runId>.md`: parameters and timings.

## Files

| file | content |
|---|---|
| `eval.py` | driver |
| `pic_plotter.py` | `PICVisualizer`: initial conditions, time loop for NEOPIC and the references, diagnostics |
| `plot_diagnostics.py` | plots of the saved diagnostics |
| `dynamics.py` | pushers (leapfrog, Boris), implicit Newton-Krylov and Picard integrators |
| `field.py`, `FourierKernels.py`, `interpolation.py` | PIC and PIF field solvers (periodic and free-space), shape functions, scatter/gather |
| `initial_conditions.py` | sampling of the initial distributions |
| `energy.py`, `landau_decay.py`, `utilities.py` | energies, decay rates, CuPy-Torch conversion |

## Training

Models are trained with `scripts/train.py --config <config>.yaml` (DDP with `torch.distributed.run`);
see `scripts/README.md`.
