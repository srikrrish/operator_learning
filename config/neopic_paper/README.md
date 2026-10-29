# Configurations of the NEOPIC paper

Training and evaluation configs for the experiments of the NEOPIC paper
(S. Muralikrishnan et al., "NEOPIC: A Neural Operator Framework for Particle-based Kinetic
Plasma Simulations"). The folder structure is that of the original run directories.

## Paths

All file paths (`picFile`, `dataFile`, `checkpoint`, `bhFile`, `trainDir`, `evalDir`) are
relative to one root folder, from which the scripts are started:

```
<root>/
  Datasets_2D_electrostatic_plasma/       training data of the 2D-2V weak Landau models
  3D_training_4_mini_apps_BH_PIF/         training data of the 3D-3V 4-mini-app model
  particle_weakLandau/                    copies of the folders below (models, PEPC files, outputs)
  2D_mini_apps_different_methods_test/
  3D_mini_apps_test/
```

```bash
cd <root>
python <repo>/scripts/pic_eval/eval.py --config <repo>/config/neopic_paper/<experiment>/config.yaml
python <repo>/scripts/train.py --config <repo>/config/neopic_paper/<experiment>/<training config>.yaml
```

## Experiments

| folder | experiment |
|---|---|
| `particle_weakLandau/<N>k/` | 2D-2V weak Landau damping, models trained with N = 50k ... 800k particles: `electrostatic_plasma2d_pif_train_weakLandau.yaml` (training, 250 epochs) |
| `particle_weakLandau/<N>k/velocity/eval_<M>k/` | in-loop inference of each model with M particles (`predOnly`); the high-fidelity PIF reference (`pifShapeOrder: 4`), which does not depend on the model, is computed only in the `50k` runs |
| `particle_weakLandau/800k/cic_shape/eval_<M>k/` | the same for the 800k model against the CIC PIF reference (`pifShapeOrder: 2`), i.e. the shape function of the training data; the reference is computed in these runs |
| `particle_weakLandau/800k/eval_k<k>_1600k/` | 800k model at different wave numbers k |
| `particle_weakLandau/800k/{weakLandau,strongLandau,tsi,bti}/cic_shape/` | 800k model applied to the other mini-apps |
| `2D_mini_apps_different_methods_test/training_data/<case>/` | generation of the 2D-2V multi-app training data |
| `2D_mini_apps_different_methods_test/training/` | training of the 2D-2V multi-app model |
| `2D_mini_apps_different_methods_test/model_evaluations/<case>/` | 2D-2V multi-app model on weak/strong Landau damping, two-stream (tsi), bump-on-tail (bti) and the magnetized cyclotron case |
| `2D_mini_apps_different_methods_test/model_evaluations/cyclotron/` | cyclotron case against the PEPC Barnes-Hut reference (`v0AtMinusHalf: true`, initial velocity at t = -dt/2 as in PEPC) |
| `2D_mini_apps_different_methods_test/model_evaluations/cyclotron/dt_convergence/` | time-step convergence (second-order Boris start, `v0AtMinusHalf` false) |
| `3D_mini_apps_test/training/` | training of the 3D-3V multi-app models (5 and 4 mini-apps) |
| `3D_mini_apps_test/model_evaluations/<case>/` | 3D-3V multi-app models on weak/strong Landau damping (Barnes-Hut reference), tsi, bti and the Penning trap |
| `*/computational_complexity/` | cost measurements; the paper's timings were made with the `nufft-fixes` branch |

The eval options are described in `scripts/pic_eval/README.md`. Runs that compare with the
Barnes-Hut code PEPC read its output from `bhFile`; the cyclotron case also takes its initial
particles from there.
