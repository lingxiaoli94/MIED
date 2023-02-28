# Sampling with Mollified Interaction Energy Descent
This repository contains the source code for the ICLR 2023 paper ([link](http://arxiv.org/abs/2210.13400)) "Sampling with Mollified Interaction Energy Descent" by Lingxiao Li, Qiang Liu, Anna Korba, Mikhail Yurochkin, Justin Solomon.

## Dependencies
The only core dependency is Pytorch. See `requirements.txt`.

## Code organization
There are two folders under the project directory: `mied` and `tests`.

`mied` folder is a python package that needs to be installed in order to run the code. Within this folder,
* `problems` folder contains classes inherited from `ProblemBase` defined in `problem_base.py`. Each constrained sampling problem is determined by the log probability of the target distribution and the constraints, described either using reparametrization or a set of inequalities.
* `solvers` folder contains implemented sampling algorithms and tools for handling inequalities constraints. We implemented a few popular particle-based algorithms such as SVGD (`svgd.py`), KSDD (`ksdd.py`), Mirror Langevin Monte Carlo (`lmc.py`), and a baseline independent particle descent (`ipd.y`, not strictly a sampling algorithm). Our proposed algoritm is MIED (`mied.py`).
* `utils`: utility folder.
* `validators`: tools for validation such as plotting densities and computing metrics.

`tests` folder contains working directories of the experiments with entry-point python scripts named `run.py`.

## How to run
First, you will need to install the package `mied` in order to run it. This can be done by either `pip install -e .` (using pip) or `conda develop .` (using conda).
Either way, you will be able to modify the source code and the changes will be reflected immediately the next time you use the package `mied`.

There is `script.sh` in each experiment folder in `tests` that provides example commands to run the code. This script must be called from the corresponding experiment folder.
Within each experiment folder, the checkpoints, results, and the config YAML file of a particular run will be stored under `exps` folder.
We use wandb to log intermediate results.
Evaluation is baked in each `run.py`.

The list of command line arguments for `run.py` is defined in the same script and in `utils/ec.py` --- check out `utils/ec.py` for details on argument parsing. For example, `--method` determines which sampling algorithm to run (it can be `MIED`, `SVGD`, `KSDD`, etc.), `--projector` determines how to handle the constraints (`--projector=DB` uses dynamic barrier which is no-op if there is no constraint), and `--reparam` determines what reparameterization map to use.
