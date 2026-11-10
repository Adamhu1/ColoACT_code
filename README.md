# ColoACT

Research code for **ColoACT: Multi-Cue Action Chunking for Smooth Autonomous
Colon Navigation on a Self-Propelled Endoscopic Robot**.


## Repository layout

```text
src/coloact/       Training, models, evaluation, and visualization
configs/           Example configuration files
scripts/           User-facing command-line helpers
ros2_ws/           ROS 2 deployment and robot-control package
tests/             Automated tests
docs/              Training and deployment notes
examples/          Minimal usage examples
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the ROS 2 dependencies separately before using `ros2_ws`.

## Training

The main model is in `src/coloact/models/act.py` and the training entry point
is `src/coloact/training/train_act.py`. Before running it, replace the example
dataset path in `configs/act_rgbde.yaml` and review the training configuration.

Datasets and model checkpoints are not included in this repository.

## Deployment

Build the ROS 2 package from `ros2_ws` after installing ROS 2 and the required
hardware drivers. Review the launch files and replace all device-specific
parameters before operating hardware.

## Reproducibility and safety

This repository contains research software. Validate all commands in simulation
or with the robot lifted from the workspace before enabling motor actuation.
Never commit patient data, animal data, credentials, or private infrastructure
paths.

## Citation

If you use this code or build on this work, please cite:

**Paper:** [ColoACT: Multi-Cue Action Chunking for Smooth Autonomous Colon Navigation on a Self-Propelled Endoscopic Robot](https://arxiv.org/abs/2610.01258)<br>
**Project page:** [adamhu1.github.io/ColoACT](https://adamhu1.github.io/ColoACT/)

```bibtex
@article{hu2026coloact,
	title         = {ColoACT: Multi-Cue Action Chunking for Smooth Autonomous Colon Navigation on a Self-Propelled Endoscopic Robot},
	author        = {Hu, Jian and He, Shujing and Chang, Leixin and Li, Zongze and Huang, Ding and Shi, Chaoyang and Hu, Chengzhi},
	journal       = {arXiv preprint arXiv:2610.01258},
	year          = {2026},
	doi           = {10.48550/arXiv.2610.01258}
}
```

