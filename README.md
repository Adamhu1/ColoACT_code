# ColoACT

<p align="center">
	<strong>Multi-Cue Action Chunking for Smooth Autonomous Colon Navigation on a Self-Propelled Endoscopic Robot</strong>
</p>

<p align="center">
	<a href="https://adamhu1.github.io/ColoACT/">Project page</a> |
	<a href="https://arxiv.org/abs/2610.01258">Paper</a> |
	<a href="https://github.com/Adamhu1/ColoACT_code">Code</a>
</p>

<p align="center">
	<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%3E%3D3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9 or newer"></a>
	<a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
	<a href="https://arxiv.org/abs/2610.01258"><img src="https://img.shields.io/badge/arXiv-2610.01258-b31b1b.svg" alt="arXiv 2610.01258"></a>
</p>

Research code for autonomous colon navigation with a self-propelled endoscopic
robot. ColoACT combines RGB-D-E perception with an Action Chunking Transformer
(ACT) policy and temporal ensembling for smooth continuous control.

## Overview

ColoACT provides the core training, evaluation, visualization, and ROS 2
deployment components used in the project:

- RGB-D-E observations using RGB, relative depth, and a pseudo-elevation map.
- Overlapping action chunks fused with temporal ensembling.
- ACT, behavior-cloning, chunking-BC, and diffusion-policy implementations.
- ROS 2 nodes for inference, data collection, joystick control, and deployment.

## Installation

Create an environment and install the Python package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Install ROS 2 and the hardware-driver dependencies separately before using
`ros2_ws`.

## Training

The main ACT implementation is in
[`src/coloact/models/act.py`](src/coloact/models/act.py), with its training entry
point at [`src/coloact/training/train_act.py`](src/coloact/training/train_act.py).
Training notes and dataset requirements are available in
[`docs/training.md`](docs/training.md).

Before training, update the example dataset path and machine-specific settings
in [`configs/act_rgbde.yaml`](configs/act_rgbde.yaml). Datasets and model
checkpoints are not included in this repository.

## Evaluation

Evaluation utilities are provided under
[`src/coloact/evaluation`](src/coloact/evaluation), including open-loop,
ADE/FDE, jitter, and temporal-ensembling analysis. See
[`docs/evaluation.md`](docs/evaluation.md) for the evaluation workflow.

## Deployment

The ROS 2 package is under
[`ros2_ws/src/coloact_robot`](ros2_ws/src/coloact_robot). It contains ACT
inference, perception, data-collection, joystick, and motor-control nodes.
Review the deployment notes in [`docs/deployment.md`](docs/deployment.md)
before building the package or connecting hardware.

## Code Structure

```text
src/coloact/       Training, models, evaluation, and visualization
configs/           Example configuration files
scripts/           User-facing command-line helpers
ros2_ws/           ROS 2 deployment and robot-control package
tests/             Automated tests
docs/              Training, evaluation, and deployment notes
examples/          Minimal usage examples
```

## Reproducibility and Safety

This repository contains research software. Validate commands in simulation or
with the robot lifted from the workspace before enabling motor actuation.
Review camera topics, model paths, serial-port settings, and emergency-stop
behavior locally. Never commit patient data, animal data, credentials, or
private infrastructure paths.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{hu2026coloact,
	title         = {ColoACT: Multi-Cue Action Chunking for Smooth Autonomous Colon Navigation on a Self-Propelled Endoscopic Robot},
	author        = {Hu, Jian and He, Shujing and Chang, Leixin and Li, Zongze and Huang, Ding and Shi, Chaoyang and Hu, Chengzhi},
	journal       = {arXiv preprint arXiv:2610.01258},
	year          = {2026},
	doi           = {10.48550/arXiv.2610.01258}
}
```

## License

This project is released under the [MIT License](LICENSE).

