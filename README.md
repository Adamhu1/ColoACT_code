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

See the project paper for the citation information.

