# Training

The training package supports the main ACT policy, behavior-cloning baselines,
and the exploratory diffusion-policy implementation.

Use a private H5 dataset containing the keys expected by
`coloact.data.dataset.ColonDatasetDiffusion`: `rgb`, `depth`, `elevation`, and
`action`. Do not add the dataset to Git.

The example configuration is in `configs/act_rgbde.yaml`. The current research
training scripts still expose their configuration as Python constants; update
those values for the local machine before running them.

