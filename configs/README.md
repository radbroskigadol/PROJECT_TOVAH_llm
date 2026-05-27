# Config references

These YAML files are reference launch plans for buyers. The current CLI accepts
flags directly rather than loading YAML. The files exist so a buyer can map the
recommended ladder into their own launcher, Hydra setup, Slurm scripts, or
DeepSpeed/Accelerate configuration.

Use `python -m tovah_v14.training.scale_ladder` for machine-readable memory
planning without allocating a model.
