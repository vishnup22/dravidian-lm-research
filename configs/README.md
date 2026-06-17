# Config Layout

This directory holds experiment manifests and reusable configuration templates.

Current status:

- the codebase has been restructured to support config-driven experiments
- the Python entrypoints are still argument-driven
- these YAML files define the intended experiment shape for future runs

Recommended convention:

- `configs/data/`: corpus and split configuration
- `configs/models/`: model architecture settings
- `configs/experiments/`: concrete experiment manifests
