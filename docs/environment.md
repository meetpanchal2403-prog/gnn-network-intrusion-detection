# Environment and dependency guidance

## What is known

The project uses Python and the packages listed in the repository's
`requirements.txt`. The notebook contains a recorded environment check that
reported:

- Python 3.12 (visible in the saved package-installation transcript before
  notebook cleanup)
- PyTorch 2.14.0+cu130
- PyTorch Geometric 2.8.0.post1

These are observations from one development environment, not a verified
compatibility matrix or a tested installation recipe for other machines.
The notebook's package-installation transcripts were removed because they
contained machine-specific paths and platform-specific installation logs.

## Installation

1. Create and activate a virtual environment using a Python version compatible
   with the selected PyTorch release.
2. Install the general dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Install PyTorch for your operating system and hardware using the official
   PyTorch installation selector:
   https://pytorch.org/get-started/locally/
4. Install PyTorch Geometric using its official installation instructions:
   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
5. Verify the installation:

   ```bash
   python -c "import torch, torch_geometric; print(torch.__version__); print(torch_geometric.__version__)"
   ```

## Reproducibility status

The dependency list is not version-pinned. A complete lockfile has not been
created, and a clean-clone execution of the full research pipeline has not
been verified. The versions above should therefore be treated as environment
observations, not as a guarantee that the project runs reproducibly with
those versions.

PyTorch and PyTorch Geometric installation commands depend on the operating
system, Python version, and CPU/CUDA build. Follow the official installation
instructions rather than assuming that one command works on every platform.

Before claiming full reproducibility, verify the complete workflow in a clean
environment with the required data, preprocessing artifacts, and model files.
