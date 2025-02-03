"""Adaptive Kernel setup utility."""

from setuptools import find_packages, setup

setup(
    name =              "adaptive-kernel",
    version =           "1.0.0",
    author =            (
                            "Gabriel C. Trahan, "
                            "Mithun Ranjan Kar, "
                            "Sahan Ahmad"
                        ),
    author_email =      (
                            "gabriel.trahan1@louisiana.edu, "
                            "mithun-ranjan.kar1@louisiana.edu, "
                            "sahan.ahmad1@louisiana.edu"
                        ),
    description =       (
                            "Research and experimentation focused on exploring the intrinsic "
                            "properties and effects of convolution neural network kernels on image "
                            "processing tasks."
                        ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/Adaptive-Kernels",
    packages =          find_packages(),
    python_requires =   ">=3.10",
    install_requires =  [
                            "matplotlib",
                            "numpy",
                            "pandas",
                            "scikit-learn",
                            "termcolor",
                            "torch",
                            "torchvision",
                            "tqdm"
                        ]
)