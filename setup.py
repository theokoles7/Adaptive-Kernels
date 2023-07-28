import os
from setuptools import find_packages, setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name =              "hi-lo",
    version =           "1.0.0",
    author =            "Gabriel C. Trahan, Sahan Ahmad",
    author_email =      "gabriel.trahan1@louisiana.edu, sahan.ahmad1@louisiana.edu",
    description =       (
        "Code package used to determine the efficacy of probability distributions on convolution of high and low resolution images."
        "The code for thiss project is adapted from KernelSet (Sahan Ahmad), adapted from CBS (Samarth Sihnha)."
    ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/HiLo-Resolution",
    long_description =  read('README.md'),
    packages =          find_packages()
)