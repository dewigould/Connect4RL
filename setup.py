from __future__ import absolute_import
import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    name="Connect4_ReinforcementLearning",
    version="1.0.0",
    description="A3C reinforcement learning implemented to play Connect 4",
    long_description = long_description,
    long_description_content_type="text/markdown",
    author="Dewi Gould",
    author_email="dewi.gould@maths.ox.ac.uk",
    url = "",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "gym",
        "chainer",
        "chainerrl",
        "os",
        "argparse",
        "logging",
        "multiprocessing"
        ],
)
