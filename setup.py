from setuptools import setup, find_packages

setup(
    name="dankpy",
    version="0.0.1",
    author="Daniel Kadyrov",
    author_email="dkadyrov@stevens.edu",
    packages=find_packages(),
    license="LICENSE.txt",
    description="DANKPY is a set of functions and utilities built and used by Daniel Kadyrov",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "glob2",
        "librosa",
        "vincenty",
        "matplotlib",
        "plotly",
        "soundfile",
        "scipy",
        "simplekml",
        "planar",
        "lxml",
    ],
)
