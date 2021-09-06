from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="avia-missing-airplanes",
    packages=find_packages(),
    version="0.1.0",
    description="Image classification",
    author="Roman Zaev",
    license="MIT",
)