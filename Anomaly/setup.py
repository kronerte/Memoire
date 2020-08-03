import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmd", # Replace with your own username
    packages=find_packages(),
    version="0.1.4",
    author="Etienne Kronert",
    author_email="etienne.kronert@inria.fr",
    description="Anomaly Detection with MMD",
    long_description= long_description,
    url = "https://github.com/kronerte/Memoire",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)