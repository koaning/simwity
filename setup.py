import pathlib
from setuptools import setup, find_packages


base_packages = ["srsly>=2.4.6", "hnswlib>=0.7.0", "tqdm>=4.64.1"]

test_packages = [
    "interrogate>=1.5.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
]

all_packages = base_packages
dev_packages = all_packages + test_packages


setup(
    name="simsity",
    version="0.6.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Super Simple Similarity Service",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/koaning/simsity/",
    project_urls={
        "Documentation": "https://github.com/koaning/simsity/",
        "Source Code": "https://github.com/koaning/simsity/",
        "Issue Tracker": "https://github.com/koaning/simsity/issues",
    },
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
