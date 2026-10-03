from setuptools import find_packages, setup

setup(
    name="emerald",
    version="2.2.0",
    package_dir={"": "."},  # Explicit mapping
    packages=find_packages(where="."),
    description="A library for modeling 1D atoms",
    author="Gabriel A. Amici",
    install_requires=[
        "numpy",
        "scipy"
    ],
)
