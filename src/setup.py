from setuptools import setup, find_packages

setup(
    name="emerald",
    version="1.0.0",
    package_dir={"": "."},  # Explicit mapping
    packages=find_packages(where="."),
    description="A library for Morse-soft-Coulomb potential analysis",
    author="Gabriel A. Amici",
    install_requires=[
        "numpy",
        "numba",
        "scipy"
    ],
)
