from setuptools import setup, find_packages

setup(
    name="beme",
    version="0.1.0",
    description="Bounty-driven Evolutionary Market Ensemble for multi-label classification",
    author="Mehmet Selman Yilmaz",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    python_requires=">=3.8",
)
