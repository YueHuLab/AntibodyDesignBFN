from setuptools import setup, find_packages

setup(
    name="antibody_bfn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "tqdm",
        "pyyaml",
        "biopython",
        "easydict",
        "joblib",
        "lmdb",
        "tensorboard",
    ],
)
