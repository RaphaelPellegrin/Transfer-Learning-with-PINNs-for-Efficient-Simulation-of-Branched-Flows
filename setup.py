"""Setup for the PINNs project."""


from setuptools import find_packages, setup

setup(
    name="pinns",
    version="1.0.0",
    python_requires=">3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==2.0.0",
        "click",
        "tqdm",
        "numpy>=1.24.3,<2.0.0",
        "pandas",
        "matplotlib",
        "scipy",
        "torch-scatter",
        "torch-sparse",
        "path",
        "fastjsonschema",
        "path",
        "imageio",
        "imageio-ffmpeg",
    ],
    extras_require={
        "dev": [
            "mypy",
            "pandas-stubs",
            "type-python-dateutil",
            "pydocstyle",
            "black",
            "isort",
            "docformatter",
            "ruff",
            "pylint",
        ],
        "test": [
            "pytest",
            "coverage",
            "pytest-cov",
            "pytest-random-order",
            "torch",
            "torch-scatter",
            "torch-sparse",
            "numpy",
        ],
    },
)
