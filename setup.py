from setuptools import setup, find_packages

setup(
    name             = "xuzu",
    version          = "1.0.0",
    author           = "Hamza A",
    author_email     = "hamza@terminalbio.io",
    description      = "XUZU: Multi-Modal Nucleotide Language Model for De Novo Aptamer Design",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://terminalbio.io/xuzu",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require = {
        "eval": ["RNA>=2.6.0"],
        "dock": ["meeko>=0.5.0"],
        "dev":  ["pytest", "black", "mypy"],
    },
    entry_points = {
        "console_scripts": [
            "xuzu-train=train:main",
            "xuzu-design=design:main",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
