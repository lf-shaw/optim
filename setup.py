from setuptools import setup, find_packages

setup(
    name="optim",
    version="1.1.1",
    packages=find_packages(),
    python_requires=">=3.8",
    # TODO 增加版本要求
    install_requires=["pandas", "mosek", "numpy", "scipy"],
)
