"""optim 的 wheel 构建配置。

公共 facade、数据契约和数据适配层保留为带内联类型注解的 Python；只有 ``optim._core``
中的数值实现编译为扩展模块。构建过程不会生成或分发 core 的 ``.pyi``。
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.sdist import sdist as sdist_orig


ROOT = Path(__file__).parent.resolve()
CORE_SOURCES = (
    "optim/_core/canonical.py",
    "optim/_core/contracts.py",
    "optim/_core/engine.py",
    "optim/_core/factor_qcqp.py",
    "optim/_core/backends/base.py",
    "optim/_core/backends/clarabel.py",
    "optim/_core/backends/highs.py",
    "optim/_core/backends/mosek.py",
    "optim/_core/backends/piqp.py",
)


def _extension_name(source: str) -> str:
    """把仓库相对 Python 路径转换成扩展模块全名。"""

    return source.removesuffix(".py").replace("/", ".")


class PublicFacadeBuildPy(build_py_orig):
    """复制公共 Python 模块，但排除 core 实现源码。"""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "optim._core" or package.startswith("optim._core."):
            return [item for item in modules if item[1] == "__init__"]
        return modules


class BlockedSourceDistribution(sdist_orig):
    """阻止误生成会包含 core Python 源码的 sdist。"""

    def run(self):
        raise RuntimeError(
            "optim is distributed as a platform wheel only; source distributions "
            "are disabled because they would expose optim._core sources"
        )


extensions = [
    Extension(_extension_name(source), [source])
    for source in CORE_SOURCES
]

_CYTHON_BUILD_DIR = Path(tempfile.mkdtemp(prefix="optim-cython-"))
atexit.register(shutil.rmtree, _CYTHON_BUILD_DIR, ignore_errors=True)


setup(
    name="optim",
    version="3.0.0",
    packages=find_packages(include=("optim", "optim.*")),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "embedsignature": False,
            "emit_code_comments": False,
        },
        annotate=False,
        build_dir=str(_CYTHON_BUILD_DIR),
    ),
    package_data={
        "optim": [
            "py.typed",
            "LIBRARY.toml",
            "references/*.md",
        ]
    },
    include_package_data=False,
    cmdclass={
        "build_py": PublicFacadeBuildPy,
        "sdist": BlockedSourceDistribution,
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.2,<3",
        "scipy>=1.15,<2",
        "pandas>=2.2",
        "bottleneck>=1.4,<2",
        "tqdm>=4.66,<5",
        "highspy>=1.15,<2",
        "piqp>=0.6.4",
        "clarabel==0.11.1",
        "Mosek>=11.2,<12",
    ],
)
