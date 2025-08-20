import os
import glob
import shutil
from pathlib import Path
from setuptools import setup, Command
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.build_ext import build_ext as build_ext_orig
from Cython.Build import cythonize


# ----------------------------
# 自定义命令：生成包含 docstring 的存根文件
# ----------------------------
class GenerateStubsCommand(Command):
    """通过 mypy stubgen 生成含 docstring 的存根文件"""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 生成 stubs
        os.system("stubgen --no-import --include-docstrings ./optim -o ./stubs")

        # 复制存根文件到源码目录
        stubs_dir = Path("./stubs/optim")
        target_dir = Path("./optim")

        for pyi_file in stubs_dir.rglob("*.pyi"):
            target_path = target_dir / pyi_file.relative_to(stubs_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pyi_file, target_path)

        shutil.rmtree("./stubs", ignore_errors=True)


# ----------------------------
# 自定义 build_py：集成存根生成
# ----------------------------
class CustomBuildPy(build_py_orig):
    """在 build_py 阶段生成存根文件"""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (
                pkg,
                mod,
                file,
            )
            for (
                pkg,
                mod,
                file,
            ) in modules
            if mod == "__init__"
        ]

    def run(self):
        self.run_command("generate_stubs")
        super().run()


# ----------------------------
# 自定义 build_ext：编译后清理中间文件
# ----------------------------
class CustomBuildExt(build_ext_orig):
    """在 build_ext 阶段完成后清理中间文件"""

    def run(self):
        super().run()
        # 清理中间文件（.c 和临时 .pyi）
        for pattern in ["optim/*.c", "optim/*.pyi"]:
            for f in glob.glob(pattern):
                os.remove(f)


# ----------------------------
# 主配置
# ----------------------------
setup(
    name="optim",
    version="2.0.2",
    ext_modules=cythonize(
        "optim/*.py",
        exclude=["optim/__init__.py"],
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,  # 必须保留签名
        },
    ),
    packages=["optim"],
    package_data={
        "optim": ["*.pyi", "*.so", "*.pyd"],  # 最终分发包包含 .pyi
    },
    exclude_package_data={
        "optim": ["*.py", "*.c"],  # 排除源码和中间文件
    },
    include_package_data=True,
    cmdclass={
        "generate_stubs": GenerateStubsCommand,
        "build_py": CustomBuildPy,  # 生成存根 + 常规构建
        "build_ext": CustomBuildExt,  # 覆盖原始 build_ext
    },
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "mosek",
        "bottleneck",
        "tqdm",
    ],
)
