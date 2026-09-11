#!/usr/bin/env python3
"""校验 optim wheel 的资源、二进制核心和源码暴露边界。"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


REQUIRED_RESOURCES = {
    "optim/LIBRARY.toml",
    "optim/references/api_overview.md",
    "optim/references/gotchas.md",
    "optim/references/recipes.md",
    "optim/py.typed",
}
ALLOWED_IMPLEMENTATION_PYTHON = {
    "optim/_core/__init__.py",
    "optim/_core/backends/__init__.py",
    "optim/_impl/__init__.py",
}
REQUIRED_IMPL_MODULES = {
    "compiler",
    "solver_adapter",
    "diagnostic_engine",
    "asset_bounds",
    "solution",
}
REQUIRED_CORE_MODULES = {
    "canonical",
    "contracts",
    "dual_bounds",
    "engine",
    "factor_conic",
    "thread_control",
    "backends/base",
    "backends/clarabel",
    "backends/highs",
    "backends/mosek",
    "backends/piqp",
}


def validate(path: Path) -> None:
    """检查一个 wheel；发现缺失资源或核心源码泄漏时终止构建。"""

    if not path.is_file() or path.suffix != ".whl":
        raise SystemExit(f"wheel 不存在或扩展名无效: {path}")
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        obsolete = sorted(
            names & {"optim/opt.py", "optim/linopt.py", "optim/solver.py"}
        )
        obsolete += sorted(
            names
            & {
                "optim/model/compiler.py",
                "optim/_solver_adapter.py",
                "optim/_diagnostic_engine.py",
                "optim/asset_bounds.py",
                "optim/solution.py",
            }
        )
        if obsolete:
            raise SystemExit(f"wheel 包含已删除的旧模块，请清理构建目录: {obsolete}")
        missing = sorted(REQUIRED_RESOURCES - names)
        if missing:
            raise SystemExit(f"wheel 缺少必要资源: {missing}")
        library = tomllib.loads(archive.read("optim/LIBRARY.toml").decode("utf-8"))
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")

    wheel_version_match = re.search(r"^Version: (.+)$", metadata, re.MULTILINE)
    wheel_version = (
        None if wheel_version_match is None else wheel_version_match.group(1)
    )
    catalog_version = library.get("meta", {}).get("version")
    if wheel_version != catalog_version:
        raise SystemExit(
            "wheel 元数据与 LIBRARY.toml 版本不一致: "
            f"wheel={wheel_version!r}, catalog={catalog_version!r}"
        )

    core_python = {
        name
        for name in names
        if name.startswith(("optim/_core/", "optim/_impl/")) and name.endswith(".py")
    }
    unexpected_python = sorted(core_python - ALLOWED_IMPLEMENTATION_PYTHON)
    if unexpected_python:
        raise SystemExit(f"wheel 暴露了 core/impl Python 实现: {unexpected_python}")

    forbidden = sorted(
        name
        for name in names
        if name.startswith(("optim/_core/", "optim/_impl/"))
        and PurePosixPath(name).suffix in {".pyi", ".c", ".cpp", ".html"}
    )
    if forbidden:
        raise SystemExit(f"wheel 包含禁止的 core/impl 中间文件: {forbidden}")

    compiled = {
        name
        for name in names
        if name.startswith("optim/_core/") and name.endswith((".so", ".pyd"))
    }
    missing_modules = sorted(
        module
        for module in REQUIRED_CORE_MODULES
        if not any(name.startswith(f"optim/_core/{module}.") for name in compiled)
    )
    if missing_modules:
        raise SystemExit(f"wheel 缺少编译后的 core 模块: {missing_modules}")

    impl_compiled = {
        name
        for name in names
        if name.startswith("optim/_impl/") and name.endswith((".so", ".pyd"))
    }
    missing_impl = sorted(
        module
        for module in REQUIRED_IMPL_MODULES
        if not any(name.startswith(f"optim/_impl/{module}.") for name in impl_compiled)
    )
    if missing_impl:
        raise SystemExit(f"wheel 缺少编译后的 impl 模块: {missing_impl}")

    print(
        f"✅ wheel 校验通过：{len(compiled)} 个 core 扩展，{len(impl_compiled)} 个 impl 扩展，"
        f"{len(REQUIRED_RESOURCES)} 个公共资源"
    )


def main() -> None:
    """命令行入口。"""

    if len(sys.argv) != 2:
        raise SystemExit("用法: validate_wheel.py /path/to/optim-*.whl")
    validate(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
