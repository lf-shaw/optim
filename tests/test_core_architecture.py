"""锁定数值核心的单向依赖和轻量根入口。"""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


_PACKAGE_ROOT = Path(__file__).parents[1] / "optim"
_CORE_ROOT = _PACKAGE_ROOT / "_core"


def test_core_never_imports_parent_implementation() -> None:
    """core 可以依赖自身子模块，但不能逃逸到 ``optim`` 上层。"""

    violations: list[str] = []
    for path in sorted(_CORE_ROOT.rglob("*.py")):
        relative = path.relative_to(_CORE_ROOT)
        internal_depth = len(relative.parent.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level > internal_depth + 1:
                violations.append(
                    f"{relative}:{node.lineno} relative level {node.level} escapes core"
                )
            if (
                node.level == 0
                and node.module is not None
                and node.module.startswith("optim.")
                and not node.module.startswith("optim._core")
            ):
                violations.append(
                    f"{relative}:{node.lineno} imports upper module {node.module}"
                )
    assert not violations, violations


def test_upper_implementation_uses_only_core_root_exports() -> None:
    """上层实现不能绕过 ``optim._core`` 根入口依赖后端内部模块。"""

    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if _CORE_ROOT in path.parents:
            continue
        relative = path.relative_to(_PACKAGE_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith("_core.") or node.module.startswith(
                "optim._core."
            ):
                violations.append(
                    f"{relative}:{node.lineno} bypasses the core root: {node.module}"
                )
    assert not violations, violations


def test_every_core_implementation_module_is_an_explicit_extension() -> None:
    """新增 core 实现文件时必须同步进入选择性 Cython 构建清单。"""

    setup_tree = ast.parse(
        (_PACKAGE_ROOT.parent / "setup.py").read_text(encoding="utf-8")
    )
    declared: set[str] | None = None
    for node in setup_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "CORE_SOURCES"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        declared = set(value)
        break
    assert declared is not None, "setup.py must declare CORE_SOURCES as literals"

    expected = {
        str(path.relative_to(_PACKAGE_ROOT.parent))
        for path in _CORE_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert declared == expected
    assert (_PACKAGE_ROOT / "py.typed").is_file()


def test_distribution_version_is_generated_from_release_tags() -> None:
    """发行版本只能由 vX.Y.Z tag 生成，catalog 源文件保留动态占位符。"""

    with (_PACKAGE_ROOT.parent / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)
    with (_PACKAGE_ROOT / "LIBRARY.toml").open("rb") as stream:
        catalog_version = tomllib.load(stream)["meta"]["version"]

    assert configuration["project"]["dynamic"] == ["version"]
    scm = configuration["tool"]["setuptools_scm"]
    assert scm["version_file"] == "optim/_version.py"
    assert scm["tag_regex"] == r"^v(?P<version>\d+\.\d+\.\d+)$"
    assert scm["fallback_version"] == "3.0.0"
    assert catalog_version == "dynamic"
