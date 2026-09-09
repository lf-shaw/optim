"""防止统一优化架构的公共文档与字段说明发生回退。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_PACKAGE_ROOT = Path(__file__).parents[1] / "optim"
_CHINESE = re.compile(r"[\u4e00-\u9fff]")


def _module_trees() -> tuple[tuple[Path, ast.Module], ...]:
    """读取当前架构的全部模块，不再豁免旧求解器。"""

    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
    )


@pytest.mark.parametrize(("path", "tree"), _module_trees())
def test_public_objects_have_chinese_docstrings(path: Path, tree: ast.Module) -> None:
    """公共类、函数和方法必须提供包含中文说明的 docstring。"""

    missing: list[str] = []
    non_chinese: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            missing.append(f"{node.name}:{node.lineno}")
        elif not _CHINESE.search(docstring):
            non_chinese.append(f"{node.name}:{node.lineno}")
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method.name.startswith("_"):
                continue
            method_docstring = ast.get_docstring(method, clean=False)
            label = f"{node.name}.{method.name}:{method.lineno}"
            if not method_docstring:
                missing.append(label)
            elif not _CHINESE.search(method_docstring):
                non_chinese.append(label)

    assert not missing, f"{path} 缺少公共 docstring：{missing}"
    assert not non_chinese, f"{path} 公共 docstring 缺少中文说明：{non_chinese}"


@pytest.mark.parametrize(("path", "tree"), _module_trees())
def test_dataclass_docstrings_name_every_field(path: Path, tree: ast.Module) -> None:
    """每个 dataclass 的类文档必须逐项说明全部声明字段。"""

    missing: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            )
            for decorator in node.decorator_list
        )
        if not is_dataclass:
            continue
        fields = [
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]
        docstring = ast.get_docstring(node, clean=False) or ""
        undocumented = [field for field in fields if field not in docstring]
        if undocumented:
            missing[node.name] = undocumented

    assert not missing, f"{path} 的 dataclass 字段缺少说明：{missing}"
