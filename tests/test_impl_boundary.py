"""业务实现层集中发布，同时保持数学核心不反向依赖上层。"""

import ast
from importlib import import_module
from pathlib import Path


def test_core_has_no_upper_layer_imports():
    """所有 core 相对导入必须落在 core 内，禁止绝对上层导入。"""
    root = Path(__file__).parents[1] / "optim" / "_core"
    for path in root.rglob("*.py"):
        depth = len(path.relative_to(root).parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= depth, (path, node.lineno)
                if not node.level and node.module:
                    assert not node.module.startswith("optim"), (path, node.lineno)
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("optim") for alias in node.names)


def test_impl_modules_and_model_convenience_exports():
    """五个实现模块集中在同一目录，原 model 便捷入口指向同一个实现。"""
    for name in (
        "compiler",
        "solver_adapter",
        "diagnostic_engine",
        "asset_bounds",
        "solution",
    ):
        module = import_module(f"optim._impl.{name}")
        assert Path(module.__file__).parent.name == "_impl"
    from optim.model import compile_problem
    from optim._impl.compiler import compile_problem as implementation

    assert compile_problem is implementation
