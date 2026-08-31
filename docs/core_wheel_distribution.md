# `_core` wheel 构建与分发约束

更新日期：2026-08-31。

本项目采用公共 Python facade 加选择性 Cython 数值核心。公共模块保留内联类型注解和
`optim/py.typed`；`optim._core` 的数值实现编译为平台扩展模块，不生成或分发 `.pyi`。

## 依赖方向

依赖必须保持单向：

```text
PortfolioProblem / SolverPolicy
             │
             ▼
       上层编译与适配
             │
             ▼
    CanonicalModel / CoreSolverOptions
             │
             ▼
          _core.solve
             │
             ▼
       CoreSolveResult
             │
             ▼
      OptimizationResult
```

`_core` 不得导入 `portfolio_types`、`api`、`validation`、`diagnostics` 或其他上层实现。上层
实现也只能从 `optim._core` 根入口获取轻量契约，不得直接依赖具体 backend 子模块。对应规则
由 `tests/test_core_architecture.py` 固定。

固定 `SolverPolicy` 在 `PortfolioOptimizer` 初始化时转换一次。多期求解每天只跨 core 边界
一次，并仅传当日 handle、`theta_seed` 和已换算为原始 alpha 单位的目标容差；theta 搜索、
PIQP workspace 更新和 fallback 不跨层往返。大规模 NumPy/CSC payload 按引用传递。

## 构建 wheel

先安装 `requirements.txt`，然后使用与 tuda2 一致的仓库级构建入口：

```bash
./build.sh
```

脚本默认保留 `dist/` 中既有产物，清理临时 `build/`、`optim.egg-info/` 和源码树中的编译残留，
随后调用 `python -m build --wheel --no-isolation`。开发环境尚未安装 `build` 时，会临时回退到：

```bash
python setup.py bdist_wheel
```

发布到 `~/.pypirc` 的内部仓库：

```bash
./build.sh --push -r local
```

`setup.py` 只显式 Cythonize `CORE_SOURCES`，Cython 的 `.c` 中间文件写入独立临时目录并在
构建进程退出时清理。wheel 中允许存在 `_core/__init__.py` 和
`_core/backends/__init__.py` 两个包入口，其余 core 实现必须是 `.so`/`.pyd`。

禁止构建和发布 sdist。源码分发无法在保持可重建性的同时隐藏 `_core/*.py`，因此构建命令
会明确拒绝 `sdist`；内部平台只能接收目标 Python/操作系统/架构对应的 wheel。

## 发布前检查

发布前至少检查：

```text
1. wheel 中有 optim/py.typed；
2. wheel 中没有 optim/_core/**/*.pyi；
3. 除两个 __init__.py 外，没有 optim/_core/**/*.py；
4. wheel 中没有 .c、.cpp 或 Cython 注解 HTML；
5. 从空目录安装 wheel 后，LP、QP、factor-QCQP smoke 均可求解；
6. optim._core.engine.__file__ 指向平台扩展模块，而不是 Python 源码。
7. `LIBRARY.toml`、`references/api_overview.md`、`recipes.md` 和 `gotchas.md` 已进入 wheel；
8. wheel 元数据、`optim.__version__` 和 `LIBRARY.toml` 版本一致，首个新版 tag 为 `v3.0.0`。
```

`build.sh` 会自动执行静态 wheel 内容检查。脱离源码环境的运行时验收可使用
`scripts/smoke_installed_wheel.py`。

## Git tag 版本规则

发行版本由 `setuptools-scm` 从 Git tag 自动生成，源码不手工维护 `_version.py`：

```bash
git tag v3.0.0
./build.sh
./build.sh --push -r local
```

tag 必须严格使用 `vX.Y.Z`。tag 之间的本地构建会自动生成带提交距离和 commit id 的开发
版本；`--push` 只允许干净工作树且 `HEAD` 正好具有发布 tag，避免上传开发版本。构建阶段还会
把实际 SCM 版本写入 wheel 内的 `LIBRARY.toml`，源码 catalog 保留 `dynamic` 占位符。

当前 core ABI 由 `CORE_ABI_VERSION` 标识。prepared handle 只允许同一进程、同一 wheel 版本
短期使用，不可持久化或跨进程传输。
