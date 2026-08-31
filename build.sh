#!/bin/bash
# optim wheel 构建脚本
#
# 用法：
#   ./build.sh                     # 清理并构建 wheel，保留 dist/ 中既有产物
#   ./build.sh --push              # 构建、校验后上传到 ~/.pypirc 的 [local]
#   ./build.sh --push -r prod      # 上传到指定仓库
#
# 只发布平台 wheel，不生成 sdist。构建使用当前 Python 环境中已经安装的依赖，避免在内部
# 生产环境重复创建隔离环境和下载构建依赖。
set -eo pipefail
cd "$(dirname "$0")"

PUSH=0
REPOSITORY="local"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH=1
            shift
            ;;
        -r|--repository)
            [[ $# -ge 2 ]] || { echo "缺少仓库名: $1 <repository>" >&2; exit 2; }
            REPOSITORY="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            exit 2
            ;;
    esac
done

echo "=== [1/4] 清理上次构建的临时产物 ==="
rm -rf build/ optim.egg-info/
find optim/_core -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.so' -o -name '*.pyd' \) -delete
find optim -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "=== [2/4] 编译平台 wheel ==="
if PYTHONNOUSERSITE=1 python -c "import build" 2>/dev/null; then
    PYTHONNOUSERSITE=1 python -m build --wheel --no-isolation
else
    echo "未安装 build，使用 setuptools bdist_wheel 兼容入口"
    PYTHONNOUSERSITE=1 python setup.py bdist_wheel
fi

WHL=$(ls -t dist/optim-*.whl | head -1)
echo "=== [3/4] 校验产物: $WHL ==="
PYTHONNOUSERSITE=1 python scripts/validate_wheel.py "$WHL"

if [[ "$PUSH" -eq 1 ]]; then
    echo "=== [4/4] 上传到 [$REPOSITORY] 仓库 ==="
    command -v twine >/dev/null || { echo "❌ 未找到 twine" >&2; exit 1; }
    twine upload -r "$REPOSITORY" --skip-existing "$WHL"
    echo "✅ 已上传到 [$REPOSITORY]: $(basename "$WHL")"
else
    echo "=== [4/4] 跳过上传（如需发布，加 --push）==="
fi
