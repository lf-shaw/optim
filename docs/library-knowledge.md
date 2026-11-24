# Optim AI 知识维护

文档优先 library knowledge v2。作者维护简介、公开模块范围、源码 docstring 和 Markdown 参考。
api.modules 包含 optim、optim.data 和 optim.integrations.tuda2；不扫描私有 core/impl，不维护手写签名、章节或约束映射。
__all__、公开 facade 与 dataclass 文档提供本库事实；SCM/METADATA 是唯一版本源，构建不再改写 catalog 版本。
类型字段、嵌套目标/约束、状态和权限/数据前提应就近说明，不要求每函数有独立 recipe。

`optim/references/task_entries.toml` 提供常见任务的入口选择。它采用独立的 schema v1，
按领域、任务和路由组织，只引用有文档的公开符号。参数、返回值和异常的说明仍以对应 docstring 为准。

## 验证与发布

保留已有业务/文档/字段测试；tests/test_library_knowledge.py 执行 R0 合成 LP/QP/Factor-QCQP、
验证输入错误、失效结果访问、示例语法和任务路由。平台另行验证检索和回答质量。
发布时对 Python 3.11 和 3.12 wheel 分别执行 `scripts/validate_wheel.py`、
`scripts/smoke_installed_wheel.py` 和平台的 `--verify-api` 检查。确认任务目录和 Markdown
已打包、路由能解析到公开 API、私有实现以扩展模块分发后，向指定仓库上传并比对回下载的 SHA-256。
