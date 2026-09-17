# Optim AI 知识维护

文档优先 library knowledge v2：作者只维护简介/公开模块范围、源码 docstring 和普通 Markdown。
api.modules 包含 optim、optim.data 和 optim.integrations.tuda2；不扫描私有 core/impl，不维护手写签名、章节或约束映射。
__all__、公开 facade 与 dataclass 文档提供本库事实；SCM/METADATA 是唯一版本源，构建不再改写 catalog 版本。
类型字段、嵌套目标/约束、状态和权限/数据前提应就近说明，不要求每函数有独立 recipe。

## 验证与发布

保留已有业务/文档/字段测试；tests/test_library_knowledge.py 执行 R0 合成 LP/QP/Factor-QCQP、
验证输入错误和失效结果访问、其他示例语法。普通数据夹具允许，AI 问题映射文件不要求。
平台维护真正的检索/回答质量评测；源码测试及 docstring 非空不能代替生产消费验收。
Python 3.11 新 wheel 经 scripts/validate_wheel.py、scripts/smoke_installed_wheel.py 和平台 --verify-api --report 核验；
源码/制品 Markdown 一致、私有实现仍为二进制且不暴露内部算法 docstring，才发布本地私有 PyPI 并下载比对 SHA-256。
不上传 MyQNAP 库包，不在资料验收前构建 CPU kernel，不把作者合同更新等同于 Gray 检索接线完成。
