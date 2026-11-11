# 实验性源码候选验证范围

本目录是本地实验性源码候选，不是公开发行版，也尚未通过本轮发布验收。包版本均为 0.1.0。

此前独立验证覆盖离线 Core/Fake/Scripted 闭环、本地 Chromium Web 参考场景和离线 LLM/retrieval 接口；这些历史结果不能替代本次从公开目录重新运行。当前提交仍有安装测试可移植性、scratch 环境约定和公开说明缺口，不能使用“本候选已验证”概括全部能力。

真实模型 API 为 `not_run`；provider cleanup 的取消/挂起情形仅有源码审查，运行行为仍为 `unknown`。不承诺生产可靠性、任意网站、模型收益、跨平台兼容或 OS 级隔离。后续 replay、复合验证和长期语义记忆不属于本候选交付范围。

公开回归区分离线测试、本地 Web 测试与五包安装测试。`tests/integration/test_llm_retrieval_evidence.py::test_the_demo_never_touches_the_governance_event_log` 依赖未公开的内部历史，须显式 deselect，不能把该节点计为通过。内部导出器和导出器测试不属于公开目录；公开安装测试不应依赖它们。

完整本轮原始证据保留在维护者内部审计记录，未包含在公开目录。该说明不替代可运行的安装和测试指引。
