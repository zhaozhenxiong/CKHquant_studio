# KHQuant 1.0.2

主站提供结构分析、全市场扫描、我的关注、预测与模型、数据与任务五个入口。结构与量价、Kronos、均线 MLP 及其同输入 L2 方法在分析与扫描间共享请求和保存身份；历史参考模型保留明确的对照入口。

“我的关注”分为关注股票和模拟账户，回测、保存续算、费用后对照集中到后者；预测/回测/学习参数独立。核验历史与原未来预测同图，改变显示范围不重算。来源 B/S、截止日方法 B/S、Kronos 采样摘要、评分及资格分别展示。均线模型 `ma_trend_v1` 有 20 项输入，与旧 `mixed_v1` 区分；配对 L2 与 MLP 共用输入、预处理和候选产物。预测采样频率不是校准概率，模型方向不等于目标仓位或成交。

扫描按冻结人口与预算分批，保留失败、不可用和延后项；续批核验数据、日历、模型和代码身份。独立模拟由 Broker/Ledger 产生真实的模拟成交、费用与库存变化。已保存账户保持原身份，训练或升级不会自动晋升模型、重启取消任务或绕过旧账户版本限制。

新股票入口排除北交所。板块当前计算使用沪深 A 股投影，历史混合快照与统计保持可查；当前范围尚未生成时显示待补齐。任务和结果采用分页；清理先核验引用与精确计划，保护原始行情、模型、账户和关注。

- [关注股票、模拟账户与历史预测图](docs/watchlist_accounts_and_prediction_chart.md)
- [组合预测、截止日方向与板块统计](docs/method_prediction_and_board_views.md)
- [v1.0.2 验证范围](docs/v1_0_2_acceptance.md)
- [组合方法与任务维护](docs/ma_methods_and_task_maintenance_20261015.md)
- [轻量研究流程](docs/agent_research.md)
- [北交所退出与板块范围](docs/beijing_retirement.md)
- [计算与来源优化](docs/compute_optimization_implementation_20261016.md)
- [缓存与留存](docs/research_storage_optimization_20261016.md)
- [参数](../PARAMETERS.md)、[安装](../../INSTALL_NATIVE.md)、[本版发行验证](../../RELEASE_VALIDATION.md)

公开包不含行情、模型权重、个人状态或私有验收。历史版本记录仅说明历史事实；最终安装平台和设备以本版实际回执为准。
