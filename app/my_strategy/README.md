# KHQuant 1.0.0

结构与量价、冻结 Kronos、本地 Agent 和轻量 PyTorch MLP 共同支持可复现的持仓研究。五个主页面为结构分析、全市场扫描、我的关注、预测与回测、数据与任务；历史参考模型保留为对照入口。

普通结构页只显示已确认结构和来源 B/S。模拟账户单独记录真实计划与成交数量、前后仓位、费用、净值、夏普与回撤。硬风险退出独立执行，缺少模型不制造零概率，也不阻挡风险卖出。Kronos 与 MLP 的预测表现需要独立检验，软件发行不授予自动增仓资格。

通过 `khquant light-research --help` 使用预测、训练、观察人口、回测和每日续算入口。小型 MLP 与同输入 L2 采用按时间隔离的训练、校准、政策及测试段；B 融合数据不足时保持待补齐。账户仅在明确启用后随每日数据更新续算，不自动训练或重启已取消的大任务。

结构补齐先预览差异，再按预算复用、追加或逐股重建。局部结构不会冒充全市场完整发布。原始行情、必要模型和保存账户受保护；预览和无引用衍生缓存采用有限留存。

公开包只含代码、原生运行库及脱敏验证材料。模型通过 `KHQUANT_KRONOS_ROOT` 指向独立本机目录；业务数据写入 `KHQUANT_HOME`，不写入 site-packages。运行 `khquant verify-workspace --target-date YYYY-MM-DD --strict --json` 进行只读健康核验。

见 [Agent 与持仓研究](docs/agent_research.md)、[本版验收](docs/v1_0_0_acceptance.md)、[参数说明](../PARAMETERS.md) 和 [安装说明](../../INSTALL_NATIVE.md)。
