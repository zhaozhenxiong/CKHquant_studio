# 当前目录

- `vendor/czsc`：冻结第三方 CZSC 1.0.1 源码与许可证。
- `vendor/wheels`：按平台、来源与 SHA 核验的原生资产。
- `my_strategy/adapters`：原始日线读取、日期、价格及单位边界。
- `my_strategy/services`：因果结构/量价、趋势模型、参考 B/S、风险与仓位、训练人口及费用资格。
- `my_strategy/execution`：仅用于内部研究标签和验收的 BrokerSimulator 与 Ledger。
- `my_strategy/storage`：原始行情、CZSC 结果索引、训练任务及持久关注。
- `my_strategy/web_dashboard`：分析、扫描、关注、训练、参数、数据与任务六页面及 API。
- `my_strategy/core`：配置、路径与运行上下文。
- `my_strategy/configs`：当前策略、公开空模型选择和行情配置。
- `my_strategy/scripts`：行情更新、安装验证、打包、诊断与审计。
- `my_strategy/docs`：当前公开操作与资格说明。
- `my_strategy/knowledge_base`、`knowledge_base_system`：本地追加审计，默认不进入源码发行包。

运行数据、个人状态、训练权重与秘密不纳入公开包。业务路径来自 `.env` 或正式 wheel 的状态目录配置；面向用户的回测、账本和手动持仓已退役。
