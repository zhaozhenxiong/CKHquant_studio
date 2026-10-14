# CZSC 量价研究工作台 0.1.7

当前公开模型族为 `current`，内部策略为 `reference_position_v1`。核验行情、日历和因果 native 特征形成 58 项输入，经纯价格方向模型、独立温度校准和政策预测基线检查，输出每股参考 B/S、目标仓位与相对上一参考目标的变化。来源结构 B/S 独立展示，不执行模拟订单、成交或组合资金分配。

六页面为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务。关注与扫描星标统一持久保存。行情、结果、任务与关注分别落在配置的业务路径中，不写 site-packages。

新训练字段和默认值由 `services/reference_training.py` 提供；不读取旧 `czsc_trend_strategy.json` 的成交研究配置。`khquant train-reference` 及训练中心要求完整有效 native 来源，经哈希、人口、日历和生成代码核验后复用；新标签和权重独立生成。预处理、内部选型、校准、政策与留出评价严格按时间分离。默认 CUDA → MPS → CPU，明确不可用设备报错；精确价格、CZSC 结构和状态协调保留 CPU。

每股目标受风险上限、方向优势、仓位档位和滞回约束。有效 WAIT 保留参考；评分缺失或基线未通过时保留上次值并标记 stale，不因缺失制造卖点。可靠硬风险失效可清零。手工仓位仅用于额外比较，不改变参考历史。

本机已完成截至 2026-10-12 来源的 3,194 股、五个窗口 MPS 训练和技术验证；当前模型未通过预测基线，不启用积极增仓目标。公开包不带行情、权重或个人库。训练完成与软件发行均不授予独立前瞻资格。

运行 `python -m my_strategy.cli --help` 或 `khquant --help` 查看入口；旧 `train-trend` 被拒绝。路径由 `core.paths` 和环境配置解析。见 [训练指南](docs/current_model_training.md)、[参考架构](docs/reference_position.md)、[训练验证](docs/reference_position_validation_20261014.md)、[参数说明](../PARAMETERS.md) 和 [安装说明](../../INSTALL_NATIVE.md)。历史成交研究文档只供查证。
