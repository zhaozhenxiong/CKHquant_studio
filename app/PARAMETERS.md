# KHQuant 0.1.7 参考模型参数

当前公开模型族为 `current`，内部策略为 `reference_position_v1`。网页及 `train-reference --config` 使用下列同一扁平参数契约，定义来自 `my_strategy/services/reference_training.py`。旧成交研究的手续费、8%/2R、质量头和 40/20/30/10 权重参数不适用于新训练，传入未知字段会被拒绝。

## 参数和默认值

| 字段 | 默认值 | 含义与约束 |
| --- | ---: | --- |
| `device` | `auto` | 自动优先 CUDA、其次 MPS、最后 CPU；可显式选择 `cuda`、`mps` 或 `cpu`，不可用时拒绝 |
| `feature_start` | `2023-01-01` | ISO 日期，必须与冻结 native 来源的人口起始日一致 |
| `end` | `null` | 省略时使用所选完整来源截止日；显式日期须与来源一致，不自动补入新行情 |
| `source_native_run_id` | `null` | 完整有效 native 来源 ID；省略时查找完成来源，仍须通过全部核验 |
| `horizon_sessions` | `10` | 方向目标周期，只支持 10 或 20 个核验交易日 |
| `epochs` | `12` | 训练轮数，整数 1–200 |
| `batch_size` | `8192` | 数值训练行批，整数 64–65536；不是股票池数量上限 |
| `cpu_workers` | `4` | 数据准备线程数，整数 1–16；不表示 GPU 数量 |
| `seed` | `42` | 随机种子，整数 0–2147483647 |
| `max_stock_weight` | `0.20` | 每股参考目标上限，比例值，范围 (0,1] |
| `risk_budget` | `0.01` | 每股参考风险预算，比例值，范围 (0,1] |
| `position_step` | `0.025` | 参考仓位变化档位，即 2.5 个百分点；范围 (0,1]，不得大于每股上限 |
| `hysteresis` | `0.01` | 参考目标切换滞回，即 1 个百分点；须为正数且小于半个仓位档位 |

比例字段使用 0 至 1 的数值，例如 20% 写作 0.20。数值必须有限。表中是**新训练计划默认值**，选定模型在运行时使用其冻结参数；不得用运行时后备参数覆盖模型报告。

完整训练覆盖来源中全部历史逐日流动性合格股票并集。来源的人口、逐日资格和流动性设置被冻结，当前表单不另提供股票数量截断、准备分组或流动性门槛覆盖。更换源数据日期、人口、日历或因果生成码，需要新的完整来源；仅修改训练轮数、设备或行批不会让旧来源自动加入新行情。

## 预检与提交

1. 准备完整、无失败的 raw/native 来源和已核验日历。在参数页保存个人默认会更新修订号；并发修订冲突时先刷新。
2. 在训练页选择本次参数和来源，再预检。预检冻结参数、来源、人口、日历、时间窗口及代码身份。来源的资产 SHA、生成代码和来源链必须有效；缺完整来源会阻断。
3. 预检通过且参数未改变后提交冻结计划。阶段进度只表示实际完成数，不推算 GPU 利用率或总体剩余时间。修改参数需要重新预检，不能改运行中的冻结任务。
4. 训练完成后分别查看技术完整性、政策预测门槛和留出指标。重新核验检查已有资产，不修改权重或把未通过的门槛改为通过。

失败的新参考训练可以通过任务 API 的 `resume_reference_run_id` 在原 run 接续。恢复严格核验原数值代码、来源、完整数据集和完成权重，恢复适配器单独冻结；旧成交模型不能通过该入口恢复为参考模型。

## CLI 示例

```sh
khquant train-reference --source-native-run-id YOUR_VALID_NATIVE_RUN --device auto --preflight-only --json

khquant train-reference --source-native-run-id YOUR_VALID_NATIVE_RUN --device auto --config /path/to/reference-parameters.json --json

khquant analyze --symbol 600519.SH --usage-mode historical --device auto --json
khquant scan --stocks 600519.SH,000001.SZ --usage-mode historical --device auto --json
```

配置 JSON 可以只填写需要覆盖的字段，例如 `{"epochs": 12, "batch_size": 8192, "cpu_workers": 4}`。CLI 的 `--device` 默认 `auto`，会覆盖配置 JSON 中的设备；显式来源参数优先于配置中的来源 ID。旧 `train-trend` 入口被拒绝；旧 `model-diagnostics` 为历史成交诊断，不用于新参考验证。新结果主要报告为 `reports/reference_training.json`，网页可提交独立完整性复核。

## 仓位、模型与验证边界

每股参考变化 = 当日参考目标 − 上次参考目标。风险上限为单股上限与风险预算/止损距离的较小值，并向下对齐仓位档位；方向优势结合档位和滞回生成参考目标。各股比例不汇总为资金组合，手工仓位只作额外比较。来源 B/S 和目标变化 B/S 分开；没有实际参考变化时不重复生成最终点。

WAIT 保持目标；评分缺失或政策基线未通过时保留最后参考并标记 stale，不把缺失变为卖出；可靠价格触发已冻结硬风险边界时可减至零。来源 B/S、风险上限或训练完成都不能单独取得积极增仓资格。

58 项因果特征输入 Torch 线性或小 MLP。纯价格三分类标签使用下一核验日开盘至周期末收盘，未成熟或缺行情保持不可用。预处理只拟合训练段，选型只用训练内部划分，独立校准仅拟合正温度。政策段 LogLoss 和 Brier 均不得劣于冻结训练类别先验；留出评价不反选模型。新训练前冻结仓位政策，本版不搜索仓位映射参数。

本机来源截止 2026-10-12 的 3,194 股、五个窗口 MPS 训练和技术验证已完成，**当前预测门槛未通过，不启用积极增仓目标**。软件发行不附带这些研究权重，不授予独立前瞻资格。架构与实际指标见 [参考架构](my_strategy/docs/reference_position.md) 和 [训练验证报告](my_strategy/docs/reference_position_validation_20261014.md)。

历史 0.1.4 参数位于 [归档](release_history/0.1.4/PARAMETERS.md)；旧发行记录只供查证，不是当前入口或参数契约。
