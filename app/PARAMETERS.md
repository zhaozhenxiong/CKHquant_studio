# KHQuant 0.1.2 参数使用说明

当前实现已作为正式软件版封存。后续修改自己的参数副本即可，原始发布包、默认协议和训练运行可用于对照。

默认协议：`my_strategy/configs/czsc_high_return_research.json`（相对本文件所在 app 目录）。独立可编辑副本：`parameter_profiles/high-return-custom.json`。两者在发布时内容完全一致；原始副本另存于 `parameter_profiles/high-return-0.1.2.json`。

| 参数位置 | 当前值 | 用途 |
|---|---:|---|
| `signal_points.ma_entry_threshold` | 0.55 | 均线 ML 入场阈值 |
| `signal_points.ma_exit_threshold` | 0.55 | 均线 ML 持仓退出阈值 |
| `signal_points.risk_fraction` | 0.01 | 单次初始风险预算 |
| `signal_points.atr_stop_multiple` | 2.0 | ATR 止损距离 |
| `signal_points.ma_cooldown_bars` | 5 | 均线入场冷却 |
| `high_return.min_net_return` | 0.08 | 完整扣费回合的高收益标签 |
| `high_return.min_r_multiple` | 2.0 | 高收益标签的初始风险倍数 |
| `high_return.max_holding_sessions` | 60 | 核验持仓交易日的退出意图上限 |
| `high_return.exit_advantage_margin` | 0.002 | 退出相对续持优势门槛 |
| `entry_threshold_grid` | 0.05 / 0.1 / 0.2 / 0.35 / 0.5 | L2 入场阈值搜索 |
| `exit_threshold_grid` | 0.55 / 0.65 | L2 退出阈值搜索 |
| `epochs` / `batch_size` | 20 / 2048 | 训练轮数 / 批次 |

在项目根目录执行；已安装环境也可用 `khquant` 代替 `python -m my_strategy.cli`。示例沿用当前本机特征源，需先激活已有 Python 3.12 环境：

```bash
python -m my_strategy.cli research-high-return-train \
  --source-training-run-id 20261009-190909-196001-20261008-czsc-wyckoff-research-train-NO_GIT \
  --end 2026-10-08 --device auto --cpu-workers 2 \
  --config app/parameter_profiles/high-return-custom.json --json
```

训练完成会返回新的 `run_id`；将下方 `NEW_TRAINING_RUN_ID` 替换为该值，回测自动读取这个新模型保存的冻结参数：

```bash
python -m my_strategy.cli research-high-return-backtest \
  --model-run-id NEW_TRAINING_RUN_ID --start 2023-01-01 --end 2026-10-08 \
  --device auto --cpu-workers 2 --json
```

信号、止损、标签、持仓退出和模型输入协议会进入模型契约。改这些参数时按上述入口重新训练，再用新运行回测；原始 79 个头及其历史结果保留。训练轮数和搜索参数的修改也需要新训练。已经完成的模型使用训练时冻结的参数；修改副本不会自动重算旧概率或旧账本，也不会自动改变普通网页评分。回测改搜索网格不会重选旧 bundle 的已保存阈值。

本版架构比较固定为 `architectures: ["linear", "tree", "mlp"]`。`publication_allowed` 和 `observed_windows_diagnostic_only` 由实现固定为 `false` / `true`；在 JSON 中修改它们不会授予模型收益认证或改变应用模式。

本版完整回测及持仓契约固定每股账户 `initial_cash: 100000`，保留该值。学习率、MLP 隐藏层和树深度等为源码中的固定设置，并非当前 JSON 的有效旋钮。若覆盖 `strategy`，需放入完整 `czsc_strategy` 对象；局部 `strategy.execution` 不会自动补齐默认项。

`--config` 是高收益训练/回测支持的独立覆盖入口。保留包内默认配置文件原字节，可以继续核验旧运行的 29 项代码/配置冻结身份。直接改包内默认协议会使旧运行的严格冻结检查失败；用独立副本调参更方便保留原始对照。原始版本号及 `ma_entry_alignment` 建议随本版保持，特征源和日期必须与实际数据匹配；后续新数据应先生成新的真实特征源。

本机特征源和模型未附带在公开可分发的程序包中，新机器需自行准备匹配的来源和核验日历。软件正式发布时，新组合扣费收益仍为 −0.4142365082%，独立未来验证待完成；这些状态随基线记录保留。

仅安装 wheel 时，本说明与两个参数 JSON 作为同目录附件提供，未内置于 wheel。将 `--config` 换为附件的绝对路径即可；本机特征源运行仍需可用。普通 CLI 查看高收益评分时显式增加 `--model-family high_return`，网页已默认显示高收益研究；修改普通评分模式与应用研究模型改变订单是各自的入口。
