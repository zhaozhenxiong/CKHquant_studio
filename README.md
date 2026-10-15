# KHQuant · 结构、预测与持仓研究

**v1.0.0** 提供结构与量价分析、本地 Kronos 预测、轻量 PyTorch MLP 融合、独立持仓模拟及按缺口补齐结构。软件源码和安装包在 [CKHquant_studio](https://github.com/zhaozhenxiong/CKHquant_studio)，本版资产在 [v1.0.0 Release](https://github.com/zhaozhenxiong/CKHquant_studio/releases/tag/v1.0.0)。公开包不含行情、模型权重、个人账户或私有审计。

结构页保留已确认的来源 B/S、确认时间和量价证据。预测与回测页用 Broker/Ledger 记录实际模拟买卖：收盘决策在下一核验开盘执行，整手、T+1、费用、拒单及待退出均可追溯。部分减仓同时显示负股数、剩余股数、计划仓位与实际成交仓位。

本地 Agent 分别整理结构、量价、Kronos 和风险证据；MLP 作为融合层。历史回测与每日续算共用逐日观察日程，持仓风险检查独立运行。模型缺失、过期或无效有明确原因，不以缺失冒充零分。历史候选研究、影子研究续算与已获资格的纸上账户分别标识，训练完成不会自动授予增仓资格。

## 安装和启动

使用原生 Python 3.12。下载 `KHQuant-1.0.0.zip`，按随附 `SHA256SUMS.txt` 核验后解压。升级保留自己的 `.env` 和业务数据。

Windows x64：

```powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
```

Apple Silicon Mac / Linux x64：

```bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
```

浏览器打开 <http://127.0.0.1:8124>。安装器核验冻结 CZSC 原生 wheel 的平台和 SHA。公开空模型安装可以查看结构功能；Kronos 需要自己的本地模型、配置和 tokenizer，状态页会报告缺失项，程序不暗中下载权重。配置与迁移参见 [轻量工作流](app/my_strategy/docs/agent_research.md) 及 [安装说明](INSTALL_NATIVE.md)。

## 日常使用

1. 更新自己的原始行情与核验交易日历。
2. 在结构页选股与日期，查看来源证据；来源点不等于已成交。
3. 在预测与回测页训练有界小池 MLP 候选，和结构量价、Kronos、同输入 L2 账户比较。
4. 板块图点击“预览结构补齐”，检查直接复用、追加、逐股重建、缺行情及延后数量，再执行本批。
5. 已明确启用的账户及持仓/关注小池随每日更新续算；日更不训练，不自动恢复取消任务。

```bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant light-research status
khquant light-research score --symbols 002436.SZ,600519.SH --end 2025-12-31 --model-type mlp --variant A --device cpu
khquant light-research simulate --symbols 002436.SZ,600519.SH --start 2026-01-01 --end 2026-10-14 --save
```

实际 CLI 子命令以 `khquant --help` 为准。MLP A 使用结构量价和缺失标记，B 另需冻结人口的历史 Kronos 摘要；覆盖不足保持待补齐。默认小网络、有限候选、独立校准及同输入 L2 对照，不全量微调 Kronos。首次结构准备可能逐股重建；只有恢复等价核验通过才使用原生状态续接，量价滚动指标仍保留逐股向量计算。

## 结果与验证边界

查看 [v1.0.0 功能验收](app/my_strategy/docs/v1_0_0_acceptance.md)、[发行验证](RELEASE_VALIDATION.md) 和随资产的 `release-validation.json`。安装平台、设备、测试和 SHA 以本版实际报告为准；上一版本证据不代替本版。

历史窗口、未复权价格、公司行为及历史交易规则的完整性会影响回测。Kronos 预训练范围未知；历史研究不能当独立前瞻认证。收益、夏普、回撤、换手、实际成交和完整持仓周期一起报告，未通过的资格保持待定。

默认预览为7天/8 GiB可清理缓存；已保存账户、当前/回滚模型与必要引用受保护。原始行情不按缓存期限删除。迁移使用代码包和明确选择的私有状态包，详见 [RUNBOOK.md](RUNBOOK.md)。
