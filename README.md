# KHQuant · 结构、预测与持仓研究

**v1.0.1** 提供结构与量价分析、本地 Kronos 预测、均线 MLP 与同输入 L2 对照、有界全市场扫描和独立持仓模拟。源码和发行资产见 [CKHquant_studio](https://github.com/zhaozhenxiong/CKHquant_studio)。公开包不含行情、模型权重、个人账户或私有审计。

结构分析与全市场扫描共用五个选项：结构与量价、结构与量价 + Kronos、结构与量价 + Kronos + 均线模型 MLP、结构与量价 + Kronos + 同输入 L2，以及明确标识的历史参考模型。来源 B/S、预测摘要与评分证据分别展示；Kronos 上涨采样比例不是校准概率。扫描保留失败、不可用和预算延后的股票，显式续批只处理未完成项。

预测与回测页用 Broker/Ledger 记录模拟成交：收盘决策在下一核验开盘执行，保留整手、T+1、费用、拒单和待退出。来源点和模型方向都不是已成交或目标仓位。历史候选研究、影子研究续算与已获资格的纸上账户分别标识，训练完成或软件升级不授予独立前瞻资格。

## 安装和启动

使用原生 Python 3.12。取得 `KHQuant-1.0.1.zip` 后，按随附 `SHA256SUMS.txt` 核验并解压。升级保留自己的 `.env` 和业务数据。本版最终安装验证的状态见 [发行验证](RELEASE_VALIDATION.md)。

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

浏览器打开 <http://127.0.0.1:8124>。安装器核验冻结 CZSC 原生 wheel 的平台和 SHA。Kronos 需要自己的本地模型、配置和 tokenizer；缺失项在状态页明确报告，不自动下载权重。详见 [安装说明](INSTALL_NATIVE.md)。

## 日常使用

1. 更新自己的沪深行情与核验交易日历。新更新、搜索、分析和扫描排除北交所。
2. 在结构页选股、日期与方法；分析跳转到模拟研究时携带相同模型和预测参数。
3. 在预测与回测页训练有界均线 MLP 候选，查看同一产物的同输入 L2 对照。旧混合模型保留原输入身份。
4. 全市场扫描按预算执行，检查完整人口及缺失原因后显式续批。板块使用独立沪深 A 股计算范围；没有对应结果时先补齐，历史版本仍可查看。
5. 在数据与任务页筛选、折叠或删除终态记录；回收产物先预览精确清单。已经启用的账户随每日更新核验续算，日更不训练、不自动恢复取消任务。

```bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant light-research status
khquant light-research score --symbols 002436.SZ,600519.SH --end 2025-12-31 --model-type mlp --variant A --feature-profile ma_trend_v1 --device cpu
khquant scan --stocks 002436.SZ,600519.SH --analysis-method structure_kronos --end 2026-01-30 --max-predictions 2 --budget-seconds 300 --device cpu
```

示例股票和日期仅说明参数格式。实际子命令以 `khquant --help` 为准；CLI 新请求的历史兼容默认值不等于网页选择，组合方法应显式传入 `--analysis-method`。均线版本 `ma_trend_v1` 有 20 项特征，配对 L2 使用相同预处理；`mixed_v1` 保留 A/B 旧输入。详见 [方法与任务维护](app/my_strategy/docs/ma_methods_and_task_maintenance_20261015.md) 和 [参数口径](app/PARAMETERS.md)。

## 结果、性能与留存

本版减少缓存全目录巡视、写锁、重复编码和任务列表解码，采用有界 CPU 来源准备、单实例预测及服务端分页。来源复用和增量追加仍核验完整身份；历史修订、代码或数据变化会失效。预算停止新增计算，必要的退出、核验与保存可能超出时间上限。小样本性能结果不能作为全市场完成时间或整体磁盘节省承诺。

历史数据完整性、未复权价格及未知的 Kronos 预训练范围限制研究解释；独立前瞻资格不足保持待定。旧账户的历史结果可以查看，继续计算仍核验其冻结版本，不自动改写旧身份。

默认预览留存 7 天，可清理缓存配额 8 GiB；保存账户、必要模型与引用受保护。原始行情不按缓存期限删除，北交所历史数据退出使用单独离线维护流程。迁移见 [RUNBOOK.md](RUNBOOK.md)，变更见 [发行说明](RELEASE_NOTES.md)。
