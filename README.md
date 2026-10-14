# CZSC量价研究工作台

当前软件版：**0.1.6（2026-10-14）**。仓库为 [zhaozhenxiong/CKHquant_studio](https://github.com/zhaozhenxiong/CKHquant_studio)，发行入口为 [v0.1.6](https://github.com/zhaozhenxiong/CKHquant_studio/releases/tag/v0.1.6)。本版提供行情分析参考、完整来源 B/S、最终共识 B/S 与仓位参考，公开包不包含行情、个人记录、训练权重或研究账本。软件发行与模型资格分别核验；发布准备时新全量训练尚未完成，模型独立收益认证保持 **pending**。

六个网页入口为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务。策略回测、资金与成交账本、手动持仓入口及专用 API/CLI 已退役。训练内部保留真实 Broker 模拟，用于生成标签、费用后比较与资格审计。

来源 B/S 展示结构、量价和合格均线 ML 的独立已确认意见；最终 B/S 另按原共识条件判断。来源 S 未达到最终卖方门槛时仍可查看。缺合格趋势模型时明确“未评分”，不将缺失概率补零。风险仓位上限与积极配置目标分开：仅在独立参考质量、原质量及共识门槛、来源、时点、价格和风险均合格时生成正目标，其余保持 pending。详见 [B/S 与仓位参考](app/my_strategy/docs/bs_position_reference.md)。

本版修复三分类校准对类别排序的偏移，以及第二层退出读取错误人口的问题。校准只拟合正温度，退出和退出优势使用预登记 E1 实际持仓配对；旧模型直接退役，修复后重新完整训练。费用后至少 8% 且至少 2R、质量 10%、原买卖共识、风险、成熟与自动选模门槛保持不变。

## 安装和启动

使用原生 **Python 3.12**。下载 `KHQuant-0.1.6.zip` 和随附 `SHA256SUMS.txt`，核验后解压。首次安装创建 `.venv`，升级保留已有 `.env` 和本机路径。

Windows x64：

```powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
```

Apple Silicon Mac（原生 ARM64 Python）：

```bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
```

Linux x64：

```bash
python3.12 install.py --device cpu
./khquant-native.sh dashboard
```

浏览器打开 <http://127.0.0.1:8124>。安装器按本机 Torch 能力选择设备；训练冻结自己的明确设备参数。支持的线性/MLP 阶段可用 MPS/CUDA，CZSC、方向/树模型、真实 Broker 和样本人口准备在 CPU。明确请求不可用设备会报错，不静默回退。各平台实际验收以随 Release 的 `release-validation.json` 和资产 SHA 为准。

安装器仅接受经冻结 CZSC 1.0.1 源码、目标平台和物理 SHA 核验的原生 wheel。GitHub 自动生成的 Source code 归档不等同已验收发行包，可能缺少匹配平台的原生资产；缺匹配 wheel 时需要 Rust 和本机编译工具。完整说明见 [INSTALL_NATIVE.md](INSTALL_NATIVE.md)。

## 首次使用

先在“数据与任务”准备自己的行情和已核验交易日历，再在“模型训练”检查并提交完整训练计划。空模型安装可查看结构与量价事实；趋势概率和合格目标仓位明确不可用。默认分析只读取真实对比合格且已生效、在信号时点可用的模型选择。

```bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant analyze --symbol 600519.SH --usage-mode historical --device cpu --json
khquant scan --stocks 600519.SH,000001.SZ --usage-mode historical --device cpu --json
khquant train-trend --end YYYY-MM-DD --calendar-run-id YOUR_VERIFIED_CALENDAR_RUN --stock-batch-size 360 --cpu-workers 6 --device mps
khquant model-diagnostics --model-run-id YOUR_CURRENT_TRAINING_RUN --cpu-workers 2 --json
```

占位日期与 ID 替换为自己的核验输入；无 MPS 的机器使用 `--device cpu`。完整训练覆盖全部历史逐日合格股票并集，默认流动性为过去 20 个完整核验交易日成交额中位数至少 3000 万元。360 是股票准备分组上限，2048 是神经训练行批，均不截断人口。参数修改需要新预检，不改已提交任务。详见 [参数说明](app/PARAMETERS.md) 和 [训练指南](app/my_strategy/docs/current_model_training.md)。

## 分发与迁移

默认只导出代码。数据和训练状态属于自己的私有状态包，公开 Release 不包含它们；关注列表可通过独立的 `--include-personal` 选项导出。

```bash
python app/build_release.py --output-dir dist/releases/0.1.6-rebuild
khquant package --target ../khquant-code
khquant package --target ../khquant-state --include-data --include-artifacts all
khquant package --verify ../khquant-state
```

wheel 用户导出时用 `--from-project /path/to/KHQuant` 指定源码安装来源。模型、行情来源、核验日历和证据依赖必须完整核验；导入不会自动晋升模型。跨机器模型状态迁移与软件安装分别验收。

## 研究边界

训练内部使用下一核验开盘尝试、T+1、整手、涨跌停、流动性、费用、拒单和真实清仓现金流。高收益标签同时要求完整费用后至少 8% 且至少 2R；拒单、未清仓及未成熟分支不能当负样本。保留严格先验、原机会原子依赖清洗、最少样本与类别资格。

自动替换仅是研究模型选择，仍需完整冻结人口、同口径基线与现任比较、费用压力和实际资产审计。独立参考质量另有真实人口、头和费用资格，不能由风险上限或规则分数替代。正式独立前瞻认证继续单独核验；目录名 production、训练完成或软件发行均不授予该资格。未复权价格、历史 ST、行业与成员资料不足随结果披露。历史软件记录见 [RELEASE_NOTES.md](RELEASE_NOTES.md)，旧菜单和模型身份只供查证。
