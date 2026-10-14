# CZSC量价研究工作台

当前软件版：**0.1.7（2026-10-15）**。仓库为 [zhaozhenxiong/CKHquant_studio](https://github.com/zhaozhenxiong/CKHquant_studio)，发行入口为 [v0.1.7](https://github.com/zhaozhenxiong/CKHquant_studio/releases/tag/v0.1.7)。本版默认使用 `reference_position_v1`，输出参考买卖点、每股参考目标仓位及相对上一参考目标的增减幅度，不执行模拟成交或组合资金分配。公开包不包含行情、个人记录、训练权重或研究账本。

六个网页入口为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务。来源结构 B/S 独立展示；最终参考 B/S 仅在目标仓位实际变化时产生，支持部分减仓。WAIT 保持参考目标，评分缺失时保留上次参考并标记 stale；可靠价格触发既有硬风险边界时可清零。手工仓位仅用于独立比较，不改变参考轨迹。详见 [新架构](app/my_strategy/docs/reference_position.md)。

本机完整训练已覆盖 **3,194 股、四个滚动留出窗口和一个当前参考模型**，使用 MPS 完成拟合、校准及数值评价，来源截止 **2026-10-12**。技术完整性和真实分析/扫描验证通过；**当前模型未通过预测基线，不启用积极增仓目标**。这些本地研究资产不随软件发布。完整指标见 [训练验证报告](app/my_strategy/docs/reference_position_validation_20261014.md)。

## 安装和启动

使用原生 **Python 3.12**。下载 `KHQuant-0.1.7.zip` 和随附 `SHA256SUMS.txt`，核验后解压。首次安装创建 `.venv`，升级保留已有 `.env` 和本机路径。

Windows x64：

```powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
```

Apple Silicon Mac（原生 ARM64 Python）及 Linux x64：

```bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
```

浏览器打开 <http://127.0.0.1:8124>。新数值模型自动选择 **CUDA → MPS → CPU**；训练、概率校准和批推理使用 Torch，文件读取、精确价格、因果结构与状态协调保留 CPU。明确请求不可用设备会报错。支持路线与本版实际验收范围分别记录，以随 Release 的 `release-validation.json` 和资产 SHA 为准。

安装器只接受经冻结 CZSC 1.0.1 源码、目标平台和物理 SHA 核验的原生 wheel。GitHub 自动 Source code 归档不等同已验收发行包；缺匹配 wheel 时需要 Rust 和本机编译工具。见 [INSTALL_NATIVE.md](INSTALL_NATIVE.md)。

## 首次使用

先准备自己的行情和已核验日历。新训练还要求完整、无失败、来源与生成代码可核验的 native 结构快照；`train-reference` 复用这些输入并重建新标签和权重。没有完整来源时预检会阻断，更新原始行情不会自动扩展已有快照。公开空模型安装可查看结构事实，无法提供合格模型概率和积极增仓目标。

```bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant analyze --symbol 600519.SH --usage-mode historical --device auto --json
khquant scan --stocks 600519.SH,000001.SZ --usage-mode historical --device auto --json
khquant train-reference --source-native-run-id YOUR_VALID_NATIVE_RUN --device auto --preflight-only --json
khquant train-reference --source-native-run-id YOUR_VALID_NATIVE_RUN --device auto --json
```

将来源 ID 替换为自己的完整核验来源。也可在“模型训练”预检并提交计划；参数改变需重新预检。旧 `train-trend` 已退役。参数默认训练 12 轮、每批 8192 行、数据准备线程 4 个，处理批次不截断历史合格人口。见 [参数说明](app/PARAMETERS.md) 和 [训练指南](app/my_strategy/docs/current_model_training.md)。

## 分发与迁移

默认只导出代码。行情、来源、模型与训练状态属于私有状态包，关注列表通过 `--include-personal` 独立选择。

```bash
python app/build_release.py --output-dir dist/releases/0.1.7-rebuild
khquant package --target ../khquant-code
khquant package --target ../khquant-state --include-data --include-artifacts all
khquant package --verify ../khquant-state
```

wheel 用户导出时用 `--from-project /path/to/KHQuant` 指定源码安装来源。模型、来源、核验日历及证据依赖必须完整，导入不会自动授予资格。跨机器模型状态迁移与软件安装分别验收。

## 验证边界

新目标为下一核验日开盘至未来第 10 或 20 个核验交易日收盘的纯价格方向，不把参考价当成交价。预处理仅训练段拟合，选型仅用训练内部时间划分，校准、政策基线检查与留出评价相互隔离。政策段 LogLoss 与 Brier 均不得劣于冻结训练先验基线；测试段不参与选型。新架构不使用旧 Broker 标签、费用后 8%/2R 质量头或旧权重。

训练完成、技术验证、预测门槛与实际冻结后的独立前瞻资格分别判定。历史研究重建不能冒充当时发布的模型，方向指标不能冒充交易收益。历史记录见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。
