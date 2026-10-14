# KHQuant 0.1.7 原生安装与验收

## 环境与安装

使用 Python **3.12**。平台路线为 Windows x64 CPU/CUDA、Linux x64 CPU/CUDA、Apple Silicon macOS ARM64 CPU/MPS；Intel Mac 不在本版矩阵。支持路线与实际验证范围分别记录，以随 Release 的 `release-validation.json` 为准。MPS 需要原生 ARM64 Python、可用 Torch 后端及真实算子检查。

下载 `KHQuant-0.1.7.zip` 与 `SHA256SUMS.txt`，核验后解压，在根目录执行：

```powershell
# Windows
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
```

```bash
# Apple Silicon Mac / Linux
python3.12 install.py --device auto
./khquant-native.sh dashboard
```

Linux 可明确使用 `--device cpu`。安装器创建 `.venv`，安装锁定依赖、平台 Torch、冻结 CZSC 和 KHQuant CLI。已有环境保留 `.env` 中的环境及状态路径；升级不会自动更新行情、训练模型或授予资格。

```bash
python3.12 install.py --dry-run --device auto
python3.12 install.py --device mps
python3.12 install.py --device cpu --data-dir /path/to/data --artifact-dir /path/to/artifacts
python3.12 install.py --upgrade --device auto
```

明确请求不可用设备会报错，不静默回退。新参考模型的训练、概率校准与批推理使用 Torch，auto 优先 CUDA，其次 MPS，最后 CPU；文件读取、精确价格、因果 CZSC 结构和状态协调保留 CPU。各阶段实际设备以报告及模型 manifest 为准，不能用设备可用性推断 GPU 利用率。新链路不执行 Broker 成交模拟。

## 原生依赖

CZSC 保持冻结 **1.0.1** 源码。安装器按 Python ABI、操作系统和架构选择 wheel，核验物理 SHA 和冻结源码身份；Release 的 `app/vendor/wheels/wheel-index.json` 记录构建来源。同名 PyPI 版本不保证该来源身份，不能任意替代。

```bash
python3.12 install.py --wheel-dir /path/to/downloaded-native-wheels --device auto
```

无匹配 wheel 时需要 Rust 和 C/C++ 工具链从冻结源码构建；Mac 还需 Xcode command-line tools。GitHub 自动 Source code 归档没有随 Release 的原生资产，首次安装推荐完整发行 ZIP。

维护者构建命令：

```bash
python app/build_czsc_runtime.py --out app/vendor/wheels --no-install
python app/build_release.py --output-dir dist/releases/0.1.7-rebuild --wheel-dir app/vendor/wheels
```

`build_release.py` 可重复传 `--wheel-dir` 合并平台资产，生成版本、来源、完整性与 SHA 清单。源码提交及工作区变化记入 `release.json`；最终发行的提交和物理资产身份由独立发行清单核验。

## 安装后检查

激活环境后使用 `khquant`，或运行 Windows 的 `.\khquant-native.bat`、Mac/Linux 的 `./khquant-native.sh`。

```bash
khquant doctor --check-write --strict --json
khquant dashboard
```

doctor 核验冻结来源、路径、数据库与真实 Torch 算子。空库可以安装，分析前需准备自己的行情和核验日历。六个页面为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务；资金账本、策略回测和手动持仓入口已经删除。

隔离安装冒烟使用明确标记的合成输入：

```bash
python app/verify_installation.py --device cpu --output installation-cpu.json
# Apple Silicon 真机实际执行
python app/verify_installation.py --device mps --output installation-mps.json
```

本版安装验收应包含实际 installed wheel 身份与 SHA、严格 doctor、新参考模型合成训练保存/重载/推理、分析与扫描、六页面、训练/参数 API、关注跨重启保存、退役接口拒绝，以及空模型的未评分和参考目标不可用。支持的设备阶段需实际执行。实际完成项目及结果以 release-validation.json 为准；工程验证不认证市场收益，历史版本 passed 不代替本版资产验证。

正式 wheel 状态默认位于 `~/.khquant`，可配置 `KHQUANT_HOME`、`KHQUANT_DATA_ROOT`、`KHQUANT_ARTIFACT_ROOT` 等路径。源码安装使用仓库 `.env`。网页默认绑定 `127.0.0.1:8124`，非本机绑定需要 `KHQUANT_API_TOKEN`。Tushare 行情和核验日历需要用户自己的凭据；未配置时仍可初始化免费备用来源。

## 状态导出与迁移

默认只导出代码，排除秘密、环境、业务数据和本地审计提示词。行情/模型属于私有状态，关注需要独立选择；公开 Release 不包含这些资产。

```bash
khquant package --target ../khquant-code
khquant package --target ../khquant-state --include-data --include-artifacts all
khquant package --target ../khquant-private-state --include-data --include-artifacts all --include-personal
khquant package --verify ../khquant-state
```

`--include-personal` 用于关注记录，手动持仓已退役。`--include-data` 使用一致 SQLite backup；模型、日历、manifest 和证据须保持完整来源、引用边界与 SHA。缺资产或不完整依赖会拒绝生成合格包。

wheel 的 site-packages 不是源码分发来源。先解压源码 Release，再明确指定：

```bash
khquant package --from-project /path/to/KHQuant --target /path/to/khquant-state --include-data --include-artifacts all
```

导出读取指定来源的 `.env`；迁移已有 wheel 状态时将该文件的业务路径指向实际状态位置。目标机先核验包再安装环境，模型导入不自动晋升。完整模型来源、人口、核验日历与路径依赖的跨机器迁移另行验收；旧模型档案不能恢复为当前合格选择。

## 模型与仓位参考

公开配置不绑定开发机任务或模型。先准备自己的完整 raw/native 来源与核验日历，在“模型训练”预检并提交新冻结计划。新入口复用通过哈希、人口、日历和生成代码核验的来源，不会将新行情自动补入旧快照；缺来源时明确阻断。默认训练参数为 device=auto、epochs=12、batch_size=8192、cpu_workers=4。参数只影响新计划，不能修改已提交任务。

来源 B/S 与参考目标变化分别展示。每股参考变化等于当日目标减上一参考目标，各股比例不合并为组合资金分配。政策基线未通过或评分缺失时不生成积极增仓目标；WAIT 保持参考值，可靠硬风险失效可清零。

本机基于截至 2026-10-12 的来源完成了 3,194 股、五个窗口的 MPS 训练及技术验证，但当前模型预测基线未通过。该结果不表示公开包附带可用模型，也不授予独立前瞻收益资格。见 [训练指南](app/my_strategy/docs/current_model_training.md)、[参数说明](app/PARAMETERS.md)、[参考架构](app/my_strategy/docs/reference_position.md) 和 [实际训练结果](app/my_strategy/docs/reference_position_validation_20261014.md)。
