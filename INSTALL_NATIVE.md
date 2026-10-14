# KHQuant 0.1.6 原生安装与验收

## 环境与安装

使用 Python **3.12**。平台路线为 Windows x64 CPU/CUDA、Linux x64 CPU/CUDA、Apple Silicon macOS ARM64 CPU/MPS；Intel Mac 不在本版矩阵。支持路线与实际验证范围分别记录，以随 Release 的 `release-validation.json` 为准。MPS 需要原生 ARM64 Python、可用 Torch 后端及真实算子检查。

下载 `KHQuant-0.1.6.zip` 与 `SHA256SUMS.txt`，核验后解压，在根目录执行：

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

明确请求不可用设备会报错，不静默回退。MPS/CUDA 用于支持的线性/MLP 训练与神经批推理；CZSC、字典、三分类方向、树模型、真实 Broker 和人口准备在 CPU。各阶段实际设备以任务及头 manifest 为准，不能用设备可用性推断 GPU 利用率。

## 原生依赖

CZSC 保持冻结 **1.0.1** 源码。安装器按 Python ABI、操作系统和架构选择 wheel，核验物理 SHA 和冻结源码身份；Release 的 `app/vendor/wheels/wheel-index.json` 记录构建来源。同名 PyPI 版本不保证该来源身份，不能任意替代。

```bash
python3.12 install.py --wheel-dir /path/to/downloaded-native-wheels --device auto
```

无匹配 wheel 时需要 Rust 和 C/C++ 工具链从冻结源码构建；Mac 还需 Xcode command-line tools。GitHub 自动 Source code 归档没有随 Release 的原生资产，首次安装推荐完整发行 ZIP。

维护者构建命令：

```bash
python app/build_czsc_runtime.py --out app/vendor/wheels --no-install
python app/build_release.py --output-dir dist/releases/0.1.6-rebuild --wheel-dir app/vendor/wheels
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

检查包含实际 installed wheel 身份与 SHA、严格 doctor、合成训练保存/重载/推理、分析与扫描、六页面、训练/参数 API、关注跨重启保存、退役接口拒绝，以及无合格模型时的趋势未评分和仓位 pending。支持的设备阶段需实际执行。工程验证不认证市场收益，历史版本 passed 不代替本版资产验证。

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

公开配置不绑定开发机任务或模型。先准备自己的来源与日历，再在“模型训练”检查并提交新冻结计划。完整训练包含全部历史逐日合格股票并集；默认 20 日成交额中位数至少 3000 万元，股票准备分组 360、CPU 准备进程 6、神经行批 2048。参数只影响新计划，不能修改已提交训练。

来源 B/S、最终共识与风险上限分别展示。缺合格趋势头时保持未评分；独立参考质量及费用资格未通过时，积极目标仓位保持 pending。软件 0.1.6 发布不表示后台全量训练已完成，不发布候选权重，不授予独立前瞻收益资格。详见 [训练指南](app/my_strategy/docs/current_model_training.md)、[参数说明](app/PARAMETERS.md) 和 [B/S 与仓位参考](app/my_strategy/docs/bs_position_reference.md)。
