# KHQuant 0.1.5 原生安装与验收

## 环境和设备

本版固定 Python 3.12。支持路线为 Windows x64 CPU/CUDA、Linux x64 CPU/CUDA、Apple Silicon macOS ARM64 CPU/MPS。Intel Mac 未列入本版安装矩阵。Mac 使用原生 ARM64 Python；系统最低版本以锁定 PyTorch wheel 和真机验证为准，MPS 可用性由 torch.backends.mps.is_available() 及实际算子执行检查。

~~~powershell
# Windows；Mac/Linux 使用 python3.12
py -3.12 install.py --device auto
~~~

完整安装会安装基础运行依赖、对应平台 Torch、冻结 CZSC 运行时以及 KHQuant CLI。新环境使用 .venv，已有环境保留 .env 中的 KHQUANT_VENV_DIR。--gpu 保留为 CUDA 安装兼容选项。

~~~bash
python3.12 install.py --dry-run --device auto
python3.12 install.py --device mps
python3.12 install.py --device cpu --data-dir /path/to/data --artifact-dir /path/to/artifacts
python3.12 install.py --upgrade --device auto
~~~

升级前使用下文状态包命令备份数据。安装器保留已有路径与凭据，升级不会自动更新行情或晋升模型；安装结束执行严格 doctor，失败原因会保留。

## 原生依赖资产

安装器先验证附件源码，再按目标 Python ABI、操作系统和架构选择 wheel，并核验 SHA-256 与冻结源码身份。Release 中的 app/vendor/wheels/wheel-index.json 记录平台构建来源。

~~~bash
python3.12 install.py --wheel-dir /path/to/downloaded-native-wheels --device auto
~~~

缺少匹配 wheel 时，从冻结源码构建，需要 Rust、C/C++ 工具链。Mac 源码构建还需 Xcode command-line tools。普通用户可使用 CI 预构建的原生资产，避免本机构建。不能替换为同版本 PyPI CZSC；同名版本不保证附件算法一致。

CI 和维护者构建命令：

~~~bash
python app/build_czsc_runtime.py --out app/vendor/wheels --no-install
python app/build_release.py --output-dir dist/releases/0.1.5-rebuild --wheel-dir app/vendor/wheels
~~~

build_release.py 支持重复 --wheel-dir 合并三平台 CI 资产。它验证原生资产并生成源码 ZIP、完整性清单、版本来源及 SHA256SUMS.txt；不会上传远端。工作区未提交内容会包含在包中，并在 release.json 标明。

## 安装后检查

Windows 用 .\khquant-native.bat，Mac/Linux 用 ./khquant-native.sh。激活环境后可直接使用 khquant。

~~~bash
khquant doctor --check-write --strict --json
khquant dashboard
~~~

doctor 核验冻结来源、路径、数据库和真实 Torch 训练/推理算子。CPU 路径同样核验 Torch；空库可安装，但分析前需要行情。设备状态与数据状态分别报告。

完整分发冒烟检查使用明确标记的合成数据，不会更改真实行情或个人记录：

~~~bash
# 使用安装后的环境执行
python app/verify_installation.py --device cpu --output installation-cpu.json
# Apple Silicon 真机必须实际执行这一项
python app/verify_installation.py --device mps --output installation-mps.json
~~~

该检查涵盖严格 doctor、模型训练/保存/重新加载、分析、扫描和回测、行情更新模块入口导入、八个网页入口及训练／参数API、关注与持仓跨服务重启保存，并单独检查默认趋势策略在空模型目录下明确不可用。源码单元测试另检验设备失败、时点资格、CPU/MPS 数值边界、源哈希及打包依赖。runtime-manifest.json 分别记录本版各平台实际验证状态，旧版本验证仅作历史记录。

正式 wheel 的运行状态默认写入 ~/.khquant；可通过 KHQUANT_HOME、KHQUANT_DATA_ROOT、KHQUANT_ARTIFACT_ROOT 指定位置。源码/Release 安装保留仓库 .env 路径契约。网页默认绑定 127.0.0.1:8124；非本机绑定需要 KHQUANT_API_TOKEN。

完整首次安装使用源码 Release 中的 `install.py`；主程序 wheel 用于已经配置对应 Torch 和冻结 CZSC 的环境。直接依赖 PyPI 自动安装的同版本 CZSC 不符合冻结来源校验。无 Tushare Token 时仍可初始化免费备用来源；Tushare 行情和核验日历下载需要单独配置用户自己的凭据。

## 数据与模型状态包

以下导出命令在源码仓库或解压的源码 Release 环境中运行。正式 wheel 的 site-packages 缺少源码安装器，不是可分发来源；直接导出会明确报错。

~~~bash
khquant package --target ../khquant-code
khquant package --target ../khquant-state --include-data --include-artifacts all
khquant package --target ../khquant-private-state --include-data --include-artifacts all --include-personal
khquant package --verify ../khquant-state
~~~

wheel 用户先克隆仓库或解压源码 Release，再显式指定源码目录：

~~~bash
khquant package --from-project /path/to/KHQuant --target /path/to/khquant-state --include-data --include-artifacts all
~~~

导出状态来自指定源码目录的 `.env` 路径配置。迁移 wheel 安装的现有状态时，将该配置中的 KHQUANT_DATA_ROOT、KHQUANT_ARTIFACT_ROOT、KHQUANT_METADATA_ROOT 和 KHQUANT_RAW_DB 指向实际状态位置；目录标识和源码安装器必须保留。wheel 用户仍可直接用 `khquant package --verify /path/to/khquant-state` 核验已有包。

默认只打代码，排除环境、秘密、运行数据和本地审计提示词；本地审计记录继续保留。--include-data 包含原始行情、结果与任务数据库。关注与手动持仓需要独立的 --include-personal。

latest 从最新运行开始收集模型、日历及发布证据依赖；all 保留所有当前支持的 CZSC/研究/日历/认证运行。SQLite 使用一致 backup；模型 manifest、checkpoint 及历史发布事件不改写。缺引用、哈希不符或不完整发布状态将拒绝生成合格包。

目标机器先 package --verify，再运行 install.py 创建自己的环境。导入已有训练候选不会自动晋升；模型资格仍按原日期及证据核验。

当前体系的模型、原始／原生来源、核验日历、逐日人口、头manifest与账本须按完整冻结依赖核验。跨机器状态迁移另行验收；不能将历史档案或目录里的任意权重恢复为当前合格模型。源码发行包只包含程序、配置与冻结原生依赖，不包含本机行情库、训练权重、个人记录或研究账本。

## 当前模型训练与本机研究选择

网页包含八个入口，其中“模型训练”提供默认参数、预检、后台训练、真实阶段进度、逐层样本流失诊断、复核及研究选择结果；“参数设置”保存以后的训练默认，新修订不改历史模型或任务。使用说明见 [当前训练指南](app/my_strategy/docs/current_model_training.md)。

完整训练使用历史逐日合格股票的并集，流动性默认要求过去20个完整核验交易日成交额中位数至少3000万元。股票准备分组默认360，仅分配资源，不限制人口；CPU准备进程数与默认2048行神经训练批分别控制不同阶段。MPS用于支持的神经训练与批推理，CZSC、字典、方向／树模型、Broker/state在CPU；实际设备以任务和头manifest报告为准，不推断未知GPU利用率。

~~~bash
khquant train-trend --end YYYY-MM-DD --calendar-run-id YOUR_VERIFIED_CALENDAR_RUN --stock-batch-size 360 --cpu-workers 6 --device mps
khquant model-diagnostics --model-run-id YOUR_CURRENT_TRAINING_RUN --cpu-workers 2 --json
~~~

占位日期与ID须替换为本机已核验输入；CPU机器明确使用`--device cpu`，MPS不可用时拒绝，不静默回退。公开配置不绑定开发机运行；当前分析／扫描／回测只使用真实对比已通过且已生效的研究选择，无合格模型时明确观察／不可用。升级不会自动训练、替换模型或授予收益资格。

## 0.1.5实际验收状态

Windows、Linux及macOS的支持路线与本版实际安装结果分别记录。此文档准备时，0.1.5发布资产的跨平台安装验收仍待CI和真机报告；旧版passed不能替代新wheel的验证。软件生产Release与模型生产资格分别判定：本机全量训练仍在进行，正式独立模型资格继续pending，发行包没有模型权重，也不声称已通过前瞻收益认证。
