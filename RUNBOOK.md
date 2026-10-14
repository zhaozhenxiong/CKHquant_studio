# KHQuant 0.1.7 运行说明

从本目录执行 `python3.12 install.py --device auto`（Windows 使用 `py -3.12`），随后运行 `./khquant-native.sh dashboard` 或 `.\khquant-native.bat dashboard`。网页默认地址为 <http://127.0.0.1:8124>。

先运行 `khquant doctor --check-write --strict --json` 核验环境，再准备自己的行情、已核验日历与完整 native 来源。六个页面为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务。来源 B/S 与参考目标变化分别显示；每股增减幅度相对上一参考目标计算，不执行模拟成交或组合资金分配。

在“模型训练”先预检再提交，或运行 `khquant train-reference --source-native-run-id YOUR_VALID_NATIVE_RUN --device auto`。来源必须完整且来源链、人口、日历、生成代码和资产 SHA 可核验；新行情不自动补入冻结来源。训练默认 12 轮、8192 行批、4 个准备线程。Torch 训练、校准和批推理优先 CUDA、其次 MPS、最后 CPU；精确价格与因果结构保留 CPU。明确请求不可用设备会阻断。旧 `train-trend` 入口已退役。

本机来源截止 2026-10-12 的 3,194 股 MPS 训练和技术验证已经完成，但当前预测基线未通过，不启用积极增仓目标。公开安装不带这些研究权重；技术完成、预测门槛和前瞻资格分别判定。缺评分保留最后参考值并标记 stale，WAIT 保持，可靠硬风险失效可清零。

运行路径由 `.env` 与 `core.paths` 解析，正式 wheel 状态默认位于 `~/.khquant`。用 `khquant package --verify /path/to/package` 核验导出包；默认包只含代码，数据和模型状态需显式选择，关注列表另需 `--include-personal`。公开发行包不带行情、模型权重、研究账本、秘密或个人记录。实际平台安装验证以随 Release 的 `release-validation.json` 为准。见 [安装说明](INSTALL_NATIVE.md)、[参数说明](app/PARAMETERS.md) 和 [参考架构](app/my_strategy/docs/reference_position.md)。
