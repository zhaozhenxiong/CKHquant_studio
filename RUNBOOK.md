# KHQuant 0.1.6 运行说明

从本目录执行 `python3.12 install.py --device auto`（Windows 使用 `py -3.12`），随后运行 `./khquant-native.sh dashboard` 或 `.\khquant-native.bat dashboard`。网页默认地址为 <http://127.0.0.1:8124>。

先运行 `khquant doctor --check-write --strict --json` 核验环境，再准备自己的行情及已核验交易日历。六个页面为结构分析、全市场扫描、我的关注、模型训练、参数设置、数据与任务。来源 B/S 与最终共识分别显示；风险上限不是积极配置目标。缺合格模型或独立参考质量时保持未评分/pending。

在“模型训练”先预检再提交。训练冻结来源、人口、日历、参数和代码，阶段进度只表示本阶段实际完成数。MPS/CUDA 用于支持的神经阶段，结构和真实 Broker 人口准备在 CPU；请求不可用设备会阻断。

运行路径由 `.env` 与 `core.paths` 解析，正式 wheel 状态默认位于 `~/.khquant`。用 `khquant package --verify /path/to/package` 核验导出包；默认包只含代码，数据和模型状态需显式选择，关注列表另需 `--include-personal`。公开发行包不带行情、模型权重、研究账本、秘密或个人记录。详见 [安装说明](INSTALL_NATIVE.md)、[参数说明](app/PARAMETERS.md) 和 [B/S 与仓位参考](app/my_strategy/docs/bs_position_reference.md)。
