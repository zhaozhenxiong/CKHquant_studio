# CZSC量价研究工作台

当前正式软件版：**0.1.3（2026-10-12）**。当前策略名称为“趋势共识策略”，使用均线上涨／下跌／中性概率、结构与威科夫规则强度及高收益候选质量，权重为40%／20%／30%／10%。主图显示绑定研究账户的最终B/S，点击查看多Agent评分；参考价与实际Broker成交分别记录。参数说明见 [当前趋势指南](app/my_strategy/docs/trend_consensus.md) 和 [参数说明](app/PARAMETERS.md)。

有效原机会逐日重评使用独立 `czsc_trend_consensus_v2` 模型族，账户规则和完整重建入口见[有效机会重评指南](app/my_strategy/docs/trend_reassessment.md)。扫描新增[最近30交易日市场状态](app/my_strategy/docs/market_history.md)，展示覆盖、行业、量价事件、方向与实际研究账户动作，并保留同一运行的股票明细与分析钻取。

六个网页入口：结构分析、市场扫描、策略回测、我的关注、我的持仓、数据与任务。当前组合研究使用冻结 CZSC 1.0.1、均线量价和 PyTorch；支持行情更新、研究训练、模型推理、按日期路由及状态打包。

## 安装和启动

安装原生 Python **3.12**，克隆仓库或解压 Release。首次安装默认创建 .venv；升级保留已有 .env 配置和环境路径。

Windows x64：

~~~powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
~~~

Apple Silicon Mac（原生 ARM64 Python）：

~~~bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
~~~

Linux x64：

~~~bash
python3.12 install.py --device cpu
./khquant-native.sh dashboard
~~~

浏览器打开 <http://127.0.0.1:8124>。安装包含 PyTorch 研究依赖；auto 在 NVIDIA 环境选择 CUDA，在 Apple Silicon 选择 MPS，其余选择 CPU。可以明确使用 --device cpu、--device cuda:0 或 --device mps。明确请求不可用的设备会报错。

安装器只接受经冻结源码校验的 CZSC wheel。Release 应携带目标平台 wheel 和 wheel-index.json；可通过 --wheel-dir 使用单独下载的资产。0.1.3 正式源码 ZIP 携带经过冻结源码及物理 SHA 校验的 Windows x64、macOS ARM64、Linux x64 CPU 原生 wheel。GitHub 自动 Source code 归档没有这些 Release 原生资产，首次安装推荐下载 KHQuant-0.1.3.zip。源码安装缺 wheel 时需要 Rust 与本机编译工具。平台测试状态见 runtime-manifest.json，MPS 验收必须使用真实 Mac。

## 首次使用

代码包不包含私人行情库、训练模型、关注或持仓。首次运行可用“数据与任务”下载行情，或安装时指定 --raw-db。公开配置自动查找本机具备资格的趋势模型，未训练时明确显示模型不可用；安装不会生成概率、模型或买卖点。研究训练需要已核验的交易日历；可导入包含日历和模型依赖的状态包。软件安装完成与行情、模型资源就绪分别显示。

激活环境后，khquant 与 python -m my_strategy.cli 共用入口：

~~~bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant analyze --symbol 600519.SH --model-family high_return --json
khquant scan --stocks 600519.SH,000001.SZ --model-family high_return --device auto --json
khquant backtest --stocks 600519.SH --model-family high_return --model-run-id MATCHING_TREND_RUN --start 2026-01-01 --device cpu --json
khquant research-train --end YYYY-MM-DD --calendar-run-id VERIFIED_CALENDAR_RUN --device auto
~~~

`MATCHING_TREND_RUN` 须替换为本机已完成的趋势共识运行 ID。新机器先生成匹配的核验日历、原生来源运行，再按[当前趋势指南](app/my_strategy/docs/trend_consensus.md)训练；不要使用文档历史中的本机运行 ID。旧 high_return v1-v4 的离线研究使用自身历史入口和身份。

网页、CLI 和模型推理按本机选择运行设备，模型原训练设备只保留为历史来源记录。MPS 加速神经网络训练和推理；精确结构、价格比较及成交账本由 CPU 执行。任务结果区分设备可用性和实际工作。

## 分发和迁移

源码 Release，不含业务数据和个人记录：

~~~bash
python app/build_release.py --output-dir dist/releases/0.1.3-rebuild
~~~

在源码仓库或解压的源码 Release 环境中打包研究状态，目标目录须不存在：

~~~bash
khquant package --target ../khquant-state --include-data --include-artifacts all
~~~

正式 wheel 安装只提供运行代码，不能从 site-packages 导出源码包。先克隆仓库或解压源码 Release，再用 `--from-project /path/to/KHQuant` 指定来源。导出状态采用该来源目录的路径配置；迁移 wheel 的现有状态时，先将来源目录的 `.env` 指向实际数据、模型 artifacts 和 metadata 目录。

需要自己迁移关注与手动持仓时，显式追加 --include-personal。既有单模型状态按受支持的依赖契约导出；本版新 dual/Wyckoff/趋势共识嵌套模型状态尚未完成完整迁移验收，不能保证 latest/all 迁移全部新研究状态。模型哈希、历史时间和影子资格保持不变。安装方式及验收详见 [INSTALL_NATIVE.md](INSTALL_NATIVE.md)。

## 研究边界

默认组合研究；--structure-only 选择原生结构对照。收盘信号在下一可执行交易日开盘尝试成交，费用、拒单、每日净值和期末持仓均留存。多股回测采用固定等额独立账户。当前使用未复权行情，历史 ST、分红送转等限制随结果披露。

训练候选只有通过独立、完整股票池的费用后账本门槛才可晋升。目录名 production 不代表已正式发布；迁移和安装也不会赋予发布资格。旧 V2/Z1 模型与派生仓库已退役。

当前趋势策略的高收益目标要求完整扣费收益至少8%且至少2R，持仓退出由冻结规则与独立MA退出头解析。独立未来收益验证仍为 pending，软件发行不授予模型生产资格。

以下0.1.2结果保留为历史对照，不是当前0.1.3策略表现：0.1.2 的高收益模型目标是完整实际回合费用后收益至少 8% 且至少 2R，独立持仓退出比较下一核验开盘退出与冻结续持。原固定 10 日模型保留自己的身份和目标。当前新组合连续扣费收益为 −0.4142365082%，候选点并集参考为 −0.2856443810%；独立未来验证待完成。正式软件发布保留这些实证状态，普通分析/扫描/回测继续使用既有影子评分契约；应用模型改变订单使用显式高收益离线研究入口。详见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。
