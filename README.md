# CZSC量价研究工作台

当前正式软件版：**0.1.5（2026-10-13）**，仓库为 [zhaozhenxiong/CKHquant_studio](https://github.com/zhaozhenxiong/CKHquant_studio)。软件发行与模型资格分别记录：本次发布不包含行情、个人记录、研究账本或训练权重；发布准备时本机全量模型训练仍在进行，正式模型收益资格继续 **pending**。

当前只有一个趋势共识模型体系，内部策略为 `trend_consensus`。均线上涨／下跌／中性概率、结构、威科夫量价和高收益候选质量按40%／20%／30%／10%组合；缺失概率不补零。有效原机会在实际空仓且风险合格时逐日重评，质量拒绝不消耗机会；真实持仓固定入场信号时的MA退出头，不在持仓中途升级。主图的参考价与实际Broker成交分别记录。

八个网页入口：结构分析、全市场扫描、策略回测、我的关注、我的持仓、模型训练、参数设置、数据与任务。训练、预检、默认参数、逐层样本诊断与研究模型选择见 [当前训练指南](app/my_strategy/docs/current_model_training.md)；扫描的最近30核验交易日状态见 [市场状态指南](app/my_strategy/docs/market_history.md)。

## 安装和启动

使用原生 Python **3.12**。首次安装默认创建`.venv`；升级保留已有`.env`与本机路径。

Windows x64：

~~~powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
~~~

Apple Silicon Mac（原生ARM64 Python）：

~~~bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
~~~

Linux x64：

~~~bash
python3.12 install.py --device cpu
./khquant-native.sh dashboard
~~~

浏览器打开 <http://127.0.0.1:8124>。安装器的auto选择可用Torch后端；训练任务仍冻结自己的明确设备参数。可以明确请求CPU、CUDA或MPS；不可用的明确设备会报错，不静默回退。

首次安装推荐下载`KHQuant-0.1.5.zip`及其SHA256清单。安装器仅接受经附件源码、目标平台与物理SHA核验的CZSC原生wheel；GitHub自动Source code归档不含Release原生资产。缺匹配wheel时需Rust和本机编译工具。0.1.5各平台的实际安装验收以`runtime-manifest.json`及随附验证报告为准，不能沿用0.1.4结果；MPS须有真实Apple Silicon算子证据，CUDA与Intel Mac不自动获得验收资格。

## 首次使用与当前CLI

代码包不带私人行情库、模型或选择指针。先在“数据与任务”准备行情和已核验交易日历；空模型环境显示缺模型／不可用，安装不会生成模型概率或买卖点。当前研究仅使用经过真实对比并已生效的选择；没有合格选择时保持观察。旧并行模型与兼容入口已退役，不能用历史权重冒充当前模型。

激活环境后，`khquant`与`python -m my_strategy.cli`共用当前入口：

~~~bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant analyze --symbol 600519.SH --usage-mode historical --device cpu --json
khquant scan --stocks 600519.SH,000001.SZ --usage-mode historical --device cpu --json
khquant backtest --stocks 600519.SH --start 2026-01-01 --device cpu --json
khquant train-trend --end YYYY-MM-DD --calendar-run-id YOUR_VERIFIED_CALENDAR_RUN --stock-batch-size 360 --cpu-workers 6 --device mps
khquant model-diagnostics --model-run-id YOUR_CURRENT_TRAINING_RUN --cpu-workers 2 --json
~~~

占位日期和运行ID须替换为自己的核验输入。完整训练默认包含全部历史合格股票：过去20个完整核验交易日成交额中位数至少3000万元。360是准备分组上限，不是360股并行，也不截断股票池；CPU准备进程数控制CPU并行，神经训练行批默认2048行。网页预检冻结实际截止日、历史合格并集和当日资格，修改参数须重新预检。

MPS用于支持的线性／MLP训练和神经批推理；字典、CZSC、方向／树模型及真实Broker/state仍在CPU。设备可用、请求设备与实际执行报告分别展示；没有报告时不推断GPU利用率或预计完成时间。网页设备建议只改本次表单，不保存默认或修改已提交任务。

## 分发与状态迁移

源码发行包仅含程序、配置和冻结原生依赖：

~~~bash
python app/build_release.py --output-dir dist/releases/0.1.5-rebuild
~~~

状态导出属于用户自己的操作，默认不进入公开Release：

~~~bash
khquant package --target ../khquant-code
khquant package --target ../khquant-state --include-data --include-artifacts all
khquant package --verify ../khquant-state
~~~

需要迁移关注和手动持仓时显式追加`--include-personal`。wheel用户从源码Release导出时指定`--from-project /path/to/KHQuant`，并将该来源目录的路径配置指向实际状态。缺引用或哈希不一致会阻断核验；历史档案不恢复当前交易资格。跨机器完整模型依赖与路径必须另行验证，软件安装通过不能代替状态包验收。详见 [INSTALL_NATIVE.md](INSTALL_NATIVE.md)。

## 交易、训练与认证边界

收盘信号在下一核验且可执行交易日尝试成交；T+1、整手、涨跌停、流动性、真实费用、拒单、每日净值与期末持仓持续核验。多股回测采用等额独立账户；手动持仓不能伪作模型退出的实际Broker账本。未复权行情、历史ST与历史成员资料不足随结果披露。

高收益标签保持完整清仓后扣费净收益至少8%且至少2R；训练至少100行、校准至少30行且两段均有两类，并保留完整成熟、严格先验与原机会原子依赖清洗。逐折诊断解释数据、目标与过滤流失，不为凑正类降低标准。

自动替换仅是研究模型选择，要求完整冻结人口至少60股、各开发窗口至少50笔完整往返且总计至少200笔，并通过同口径真实账本、费用压力与实际资产审计。缺正类、关键头或证据时保持阻断。正式独立前瞻认证另需冻结后的真实独立窗口，现状仍为pending；目录名production与软件正式发布均不授予模型生产资格。历史版本及旧研究结果仅保留在 [RELEASE_NOTES.md](RELEASE_NOTES.md) 作为档案。
