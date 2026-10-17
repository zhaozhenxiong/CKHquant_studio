# KHQuant v1.0.4 原生安装

安装 Python 3.12，下载本版 `KHQuant-1.0.4.zip`，按 `SHA256SUMS.txt` 核验并解压。不要把上一版本的安装证明用于本版。

Windows x64：`py -3.12 install.py --device auto`，随后 `.\khquant-native.bat dashboard`。
Apple Silicon macOS / Linux x64：`python3.12 install.py --device auto`，随后 `./khquant-native.sh dashboard`。

安装器按平台选择 SHA 核验的 CZSC 1.0.1 原生 wheel，创建独立虚拟环境。浏览器打开 `http://127.0.0.1:8124`，主入口是结构分析、全市场扫描、我的关注、预测与模型、数据与任务。“我的关注 → 模拟账户”包含回测、保存与每日续算、费用后对照；预测与模型保留预测及学习。原 `#simulation` 链接仍可用。旧参考训练与参数仍可通过明确的高级入口访问。

## 模型与业务目录

公开包没有行情、Kronos权重、已训练评分器或账户。可先查看界面和结构功能；预测资产不足明确报告，不自动下载、不启动旧全市场计算或训练。

通过 `KHQUANT_KRONOS_ROOT` 指定自己的单份资产目录，包含：

```
config.json
model.safetensors
tokenizer/config.json
tokenizer/model.safetensors
```

程序核验冻结版本与SHA：Kronos-base `2b554741eca47781b64468546e77fef3e85130e6`，Kronos-Tokenizer-base `0e0117387f39004a9016484a186a908917e22426`。不匹配时停止预测，不自动替换已有文件。源码安装默认本地 `app/my_strategy/kronos_model`；独立wheel默认用户 `KHQUANT_HOME/models/Kronos-base`，避免写入site-packages。显式环境变量优先。

已有正确base权重时，可明确运行以下维护动作补缺配置/tokenizer；它不下载base，不覆盖不匹配文件：

```bash
python -c 'from my_strategy.services.kronos_assets import prepare_assets; print(prepare_assets())'
khquant light-research status
```

上面的Python须来自已安装虚拟环境。原始行情、核验日历、个人账户及必要模型按原业务目录迁移，升级前保留 `.env`。普通代码打包排除这些状态。需要私有迁移时明确选择数据范围，不能将私有迁移包作为公开Release资产。

## 本版安装验证

本版最终发行资产的 Windows、Linux 与 macOS 安装验证目前待完成；CPU/MPS 分别记录实际结果，不能沿用 v1.0.1 或开发机的通过状态。公开验证门槛见 [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)。

最终安装probe从site-packages导入应用，验证本版版本、wheel SHA、原生来源、五个主入口、轻量API和隔离合成账户。它实际训练小MLP、验证同输入L2、原生跨日等价、风险卖出与续算；不会读取真实行情或业务模型。

平台结果见本版 `release-validation.json`：最终同字节应用wheel须分别验证Windows x64 CPU、Linux x64 CPU、Apple Silicon CPU/MPS。CUDA及Intel Mac未验证时明确保留该状态。MPS的训练验证不代表所有数值步骤都在GPU执行；校准和推理设备以报告为准。
