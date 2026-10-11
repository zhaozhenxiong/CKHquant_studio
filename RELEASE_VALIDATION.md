# KHQuant 0.1.2 GitHub 发行验证

2026-10-11。本次发布完整的独立 MA / 结构 / 威科夫买卖点、真实交易区间和 L2 高收益入场/持仓退出实现，并保留 GitHub 原有 Windows UTF-8 与克隆日历 SHA 修复。

- 新源码完整回归：1,296 通过、10 项跳过、1 项既有 Starlette 弃用提示，69.52 秒。随后单独复核 Windows 原生重建的 wheel 索引回归和语法门槛。
- 29 项冻结训练核心、默认参数与本机正式基线逐项 SHA 一致；公共159项中155项一致，CLI、runtime_env、doctor、dual calendar 四项保留上游兼容性修复。
- 新主程序 wheel 独立安装后在真实 Apple Silicon / Python 3.12 通过 CPU 和 MPS：严格 doctor、合成训练保存/重载/40行推理、结构分析/扫描/回测、六入口与关注/持仓跨重启保存。复用既有锁定依赖，不宣称重新下载依赖的全新环境。
- 保留226项冻结CZSC源码及Windows x64/macOS ARM64原生资产的原SHA；CI重建得到的新物理wheel SHA由各自索引记录，不能用历史附件wheel SHA比较新构建。
- GitHub 0.1.2 Windows/Linux CPU CI 单独执行；CUDA、Intel Mac、完整新嵌套模型状态迁移未认证。源码包暂只附Windows x64/macOS ARM64原生资产。实际完成状态以 Actions 和发布验收资产为准。

公开资产仅含代码、原生运行时与参数；不含行情、训练模型、真实研究账本或私人状态。本机79头权重及原始科学冻结资产保持原样。高收益实证仍为连续扣费 −0.4142365082%，参考 −0.2856443810%；收益改善未通过，独立未来验证 pending。正式软件发布不授予模型市场收益资格。

安装资产绑定发布Git提交及逐文件SHA，见 release.json、image-manifest.json、RELEASE_MANIFEST.json 和 SHA256SUMS.txt。构建宿主绝对路径仅保留本机审计材料。

---

# KHQuant 0.1.1 GitHub 发行验证

2026-10-10。最终主程序运行时代码与 25a66bc96ff180837e438b5ee6bae4b9c3c8d03c 的内容一致；后续变更仅安装验证脚本、回归测试及发行记录。

- Apple Silicon / Python 3.12 的完整语法检查、829 项源码测试（7 项平台跳过）与严格 doctor 通过。
- Windows x64 / Python 3.12 全流程通过：新虚拟环境、锁定依赖下载、随包原生 CZSC 的实际安装 SHA 核验、829 项源码测试（7 项平台跳过）、主程序 wheel 实际安装及 site-packages 身份、隔离功能冒烟。真实运行见 [Windows 验收](https://github.com/zhaozhenxiong/CZSC_KHquant_studio/actions/runs/37965437521)，源码提交为 `edc2808274cdcc48f36f3eb8a99861bce701efe1`。
- 公开主程序 wheel 在 macOS 上独立安装，CPU 与真实 MPS 均通过：严格 doctor、训练保存与重新加载、各 40 行实际推理、六个网页入口、关注及持仓跨重启保存。Mac 验收复用既有锁定依赖；Windows 验收使用同提交在 CI 构建的 wheel，物理 SHA 单独记录。
- 226 项冻结 CZSC 源码逐项 SHA 通过；Windows x64 与 macOS ARM64 随包原生资产来源及 SHA 保持冻结。114 个主程序代码文件、23 项配置/网页资源与 Git 源码字节一致；安装环境的 365 项运行时文件与公开 wheel 一致。
- Windows 中文 JSON 输出、验证报告输出及 CRLF 克隆日历绑定均已修复。真实 cp1252 子进程回归覆盖中文文本；日历测试覆盖 CRLF 有效路由、字节篡改拒绝和旧无新字段 LF 报告兼容。

公开主程序 wheel SHA-256：`339de97efcc43e85d81e2bdb7db58b083ce9ffdab68b11fbe196fbdb3c3d29fc`。

Windows CI 平台构建 wheel SHA-256：`c467e84f55b77d68754fbf2ec000583fbec32273e1b128ba5b0e2caaca066265`。

公开 wheel 运行时内容树 SHA-256：`b6932ec6197df77e893cc011ae5c564c814d677f123fdb1d06a3a15a8f3ab287`（按 wheel 路径排序，UTF-8 的 `path SHA` 行以 LF 连接，无末尾换行）。

源码 ZIP 从干净 main 生成，最终提交与完整性见 release.json、image-manifest.json 和 SHA256SUMS.txt。公开 image-manifest 的来源路径使用仓库相对 app；真实构建宿主路径只保留在本地审计材料。

Linux CPU 曾在较早的 0.1.1 源码提交 4c5312c3 通过 CI；最终源码的跨平台原生重建 CI 继续运行，发行包不附 Linux 原生 wheel。CUDA、Intel Mac、完整新嵌套模型状态迁移未认证。

发行包不含行情库、模型权重、研究账本或私人状态。模型继续保持影子；P10 三专家目标为 10 个交易日费用后正收益，实际 policy-entry 目标为冻结续持策略完整往返的费用后正收益，实际持仓退出为独立目标。工程验收不认证市场收益，原有本地 0.1.1 资产与科学冻结记录保持原样。
