# 无 Git 源码变更审计

`ai_change_logger.py start` 在真实 Git 工作区继续使用原 Git 流程；无 Git 时保存 `source_audit_v1` 文件 SHA 清单和压缩脱敏文本快照。不会初始化 Git，不构造提交历史。

采集范围由脚本内 `SOURCE_DIRS`、扩展名与根目录文件白名单决定，只包括代码、配置模板和文档。`.env`、`.env.local`、业务 `data`、`artifacts`、知识库、模型目录、日志与符号链接不读取；仅允许脱敏后的 `.env.example` 模板。根目录任意 JSON 库存、业务报告不会进入快照。快照记录原文件 SHA 和脱敏文本 SHA；它用于审阅差异，不是可执行源码备份。

`finish` 比较会话起点快照，列出新增、修改、删除路径及前后 SHA，输出脱敏 unified diff。超大或读取失败的源码、原文件或祖先目录改为符号链接、读取期间 inode/大小/修改时间改变都标记 `partial` 和不可核验路径，不会误记为删除或零变化。快照缺失、损坏、SHA 不符，或旧会话没有起点快照时标记 `unavailable`；日志显示 `unknown files`，保留原会话事实。

可通过 `finish --external-diff /path/to/reviewed-deployment.patch` 附加本次部署或独立基线生成的 diff，保存来源 SHA 和压缩脱敏副本。附件明确标记为外部补充证据，不回写、不补造旧会话的起点快照，也不会把不可核验的历史改成成功采集。

脱敏覆盖 API key、token、password、cookie、secret、authorization 形式的键值，包括带引号和空格的值。原敏感配置仍须保留在被排除的业务/秘密目录内，审计快照不替代专门的秘密管理。
