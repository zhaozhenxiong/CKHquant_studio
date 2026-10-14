# 当前板块工作台与每日同步

主站 `/#scan` 的“板块”页签提供行业、概念两组活跃 Top30 和成交资金 Top30。`/boards` 跳转这个页签。浏览只读已发布索引行；“计算当前板块”使用本地数据库，“同步结构与特征”执行当前日的派生同步，“更新当前成员”显式调用知识来源。无需独立网页或额外服务。

输入边界：`khquant_raw.db` 提供已核验 OHLCV 和成交额，`native_structure.db` 提供精确身份的 CZSC／威科夫证据，`stock_basic/basic_kb.db` 提供当前同花顺成员映射。输出为 `processed/stock_fluent/basic_fluent.db`，可用 `KHQUANT_BASIC_FLUENT_DB` 改路径。当前成员只作 `current_reference`，不会补历史成员变化或进入模型训练。

活跃分数为同类型合格板块中 20 日相对成交额和当日涨跌幅绝对值的平均百分位乘 100；并列使用平均名次，单板块为 50 分，同分按板块代码排列。成交资金榜按成员成交额降序。概念成员可重叠，板块成交额不能求和当作全市场资金，也不是主力净流入。少于五个有效成员或覆盖低于 80% 的指标保持缺失，榜单不会为了凑足 30 行填零。

数据库包含 `sync_runs`、`feature_generations`、`board_daily_features`、`stock_daily_features`、`native_stock_summaries`、`board_structure_features`、`board_rankings`、`feature_input_refs`。生成记录、依赖和数值以 SHA 保护并追加保存，股票、板块、日期和榜单均有索引。旧知识库整包缓存经原始数学计算的真实内容键核验后迁移，十二个板块与二十二个股票数值保持一致，旧数据保留。

行情与结构使用同一数据库中的 market/full 发布指针。新的行情特征先发布；原生摘要完成全部身份核验后发布完整版本。详情按生成 ID 固定成员 SHA，不混用新成员或另一个日期的结构。CZSC 和威科夫分别核验输入可用性并计算覆盖；warmup、未成熟和缺来源保留缺失，不当作有效零分。派生摘要不是板块 `.TI` 的原生结构。

日更默认执行：行情提交、质量检查、元数据更新、冻结实际合格行情日期和内容、刷新或复用当日成员、发布行情与榜单、原生 prepare-only、摘要核验、完整发布。原生阶段不会训练模型。相同身份复用缓存；新增交易日或依赖改变仍安全全历史重放，尚不承诺保存引擎状态的跨日增量重建。来源限额会停止并保留续跑身份，不能把不完整成员标成完整。

独立同步使用 SQLite 进程租约、PID 创建时间、心跳和阶段检查点。活着但长时间没有心跳的 owner 不能被第二个进程夺取。失败和取消保留已提交行情、已发布 market 及原生恢复 ID；输入变化必须启动新同步。TaskManager 保存操作结果和部分完成状态，服务重启后可以读取和重试。

```sh
python -m my_strategy.cli build-board-fluent --help
python -m my_strategy.cli sync-daily-features --help
python -m my_strategy.cli refresh-native-cache --help
python -m my_strategy.cli update-data --mode local --json
```

`update-data --skip-feature-sync` 仅供明确的维护操作。迁移和验收可以指定 `--raw-db`、`--kb-db`、`--fluent-db`、`--source-root` 与 `--artifacts-root`；实际生成代码根和产物根分别核验，不能用路径或数据库时间戳替代真实内容、日历、代码及原生 blob 的身份。GET 的时间戳检查仅提示是否需要核验，不授予缓存复用资格。
