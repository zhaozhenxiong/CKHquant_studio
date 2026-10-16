# v1.0.1 任务、状态与结果索引

## 持久任务

`storage/task_journal.py` 在独立 SQLite 中保存接单、owner PID/create-time、状态、进度、心跳、取消标记和结果引用。先提交事务再入队，活动 fingerprint 有唯一索引。进度最多每秒写一次，长操作另有 5 秒心跳。跨服务取消使用持久标记；owner 退出后任务标为 interrupted，不自动恢复。

业务输出提交前持久登记 namespace/key/预期值 SHA intent，随后发布 CAS 并绑定终态引用。跨库提交间崩溃仍为 interrupted；只有实际 CAS SHA、身份及预期值 SHA 一致，才通过 recovered_outputs 展示产物，不能推定任务成功。服务刷新核验死亡 owner，新接单不会重启旧任务。旧终态 CAS 摘要按 SHA 导入一次，首次兼容成本与历史任务量相关。

统一历史通过 SQL 筛选、分页，仅解码本页。legacy 展示索引不修改任务 payload 或结果。overview 在同一连接事务读取活动任务与历史页；前端合并请求，隐藏页面停止密集轮询。清理仍核验真实引用，展示投影不冒充额外业务引用。

## 状态摘要

原始库只读。独立 `raw_status_summary.db` 记录深查结果及文件观察代次，包括路径、DB/WAL inode/size/mtime/ctime 和 SQLite 文件头。它只使状态摘要失效，不替代来源、价格或模型的实际内容 SHA。提交、替换和 WAL 变化使摘要失效；核验期间变化不发布新代。失败保留旧摘要并标记 stale/error。`GET /api/data/status?refresh=true` 显式深查。

热状态查询复用核验摘要；冷启动或新行情后仍需深查，不能以热路径延迟承诺首次响应时间。

## 扫描结果

完整报告保存时提交绑定报告 SHA 的摘要与逐行索引。同一运行的报告和索引更新用跨进程锁串行。`page/page_size/q/category/symbols` 查询只读取所需行；旧不带分页的接口保留完整 compact 返回。旧报告首次访问在外部索引惰性建页，不改报告字节。网页首页、过滤与翻页使用服务端分页。

合成性能基准可在隔离临时目录复跑：

```bash
python -m my_strategy.scripts.benchmark_runtime_metadata --output ./runtime-metadata-benchmark.json
```

该基准衡量元数据与分页开销，不能代替全市场数值计算吞吐或整机 RSS 测量。

## 正确性边界

回归覆盖接单、运行、检查点、结果引用提交、取消和终态的进程退出；CAS 提交前后意图恢复；PID 重用、跨实例去重和取消；队列拒收；状态并发刷新、WAL 失效及核验中提交；旧报告迁移、SHA 绑定、分页拼接等价和大列表只读取本页。标签影响审计不改变旧模型、人口、权重或资格；新训练使用新的标签身份。
