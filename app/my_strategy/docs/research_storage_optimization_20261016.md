# v1.0.1 研究缓存与原生对象留存

## 已实施的缓存优化

`ResearchStore` 保持完整身份、不可变 SHA 文件、跨进程配额、多个 pin owner、读写租约和 GC 墓碑合同。新增 `object_usage` 与 `research_totals` 为派生索引；首次在同一 SQLite 事务内从原表构建，后续由触发器增量更新。原表、对象字节、TTL 与已有 pin/lease 不迁移重写。外部维护连接删除 result 时，触发器也维护计数和级联关系。

普通 put 先做索引探测；只有到期、配额压力、待重试删除或孤儿巡视到期才执行 GC。过期候选由 `MAX(expires_at, lease_until)` 索引取得，压力回收从访问时间索引按需读取。共享 SHA 的全部身份、pin 和 lease 都参与对象级保护。显式 GC/dry-run 仍巡视完整自有对象目录。

缓存 GET 只读索引并打开文件描述符，在锁外读取、校验 SHA/大小和解压。GC 抢先删除且索引已退休时返回明确 cache miss；索引仍有效但文件丢失则报错。已打开描述符在 POSIX unlink 后仍可读完整内容；不允许删除打开文件的平台保留 pending GC 等待读者关闭。访问时间以 30 秒粒度更新；热点读取、`get_setting`、`list`、`is_pinned` 无 SQLite 写事务。LRU 的访问时间精度因此为 30 秒，身份、TTL 和内容资格没有放宽。

批量接口：

- `put_many(items, lease_owner=None, lease_ttl_seconds=300, refresh_keys=())`：items 含 identity/result/namespace 和可选 TTL/pin；原子发布并原子加租约，返回顺序与输入一致。
- `acquire_leases(keys, owner, ttl_seconds=300)` 与 `release_leases(keys, owner)`：批量全有或全无。
- `put(..., refresh_keys=chunk_keys)`：在 manifest 同一事务内验证并延长所有依赖至至少 manifest 到期时刻；缺失则全部回滚。
- `get_by_key(key, touch=False)`：只读核验，不更改访问时间。
- `refresh_expiry(keys, ttl_seconds=None)`：只能延长仍有效的结果，不能复活过期对象。

## 小型隔离基准

使用真实 SHA 文件的合成 1,000/5,000 条元数据夹具，单 CPU 进程，没有业务数据库、模型、扫描或 GPU 运算。时钟固定，旧新实例接收相同输入。下表为 5,000 条夹具中位耗时，不是生产扫描端到端加速率。

| 操作 | 原实现 | 新实现 |
|---|---:|---:|
| 单条 put（30 次） | 19.096 ms | 0.693 ms |
| 配额字节读取（30 次） | 2.977 ms | 0.00167 ms |
| 无压力索引 GC（20 次） | 15.815 ms | 0.01123 ms |
| namespace 列表 20 行（30 次） | 1.991 ms | 0.07462 ms |
| 32 块发布及加租约（3 批） | 618.426 ms | 14.678 ms |
| 热 GET（100 次） | 0.11125 ms | 0.10408 ms |

100 次热 GET 的 SQLite 写事务从 100 次变为 0 次。单次读取延迟基本相同，主要收益是释放写锁。5,126 个最终对象的逻辑和分配字节完全一致；索引数据库由 2,646,016 增至 4,861,952 字节，增加约 2.11 MiB。首次派生索引迁移 29.3 ms。1,000 条下单 put 3.979→0.561 ms；新实现的增长与旧实现全表扫描不同。

复跑：

```bash
python -m my_strategy.scripts.benchmark_research_store \
  --baseline-source /absolute/path/to/saved/research_store.py \
  --output /absolute/path/to/cache_benchmark.json
```

## 原生对象只读清单

`inspect_native_storage` 输出 inode、全部已发现 hardlink 路径、未知链接数、索引代际及 run/source/model/account/board/task 引用。不按年龄删除，不提供 apply 操作。索引引用、运行别名、活跃任务或未知链接保持保护；没有已知引用也只能归类待核验。报告的 `deletable_bytes` 固定为 0，`ready_for_gc` 为 false。

运行中的清单使用有限引用文本预算，所有未扫描/过大/不可读来源均进入 coverage gaps；这不是完整全局无引用证明。默认不读取全部原生内容计算 SHA，也不读取原始行情表。未来确切退休需冻结依赖写者，完成所有扫描、模型、账户、板块、回滚引用核验，再核验实际内容和所有链接。

```bash
python -m my_strategy.scripts.inspect_native_storage \
  --project-root /absolute/path/to/app \
  --max-reference-mib 512 \
  --output /absolute/path/outside/business/data/native_inventory.json
```

显式 `--verify-content` 会读取全部对象，应在受控维护时执行；`--alias-root` 可补充其他已知硬链接所在目录。

## 原生 lossless delta 归档格式与恢复

`native_lossless_delta_v1` 将顶层记录流以 128 行检查点、字典字段变化及累积数组后缀保存。封装绑定完整 canonical JSON 的 SHA 和字节长度。恢复保留所有记录、事件与字段，区分 int/float/bool 和负零，恢复行之间不共享可变对象。普通旧 gzip JSON 仍能读取。

该格式须验证完整 JSON canonical 等价及源文件身份不变。压缩节省与解码开销依内容而变；本版默认继续旧 schema，新 codec 只用于显式归档副本。小样本不能外推全库释放量。

创建只读源的独立归档副本及 manifest：

```bash
python -m my_strategy.scripts.archive_native_payload create \
  --source /absolute/path/to/native.json.gz \
  --archive /absolute/path/to/archive/native.delta.json.gz
```

恢复到新的独立位置：

```bash
python -m my_strategy.scripts.archive_native_payload restore \
  --archive /absolute/path/to/archive/native.delta.json.gz \
  --manifest /absolute/path/to/archive/native.delta.json.gz.manifest.json \
  --output /absolute/path/to/restored/native.json.gz
```

工具拒绝覆盖任意已有文件，也拒绝恢复到 manifest 记录的原始路径。manifest 同时绑定原始 gzip SHA、归档 SHA、完整逻辑 SHA 和恢复 gzip SHA。**完整 JSON 等价不代表压缩文件字节身份不变**：恢复的是新的 canonical legacy gzip，有自己的物理 SHA，不能替换冻结 source/CAS 指针。归档工具只生成副本，不迁移现有 CAS 或索引。

验收覆盖配额并发、共享 SHA 多 owner、lease 缩短/续期、事务回滚、文件发布中断、GC 删除重试、读者/GC 两种竞态、迁移无原表改写、只读路径不争写锁、manifest 依赖 TTL、hardlink/symlink/缺证据保护，以及 codec 追加、修订、周月边界、重启与损坏拒绝。真实市场端到端性能和全库 codec 收益不能由上述小型基准替代。

新增索引、显式状态、保护备份和回滚副本也会占用空间；本版不承诺整体磁盘净缩减。上述合成 CPU 微基准不构成 Windows/Linux 或最终发行 wheel 的安装证明。
