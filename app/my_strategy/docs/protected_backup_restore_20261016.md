# 保护性备份恢复

保护性备份是原始行情及账户、模型、指针的本地恢复点，并非完整工作区镜像。它不包含全部扫描产物、任务历史、原生结构缓存和临时研究缓存。原始行情另有经过完整解压、物理 SHA 和 SQLite 完整性核验的 `.db.gz` 备份。

账户、模型恢复必须使用恢复器，不能直接把备份内的 `research.db.gz` 解压覆盖运行数据库。原始备份保留了完整研究索引，但只复制永久保护对象；直接覆盖会使临时缓存索引指向不存在的文件。恢复器仅在新的独立目录中删除这些未备份且无永久保护的索引，将它们恢复为缓存未命中。缺少任何 pin 或永久 namespace 对象时，恢复失败，不发布不完整目录。

在含 `my_strategy` 的已核验源码目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m my_strategy.scripts.restore_protected_assets \
  --backup /absolute/path/to/protected-backup \
  --destination /absolute/path/to/new-isolated-restored-directory
```

目标必须不存在，且不能位于备份目录之内。程序不会停止服务，也不会把恢复目录安装到生产。仅当全部任务和写者已停止、确认目标布局与备份范围、完成独立恢复验收后，才能另行执行受控安装。原始备份、manifest、源对象和账户身份均不改变。

恢复器逐一核验所有文件与压缩数据库 SHA、SQLite 完整性与外键；所有剩余研究结果对应的对象必须存在、物理 SHA/长度正确且能完整解压 JSON。原有账户设置、模型指针、pin 原样保留。对于历史上缺少 pin 的永久对象，恢复副本增加 `protected-recovery:<备份manifest SHA>` owner，避免后续普通写入触发 GC 时淘汰这些恢复结果。恢复报告明确列出新增 pin 数；恢复不会绕过冻结代码身份或赋予旧账户次日续跑资格。

研究对象计数、共享引用、配额、租约和触发器随恢复迁移，并用独立 SQL 全量重算校验。恢复结果目录内的 `restore-receipt.json` 记录旧备份 manifest SHA、恢复数据库 SHA、删除的临时索引数量及 namespace、保护对象与计数。

恢复器测试还覆盖共享 SHA、旧结果过期后的 GC、触发器后续写入、模型或压缩包损坏、保护对象缺失、路径越界和拒绝覆盖现有目录。若缺少冻结代码的精确版本，保存账本和模型仍可验收，但不能宣称账户已恢复计算能力。
