# 中断训练的阶段恢复

训练日志的 JSON 事件可能与多个 worker 的普通日志粘在同一物理行。训练中心逐个解码完整 JSON 对象，原始 stdout 保留在 `training.log`，语法异常写入 `training.parser.jsonl`。有效事件仍须满足冻结人口、fold、scope、单调进度和实际设备约束；非零退出码仍为失败。

`resume-trend-training` 用于趋势、政策选择、E0 和 MA 退出头已经完成、在 E1 逐股阶段中断且尚无完整报告的模型 run。恢复适配器保持原二十五个生成源码及原配置身份，核验原始/native 来源、日历、模型 checkpoint、schema、逐股 manifest、表文件和账户来源。已完成趋势特征、趋势头、政策选择、E0 与 MA 退出头直接读取；E1 只提交没有完整 manifest 的股票。已有完整 manifest 的 SHA 或契约失效时停止，不能默默把错误缓存当作新结果。恢复适配器单独记录源码 SHA、进程锁、attempt、复用量和新增计算量，所有后续训练、诊断、费用验收与选模门槛照原管线执行。

```sh
python -m my_strategy.cli resume-trend-training --resume-training-run-id RUN_ID --verify-only
```

通过来源复用预检后，可给 `POST /api/training/jobs` 同时提交 `plan_id`、原 `parameters`、`resume_training_run_id` 和 `resume_from_job_id`。恢复仅允许完整 native 来源与 complete 阶段。CLI `run-training-job --plan-id PLAN_ID --resume-training-run-id RUN_ID` 使用相同 TaskManager、任务数据库、训练资源租约和取消流程；独立 owner 进程适用于主站正有其他任务时，不需要重启网页。

恢复不会把其他 run 的旧模型导入当前模型，也不会将当前同花顺成员加入原训练。原截止日保持不变，新日更行情不会改变这个冻结训练。阶段恢复成功意味着继续运行，最终模型是否合格仍以完整报告和费用后比较为准。
