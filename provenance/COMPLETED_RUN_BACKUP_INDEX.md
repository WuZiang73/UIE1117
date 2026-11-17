# 已完成轨迹的滚动独立备份

位置：`/data1/wza/wwe/four_stage_completed_runs_backup_20261116/`。

每个完成200轮并写入完整Val/Test结果的轨迹，按`runs/<dataset>/seed<7或43>/<mode>/`复制。每份包含9个权重文件、最后完整Raw/EMA/AdamW/scheduler/所有随机状态、逐轮曲线、逐步输入/GT/DropPath轨迹、Val选轮和逐图评测记录。复用轨迹的原始来源标记同时保留。

`BACKUP_INDEX.json`记录实际已备份条数及各份清单。每份`MANIFEST.json`验证SHA与独立inode；`MODEL_LOAD_AUDIT.json`记录从该副本实际strict加载9个文件/10个恢复模型状态，构造代码来自四阶段完整代码备份。只有复制和实际加载均通过，才计入已备份数。冻结DINO教师等完整代码资产仍在`four_stage_code_backup_complete_20261116/`各阶段中。

CPU辅助程序：`tools/backup_completed_runs.py --watch`，每20分钟扫描新完成轨迹。句柄见`LATEST_COMPLETED_BACKUP_LAUNCH.json`，动态摘要见`LATEST_COMPLETED_RUN_BACKUP_STATUS.json`；日志在备份根目录`logs/completed_backups.log`。程序仅处理完成轨迹，不分配GPU或改变训练与指标。写入活动状态的文件使用既有归档排除前缀，避免干扰最终封存。

滚动备份是训练过程中的权重保障，不等于全任务完成。最终仍需40条轨迹、160份配对评价、48份外部评价、全表Excel和包含数据/代码/教师/全部权重的独立完整归档：`/data1/wza/wwe/four_stage_all_datasets_backup_20261116/`。

## 已封存的全部外部结果

`external_results/`已封存三个外部集合的48份V2评测、1792条逐图记录、固定评价器锁及指标口径说明、四版本成本测量和独立参数核算。16份UIEB Val-best源权重位于同一备份根目录的`runs/UIEB/`，八条UIEB轨迹共80模型状态已从副本实际加载；每个外部分数都核对其对应权重SHA。

该目录的`paper_comparison_external_complete_paired17.xlsx`是外部评测完成时的独立Excel快照：Table2完整，五个配对数据集仍仅17/40条完成。它不替代实验根目录中持续更新的总Excel。证据为该目录`EXTERNAL_COMPLETION_AUDIT.json`与`MANIFEST.json`，全目标仍保持未完成。
