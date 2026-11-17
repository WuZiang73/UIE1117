# 四个累计版本的代码备份位置

四个版本全部通过实际模型构造、两初值模型SHA、源码/初值/配置校验；第一份与第二份为独立文件副本。当前神经网络和训练数学已经冻结，后续外部评测工具作为独立补充归档，不覆盖已冻结条目。

| 累计版本 | 本实验内完整代码备份 | 根外第二份独立备份 |
|---|---|---|
| Baseline | `/data1/wza/wwe/four_stage_all_datasets_20261116/code_backups/baseline/` | `/data1/wza/wwe/four_stage_code_backup_20261116/baseline/` |
| Baseline+C0 | `/data1/wza/wwe/four_stage_all_datasets_20261116/code_backups/c0/` | `/data1/wza/wwe/four_stage_code_backup_20261116/c0/` |
| Baseline+C0+ISP | `/data1/wza/wwe/four_stage_all_datasets_20261116/code_backups/isp_control/` | `/data1/wza/wwe/four_stage_code_backup_20261116/isp_control/` |
| Baseline+C0+ISP+最新DINO交互 | `/data1/wza/wwe/four_stage_all_datasets_20261116/code_backups/dino_cross_stage/` | `/data1/wza/wwe/four_stage_code_backup_20261116/dino_cross_stage/` |

每版的`STAGE.json`给出实际mode；`source/`为完整神经网络及依赖代码，`tools/`有构造、训练、断点恢复和当前评测入口，`initializations/`有native7/43，`teachers/`有本地冻结DINO权重，PLAN/CONFIG_LOCK/MANIFEST固定配方与哈希。仅DINO版本前向实际调用教师。数据来自父实验已独立复制并哈希的共同快照，源数据不被修改。

实际构造入口为`tools/frozen_factory.py:create_model(mode,7或43)`。Baseline是已接受的480995参数版本；原文0.734M WWE作为论文参考另列，不擅自替换当前baseline。

证明：本实验`CODE_BACKUP_AUDIT.json`、`SECOND_CODE_BACKUP_AUDIT.json`以及第二份根目录`BACKUP_AUDIT.json`。14份已完成实验的原来源及126权重/140状态实际加载证明见`IMPORTED_REUSE_AUDIT.json`，均明确标记复用、不计新增训练。

正式队列和结果仅写本实验根目录，勿从代码备份启动另一个抢卡器。主Excel为`four_stage_paper_comparison.xlsx`。所有数据集训练、Table2外部指标及最终全量权重归档仍需依完成审计判断，不因代码备份完成就宣称整个目标完成。

## 包含完整外部评价器的最新版备份

2026-11-16追加独立版本：`/data1/wza/wwe/four_stage_code_backup_complete_20261116/`，内含baseline、c0、isp_control、dino_cross_stage四目录。每份还收录已实际strict加载验证的URanker权重、NIQE参数、固定色卡参考/取样标注、外部指标代码和环境清单，另有MANIFEST。证明见本实验`COMPLETE_CODE_BACKUP_AUDIT.json`。旧两套冻结备份及其MANIFEST保持原样。

NR指标中作者Python与原文旧MATLAB、色卡ROI协议的差异已经在Excel行旁注明，不能将不同协议的已发表参考分数当成本机重跑结果。

## 2026-11-16追加的版本化评价与审计材料

最新版完整备份每个阶段下，`evaluation_v2/`保存统一pad-to8/crop的外部评价器，`archive_tools_v2/`保存完整归档审计工具；均有独立清单，恢复到新工作副本时按版本叠加，不修改原封存副本。

`report_audit_v1/`新增一手来源指标口径核查、独立结果审计脚本和一次部分完成快照。它核查现有15条完成轨迹/60份配对评价及42份外部评价，不能作为40条训练或最终归档已经完成的证明。完整性证据见`checks/REPORT_EVIDENCE_BACKUP_AUDIT.json`。

`excel_audit_v1/`另存独立Excel审计脚本及一份已验证快照：20条论文行/200个数值、当前两初值均值、逐图记录、曲线及完成轨迹权重SHA。它与导出器的回读检查相互独立，只证明快照内容，不宣称剩余训练或最终归档完成。

`rolling_backup_v1/`保存CPU滚动权重备份辅助程序与一份部分完成快照。完整位置与范围见`COMPLETED_RUN_BACKUP_INDEX.md`；真正全量归档完成仍按最终审计判断。

`profiled_resource_v1/`保存当前全局协调器工具、实测显存准入政策与进程交接证据。用于在原协调器备份的新工作副本上恢复资源层；四个阶段的神经网络、训练及评价公式继续使用原冻结版本。详情见`GPU_ADMISSION_REVIEW.md`。

`dark_data_compat_v1/`补充EUVP-Dark的大写JPEG/`.jpeg`读取兼容入口、协议锁、实际11100张input/GT解码及CPU枚举/像素等价证据、已保留8次零步失败的修复审计，以及当前协调器和Excel溯源代码。Dark必须按`DARK_DATA_COMPATIBILITY.md`使用专用入口；原始61源码/训练数学及32完成轨迹保持原样。每份补充均有独立SHA清单，旧封存清单未改。

`seed7_policy_v1/`保存用户后续统一初值7的协议、36项有效范围、报告/封存/观察器辅助代码与当前协调器快照。原40项PLAN和已完成43结果仍保留；Dark43四个未开始任务退排。详情见`SEED7_POLICY.md`，恢复时在新工作副本上依版本叠加；旧清单不改。
