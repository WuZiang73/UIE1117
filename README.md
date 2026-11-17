# UIEB初值7：四阶段代码与EMA权重存档

这份独立副本对应同一组UIEB消融：**Baseline → C0 → 原始ISP正确对齐 → DINO跨阶段交互**。四个恢复模型均完成200轮训练，使用各自完整Val选出的EMA最佳权重，在同一89张Test上评测。

| 阶段 | `--mode` | EMA最佳Val轮数 | Test PSNR/dB | Test SSIM | 恢复网参数 |
|---|---|---:|---:|---:|---:|
| Baseline | `baseline` | 142 | 24.3583693662 | 0.9148829794 | 480995 |
| 第一个点C0 | `c0` | 137 | 24.5012224396 | 0.9156007693 | 485143 |
| 第二个点ISP | `isp_control` | 149 | 24.5960890838 | 0.9183634037 | 485143 |
| 第三个点DINO | `dino_cross_stage` | 117 | 24.6241080637 | 0.9191473126 | 645778 |

相邻PSNR增益约为+0.14285、+0.09487、+0.02802 dB。本表仅描述**UIEB、模型初值7、EMA**。初值7是在查看UIEB Test后按用户要求选定的后续默认值。Raw最后一步没有提升；UIEB两个初值的EMA均值中，DINO相对ISP为−0.014712439 dB。已有其他初值与负结果保留在`historical_context/`，不能把这份单初值链写成跨种子稳定结论。

## 文件位置

```text
checkpoints/                    四个完整恢复模型的EMA state_dict
models/{baseline,c0,isp_control,dino_cross_stage}.py
                                四个明确模型入口，调用原始构造器
source/                         全部61份模型与依赖源码，逐字节保留
source/analysis/                 C0固定上下文资产，一并保留
source/third_party/dinov2_vendor/ 本地DINOv2实现
tools/                          原实验全套Python工具，含不变的训练数学
initializations/                native_seed7/43，原始未训练初值
teachers/vitb14.pth              独立冻结DINOv2教师，约346.4 MB
results/<mode>/                 原始训练配置、曲线、Val/Test逐图结果与选轮记录
historical_context/             UIEB43和Raw记录，保留结果边界
references/                     UIEB训练输入/GT/DropPath及ISP历史轨迹
MODELS.json                     四个入口、权重、参数量、指标与SHA映射
PLAN.json                       原始完整实验计划，作为历史配方保存
EXPORT_SCOPE.json               本包实际范围：UIEB/seed7/EMA四模型
EXPORT_MANIFEST.json             本包文件SHA256与大小，独立于原始大实验锁
validation/                     原环境CPU前向输出参考
runtime.py / infer.py            独立加载和RGB256单图推理
evaluate.py                     同口径UIEB完整Test复测
retrain.py                      调用原训练函数重新训练UIEB，输出到新runs/
verify_package.py               哈希、严格加载及CPU前向复验
```

四份`checkpoints/*.pth`均为**完整模型**，不能把后面几个当增量补丁叠加到前一个模型，也不能只凭张量形状把ISP权重加载到C0。二者参数量相同，但前向方法不同。

## 模型具体对应

统一构造入口为`tools/frozen_factory.py:create_model(mode,7,device)`。`models/`的四个入口只封装该工厂，不另写网络。

| mode | 原始定义 | 必须保留的构造设置 |
|---|---|---|
| baseline | `source/baseline_model.py:BaselineUNet` | `variant=None, semantic=None` |
| c0 | 同一`BaselineUNet`，以及`dark_context_v2.py`、`dark_modules_v2.py` | `variant='M2v2R', semantic=None` |
| isp_control | `source/optical_regional_isp.py:RegionalISPUNet` | `variant='M2v2R', regional_isp_mode='diagonal'` |
| dino_cross_stage | `source/rms_direction_interaction.py:RMSDirectionUNet`，继承原ISP | `variant='M2v2R', relation_mode='stage_disagreement'` |

最后一项在**RMSDirectionUNet这份实现**中表示居中的DINO关系加RMS幅度控制，不能换成`relation_variants.py`里同名字的另一种组合。其依赖包括`factorized_cross_stage.py`、`dino_stage_interaction.py`、`relation_variants.py`及本地DINOv2源码，全部已复制。

DINO教师在训练和推理时都需要。教师不包含在645778参数的恢复网权重内；冻结教师另有86580480参数，总计87226258参数。不能只报告恢复网成本而漏掉教师。

## 环境与验证

本机已核验环境为Python3.10.22、PyTorch1.13.1+cu117、CUDA11.7、cuDNN8500。`environment_requirements.txt`是原环境的完整依赖快照，不是保证可在任意机器直接一次安装成功的最小依赖文件；精确信息另存`provenance/INPUT_AUDIT.json`。

**从GitHub克隆后，先恢复教师文件。** 本地独立备份已有完整346MB教师；远端代码与四个恢复网权重放在Git中，教师完整副本另存Release（上传是否完成见`GITHUB_UPLOAD_RECEIPT.json`）。可通过下面的Release命令恢复，也可运行`python fetch_teacher.py`从固定Meta官方下载地址取回同一文件，两者都必须匹配完整SHA256。只有教师恢复后，本包完整清单和DINO前向校验才能全部通过。

```bash
gh release download uieb-seed7-four-stage-20261117 \
  --repo WuZiang73/UIE1117 --pattern vitb14.pth --dir teachers
```

在当前服务器可使用：

```bash
cd /data1/wza/wwe/uieb_seed7_four_stage_github_20261117
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/data1/wza/conda_envs/wwe3/bin/python verify_package.py --forward
```

该命令检查本包清单、四个完整state_dict严格加载，以及四个CPU前向与复制前参考结果逐元素一致。它验证搬迁正确，不代替从头重新训练。不同软硬件的数值差异应如实记录，不应改权重或删结果来凑分。

`PLAN.json`、旧`CONFIG_LOCK.json`等是原始大实验的历史记录，里面仍可见原服务器路径、全数据集和原40项计划；它们不能用来宣称本导出包含全40项训练产物。本包范围以`EXPORT_SCOPE.json`为准，完整性用新`EXPORT_MANIFEST.json`检验。

四项在本包独立完成了CPU完整Test89复测：PSNR依次为24.358199711、24.500941888、24.596347208、24.624065793，SSIM也依次递增；输入/GT哈希与原评测完全相同。CPU相对原GPU PSNR的最大绝对差异为0.000280552 dB。见`validation/FULL_CPU_TEST_AUDIT.json`。此处保留数值差，不替换原GPU结果。

## 单图推理

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python infer.py \
  --mode dino_cross_stage --input /absolute/path/input.png \
  --output /absolute/path/output.png --device cpu
```

其他三项替换`--mode`即可。为对齐本次UIEB评测，此入口将RGB图双线性缩放至256×256，输出也为256×256。落盘PNG会量化，不要用PNG重新计算来代替原始浮点Test指标。

在获得空闲授权GPU后，可显式设置`CUDA_VISIBLE_DEVICES=物理卡号`并用`--device cuda`。本包入口不会自行抢卡或停止任何正在运行的任务。

## 完整UIEB Test复测

数据默认不进入GitHub；保留固定划分和文件SHA的`DATA_MANIFEST.json`。本机的数据根目录是：

```text
/data1/wza/wwe/four_stage_all_datasets_20261116/data/UnderWaterDataset/UIEB
```

`evaluate.py`首先核对全部178个Test input/GT文件的名字、大小和SHA，然后按原batch24、RGB256、逐图指标平均执行。示例：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluate.py \
  --mode baseline \
  --data-root /absolute/path/UnderWaterDataset/UIEB \
  --output /absolute/path/recheck/baseline.json --device cpu
```

对四个mode各运行一次。程序写新文件，拒绝覆盖原结果，并报告相对存档分数的数值差。输入/GT张量哈希必须对齐；不重新划分数据，不增加翻转、集成或Test选轮。

## 按现有配方从头训练

本包保留原`tools/experiment.py`、`frozen_factory.py`、`strict_math.py`、`base_equivalent_operators.py`、`restoration_losses.py`、`delayed_ema.py`，没有改训练计算。`retrain.py`是独立迁移入口，调用原`train()`函数，不依赖原服务器的全局后台调度入口。

先在**本包**`data/UnderWaterDataset/UIEB`放入或链接已有固定数据，`source/UnderWaterDataset`已是相对链接。UIEB为712/89/89；模型初值7，训练随机流7，loader1007；batch24、200轮、6000步。Loss为Charbonnier+0.1×(1−SSIM)，AdamW lr0.002、wd1e−5，Cosine200轮至2e−7。DropPath为[0,.025,.05,.075,.1]，EMA从第2100步完成更新后开始，系数固定0.999。

```bash
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=7 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python retrain.py --mode baseline
```

只在已授权且有足够安全显存时执行。原始资源检查保留4090、计算能力8.9及本包`SCHEDULER_POLICY.json`所列设备；迁到别的机器时可单独适配**副本的资源编号**并记录，不改模型、损失、数据或种子。

入口先做独立一轮预检，再从相同初值正式训练，首轮逐步精确比对；新结果写`runs/UIEB/seed7/<mode>/`，四个存档主权重仍留在`checkpoints/`。中断后用相同命令加`--resume`恢复`last_full.pth`里的Raw/EMA/优化器/调度器及所有RNG，禁止从best_model代替续训。迁移入口的帮助和静态检查已做；本导出不额外发起200轮训练来冒充复现成功。

原工具中`activate_queue.py`、`prepare.py`、`finalize.py`等包含原服务器工作流路径，保留是为了全套代码审计；不要在克隆包中运行它们去修改原活动实验。

## 私有GitHub存放与恢复

四个恢复网权重合计约8.77 MB；DINO教师为346378731字节。GitHub普通Git禁止大于100MiB的单文件，所以需要Git LFS或独立大文件附件，不能把教师作为普通Git blob直接推送。[GitHub文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

若通过本机Git/LFS上传，必须在首次提交前执行`git lfs track '*.pth'`，并提交`.gitattributes`；从私有仓库恢复后执行`git lfs pull`，再运行本包校验。[Git LFS配置](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)。

**远端文件布局**：四个恢复网权重和小体积原生初值按普通Git二进制文件存储，教师完整副本通过Release存储，官方地址也提供同SHA的恢复方式。GitHub连接器写入返回403，已准备官方CLI设备授权；实际上传是否完成，以最终上传收据为准。因此`.gitattributes`只预留教师路径的LFS配置，不能把已用普通Git上传的小权重误标成LFS指针。

本地导出包含教师完整副本。若连接器不支持LFS/大文件上传，远端应明确记录教师采用固定官方地址和SHA恢复，四个恢复网主权重仍须实际上传；不能只上传四份权重的路径文字后声称已经备份。上传完成需另记仓库链接、实际分支、提交SHA和远端核验结果。

目标仓库为`WuZiang73/UIE1117`。用户在获知其为公开仓库后明确授权“先上传，传完再私有”；本次按此最新授权上传，不能把尚未改Private的仓库称为私有。数据目录、生成图片、重复训练产物、凭据和本机环境目录不进入仓库；完整原始训练代码、固定资产、所选四权重与结果证据均应保留。

## 来源

原WWE及扩展源码的原始注释和来源保留。DINOv2为Meta官方ViT-B/14无register版本，保留本地vendored源码与`LICENSE_DINOV2`，不为整个混合项目重新指定许可证。官方明确DINOv2代码与该系列权重采用Apache2.0，见[官方仓库](https://github.com/facebookresearch/dinov2#license)。
