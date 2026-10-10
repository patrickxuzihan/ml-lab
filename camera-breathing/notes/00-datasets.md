# 数据集：获取方式和使用条款

本文件按 CLAUDE.md 规则 4 记录每个数据集的获取方式和条款。数字规格（大小、帧率、分辨率等）以 `results/` 里阶段 1 的核实报告为准，这里不写。

对所有数据集都适用的做法（CLAUDE.md 规则 2、3）：

- 数据集、视频、帧和从数据集提取的信号都不进 git，只放在 Colab 本地硬盘（处理完就删）和你的 Google Drive 上。
- 真人画面不保存、不进 git、不放进报告。核实脚本对真人数据集只输出数字；保存示例画面的选项只对 SCAMPS 开放，代码里对其他数据集直接报错。
- 不转发任何数据。

## SCAMPS（使用）

- **来源**：<https://github.com/danmcduff/scampsdataset>
- **获取**：官方页面有直接下载链接，不用申请。

  | 文件 | 内容 | 大小（官方页面） |
  |---|---|---|
  | `scamps_videos_example.tar.gz` | 10 段视频和标签 | 1.2 GB |
  | `scamps_videos.tar.gz` | 全部视频和标签 | 约 600 GB，单个文件 |
  | `scamps_waveforms.tar.gz` | 只有标签（.mat） | 500 MB |
  | `scamps_waveforms_csv.tar.gz` | 只有标签（.csv） | 60 MB |
  | `ScampsTrainTestSplit.csv` | 官方的训练、验证、测试划分 | 未写 |

  链接的前缀都是 `https://facesyntheticspubwedata.z6.web.core.windows.net/neurips-2022/`。云端会话的网络白名单拦了这个域名，所以只能在 Colab 里下载；实际大小由阶段 1 脚本用 HTTP HEAD 请求记录。
- **许可**：R-UDA v1.0（Research Use of Data Agreement），我读过仓库里的 `LICENSE.txt`。要点：
  - 只能用于非商业研究的计算分析；数据和用它得到的成果都不能用于商业产品或服务。
  - 不能因为使用或转发数据收钱。
  - 转发时要保留原有的署名信息，并让接收方同样受 R-UDA 约束。我们不转发。
  - 「成果」可以包含报告研究所必需的极少量数据，例如论文图里的画面。所以图里可以放一两帧合成画面。
  - 用数据训练出的模型，只要不包含超过极少量的数据，也算「成果」。
  - 官方页面另外说明：这个数据集不是用来评估临床效果的测试集。
- **引用**：McDuff et al., *SCAMPS: Synthetics for Camera Measurement of Physiological Signals*, NeurIPS 2022 Datasets and Benchmarks Track.

## UBFC-rPPG（使用）

- **来源**：<https://sites.google.com/view/ybenezeth/ubfcrppg>
- **获取**：官方页面给出 Google Drive 共享文件夹，不用申请。做法见 GUIDE 第 7 节：添加快捷方式到「我的云端硬盘」，Colab 挂载 Drive 后读取。
- **条款**：检索摘要显示官方页面写的是数据「shared for research purpose」。云端会话打不开这个页面，所以没有核对原文。
  - TODO（你来补）：把页面上的使用条款或引用要求原文抄到这里。
- **引用**：Bobbia et al., *Unsupervised skin tissue segmentation for remote photoplethysmography*, Pattern Recognition Letters 124:82–90, 2019, doi:10.1016/j.patrec.2017.10.017。

## PURE（暂不纳入）

- **获取**：用学校邮箱发邮件给 nikr-datasets-request@tu-ilmenau.de 申请，gmail 邮箱不受理。已发出，在等回复。
- **决定**：暂不纳入。核心实验用 SCAMPS 和 UBFC-rPPG 就能完成（GUIDE 第 9 节）。拿到后再决定是否加入，届时再写读取器。
- **条款**：TODO，拿到后把协议要点记在这里。

## COHFACE（暂不申请）

- **获取**：Zenodo 上受限访问（record 4081054），约 255 GB；签协议的人必须是正式在职人员。
- **决定**：暂不申请。以后如果有条件申请，再按 GUIDE 第 4.1 节做真呼吸带验证和「再压缩一次」实验。
