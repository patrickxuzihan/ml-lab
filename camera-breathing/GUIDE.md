# camera-breathing 项目指导

> 这份文件写给两个读者：**你**照着它推进项目，**Claude** 在云端会话里按它写代码。每个阶段最后都有「发给 Claude 的话」，复制到 [claude.ai/code](https://claude.ai/code) 的新会话里就行。
>
> 仓库根目录的 `CLAUDE.md` 是给 Claude 的长期规则，每次会话都会自动读取。
>
> 本仓库早先的两个计划（text2sql-grpo 和 frontdesk-slm）都已放弃，内容留在 git 历史和已关闭的 PR #1 里。

---

## 1. 项目要回答什么

**一句话：做一个用摄像头测呼吸的系统，系统地研究它在视频压缩下还准不准、和测心率比谁更抗压缩，并让它在压缩下更准。**

### 为什么这样定题

- 摄像头测呼吸和心率可以用在远程问诊、居家监护、车内驾驶员监测上。这些场景里，视频在到达算法之前几乎都被压缩过：手机录像、视频通话、云端存储都会压缩。算法如果只在原始视频上有效，上线就会失灵。
- 已有工作：
  - 压缩对摄像头测心率的影响已有不少研究（McDuff 等，2017 年起），结论是压缩会明显削弱心率信号。
  - 摄像头测呼吸的方法已有综述和基准（2025 年的 ACM 综述和开源工具 resPyre），但它们用的数据集本身都是压缩过的。
  - 初步检索没有找到专门研究「压缩对测呼吸的影响」、或在同一批视频上比较心率和呼吸的工作。阶段 1 要做一次正式检索来确认。
- 心率靠的是脸上皮肤颜色随心跳的极其微弱的变化，这正是压缩最先抹掉的东西。呼吸靠的是胸口起伏，是看得见的位移，按理更容易保留；但起伏只有零点几个像素时，编码器也可能把那块当成「没变」直接丢掉。所以谁更抗压缩并不是显然的。

### 四个问题

- **Q1 压缩对测呼吸的影响**：不同编码器（H.264、H.265）和码率下，测呼吸的误差怎么变？从哪一档开始失灵？
- **Q2 呼吸和心率谁更抗压缩，为什么**：在同一批视频上比较。为了分清原因是「信号类型」（运动还是颜色）还是「频率快慢」，用三条测量（第 4.2 节）：胸部运动测呼吸、脸部颜色测呼吸、脸部颜色测心率。
- **Q3 码率怎么花**：同样的码率下，帧率、分辨率和每帧画质怎么取舍，对测呼吸和测心率最好？呼吸很慢，理论上每秒一两帧就够，所以「少帧但清楚」可能比「多帧但模糊」更适合测呼吸。这对应网页视频通话（WebRTC）里一个现成的设置 `degradationPreference`：网络变差时，优先降帧率还是降画质。
- **Q4 能不能补救**：针对最容易失灵的情况做一种改进，比如训练时加入压缩过的视频，或者按码率自动选择测量方法。

### 范围和边界

- 只用公开数据集，外加可选的、只拍你自己的少量视频。不招募其他人，不收集别人的视频。
- 这是测量方法的研究，不做医疗诊断。
- 讯飞实习的代码、数据和参数一律不用，全部从零写，只参考公开文献。

---

## 2. 分工和工作流

| | Claude（claude.ai/code 云端会话） | 你 |
|---|---|---|
| 机器 | 只有 CPU：4 核、16GB 内存、30GB 硬盘，没有 GPU，也下载不了数据集 | Colab Pro（CPU 和 GPU 运行时，自带一块临时的大硬盘）+ Google Drive |
| 负责 | 写代码和测试、用程序生成的小视频做 CPU 测试、分析结果、画图、写文档 | 申请和下载数据集（都在 Colab 里完成）；在 Colab 上跑实验；把结果推回仓库；审 PR 并合并 |

你自己的电脑不跑任何东西，数据集也不经过它，直接下载到 Colab（第 7 节）。

每个阶段按这个顺序走：

1. 把「发给 Claude 的话」发到一个新会话。
2. Claude 新建分支，写代码、跑测试，然后开 PR。PR 里会写清楚你下一步要在 Colab 上运行的命令。
3. 你审 PR，看不懂的地方直接在会话里问，然后合并。
4. 如果这个阶段要跑实验，你就在 Colab 上跑命令，把 `results/` 提交并推送。
5. 开一个新会话，让 Claude 分析结果，再进入下一阶段。

几个建议：

- **先看计划，再让它写代码。** 大一点的阶段，先让 Claude 只给实现计划，你看过没问题再让它动手。
- **一个会话只做一件事。** 会话太长就开新的，并告诉它看 GUIDE.md 的哪一节。
- **每个 PR 你都要能讲清楚。** 面试官一定会追问细节（见第 10 节）。每个 PR 末尾都有中文的「学习要点」，要认真看，不懂就问。

---

## 3. 技术选型

| 项目 | 选择 | 说明 |
|---|---|---|
| 视频压缩 | ffmpeg（libx264、libx265；可选 VP9、AV1） | 所有压缩版本都由脚本生成，参数写进配置 |
| 视频读取 | PyAV 或 OpenCV | |
| 区域定位 | MediaPipe：用人体姿态找肩膀和胸部，用人脸关键点找脸 | 阶段 1 先确认画面里能看到多少胸部 |
| 运动信号 | 光流（OpenCV）或区域内的亮度 | 自己实现 |
| 颜色信号 | 不需要训练的经典方法（POS、CHROM、绿色通道） | 自己用 numpy 实现 |
| 频率估计 | 带通滤波 + 频谱（Welch 方法）+ 峰值插值 | |
| 对照实现 | resPyre（测呼吸）、rPPG-Toolbox（测心率） | 只用来核对我们的结果；用之前先看许可证 |
| 深度模型（阶段 5，可选） | 小型时序网络，PyTorch | 只有这一阶段用 GPU |
| 演示（可选） | 浏览器纯前端（MediaPipe 网页版 + 简单的运动分析），放在 GitHub Pages | 画面不离开用户的设备 |
| 实验记录 | 仓库里的 `results/` | |

---

## 4. 实验设计

### 4.1 数据

| 数据集 | 内容 | 视频 | 呼吸真值 | 心率真值 | 获取方式 | 大小 | 用途 |
|---|---|---|---|---|---|---|---|
| SCAMPS | 合成人像，2,800 段，每段约 20 秒 | 未压缩 | 有，精确 | 有，精确 | 官方 GitHub 页面有直接下载链接，不用申请；研究许可（R-UDA），不能商用 | 10 段样例 1.2 GB；全集约 600 GB（单个 tar.gz）；只有标签的 csv 60 MB | 受控压缩实验的主力 |
| UBFC-rPPG | 真人，42 段，每段 1–2 分钟 | 未压缩，640×480，30 帧/秒 | 没有，从指夹 PPG 推出近似值 | 指夹血氧仪 | Google Drive 共享文件夹，不用申请 | 约 70–140 GB（估计） | 真人、未压缩的压缩实验 |
| PURE | 真人，10 人 × 6 段，每段约 1 分钟，含头部运动 | 无损 PNG，640×480，30 帧/秒 | 同上 | 指夹血氧仪 | 用学校邮箱发邮件给 nikr-datasets-request@tu-ilmenau.de 申请，gmail 不受理；已发出，等回复。**暂不纳入**，拿到后再决定 | 几十 GB（估计） | 同上，外加头动的情况 |
| COHFACE | 真人，40 人 × 4 段，每段 1 分钟 | 已经高度压缩（二手资料说约 0.25 Mb/s） | 呼吸带 | 接触式 PPG | Zenodo 上受限访问（record 4081054），签协议的人必须是正式在职人员。**暂不申请** | 约 255 GB | 真人加真呼吸带的验证；「再压缩一次」实验 |
| 自录（可选） | 只拍你自己 | 手机最高画质 | 手机加速度计（phyphox） | 可选 | — | 小 | 单人案例、演示的测试 |

SCAMPS 和 COHFACE 的大小来自官方页面；其余是按分辨率、帧率和时长估算的，阶段 1 实测。获取方式和许可条款的细节记在 `notes/00-datasets.md`。

- **压缩实验只从未压缩（或无损）的源开始**：SCAMPS、UBFC-rPPG、PURE。
- **UBFC-rPPG 和 PURE 没有呼吸带。** 呼吸参考从指夹 PPG 里推：PPG 的基线、幅度和心跳间隔都会随呼吸起伏。三种推法结果一致的时间窗才用，并在局限性里写明。
- **COHFACE 如果拿到**：在真人和真呼吸带上验证我们的方法，并和 resPyre 的已发表结果对照；再做「已经压缩过的视频又经过一次视频通话」的实验。
- **阶段 1 必须先确认两件事**：画面里能看到多少胸部和肩部；SCAMPS 里的呼吸是否表现为身体起伏。看不到胸部，就改用肩部或头部的运动测呼吸，并在结论里写明。
- **任何真人画面都不进 git，也不放进公开报告。**

### 4.2 三条测量

| 编号 | 测什么 | 用什么信号 | 频段 |
|---|---|---|---|
| M1 | 呼吸 | 胸部和肩部区域的运动（光流的上下分量，或区域亮度） | 0.1–0.7 Hz（每分钟 6–42 次） |
| M2 | 呼吸 | 脸部颜色信号（POS）里随呼吸的起伏 | 同上 |
| M3 | 心率 | 脸部颜色信号（POS） | 0.7–3 Hz（每分钟 42–180 次） |

- M1 和 M2 测的是同一个量，用的是不同类型的信号；M2 和 M3 用的是同一类信号，测的是不同频段。三者一比，就能分清压缩伤害的是「颜色信号」还是「快的信号」。
- 频率估计：带通滤波后算频谱，取频段内的最高峰，再做峰值插值。呼吸用整段或 30 秒的窗口（SCAMPS 每段只有约 20 秒，就用整段），心率用 10 秒的窗口。具体参数在阶段 2 定稿，写进评测协议。

### 4.3 压缩实验

- **压缩阶梯**（初步设定，阶段 2 定稿）：
  - 编码器：H.264（libx264）、H.265（libx265）；
  - 两种码率控制：按质量（CRF 0、18、23、28、33、38、43）和按码率（2000、1000、500、250、125 kbps，模拟视频通话）；
  - 视频通话式的低延迟设置：不用 B 帧、零延迟调优、固定关键帧间隔。
- **码率分配实验（Q3）**：在 500、250、125 kbps 等几档固定码率下，组合不同的帧率（30、15、10、5）和分辨率（原始、1/2、1/4），比较三条测量的误差。
- 压缩后的视频只是中间产物：生成 → 提取信号 → 删除。只保存提取出的信号和结果。

### 4.4 统一评测协议（所有实验共用，开跑后不能改）

- **固定的内容**：区域定位方法、信号提取方法、滤波频段、窗口长度、频率估计方法、压缩参数、指标代码。这些都写在代码和配置里；改了就要重跑受影响的实验，并在 notes 里记录。
- **指标**：
  - 误差：MAE、RMSE（单位：次/分）、MAPE、Pearson 相关系数；
  - 成功率：误差在容差以内的比例。呼吸和心率各自的容差，在阶段 2 参考已发表的工作来定；
  - 频谱信噪比：真值频率附近的能量与频段内其余能量之比（dB）。它不依赖单位，用来公平地比较呼吸和心率；
  - 失灵点：成功率比未压缩时下降超过约定幅度的最高码率。具体定义在阶段 2 定。
- **统计**：bootstrap 95% 置信区间，按受试者重采样（同一个人的几段视频并不独立）。两种设置之间，在同一批视频上配对比较。
- **自检**：用程序生成的合成视频（已知呼吸和心跳频率）跑一遍全流程，误差应该接近 0；在未压缩的源上，结果要和已发表的结果大致相当。

### 4.5 改进（Q4：只做一种，做透）

根据阶段 4 的发现，二选一：

- **经典方法**：按码率或画质，自适应地选择区域、信号或窗口长度；
- **深度模型**：在 SCAMPS 上训练一个小模型，训练时混入不同码率的压缩版本（压缩增强），比较有无增强时各档码率下的误差，再到 UBFC-rPPG 或 PURE 上测泛化。要同时报告未压缩时的表现有没有变差。

### 4.6 可选扩展（时间够再做）

- 浏览器实时演示：纯前端，放在 GitHub Pages。
- 自录单人数据：手机最高画质，用 phyphox 记录加速度计作参考。
- 压缩域方法：直接用编码器里的运动矢量估呼吸，不需要完整解码视频。
- 加入音频：呼吸声和画面融合。
- 更多编码器：VP9、AV1。

---

## 5. 仓库结构

```
ml-lab/
├── .gitignore
├── CLAUDE.md                     # 给 Claude 的长期规则
├── README.md                     # 仓库首页：项目索引
└── camera-breathing/
    ├── GUIDE.md                  # 本文件
    ├── README.md                 # 项目首页：结论、图表、复现方法（阶段 6 写完整）
    ├── pyproject.toml            # 包配置，以及 ruff 和 pytest 的设置
    ├── requirements.txt          # 主要依赖，不含 torch
    ├── requirements-dev.txt      # 测试和检查工具（pytest、ruff）
    ├── requirements-train.txt    # 阶段 5 的深度模型（torch）
    ├── configs/                  # 每个实验一个配置文件
    ├── scripts/                  # 下载数据（在 Colab 上跑）、生成压缩版本、跑实验
    ├── notebooks/                # Colab 启动器，只调用 scripts/，不放逻辑
    ├── src/breathing/
    │   ├── synth.py              # 生成合成测试视频（已知呼吸和心跳频率）
    │   ├── video_io.py           # 读视频、调用 ffmpeg 压缩
    │   ├── roi.py                # 找胸部、肩部和脸部区域
    │   ├── signals.py            # 运动信号、颜色信号（POS 等）
    │   ├── rate.py               # 滤波、频谱、频率估计
    │   ├── reference.py          # 真值处理：呼吸带、从 PPG 推呼吸
    │   ├── metrics.py            # 误差、成功率、信噪比、置信区间
    │   ├── runinfo.py            # 运行记录：配置、git 版本、包版本、硬件；断点续跑的进度日志
    │   ├── datacheck.py          # 阶段 1 的数据核实
    │   └── datasets/             # 各数据集的读取器
    ├── analysis/                 # 画图、误差分析
    ├── notes/                    # 每个阶段的分析笔记（中文）
    ├── tests/                    # 单元测试（只用程序生成的合成视频）
    └── results/<run_id>/         # 每次运行的配置、逐段结果、指标
```

数据集、视频、帧、从数据集提取出的信号和模型权重都不进 git。

---

## 6. 分阶段任务

### 阶段 0：准备（约 1 天）

Claude 要做（已在这个 PR 里完成）：重写 GUIDE.md、CLAUDE.md 和 README，建立 `camera-breathing/` 目录，更新 `.gitignore`。

你要做：

1. 审并合并这个 PR。
2. 仓库改名：仓库页面 → **Settings** → **General** → **Repository name** 改成 `ml-lab` → **Rename**。旧链接会自动跳转。
3. 申请数据：在 Idiap 网站注册并申请 COHFACE；给 PURE 的作者发邮件申请。都写明你是学生、用于个人学习研究、不转发数据。
4. 打开 UBFC-rPPG 和 SCAMPS 的官方页面，看清楚下载方式：直接链接、Google Drive 共享，还是要填表。
5. Colab：在 Secrets 里存一个 GitHub fine-grained token（名字叫 `GH_TOKEN`，只给这个仓库的 Contents 读写权限）；在 Google Drive 里建一个放信号和结果的文件夹。

验收：

- [ ] 目录重组完成：GUIDE 在 `camera-breathing/` 下，根目录 README 是项目索引
- [ ] 仓库已改名为 `ml-lab`
- [ ] COHFACE 和 PURE 的申请已发出
- [ ] 知道 UBFC-rPPG 和 SCAMPS 怎么下载

发给 Claude 的话：

> 读一下 CLAUDE.md 和 camera-breathing/GUIDE.md，用你自己的话总结这个项目的目标，以及阶段 1 要做什么。这次不要改任何文件。

### 阶段 1：文献检索和数据核实（Claude 写脚本，你在 Colab 上跑，约 3–5 天）

Claude 要做：

1. 正式文献检索：有没有人研究过压缩对摄像头测呼吸的影响，有没有人在同一批视频上比较过心率和呼吸。结果写进 `notes/01-literature.md`。如果已经有人做过，提出怎么调整研究问题。
2. 每个数据集的下载和读取脚本，在 Colab 上运行。按视频分批下载，核实完的视频可以删掉。
3. 数据核实脚本：实际大小、帧率、分辨率、编码方式（ffprobe）、真值的采样率和长度；画面里胸部和肩部能看到多少（用姿态估计统计可见比例，只输出数字，不保存真人图像）。
4. 用程序生成的合成视频写单元测试。

你要做（Colab）：跑下载和核实脚本，把核实报告（只有文字和数字）推回仓库。

验收：

- [ ] `notes/01-literature.md` 写完，结论明确
- [ ] 每个拿到的数据集都有实测的规格表
- [ ] 确定了测呼吸用哪个区域（胸部、肩部还是头部）
- [ ] `pytest`、`ruff check` 和 `ruff format --check` 通过

发给 Claude 的话：

> 阅读 CLAUDE.md 和 camera-breathing/GUIDE.md，完成「阶段 1」。先给我实现计划，我确认后再写代码。完成后开 PR，PR 描述里逐条对照验收清单，写清楚我在 Colab 上要运行的命令，最后写「学习要点」。

### 阶段 2：测量流程（Claude，只用 CPU，约 5–7 天）

Claude 要做：

1. 合成测试视频生成器：已知频率的位移（模拟呼吸）和颜色变化（模拟心跳），可以加噪声，可以调帧率和分辨率。
2. M1、M2、M3 三条测量；频率估计；从 PPG 推出呼吸参考。
3. 压缩模块：调用 ffmpeg，生成第 4.3 节的各种版本。
4. 第 4.4 节的全部指标、bootstrap 置信区间、配对比较。
5. 定稿评测协议：窗口长度、容差、失灵点的定义，写进本文件第 4.4 节，并在 PR 里单独说明这次改动。

验收：

- [ ] `pytest`、`ruff check` 和 `ruff format --check` 通过
- [ ] 在合成视频上，三条测量的误差都接近 0
- [ ] 在合成视频上跑一遍小规模的压缩实验，流程能跑通
- [ ] 核心模块（信号、频率估计、指标）不依赖 torch

发给 Claude 的话：

> 完成 GUIDE.md「阶段 2」里 Claude 负责的部分。先给我实现计划，我确认后再写代码。完成后开 PR。

### 阶段 3：未压缩时的基线（Claude 写，你在 Colab 上跑，约 2–3 天）

Claude 要做：实验脚本，以及 `notebooks/colab_runner.ipynb`（只负责挂载 Drive、拉代码、装环境、调用脚本、推送结果）。

你要做（Colab）：在 SCAMPS 子集、UBFC-rPPG 以及拿到的其他数据集上跑 M1–M3，推送结果。

验收：

- [ ] 每个数据集上三条测量的基线结果，带置信区间
- [ ] 和已发表的结果（resPyre、rPPG-Toolbox 等）对照；明显更差就先排查清楚
- [ ] 从 PPG 推出的呼吸参考的可靠性分析：可用的时间窗占多少，三种推法有多一致

发给 Claude 的话：

- 写代码：
  > 完成 GUIDE.md「阶段 3」里 Claude 负责的部分，开 PR，写清楚我在 Colab 上要依次运行的命令。
- 分析结果：
  > 基线结果已经推送到 results/。请检查结果是否合理，和已发表结果对照，写进 notes/03-baseline.md。

### 阶段 4：压缩实验（约 1–2 周）

Claude 要做：压缩阶梯和码率分配实验的配置和脚本。脚本按视频记录进度，Colab 断线后能接着跑。

你要做（Colab，用 CPU 运行时就行）：跑实验，推送结果。

验收：

- [ ] 三条测量的误差、成功率、频谱信噪比随码率变化的曲线，带置信区间
- [ ] Q1、Q2、Q3 各有一段有数据支撑的结论，写进 `notes/04-compression.md`；不显著就写不显著

发给 Claude 的话：

- 写代码：
  > 完成「阶段 4」里 Claude 负责的部分，开 PR，写清楚我在 Colab 上要依次运行的命令。
- 分析结果：
  > 压缩实验的结果已经推送。请分析三条测量随码率的变化、失灵点和码率分配的结果，回答 Q1–Q3，写进 notes/04-compression.md，并建议阶段 5 做哪种改进。

### 阶段 5：改进（约 1–2 周）

Claude 要做：按第 4.5 节二选一。先在 PR 里说明选哪个、为什么，再实现。深度模型要附带 CPU 冒烟测试（极小的模型、几段合成视频、跑几步）。

你要做（Colab；深度模型要用 GPU 运行时）：跑实验，推送结果。

验收：

- [ ] 改进前后在各档码率下的对比，包括未压缩时的表现
- [ ] `notes/05-improvement.md` 记下结果和观察，失败的尝试也要记

发给 Claude 的话：

> 完成「阶段 5」里 Claude 负责的部分。先告诉我你建议做哪种改进、为什么，我确认后再写代码。完成后开 PR。

### 阶段 6：写作和演示（约 1 周）

Claude 要做：

1. 结果总表和图：误差随码率变化的曲线（呼吸对比心率）、码率分配的热力图、各条测量的失灵点。
2. 项目 README，顺序是：一句话结论 → 结果图表 → 方法 → 发现 → 局限性 → 复现方法。局限性至少写这几条：SCAMPS 是合成数据；真人数据的呼吸参考是从 PPG 推出来的（如果没拿到 COHFACE）；受试者数量有限；压缩是用 ffmpeg 模拟的，不是真实的视频通话。
3. 简历 bullet 的中英文草稿，所有数字都来自 `results/`。
4. （可选）浏览器实时演示。

验收：

- [ ] README 里每个数字都能在 `results/` 里找到出处
- [ ] 结论的措辞和置信区间一致，不显著就写不显著

发给 Claude 的话：

> 所有实验结果都在 results/ 里了。请完成「阶段 6」：图表、项目 README 和简历 bullet 草稿，开 PR。

---

## 7. 数据下载和 Colab 操作手册

### 数据集放在哪里

- **不能放进 GitHub**：许可证不允许转发，GitHub 也不收超过 100 MB 的文件。仓库里只放代码、配置和结果。
- **不用经过你的电脑**，直接下载到 Colab：
  - **SCAMPS**：官方 GitHub 页面有直接下载链接，不用申请，脚本在 Colab 里直接下载。全集是单个约 600 GB 的 tar.gz，不能按视频单独下载，所以脚本边下载边解压，一次只把一段视频放到硬盘上，处理完就删。阶段 1 只用 1.2 GB 的 10 段样例和 60 MB 的标签 csv。
  - **UBFC-rPPG**：Google Drive 共享文件夹，不用申请。在浏览器（手机也行）里打开共享链接，在文件夹上右键 →「整理」→「添加快捷方式」，放到「我的云端硬盘」根目录。Colab 挂载 Drive 后，它就在 `/content/drive/MyDrive/<文件夹名>`。别人共享的文件不占你的 Drive 空间。脚本一段一段拷到 Colab 本地，处理完就删。
  - **PURE**：用学校邮箱发邮件给 nikr-datasets-request@tu-ilmenau.de 申请，gmail 不受理。已发出，在等回复。暂不纳入；拿到后按对方给的下载方式再定。
  - **COHFACE**：在 Zenodo 上受限访问（record 4081054），约 255 GB；签协议的人必须是正式在职人员，所以暂不申请。
- **按视频分批处理**：Colab 的本地硬盘大、速度快，但运行时一断开就清空。所以下载一段 → 提取信号 → 删除视频，只把信号和结果存进 Drive。
- **Drive 空间**：免费的 15 GB 存信号和结果足够。如果想在 Drive 里缓存数据集的子集，空间可能不够，需要买 Google One，或者只缓存很小的子集。

### 每次开 Colab 的固定流程

具体的单元格在阶段 3 的 `notebooks/colab_runner.ipynb` 里提供。

1. 选运行时：信号提取和压缩实验用 CPU 运行时，消耗的计算单元少；只有阶段 5 的深度模型用 GPU。
2. 挂载 Google Drive，拉代码（第一次 clone，之后 pull），安装依赖。
3. 运行 PR 里给出的命令。
4. 用 Secrets 里的 `GH_TOKEN` 把 `results/` 提交并推送。

Colab Pro 没有后台运行：跑实验时浏览器页面要开着，电脑不能休眠（电脑本身不参与计算）。长任务按视频分批，断了就接着跑。

---

## 8. 时间和预算

按每周投入 15 小时左右估算：

| 阶段 | 时间 | 算力 |
|---|---|---|
| 0 准备 | 1 天 | — |
| 1 文献检索和数据核实 | 3–5 天 | Colab CPU，少量 |
| 2 测量流程 | 5–7 天 | — |
| 3 未压缩基线 | 2–3 天 | Colab CPU |
| 4 压缩实验 | 1–2 周 | Colab CPU（主要消耗） |
| 5 改进 | 1–2 周 | 深度模型用 Colab GPU |
| 6 写作和演示 | 约 1 周 | — |
| **合计** | **约 6–8 周** | 大部分是 CPU 计算。阶段 3 实测速度后再估算计算单元 |

---

## 9. 风险和应对

| 风险 | 应对 |
|---|---|
| 文献检索发现已经有人做过 | 改成「复现 + 扩展」：重点放在码率分配（Q3）和呼吸对比心率的拆分（Q2） |
| 画面里看不到胸部 | 用肩部或头部的运动测呼吸，结论里写明 |
| COHFACE、PURE 申请不下来 | 不依赖它们：SCAMPS + UBFC-rPPG 就能完成核心实验；真人的呼吸参考从 PPG 推，必要时加自录数据 |
| 从 PPG 推出的呼吸参考不准 | 只用三种推法一致的时间窗；报告可用比例；主要结论以 SCAMPS 的精确真值为准 |
| SCAMPS 每段只有约 20 秒，呼吸周期少 | 用整段和峰值插值；结论分数据集报告 |
| 合成数据和真人差别大 | 主要结论要求 SCAMPS 和真人数据的方向一致才下 |
| Colab 断线、空间不够 | 按视频分批、记录进度、断点续跑；视频处理完就删 |
| 隐私和许可 | 真人画面不进 git、不进报告；不转发任何数据；遵守各数据集的许可 |
| 和讯飞的工作重叠 | 全部从零写，只参考公开文献；动手前看一下实习协议里的保密条款 |
| 所有设置下差别都不显著 | 这本身也是结论，只要能把原因分析清楚 |

---

## 10. 面试前必须能回答的问题

1. 摄像头测心率、测呼吸的原理分别是什么？为什么心率信号那么弱？
2. 视频压缩是怎么工作的（帧间预测、变换和量化、颜色下采样、码率控制）？为什么它会抹掉心率信号？
3. 采样定理：测呼吸和测心率各需要多高的帧率？为什么同样的码率下，降帧率可能对测呼吸更好？
4. 频率估计：FFT 的频率分辨率由什么决定？窗口长度怎么取舍？峰值插值是什么？
5. POS、CHROM 这类方法怎么抵消光照和运动的影响？
6. 怎么从 PPG 推出呼吸？这个参考有多可靠？
7. 为什么 bootstrap 要按受试者重采样，而不是按视频片段？
8. 合成数据（SCAMPS）的价值和局限是什么？
9. 压缩增强为什么能提升鲁棒性？会不会让未压缩时的表现变差？
10. 在摄像头那一端算和在接收端算，各有什么取舍？WebRTC 的 `degradationPreference` 是什么？

---

## 11. 简历写法（数字等结果出来再填）

中文：

> 研究视频压缩对摄像头测呼吸的影响（Python、OpenCV、MediaPipe、ffmpeg）：从零实现基于胸部运动的测呼吸和基于脸部颜色的测心率流程，在合成和真人数据上系统测试 H.264/H.265 不同码率下的误差；发现 __（呼吸和心率谁更抗压缩，失灵点 __ kbps）；同等码率下 __ 的帧率组合使呼吸误差降低 __；通过 __ 把低码率下的误差降低 __（95% CI __）。

English:

> Studied how video compression affects camera-based breathing measurement (Python, OpenCV, MediaPipe, ffmpeg). Built breathing (chest motion) and heart-rate (skin color) pipelines from scratch and tested them under H.264/H.265 at a range of bitrates on synthetic and real videos; found __ (which measurement breaks first, at __ kbps); at a fixed bitrate, __ cut breathing-rate error by __; __ reduced low-bitrate error by __ (95% CI __).

---

## 12. 参考资料

- [The Impact of Video Compression on Remote Cardiac Pulse Measurement Using Imaging Photoplethysmography](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/11/McDuff_2017_Impact2.pdf)（McDuff 等，2017）
- [Effects of Video Encoding on Camera Based Heart Rate Estimation](https://doi.org/10.1109/tbme.2019.2904326)（IEEE TBME，2019）
- [Remote Heart Rate Measurement from Highly Compressed Facial Videos](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Remote_Heart_Rate_Measurement_From_Highly_Compressed_Facial_Videos_An_ICCV_2019_paper.pdf)（Yu 等，ICCV 2019）
- [A comprehensive evaluation of multiple video compression algorithms for preserving BVP signal quality](https://www.sciencedirect.com/science/article/abs/pii/S1746809424015039)（2024）
- [Remote Respiration Measurement with RGB Cameras: A Review and Benchmark](https://doi.org/10.1145/3771763)（ACM，2025）和 [resPyre](https://github.com/phuselab/resPyre)
- [SCAMPS: Synthetics for Camera Measurement of Physiological Signals](https://arxiv.org/abs/2206.04197)（McDuff 等，NeurIPS 2022），[数据和许可](https://github.com/danmcduff/scampsdataset)
- [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
- [COHFACE](https://idiap.ch/dataset/cohface) 和 [A Reproducible Study on Remote Heart Rate Measurement](https://arxiv.org/abs/1709.00962)（Heusch 等，2017）
- [rPPG-Toolbox](https://arxiv.org/abs/2210.00716)（NeurIPS 2023）
- POS 方法：Wang 等，*Algorithmic Principles of Remote PPG*（IEEE TBME，2017）
- [Claude Code 云端环境文档](https://code.claude.com/docs/en/cloud-environments)
