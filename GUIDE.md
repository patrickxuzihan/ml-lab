# text2sql-grpo 项目指导

> 这份文件写给两个读者：**你**照着它推进项目，**Claude** 在云端会话里按它写代码。每个阶段最后都有「发给 Claude 的话」，复制到 [claude.ai/code](https://claude.ai/code) 的新会话里就行。
>
> 仓库根目录的 `CLAUDE.md` 是给 Claude 的长期规则，每次会话都会自动读取。下文按 claude.ai/code 来写；如果你用的是在 issue 里 @claude 的 GitHub Actions 方式，把同样的话写进 issue 即可，`CLAUDE.md` 一样生效。

---

## 1. 项目要回答什么

**一句话：小模型做 Text-to-SQL 时，强化学习（GRPO）什么时候有用、什么时候没用、为什么。**

### 为什么这样定题

网上「Qwen + Spider + GRPO」的复现已经很多，单纯复现没有新意。但公开结果之间有一个没解释清楚的矛盾：

- **大模型上，简单奖励就够了。** Snowflake 的 Arctic-Text2SQL-R1（2025）在 7B–32B 模型上只用「执行结果对不对、语法是否有效」做奖励，GRPO 就拿到了很强的结果。
- **小模型上几乎没用。** Hugging Face 上有人公开了一个负面结果（`Anurich/slmsql-grpo-v2`）：Qwen2.5-Coder-1.5B 先做 SFT，再用只看执行结果的奖励做 GRPO，执行准确率几乎没变（Spider dev +1.0，BIRD dev −0.7）。作者第一次训练时，70% 的组里 8 个采样的奖励完全相同，优势（advantage）为 0，模型拿不到梯度；第二次只保留通过率在 0.2–0.8 之间的题，结果还是没有提升。作者的结论是：要有提升，得用过程奖励。
- **更细的奖励似乎有帮助。** 也有人在 0.5B 模型上用更细的稠密奖励（FINER-SQL）报告了不错的结果。

差别到底来自模型大小、SFT 之后的「饱和」、奖励太稀疏，还是训练设置？我们用同一个小模型、同一套评测做对照实验，把这几个因素拆开。

另外，那个负面结果只在 200 道 Spider 题和 149 道 BIRD 题上评测，±1 分本身就在误差范围内。我们用完整的 Spider dev（1,034 题），给出 95% 置信区间，模型之间用配对检验比较。

### 三个问题

- **Q1：为什么 SFT 之后 GRPO 学不动？**
  假设是：SFT 之后，大部分题 8 次采样要么全对、要么全错，组内奖励没有差异，优势就是 0。这一点在训练前就能测出来（第 4.4 节）。
- **Q2：什么办法能让它重新有效？** 候选方案：
  - (a) 只用「有时对、有时错」的题训练（难度筛选）；
  - (b) 给「差一点就对」的 SQL 部分分数（部分奖励）；
  - (c) 不做 SFT，直接从原始模型开始 RL；
  - (d) 调学习率：那个负面结果给 LoRA 用的学习率是 1e-6，对 LoRA 来说偏小，可能本身就学不动。
- **Q3：RL 和 SFT 谁泛化得更好？**
  2025 年有篇论文叫《SFT Memorizes, RL Generalizes》。我们用 CSpider 检验这个说法：CSpider 把 Spider 的问题人工翻译成了中文，数据库和 SQL 都不变。模型只用英文训练，看哪种方法在中文问题上掉得更少。

---

## 2. 分工和工作流

| | Claude（claude.ai/code 云端会话） | 你 |
|---|---|---|
| 机器 | 只有 CPU：4 核、16GB 内存、30GB 硬盘，没有 GPU | 租用的 GPU 机器（AutoDL 或 RunPod） |
| 负责 | 写代码和测试、在 CPU 上跑冒烟测试、分析结果、画图、写文档 | 审 PR 并合并；在 GPU 上跑评测和训练；把结果提交回仓库 |

每个阶段按这个顺序走：

1. 把「发给 Claude 的话」发到一个新会话。
2. Claude 新建分支，写代码、跑测试，然后开 PR。PR 里会写清楚你下一步要在 GPU 上运行的命令。
3. 你审 PR，看不懂的地方直接在会话里问，然后合并。
4. 如果这个阶段需要 GPU，你就在 GPU 机器上拉代码、跑命令，把 `results/` 提交并推送。
5. 开一个新会话，让 Claude 分析结果，再进入下一阶段。

几个建议：

- **先看计划，再让它写代码。** 大一点的阶段，先让 Claude 只给实现计划，你看过没问题再让它动手。
- **一个会话只做一件事。** 会话太长就开新的，并告诉它看 GUIDE.md 的哪一节。
- **每个 PR 你都要能讲清楚。** 代码让 Claude 写没问题，但面试官一定会追问细节（见第 10 节）。每个 PR 末尾都有中文的「学习要点」，要认真看，不懂就问。

---

## 3. 技术选型

| 项目 | 选择 | 说明 |
|---|---|---|
| 主模型 | Qwen2.5-Coder-1.5B-Instruct | Apache-2.0 许可证；和公开的负面结果是同一个模型，方便对比。全程只用这一个主模型，中途不要换 |
| 规模对照（可选） | Qwen2.5-Coder-0.5B-Instruct | 看结论会不会随规模变化。3B 版本的许可证限制更多，用之前先看模型卡 |
| 训练数据 | Spider 训练集（`train_spider.json`，7,000 题） | |
| 评测数据 | Spider dev（1,034 题，20 个数据库）、CSpider dev | 可选：BIRD dev（1,534 题，更难）；Spider-Syn、Spider-Realistic（改写了问法，用来测鲁棒性） |
| 训练框架 | Hugging Face TRL（SFTTrainer、GRPOTrainer）+ PEFT（LoRA）+ vLLM（负责生成） | |
| 评测指标 | 执行准确率（EX） | 训练奖励和日常评测用自己写的快速版；最终数字用官方的 test-suite-sql-eval 复核，并按官方难度分级（easy / medium / hard / extra）统计 |
| 实验记录 | wandb（个人免费）+ 仓库里的 `results/` | wandb 项目可以设成公开，在 README 里放链接 |

---

## 4. 实验设计

### 4.1 统一的评测协议（所有实验共用，中途不能改）

- **提示词**：系统提示 + 数据库结构（每张表的 CREATE TABLE 语句，加 3 行示例数据）+ 问题，要求只输出一条放在 ` ```sql ` 代码块里的 SQL。
  示例数据对中文问题尤其重要：问题里写的是「法国」，数据库里存的是 `France`，模型得自己对上。
- **解码**：贪心解码（temperature 0），最多生成 256 个 token。
- **指标**：执行准确率，附 bootstrap 95% 置信区间。比较两个模型时用配对检验，在同一批题上逐题比。
- **同时记录**：SQL 执行报错的比例、平均输出长度、按难度分级的执行准确率。

### 4.2 实验矩阵（主线，1.5B）

| 编号 | 起点 | 方法 | 奖励 | 训练题 | 对应问题 |
|---|---|---|---|---|---|
| B0 | 原始模型 | 不训练，直接评测 | — | — | 基线 |
| S1 | 原始模型 | LoRA SFT | — | Spider 训练集 | SFT 基线 |
| R1 | 原始模型 | GRPO | 结果奖励 | 全部 | Q2(c)：不做 SFT 直接 RL |
| R2 | S1 | GRPO | 结果奖励 | 全部 | Q1：复现「学不动」 |
| R3 | S1 | GRPO | 结果奖励 | 只保留 S1 有时对、有时错的题 | Q2(a)(d)：难度筛选、学习率 |
| R4 | S1 | GRPO | 结果奖励 + 部分奖励 | 同 R3 | Q2(b)：部分奖励 |

- **结果奖励**：执行结果和标准答案一致得 1；能执行但结果不对得 0.1；报错或格式不对得 0。和那个负面结果用的奖励一样，方便对比。
- **部分奖励**：结果不对时，再看用到的表和列与标准 SQL 的重合度（Jaccard），据此额外给 0–0.3 分。要专门检查模型会不会钻空子，比如把能想到的列全写上。
- **学习率对照**：在 R3 的设置下，LoRA 学习率 1e-6 和 1e-5 各跑 100 步，比较奖励曲线，再用更合适的那个跑完整的 R3。
- **GRPO 默认设置**：每题采样 8 个（G=8），采样温度 1.0，LoRA r=32。其余超参数在第一次跑通后固定下来，写进配置文件。

### 4.3 每次 GRPO 都要记录的诊断指标

- **有效组比例**：组内 8 个奖励不全相同的比例，这是回答 Q1 的关键证据。TRL 新版本会记录 `frac_reward_zero_std`（组内奖励标准差为 0 的比例），用 1 减去它就是有效组比例；如果没有这个指标，就自己算。
- 平均奖励、奖励标准差、KL（如果开了）、熵、输出长度、学习率。
- 训练结束后，对最终模型跑完整评测（Spider dev + CSpider dev）。

### 4.4 训练前诊断

这一步很便宜，但可能是整个项目最有说服力的发现。

在跑任何 GRPO 之前，用 B0 和 S1 对每道训练题各采样 8 次（温度 1.0），统计通过率的分布：多少题 8 次全对，多少题全错，多少题在中间。只有中间那部分题，GRPO 才能学到东西。

如果 S1 落在中间的题很少，就等于在训练前预测了 R2 会学不动，之后再用 R2 的训练曲线验证。R3、R4 的训练题也从这一步筛出来。

### 4.5 可选扩展（时间和预算够再做）

- 用 0.5B 模型跑 S1、R2、R3，看结论会不会随规模变化。
- 让模型先推理再写 SQL，和直接输出 SQL 比较。
- 在 Spider-Syn、Spider-Realistic 上测鲁棒性；在 BIRD dev 上评测。
- 中英混合训练：训练时加入一部分 CSpider 的中文问题，看中文上的差距能不能缩小。
- 把最好的 LoRA 上传到 Hugging Face，再做一个 Gradio demo：选一个示例数据库、输入问题，显示生成的 SQL 和执行结果。

---

## 5. 仓库结构

```
llm-lab/
├── CLAUDE.md                  # 给 Claude 的长期规则
├── README.md                  # 仓库首页：项目索引
└── text2sql-grpo/
    ├── GUIDE.md               # 本文件
    ├── README.md              # 项目首页：结论、结果表、图、复现方法（阶段 6 写完整）
    ├── requirements-cpu.txt   # 云端 / 本地 CPU 环境
    ├── requirements-gpu.txt   # GPU 机器环境（含 vLLM）
    ├── configs/               # 每个实验一个配置文件：sft.yaml、grpo_r1.yaml……
    ├── scripts/               # download_data.sh 和运行评测、训练的命令
    ├── src/text2sql/
    │   ├── data.py            # 读取 Spider/CSpider，生成提示词（含数据库结构）
    │   ├── sql_exec.py        # 只读、限时地执行 SQL，比较结果
    │   ├── rewards.py         # 奖励函数（纯函数）
    │   ├── metrics.py         # 执行准确率、置信区间、配对检验、按难度统计
    │   ├── generate.py        # 批量生成：vLLM（GPU）/ transformers（CPU 冒烟测试）
    │   ├── train_sft.py
    │   ├── train_grpo.py
    │   └── diagnose.py        # 训练前的 pass@8 诊断和难度筛选
    ├── analysis/              # 画图、错误分析
    ├── notes/                 # 每个阶段的分析笔记（中文）
    ├── tests/                 # 单元测试 + 小型 SQLite 测试库（tests/fixtures/）
    └── results/<run_id>/      # 每次运行的配置、指标、预测、训练曲线
```

---

## 6. 分阶段任务

### 阶段 0：准备（你，约 1 天）

1. 定下作品集用哪个 GitHub 账号（我建议用 `patrickxuzihan`），新建公开仓库 `llm-lab`。
2. 把本地 `llm-lab` 文件夹里的文件传上去，保持目录结构。两种方法任选一种：
   - **网页上传**：在仓库页面点 **Add file → Upload files**，把文件夹里的内容拖进去。`.gitignore` 是隐藏文件，在 Finder 里按 `Cmd + Shift + .` 才能看到。
   - **命令行**：
     ```bash
     cd ~/Desktop/找工作/llm-lab
     git init -b main
     git add .
     git commit -m "Add project guide and Claude instructions"
     git remote add origin https://github.com/<你的用户名>/llm-lab.git
     git push -u origin main
     ```
     提交前先确认 `git config user.email` 是这个 GitHub 账号里验证过的邮箱，否则提交不会算进贡献图。
3. 打开 [claude.ai/code](https://claude.ai/code)（需要 Pro 或 Max 订阅），连接 GitHub，授权 Claude GitHub App 访问 `llm-lab`。
4. 配置云端环境：网络访问选 **Custom**，勾选 **Also include default list of common package managers**，再在允许的域名里加上下面几行。这样冒烟测试才能下载小模型、安装 CPU 版 PyTorch。
   ```
   huggingface.co
   *.huggingface.co
   hf.co
   *.hf.co
   *.pytorch.org
   ```
   环境变量里不要放任何密钥，因为用这个环境的人都能看到。这个项目在云端也用不到密钥。
5. 注册 GPU 平台账号（见第 7 节）；需要的话再注册 wandb。

验收：

- [ ] claude.ai/code 里能选到 `llm-lab` 仓库
- [ ] 把下面这段话发出去后，Claude 能正确复述项目，并确认能访问 pypi.org 和 huggingface.co

发给 Claude 的话：

> 读一下 CLAUDE.md 和 text2sql-grpo/GUIDE.md，用你自己的话总结这个项目的目标，以及阶段 1 要做什么。然后检查环境：Python 版本，能否访问 pypi.org 和 huggingface.co。这次不要改任何文件。

### 阶段 1：核心代码（Claude，只用 CPU，约 3–5 天）

Claude 要做：

1. 搭好 Python 包结构，写两份依赖文件：`requirements-cpu.txt` 从 PyTorch 的 CPU 源安装 torch；`requirements-gpu.txt` 包含 vLLM，torch 版本跟着 vLLM 走。核心模块（`data`、`sql_exec`、`rewards`、`metrics`）不能依赖 torch。
2. `data.py`：读取 Spider/CSpider 格式的 JSON；从 SQLite 文件生成数据库结构文本（CREATE TABLE 加每表 3 行示例）；按第 4.1 节的模板生成提示词，超长时先删示例行。
3. `sql_exec.py`：以只读方式打开数据库，限时执行（默认 5 秒），并限制返回行数。比较两个结果时，没有 ORDER BY 就按多重集合比较，有 ORDER BY 就连顺序一起比。返回结构化结果：是否执行成功、报错信息、结果是否一致。
4. 从模型输出里提取 SQL：优先取 ` ```sql ` 代码块，没有代码块时用兜底规则。规则要写清楚，并且有测试。
5. `rewards.py`：实现第 4.2 节的结果奖励和部分奖励，都写成纯函数：输入模型输出、标准 SQL 和数据库路径，输出分数。解析 SQL 里的表和列可以用 `sqlglot`。
6. `metrics.py`：执行准确率、bootstrap 置信区间、配对检验（配对 bootstrap 或 McNemar 检验）。
7. `scripts/download_data.sh`：把 Spider 和 CSpider 下载到 `data/`（不进 git）。动手前先确认当前的官方下载地址和许可证。
8. 测试：用脚本在 `tests/fixtures/` 里生成 2 个小 SQLite 库和十几道题，覆盖以下情况：结果一致和不一致、ORDER BY、执行报错、超时、空结果、各种格式的 SQL 提取，以及两种奖励。

验收：

- [ ] `pytest` 全部通过，`ruff check` 没有报错
- [ ] 在测试库上用标准 SQL 当预测，执行准确率是 100%
- [ ] 死循环或特别慢的 SQL 会在时限内被中断，不会卡住
- [ ] 没装 torch 的环境也能导入核心模块
- [ ] 项目 README 写明 CPU 和 GPU 两种环境的安装命令

发给 Claude 的话：

> 阅读 CLAUDE.md 和 text2sql-grpo/GUIDE.md，完成「阶段 1」。先给我实现计划，我确认后再写代码。完成后开 PR，PR 描述里逐条对照验收清单，最后写「学习要点」。

### 阶段 2：评测脚本和基线（Claude 写，你在 GPU 上跑，约 2–3 天）

Claude 要做：

1. `generate.py`：支持两种后端。vLLM 用于 GPU 上的正式评测，transformers 用于 CPU 冒烟测试。
2. 评测命令：读取配置 → 生成 → 提取 SQL → 执行 → 计算指标 → 写入 `results/<run_id>/`。写入的内容包括配置快照、git commit、包版本、GPU 型号、`metrics.json` 和 `predictions.jsonl`。
3. 自检模式：用标准 SQL 当预测跑一遍评测，用来检查执行和比较逻辑。
4. `--smoke` 模式：在 CPU 上用很小的模型跑 5 道题，只验证流程能跑通，准确率不重要。
5. 接入官方 test-suite-sql-eval，用于最终复核和难度分级。运行时再下载，不要把它的代码复制进仓库，除非许可证允许。

你要做（RTX 4090 就够，约 1 小时）：

1. 按第 7 节开机、装环境、下载数据。
2. 先跑自检，执行准确率应该接近 100%。如果明显低于 100%，说明执行或比较逻辑有问题，先让 Claude 修好再往下走。
3. 跑 B0：原始模型在 Spider dev 和 CSpider dev 上的评测。
4. 提交并推送 `results/`。

验收：

- [ ] 自检的执行准确率接近 100%
- [ ] B0 在 Spider dev 上的结果和公开参考值大致相当（参考值让 Claude 去查）。差得很远，说明提示词或评测有问题
- [ ] B0 在 CSpider dev 上的结果已经生成

发给 Claude 的话：

- 写代码：
  > 完成 GUIDE.md「阶段 2」里 Claude 负责的部分，开 PR，写清楚我在 GPU 机器上要依次运行的命令。
- 分析结果：
  > 自检和 B0 的结果已经推送到 results/。请检查结果是否合理，和公开参考值比较，挑 20 个错例看主要错在哪里，写进 text2sql-grpo/notes/02-baseline.md。

### 阶段 3：SFT（约 2–3 天）

Claude 要做：`train_sft.py`（TRL SFTTrainer + LoRA）和 `configs/sft.yaml`。损失只算在答案部分；训练结束后自动在 Spider dev 和 CSpider dev 上评测；附带 CPU 冒烟测试。

你要做（RTX 4090，约 1 小时）：跑 S1，提交并推送结果。

验收：

- [ ] S1 在 Spider dev 上的结果高于 B0（如果没有，先排查清楚再往下走）
- [ ] 训练 loss 曲线导出成 CSV 并提交

发给 Claude 的话：

- 写代码：
  > 完成「阶段 3」里 Claude 负责的部分，开 PR。
- 分析结果：
  > S1 的结果已经推送。请和 B0 做配对比较，按难度分级看提升在哪里，再看中英文差距有什么变化，写进 notes/03-sft.md。

### 阶段 4：训练前诊断（约 1–2 天）

Claude 要做：`diagnose.py`。对训练集每道题采样 8 次，统计通过率分布并画直方图，再输出中间难度题目的 ID 列表，给 R3、R4 用。

你要做（GPU，约 1 小时）：对 B0 和 S1 各跑一次，提交并推送结果。

验收：

- [ ] 有 B0、S1 的通过率分布图
- [ ] 有筛选后的题目 ID 列表（只存 ID，不存数据本身）
- [ ] `notes/04-diagnosis.md` 里根据分布预测了 R1、R2 的有效组比例

发给 Claude 的话：

- 写代码：
  > 完成「阶段 4」里 Claude 负责的部分，开 PR。
- 分析结果：
  > 诊断结果已经推送。请画出 B0 和 S1 的通过率分布，生成 R3、R4 用的题目列表，并在 notes/04-diagnosis.md 里预测 R1、R2 的有效组比例。

### 阶段 5：GRPO 实验（约 1.5–2 周）

Claude 要做：

1. `train_grpo.py`：TRL GRPOTrainer + LoRA + vLLM，用 colocate 模式让一张卡同时负责生成和训练。奖励调用 `rewards.py`；SQL 用多进程并行执行，每条都限时。
2. 每个实验一个配置文件：R1–R4，加上学习率对照。
3. 训练中记录第 4.3 节的诊断指标；训练结束后导出 CSV，并自动评测最终模型。
4. 定期保存检查点，支持断点续训，因为租的机器可能断线。
5. CPU 冒烟测试：用极小的模型、8 道题、2 步，不用 vLLM，确认整个流程能跑通。

你要做（建议用 A100 80GB）：

1. 先跑 R2 的前 50 步，确认速度、显存和各项指标都正常，再估算每个完整实验要多久、花多少钱。
2. 按这个顺序跑：R2 → 学习率对照 → R3 → R4 → R1。每跑完一个就提交结果，先让 Claude 分析，再决定下一个要不要调整。

验收：

- [ ] 每个实验都有 `results/<run_id>/`，里面有配置、训练曲线 CSV 和最终评测
- [ ] 能用有效组比例的曲线解释 R2 的结果，不管它有没有提升
- [ ] `notes/05-grpo.md` 记下了每个实验的结果和观察，失败的尝试也要记

发给 Claude 的话：

- 写代码：
  > 完成「阶段 5」里 Claude 负责的部分（训练脚本、全部实验配置、CPU 冒烟测试），开 PR。
- 每跑完一个实验：
  > <run_id> 的结果已经推送。请分析训练曲线（尤其是有效组比例）和最终评测，和之前的实验比较，更新 notes/05-grpo.md，并建议下一个实验要不要调整参数。

### 阶段 6：分析和写作（Claude，约 3–5 天）

Claude 要做：

1. 汇总所有实验，做一张结果总表：执行准确率和 95% 置信区间、与 S1 的配对检验、按难度分级的结果、CSpider 上的差距。
2. 画图：各方法的执行准确率对比；训练中有效组比例和奖励的曲线；通过率分布；中英文差距。
3. 错误分析：从 S1 和最好的 RL 模型里各抽 100 个错例，按以下类别归类，统计各类占比：
   - 选错表或列
   - JOIN 错误
   - 聚合或分组错误
   - 嵌套查询错误
   - 条件里的值写错（包括中文问题里的值没对上英文数据）
   - 排序或 LIMIT 错误
   - 语法错误

   你抽查其中 20 条，确认归类靠谱。
4. 写项目 README，顺序是：一句话结论 → 结果表和图 → 方法 → 发现 → 局限性 → 复现命令、硬件和花费。局限性至少写三条：Spider 很可能出现在模型的预训练数据里；只用了一个主模型；随机种子数量有限。
5. 写简历 bullet 的中英文草稿，所有数字都来自 `results/`。

验收：

- [ ] README 里每个数字都能在 `results/` 里找到出处
- [ ] 结论的措辞和置信区间一致，不显著就写不显著

发给 Claude 的话：

> 所有实验结果都在 results/ 里了。请完成「阶段 6」：汇总表、图、错误分析、项目 README 和简历 bullet 草稿，开 PR。

---

## 7. GPU 操作手册

### 选平台

| 平台 | 优点 | 注意 |
|---|---|---|
| AutoDL | 便宜，用人民币支付 | 需要国内手机号。访问 GitHub 和 Hugging Face 前要先运行 `source /etc/network_turbo`（学术资源加速），或者设置 `export HF_ENDPOINT=https://hf-mirror.com`。数据和模型放在数据盘 |
| RunPod | 网络畅通，用信用卡支付美元 | 比 AutoDL 贵一些 |

### 选卡

- 评测、SFT、诊断：RTX 4090（24GB）就够。
- GRPO：建议用 A100 80GB。4090 也能跑，但要调低 vLLM 的显存占比（比如 0.3），并减少每步的题目数。
- 价格变化快，以平台当时的报价为准。

### 每次开机后的固定流程

```bash
# 1. 拉代码（第一次用 clone，之后用 git pull）
git clone https://github.com/<你的用户名>/llm-lab.git
cd llm-lab/text2sql-grpo

# 2. 装环境、下载数据（具体命令以阶段 1 完成后 README 里写的为准）
pip install -r requirements-gpu.txt
bash scripts/download_data.sh

# 3. 在 tmux 里运行，SSH 断开也不会中断
tmux new -s run
# 运行 PR 里给出的命令
# 先按 Ctrl+B 再按 D 离开 tmux；之后用 tmux attach -t run 回来查看
```

### 跑完之后

```bash
git add results/
git commit -m "Add results: <run_id>"
git push
```

- 在 GPU 机器上推送需要 GitHub 身份验证。建一个 fine-grained personal access token，只给 `llm-lab` 这一个仓库的 Contents 读写权限，推送时用它代替密码。项目结束后把它删掉。
- **用完一定关机。** 按小时计费，忘记关机是最常见的浪费。
- 检查点不要提交到 git。需要保留的 LoRA 可以上传到 Hugging Face。

---

## 8. 时间和预算

按每周投入 15 小时左右估算：

| 阶段 | 时间 | GPU 用量 |
|---|---|---|
| 0 准备 | 1 天 | — |
| 1 核心代码 | 3–5 天 | — |
| 2 评测和基线 | 2–3 天 | 约 1 小时（4090） |
| 3 SFT | 2–3 天 | 约 1 小时（4090） |
| 4 训练前诊断 | 1–2 天 | 约 1 小时 |
| 5 GRPO 实验 | 1.5–2 周 | 约 20–30 小时（A100） |
| 6 分析和写作 | 3–5 天 | — |
| **合计** | **约 5–6 周** | **约 25–35 GPU 小时，30–60 美元** |

---

## 9. 风险和应对

| 风险 | 应对 |
|---|---|
| 所有 RL 方法都没有显著提升 | 这本身就是结论。只要能用有效组比例和训练前诊断解释清楚为什么没提升，项目就是完整的，而且比一个说不清原因的 +2 分更有价值 |
| 显存不够（OOM） | 降低 vLLM 的显存占比、减少每步的题目数、缩短最大输出长度、开启梯度检查点；还不行就换 A100 |
| 模型钻奖励的空子 | 定期抽看训练中的输出；部分奖励设上限；在报告里如实记录 |
| TRL 和 vLLM 版本不兼容 | 先用 TRL 文档推荐的版本组合，跑通后锁定版本 |
| Spider 可能出现在预训练数据里 | 写进局限性；所有方法比的都是相对提升，受到的影响一样 |
| 机器断线、实验中断 | 用 tmux，定期保存检查点，支持断点续训 |

---

## 10. 面试前必须能回答的问题

1. GRPO 和 PPO 有什么区别？为什么 GRPO 不需要 value model？组内相对优势是怎么算的？
2. 为什么组内奖励全一样时梯度为 0？你是怎么测出这个问题的？
3. 只用 0/1 的结果奖励有什么问题？部分奖励为什么可能被钻空子？你实际观察到了什么？
4. SFT 和 RL 各自解决什么问题？为什么做完 SFT 之后，RL 反而可能更难学到东西？
5. KL 惩罚有什么作用？为什么有些新工作把它去掉了？
6. LoRA 的原理是什么？为什么 LoRA 的学习率通常比全参数微调高？
7. 执行准确率（EX）和精确匹配（EM）有什么区别？EX 为什么会出现「假阳性」？test-suite 评测是怎么缓解的？
8. 为什么要用配对检验和置信区间？在 1,034 道题上差 1 分，算不算显著？
9. 中文问题主要错在哪里？（比如问题里的「法国」要对上数据库里的 `France`）
10. vLLM 在 GRPO 训练里起什么作用？训练的时间主要花在哪里？

---

## 11. 简历写法（数字等结果出来再填）

中文：

> 设计对照实验，研究小模型 Text-to-SQL 中 GRPO 何时有效（Qwen2.5-Coder-1.5B，TRL + vLLM）：通过训练前的 pass@8 诊断，定位「组内奖励无差异导致无梯度」的问题，并比较难度筛选、部分奖励等方案；最优方案在 Spider dev 上的执行准确率比 SFT 提升 __（95% CI __），在中文提问（CSpider）上 RL 与 SFT 的泛化差异为 __。

English:

> Ran controlled GRPO experiments on small-model text-to-SQL (Qwen2.5-Coder-1.5B, TRL + vLLM). Diagnosed zero-variance reward groups as the reason RL stalls after SFT, then compared difficulty filtering and partial-credit rewards; the best setup improved Spider-dev execution accuracy by __ over SFT (95% CI __) and changed the English-to-Chinese (CSpider) gap by __.

---

## 12. 参考资料

- [Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL](https://arxiv.org/abs/2505.20315)（Snowflake，2025）
- [slmsql-grpo-v2 模型卡](https://huggingface.co/Anurich/slmsql-grpo-v2)：1.5B 模型上「只看结果的 GRPO 几乎没有提升」的公开负面结果
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161)（Chu 等，2025）
- [DAPO](https://arxiv.org/abs/2503.14476)（2025）：其中的「动态采样」会过滤掉组内奖励全部相同的题，思路和本项目的难度筛选一致
- Reasoning-SQL（Pourreza 等，2025）：为 Text-to-SQL 设计的部分奖励
- FINER-SQL：0.5B 模型 + 稠密奖励（在 Hugging Face 上搜 `FINER-SQL-0.5B`；写 README 前让 Claude 找到原始论文）
- [Spider](https://yale-lily.github.io/spider)（Yu 等，2018）、[CSpider](https://taolusi.github.io/CSpider-explorer/)（Min 等，2019）、[test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)（Zhong 等，2020）
- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer)
- [Claude Code 云端环境文档](https://code.claude.com/docs/en/cloud-environments)
