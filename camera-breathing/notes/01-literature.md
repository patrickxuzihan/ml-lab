# 阶段 1：文献检索

> 状态：**基本定稿**。最关键的四篇里已经读了三篇全文（SCAMPS 论文、arXiv 2606.04198、Nguyen 2024，2026-11-21），结论没有变。剩下的 ACM 综述（resPyre）只看过摘要，它的 PDF 在米兰大学机构仓库，云端访问不了，见第 5 节。其余文献的「依据」列标明是摘要还是全文。

## 1. 结论

1. **没有找到专门研究「视频压缩对摄像头测呼吸的影响」的工作。** 运动法和颜色法都没有。
   - 2026 年最新的编码器研究 [9] 和 2024 年的「复杂环境下的 rPPG」研究 [19]，读了全文确认：**都只测心率，没有测呼吸。**
   - 最接近的是 2025 年的 ACM 综述和基准（resPyre）[10]。它在讨论里定性地说：用运动测呼吸的方法，只要画质「合理」，可以接受压缩、低分辨率和低帧率；从颜色信号推呼吸的方法，在高度压缩的数据集（COHFACE、MAHNOB-HCI）上受影响很大。摘要里没有呼吸误差随码率变化的实验，**正文待核对**。
2. **没有找到在同一批视频上并列比较压缩对心率和呼吸影响的工作。**
3. **压缩对测心率的影响已经研究得很充分**（2017–2026），结论比较一致：
   - 压缩越重，心率信号越弱；
   - H.265 一般比 H.264 好；
   - MPEG-4 最差：[9] 发现它产生的块状伪影在脸部各区域之间是同步的，会骗过基于空间分解的方法；
   - 训练时加入压缩视频能部分补救。

   分歧在于颜色下采样和降分辨率的影响多大。
4. **码率怎么分配**（帧率、分辨率、画质怎么取舍）：只有心率方向的零散结果，研究的是单独降帧率、降分辨率、随机丢帧 [16–19]。**没有一篇在固定码率下比较这三者的取舍**，也没有研究呼吸的。用 WebRTC 的 `degradationPreference` 研究生理信号的工作也没有找到。
5. **所以 Q1、Q2、Q3 目前看都还是空白。** Q4 的「压缩增强训练」对心率已有人做过 [5]，对呼吸没有。

**研究问题不用调整。** 唯一剩下的风险点是 resPyre 综述的正文里有没有按码率做的呼吸实验。如果有，就按 GUIDE 第 9 节改成「复现 + 扩展」，重点放在 Q2 的拆分和 Q3 的码率分配上。到时我会在 PR 里提出，不自己改计划。

## 2. 本项目暂定的新意

1. 系统测量压缩（编码器 × 码率 × 码率控制方式）对摄像头测呼吸的影响，同时覆盖运动法（M1）和颜色法（M2）。
2. 在同一批视频上用三条测量拆开两个因素：伤害的是「颜色信号」，还是「快的信号」。
3. 在固定码率下比较帧率、分辨率和画质的取舍对呼吸和心率有什么不同，并对应到 WebRTC 的 `degradationPreference`。

## 3. 分类整理

「依据」一列：**全文**表示读过全文；**摘要**表示只看了检索结果里的摘要或片段；**转引**表示没读原文，信息来自另一篇读过全文的论文对它的介绍。

### 3.1 压缩对测心率的影响

| 文献 | 做了什么 | 主要发现 | 依据 |
|---|---|---|---|
| McDuff 等，CVPRW 2017 [1] | x264、x265，不同 CRF | 很轻的压缩就会明显降低脉搏信号的信噪比 | 摘要 |
| Rapczynski 等，IEEE TBME 2019 [2] | H.264、H.265，压缩率、分辨率、颜色下采样 | 压缩率越高心率越不准；分辨率降到一定程度以内、以及颜色下采样，对心率影响不大 | 摘要 |
| Špetlík 等，FG 2018 workshop [3] | H.264，5 人 | 降低分辨率会放大 H.264 压缩的影响；颜色下采样有负面影响，和 [2] 矛盾 | 摘要 |
| Yu 等，ICCV 2019 [4] | 高度压缩的人脸视频 | 波形变噪、峰值位置不准；提出视频增强网络来恢复 | 摘要 |
| Nowara、McDuff，ICCVW 2019 [5] | 用压缩视频训练 | 训练视频的压缩程度不低于测试视频时，信噪比和误差都有改善 | 摘要 |
| Nowara、McDuff、Veeraraghavan，BOE 2021 [23] | H.264、H.265、MPEG-4，五档 CRF | MPEG-4 最差；只用了全局投影类方法 | 转引自 [9] |
| Gudi 等，2020 [6] | 在 PURE 上比较编码 | H.265 在最低码率下最能保留脉搏成分 | 摘要 |
| Zhou 等，BSPC 2025 [7] | 比较 7 种编码器 | 推荐用帧内编码的 H.265 | 转引自 [9] |
| Sporrer 等，2024 [24] | 压缩对深度学习 rPPG 的影响 | CRF 很高时性能崩溃 | 转引自 [9] |
| Wang 等，IEEE JBHI 2025 [25] | 「保留生理信息」的视频压缩 | 提出专门为 rPPG 设计的压缩方法 | 转引自 [9] |
| Comas 等，2024（ESWA 2026）[8] | 压缩视频上的脉搏信号放大 | H.265 优于 H.264，尤其在运动大的视频上 | 摘要 |
| Ben Ahmed，arXiv 2606.04198，2026 [9] | 3 个数据集（MCD、UBFC-rPPG、UBFC-PHYS）；11 种退化：4:2:0 颜色下采样，JPEG q50/q30，H.264 和 H.265 的帧内编码和 GOP15（都是 CRF 28），MPEG-4 85/200/500 kbps；13 种算法（CHROM、POS、2SR 和作者提出的方法） | 提出「空间伪影一致性」SAC：MPEG-4 的伪影在脸部各块之间高度同步，其他编码器不会；MPEG-4 让基于空间分解的方法失效，而 CHROM 这类固定投影的方法只是平稳退化。**只测心率（0.75–2.5 Hz），没有呼吸**；所有视频都是 640×480，没有研究帧率和分辨率的取舍 | 全文 |

[9] 还提到 UBFC-rPPG 的原始码率约 218,000 kbps，和我们实测的 171–221 Mbit/s 吻合（见 `notes/01-data.md`）。

### 3.2 摄像头测呼吸：方法和基准

| 文献 | 内容 | 和本项目的关系 | 依据 |
|---|---|---|---|
| Boccignone 等，ACM Computing Surveys 2025，resPyre [10] | 运动法、rPPG 法、深度学习三类测呼吸方法的综述和开源基准 | rPPG 推呼吸最不可靠；压缩只做了定性讨论（**待全文核对**） | 摘要 |
| Wang 等，Physiol. Meas. 2022 [11] | 呼吸运动提取方法的系统比较 | M1 的方法选择 | 摘要 |
| Revisiting motion-based respiration measurement（Oulu）[12] | 像素亮度法和像素运动法的比较 | 运动法在多种条件下更好，对应 M1 用光流还是亮度 | 摘要 |
| Protopopov，arXiv 2025 [13] | 光流加人体分割，14 人 | 主要误差来自和呼吸无关的身体运动 | 摘要 |
| van Gastel 等，2016 [14]；Chen 等，2019 [15] | 从脸部 rPPG 推呼吸 | M2 的做法 | 摘要 |

### 3.3 帧率、分辨率、网络条件（对应 Q3）

| 文献 | 内容 | 依据 |
|---|---|---|
| Blackford、Estepp，2015 [16] | 帧率和分辨率降低对脉率误差影响很小（多摄像头场景） | 摘要 |
| Mironenko 等，CVPRW 2020 [17] | 帧率不稳定没有明显扭曲脉搏信号的频谱 | 摘要 |
| Álvarez Casado 等，SCIA 2023 [18] | 模拟视频通话里的丢帧、分辨率和帧率；在接收端重建信号可以缓解 | 摘要 |
| Nguyen 等，CBM 2024 [19] | 8 个数据集（含 UBFC-rPPG，文中写的帧率是 23–29，和我们的实测一致）。退化有：脸部区域分辨率、颜色位深、模糊、噪声、遮挡（墨镜、口罩）、按规律降帧率（20、15、10 帧/秒）、随机丢帧（10%、20%、50%）。**摘要提到编码器伪影，但实验里没有真正的编码器或码率实验；只测心率。** 结论：脸部区域分辨率影响很小；降低颜色位深有害；噪声比模糊危害大；传统方法对降帧率不敏感，深度学习方法表现不一；随机丢帧可以靠准确的时间戳重建信号来缓解 | 全文 |

### 3.4 呼吸参考：从 PPG 推呼吸（UBFC-rPPG 要用）

| 文献 | 内容 | 依据 |
|---|---|---|
| Charlton 等，Physiol. Meas. 2016 [20] | 比较了几百种从心电和 PPG 推呼吸的算法；PPG 比心电差；好的算法都融合了多种调制 | 摘要 |
| Charlton 等，IEEE RBME 2018 [21] | 上面这类算法的综述：基线、幅度、频率三种调制 | 摘要 |

这支持 GUIDE 第 4.1 节的做法：UBFC 的呼吸参考只用三种推法一致的时间窗，并报告可用比例。

### 3.5 SCAMPS 数据集本身 [22]（全文）

和本项目直接相关的几点：

- **呼吸怎么渲染**：论文写的是「用合成的呼吸信号控制胸部和头部的俯仰」，并说这些运动幅度很小，叠加头部转动和表情后常常难以看出（第 4 节 Breathing）。这解释了我们在阶段 1 测到的现象：肩部和头部有跟着呼吸的、不到一个像素的上下运动（`notes/01-data.md`）。
- **PPG 里故意加了呼吸成分**：合成的 PPG 波形「在呼吸频率上加了一个小的基线漂移」（第 3 节），而皮肤颜色由 PPG 驱动。所以 SCAMPS 上的 M2（从脸部颜色测呼吸）是按渲染规则「设计出来」的，强弱不一定和真人一样。分析 Q2 时，SCAMPS 上的 M2 结果要谨慎解读，主要结论要求和 UBFC 方向一致，这也是 GUIDE 第 9 节已有的要求。
- **参数范围**：呼吸频率在每分钟 8–24 次之间均匀采样，心率在 40–150 次/分之间均匀采样（第 3 节）。我们 10 段样例测到的呼吸 8.5–21.7 次/分、心率 49–142 次/分，都落在这个范围里。
- **时长和帧率**：波形按「20 秒、300 Hz」生成，然后给出逐帧的标签（第 5 节）；全集 2,800 段共 1.68M 帧，即每段 600 帧。论文没有直接写帧率，但 20 秒、600 帧对应 30 帧/秒，和脚本的假设一致。标签 csv 里的 `t_ppg` 列可以直接确认。
- **干扰因素**：2,000 段有头部转动（其中 1,200 段匀速转动，速度为每秒 10、20、30 度；800 段每秒随机换一个角度），800 段没有头部运动；1,400 段有面部动作；400 段有动态光照和背景运动。每段带哪些干扰因素，项目页面上有一个文件列出（第 5 节）。阶段 3 选 SCAMPS 子集时可以按这些因素分层，先用没有头部运动的视频做干净的压缩实验。
- **使用条款**：论文第 7 节写明数据集里的任何画面都可以用在学术论文里；不能商用（和 `notes/00-datasets.md` 一致）。
- 论文本身没有做任何压缩实验。

## 4. 检索方法

- **第一轮（2026-10-09）**：只有 WebSearch（通用搜索引擎），返回检索结果的摘要。当时云端的网络白名单拦截了 arxiv.org、dl.acm.org、PubMed、Semantic Scholar 和 OpenAlex。
- **第二轮（2026-11-21）**：经你同意，环境放开了 arxiv.org 和 dl.acm.org，我从 arXiv 下载了 [9]、[19]、[22] 三篇的全文来读。PDF 只放在会话的临时目录里，不进仓库。[9] 的参考文献里又发现了 [7]、[23]、[24]、[25] 这几篇压缩相关的工作。ACM 综述 [10] 在 dl.acm.org 上没拿到，开放获取的 PDF 在米兰大学机构仓库 air.unimi.it，不在白名单里。
- **检索式**（英文）：
  - video compression effect on camera-based respiration rate estimation
  - remote respiration measurement compressed video bitrate H.264 breathing rate robustness
  - compression impact remote photoplethysmography heart rate and respiratory rate same videos comparison
  - influence of video compression on camera-based respiratory monitoring motion optical flow chest
  - heart rate and respiratory rate from video conferencing telehealth compressed video Zoom WebRTC rPPG
  - effect of frame rate and resolution on remote photoplethysmography and respiration estimation
  - 2025 2026 video compression remote physiological measurement respiration heart rate benchmark codec bitrate
  - "respiratory rate" OR "breathing rate" video compression bitrate CRF camera-based estimation
  - respiration rate estimation from H.264 motion vectors compressed domain video breathing
  - respiratory rate estimation from photoplethysmogram review (Charlton, Karlen)
  - multi-task heart rate respiration rate from face video (MTTS-CAN, BigSmall)
- **局限**：
  - 不是系统综述，只覆盖英文；
  - 多数文献只看了摘要；
  - 搜索引擎对 2026 年的新论文收录可能不全。

## 5. 核对情况

| 优先级 | 文章 | 要核对什么 | 结果 |
|---|---|---|---|
| 1 | [10] ACM 综述和 resPyre | 正文里有没有按码率做的呼吸实验 | **待核对**：开放获取的 PDF 在 air.unimi.it，需要你下载后作为附件发来，或者同意把这个域名加进允许列表 |
| 1 | [9] arXiv 2606.04198 | 是否也测了呼吸 | 已核对：只测心率 |
| 1 | [19] Nguyen 等 2024 | 编码伪影实验的设置，有没有呼吸 | 已核对：没有真正的编码器实验，只测心率 |
| 1 | SCAMPS 论文 [22] | 呼吸怎么驱动画面；帧率和时长 | 已核对：控制胸部和头部的俯仰；每段 20 秒、600 帧，即 30 帧/秒 |
| 2 | [1]、[2]、[5]、[7]、[8]、[23] | 压缩设置和主要数字 | 阶段 2 定压缩阶梯时再读 |
| 2 | [11]、[12]、[20]、[21] | 运动提取方法和呼吸参考 | 阶段 2 定 M1 和呼吸参考时再读 |

## 6. 参考文献

1. McDuff, Blackford, Estepp. The Impact of Video Compression on Remote Cardiac Pulse Measurement Using Imaging Photoplethysmography. CVPRW 2017, pp. 1391–1399. <https://www.microsoft.com/en-us/research/wp-content/uploads/2020/11/McDuff_2017_Impact2.pdf>
2. Rapczynski et al. Effects of Video Encoding on Camera Based Heart Rate Estimation. IEEE TBME 2019. <https://doi.org/10.1109/tbme.2019.2904326>
3. Špetlík, Čech, Matas. Non-Contact Reflectance Photoplethysmography: Progress, Limitations, and Myths. FG 2018 workshop. <https://cmp.felk.cvut.cz/ftp/articles/cech/Spetlik-FGw-2018.pdf>
4. Yu et al. Remote Heart Rate Measurement from Highly Compressed Facial Videos. ICCV 2019. <https://arxiv.org/abs/1907.11921>
5. Nowara, McDuff. Combating the Impact of Video Compression on Non-Contact Vital Sign Measurement Using Supervised Learning. ICCVW 2019. <https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Nowara_Combating_the_Impact_of_Video_Compression_on_Non-Contact_Vital_Sign_ICCVW_2019_paper.pdf>
6. Gudi et al. Real-time Webcam Heart-Rate and Variability Estimation with Clean Ground Truth for Evaluation. 2020. <https://arxiv.org/abs/2012.15846>
7. Zhou et al. A comprehensive evaluation of multiple video compression algorithms for preserving BVP signal quality. Biomedical Signal Processing and Control, vol. 103, 2025. <https://www.sciencedirect.com/science/article/abs/pii/S1746809424015039>
8. Comas, Ruiz, Sukno. Deep Pulse-Signal Magnification for Remote Heart Rate Estimation in Compressed Videos. 2024; Expert Systems with Applications 2026. <https://arxiv.org/abs/2405.02652>
9. Ben Ahmed. Spatial Artifact Coherence Determines Codec Robustness in Patch-Based rPPG. arXiv 2606.04198, 2026. <https://arxiv.org/abs/2606.04198>
10. Boccignone, Cuculo, D'Amelio, Grossi, Lanzarotti, Patania. Remote Respiration Measurement with RGB Cameras: A Review and Benchmark. ACM Computing Surveys 58(5), article 114, 2025. <https://doi.org/10.1145/3771763>；开放获取版本：<https://air.unimi.it/handle/2434/1194001>；resPyre：<https://github.com/phuselab/resPyre>
11. Wang et al. Algorithmic insights of camera-based respiratory motion extraction. Physiological Measurement, 2022. <https://iopscience.iop.org/article/10.1088/1361-6579/ac5b49>
12. Revisiting motion-based respiration measurement from videos. University of Oulu. <https://www.6gflagship.com/?p=3612>
13. Protopopov. A Robust Camera-based Method for Breath Rate Measurement. 2025. <https://arxiv.org/abs/2512.03827>
14. van Gastel, Stuijk, de Haan. Robust respiration detection from remote photoplethysmography. Biomedical Optics Express, 2016.
15. Chen et al. Respiratory Rate Estimation from Face Videos. 2019. <https://arxiv.org/abs/1909.03503>
16. Blackford, Estepp. Effects of frame rate and image resolution on pulse rate measured using multiple camera imaging photoplethysmography. SPIE 2015.
17. Mironenko et al. Remote Photoplethysmography: Rarely Considered Factors. CVPRW 2020. <https://arxiv.org/abs/2004.12695>
18. Álvarez Casado et al. SCIA 2023. <https://doi.org/10.1007/978-3-031-31438-4_38>
19. Nguyen, Nguyen, Li, Bordallo López, Álvarez Casado. Evaluation of Video-Based rPPG in Challenging Environments: Artifact Mitigation and Network Resilience. Computers in Biology and Medicine, 2024. <https://arxiv.org/abs/2405.01230>
20. Charlton et al. An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram. Physiological Measurement, 2016. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5390977/>
21. Charlton et al. Breathing Rate Estimation From the Electrocardiogram and Photoplethysmogram: A Review. IEEE Reviews in Biomedical Engineering, 2018. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7612521/>
22. McDuff, Wander, Liu, Hill, Hernandez, Lester, Baltrusaitis. SCAMPS: Synthetics for Camera Measurement of Physiological Signals. NeurIPS 2022 Datasets and Benchmarks. <https://arxiv.org/abs/2206.04197>
23. Nowara, McDuff, Veeraraghavan. Systematic analysis of video-based pulse measurement from compressed videos. Biomedical Optics Express 12:494–511, 2021.（转引自 [9]）
24. Sporrer, Vance, Speth, Flynn. Examining the effects of compression on deep learning remote photoplethysmography. IS&T Electronic Imaging Symposium, 2024.（转引自 [9]）
25. Wang, Shan, Liu, Zhou, Shu. Physiological information preserving video compression for rPPG. IEEE JBHI 29:3563–3575, 2025.（转引自 [9]）
