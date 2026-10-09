# 阶段 1：文献检索

> 状态：**初版，结论只基于检索摘要**。云端会话读不到论文全文（见「检索方法」）。第 5 节列出的文章拿到全文后要逐条核对，核对完再把状态改成「定稿」。

## 1. 结论

1. **没有找到专门研究「视频压缩对摄像头测呼吸的影响」的工作。** 运动法和颜色法都没有。
   - 最接近的是 2025 年的 ACM 综述和基准（resPyre）。它在讨论里定性地说：用运动测呼吸的方法，只要画质「合理」，可以接受压缩、低分辨率和低帧率；从颜色信号（rPPG）推呼吸的方法，在高度压缩的数据集（COHFACE、MAHNOB-HCI）上受影响很大。摘要里没有呼吸误差随码率变化的实验。
2. **没有找到在同一批视频上并列比较压缩对心率和呼吸影响的工作。** 有的论文提到压缩会通过 rPPG 间接影响呼吸估计，但没有给出呼吸的数据。
3. **压缩对测心率的影响已经研究得很充分**（2017–2026）。结论比较一致：
   - 压缩越重，心率信号越弱；
   - H.265 一般比 H.264 好；
   - 训练时加入压缩视频能部分补救。

   分歧在于：颜色下采样和降分辨率影响多大，不同研究的结论不一样。
4. **码率怎么分配**（帧率、分辨率、画质怎么取舍）：只有心率方向的零散结果，包括帧率、分辨率、视频通话里的丢帧。针对呼吸的没有找到；用 WebRTC 的 `degradationPreference` 研究生理信号的工作也没有找到。
5. **所以 Q1、Q2、Q3 目前看都还是空白。** Q4 的「压缩增强训练」对心率已有人做过，对呼吸没有。

**研究问题暂时不用调整。** 但有两个风险点，要等拿到全文后核对：

- 2025 年 ACM 综述（resPyre）的正文里，有没有按码率做的呼吸实验；
- 2026 年的 arXiv 2606.04198（11 种编码退化 × 13 种方法）有没有测呼吸。

其中任何一篇如果已经系统测过呼吸随码率的变化，就按 GUIDE 第 9 节改成「复现 + 扩展」，把重点放在 Q2 的拆分（M1、M2、M3 三条测量）和 Q3 的码率分配上。到时我会在 PR 里提出，不自己改计划。

## 2. 本项目暂定的新意

1. 系统测量压缩（编码器 × 码率 × 码率控制方式）对摄像头测呼吸的影响，同时覆盖运动法（M1）和颜色法（M2）。
2. 在同一批视频上用三条测量拆开两个因素：伤害的是「颜色信号」，还是「快的信号」。
3. 在固定码率下比较帧率、分辨率和画质的取舍对呼吸和心率有什么不同，并对应到 WebRTC 的 `degradationPreference`。

## 3. 分类整理

「依据」一列：**摘要**表示只看了检索结果里的摘要或片段，**全文**表示读过全文。目前全部是摘要。

### 3.1 压缩对测心率的影响

| 文献 | 做了什么 | 主要发现 | 依据 |
|---|---|---|---|
| McDuff 等，2017 [1] | x264、x265，不同 CRF | 很轻的压缩就会明显降低脉搏信号的信噪比；码率足够高时还能保留可用的信号 | 摘要 |
| Rapczynski 等，IEEE TBME 2019 [2] | H.264、H.265，压缩率、分辨率、颜色下采样 | 压缩率越高心率越不准；分辨率降到一定程度以内、以及颜色下采样，对心率影响不大 | 摘要 |
| Špetlík 等，FG 2018 workshop [3] | H.264，5 人 | 降低分辨率会放大 H.264 压缩的影响；颜色下采样有负面影响，和 [2] 矛盾 | 摘要 |
| Yu 等，ICCV 2019 [4] | 高度压缩的人脸视频 | 波形变噪、峰值位置不准；提出视频增强网络来恢复 | 摘要 |
| Nowara、McDuff，ICCVW 2019 [5] | 用压缩视频训练 | 训练视频的压缩程度不低于测试视频时，信噪比和误差都有改善 | 摘要 |
| Gudi 等，2020 [6] | 在 PURE 上比较编码 | H.265 在最低码率下最能保留脉搏成分 | 摘要 |
| 多种压缩算法对 BVP 的评估，BSPC 2024 [7] | 多种编码器 | 未读到细节 | 摘要 |
| Comas 等，2024（ESWA 2026）[8] | 压缩视频上的脉搏信号放大 | H.265 优于 H.264，尤其在运动大的视频上 | 摘要 |
| arXiv 2606.04198，2026 [9] | 3 个数据集、11 种编码退化、13 种方法 | 方法能否扛住压缩，取决于压缩伪影在脸部各块之间是否空间一致；指出部署系统常用的低码率区间缺少研究 | 摘要 |

### 3.2 摄像头测呼吸：方法和基准

| 文献 | 内容 | 和本项目的关系 | 依据 |
|---|---|---|---|
| ACM 综述和基准，2025，resPyre [10] | 运动法和 rPPG 法测呼吸的综述和开源基准 | rPPG 推呼吸最不可靠；压缩只做了定性讨论（待全文核对） | 摘要 |
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
| Nguyen 等，CBM 2024 [19] | 编码伪影、低光、遮挡、丢帧对 rPPG 的影响 | 摘要 |

### 3.4 呼吸参考：从 PPG 推呼吸（UBFC-rPPG 要用）

| 文献 | 内容 | 依据 |
|---|---|---|
| Charlton 等，Physiol. Meas. 2016 [20] | 比较了几百种从心电和 PPG 推呼吸的算法；PPG 比心电差；好的算法都融合了多种调制 | 摘要 |
| Charlton 等，IEEE RBME 2018 [21] | 上面这类算法的综述：基线、幅度、频率三种调制 | 摘要 |

这支持 GUIDE 第 4.1 节的做法：UBFC 的呼吸参考只用三种推法一致的时间窗，并报告可用比例。

## 4. 检索方法

- **日期**：2026-10-09。
- **渠道**：只有 WebSearch（通用搜索引擎），返回检索结果的摘要。云端会话的网络白名单拦截了 arxiv.org、dl.acm.org、PubMed、Semantic Scholar 和 OpenAlex，所以读不到全文，也拿不到数据库的命中数。
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
  - 只看了摘要，可能漏掉正文里的呼吸实验；
  - 搜索引擎对 2026 年的新论文收录可能不全。

## 5. 待核对（需要全文）

| 优先级 | 文章 | 要核对什么 |
|---|---|---|
| 1 | [10] ACM 综述和 resPyre | 正文里有没有压缩实验；有的话，用了哪些码率、测了哪些方法 |
| 1 | [9] arXiv 2606.04198 | 是否也测了呼吸 |
| 1 | [19] Nguyen 等 2024 | 编码伪影实验的设置，有没有呼吸 |
| 1 | SCAMPS 论文 [22] | 呼吸是怎么驱动画面的（身体起伏、头动还是只有颜色）；帧率和每段时长。阶段 1 的脚本先假设 30 帧/秒 |
| 2 | [1]、[2]、[5]、[7]、[8] | 压缩设置和主要数字，阶段 2 定压缩阶梯时参考 |
| 2 | [11]、[12]、[20]、[21] | 阶段 2 定 M1 和呼吸参考时参考 |

## 6. 参考文献

1. McDuff, Blackford, Estepp. The Impact of Video Compression on Remote Cardiac Pulse Measurement Using Imaging Photoplethysmography. 2017. <https://www.microsoft.com/en-us/research/wp-content/uploads/2020/11/McDuff_2017_Impact2.pdf>
2. Rapczynski et al. Effects of Video Encoding on Camera Based Heart Rate Estimation. IEEE TBME 2019. <https://doi.org/10.1109/tbme.2019.2904326>
3. Špetlík, Čech, Matas. Non-Contact Reflectance Photoplethysmography: Progress, Limitations, and Myths. FG 2018 workshop. <https://cmp.felk.cvut.cz/ftp/articles/cech/Spetlik-FGw-2018.pdf>
4. Yu et al. Remote Heart Rate Measurement from Highly Compressed Facial Videos. ICCV 2019. <https://arxiv.org/abs/1907.11921>
5. Nowara, McDuff. Combating the Impact of Video Compression on Non-Contact Vital Sign Measurement Using Supervised Learning. ICCVW 2019. <https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Nowara_Combating_the_Impact_of_Video_Compression_on_Non-Contact_Vital_Sign_ICCVW_2019_paper.pdf>
6. Gudi et al. Real-time Webcam Heart-Rate and Variability Estimation with Clean Ground Truth for Evaluation. 2020. <https://arxiv.org/abs/2012.15846>
7. A comprehensive evaluation of multiple video compression algorithms for preserving BVP signal quality. Biomedical Signal Processing and Control, 2024. <https://www.sciencedirect.com/science/article/abs/pii/S1746809424015039>
8. Comas et al. Deep Pulse-Signal Magnification for Remote Heart Rate Estimation in Compressed Videos. 2024; Expert Systems with Applications 2026. <https://arxiv.org/abs/2405.02652>
9. Spatial Artifact Coherence Determines Codec Robustness in Patch-Based rPPG. 2026. <https://arxiv.org/abs/2606.04198>
10. Remote Respiration Measurement with RGB Cameras: A Review and Benchmark. ACM, 2025. <https://doi.org/10.1145/3771763>；resPyre：<https://github.com/phuselab/resPyre>
11. Wang et al. Algorithmic insights of camera-based respiratory motion extraction. Physiological Measurement, 2022. <https://iopscience.iop.org/article/10.1088/1361-6579/ac5b49>
12. Revisiting motion-based respiration measurement from videos. University of Oulu. <https://www.6gflagship.com/?p=3612>
13. Protopopov. A Robust Camera-based Method for Breath Rate Measurement. 2025. <https://arxiv.org/abs/2512.03827>
14. van Gastel, Stuijk, de Haan. Robust respiration detection from remote photoplethysmography. Biomedical Optics Express, 2016.
15. Chen et al. Respiratory Rate Estimation from Face Videos. 2019. <https://arxiv.org/abs/1909.03503>
16. Blackford, Estepp. Effects of frame rate and image resolution on pulse rate measured using multiple camera imaging photoplethysmography. SPIE 2015.
17. Mironenko et al. Remote Photoplethysmography: Rarely Considered Factors. CVPRW 2020. <https://arxiv.org/abs/2004.12695>
18. Álvarez Casado et al. SCIA 2023. <https://doi.org/10.1007/978-3-031-31438-4_38>
19. Nguyen et al. Evaluation of Video-Based rPPG in Challenging Environments: Artifact Mitigation and Network Resilience. Computers in Biology and Medicine, 2024. <https://arxiv.org/abs/2405.01230>
20. Charlton et al. An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram. Physiological Measurement, 2016. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5390977/>
21. Charlton et al. Breathing Rate Estimation From the Electrocardiogram and Photoplethysmogram: A Review. IEEE Reviews in Biomedical Engineering, 2018. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7612521/>
22. McDuff et al. SCAMPS: Synthetics for Camera Measurement of Physiological Signals. NeurIPS 2022 Datasets and Benchmarks. <https://arxiv.org/abs/2206.04197>
