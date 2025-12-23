# 深度学习中的不确定性估计（按应用领域分类）

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>

本仓库将深度学习不确定性估计领域的论文按具体应用场景进行了分类整理，主要涵盖：**🏥 医疗**、**🚗 自动驾驶**、**🛡️ 军事/安防** 以及 **📦 其他领域**。
---
## 🏥 医疗应用 (Medical Application)

> **概述**: 在医学影像分析中，不确定性量化对于建立可信的辅助诊断系统至关重要。该领域主要涵盖**肿瘤分割**、**MRI/CT 重建**以及**病理诊断**，重点在于区分可靠预测与需要医生人工复核的风险区域。

### 会议论文 (Conference Papers)

- **[Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation]** [MICCAI 2019]
  - **不确定性表征:** `体素级方差` (Voxel-wise Variance) / `预测熵` (Predictive Entropy)
  - **方法简介:** 提出一种用于半监督分割的教师-学生（Teacher-Student）框架。通过对教师模型进行随机增强（Stochastic Augmentation）产生不确定性图，并利用该图来加权一致性损失函数，迫使模型重点学习可靠区域，忽略不确定性高的模糊边界。
  - [论文链接](https://arxiv.org/abs/1806.05034) - [[PyTorch]](https://github.com/yulequan/UA-MT)

- **[Efficient Bayesian Uncertainty Estimation for nnU-Net]** [MICCAI 2022]
  - **不确定性表征:** `集成方差` (Ensemble Variance) / `熵` (Entropy)
  - **方法简介:** 将可扩展的贝叶斯方法（特别是轻量级集成和 MC-Dropout）集成到目前医学图像分割的最强基线模型 **nnU-Net** 中。该方法在不显著增加计算成本的前提下，实现了高质量的不确定性校准，为医学分割提供了一个标准化的基准。
  - [论文链接](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_51)

- **[TBraTS: Trusted Brain Tumor Segmentation]** [MICCAI 2022]
  - **不确定性表征:** `空虚度` (Vacuity, 证据不确定性)
  - **方法简介:** 将**证据深度学习 (Evidential Deep Learning)** 应用于脑肿瘤分割。通过 Dirichlet 分布对类别概率建模，网络能够输出一个代表“证据缺失”的“空虚度”分数。这使得模型能够显式地标记出数据分布外（OOD）或模棱两可的区域，提高临床安全性。
  - [论文链接](https://arxiv.org/abs/2206.09309) - [[PyTorch]](https://github.com/cocofeat/tbrats)

- **[CRISP - Reliable Uncertainty Estimation for Medical Image Segmentation]** [MICCAI 2022]
  - **不确定性表征:** `校准误差` (Calibration Error) / `风险-覆盖曲线` (Risk-Coverage Curve)
  - **方法简介:** 专注于对分割不确定性图进行严格的评估与校准。文章提出了一套重校准（Recalibration）框架，确保预测的置信度具有临床意义，即模型预测的“90%确信”应对应真实的“90%准确率”。
  - [论文链接](https://arxiv.org/abs/2206.07664)

- **[Learning to Predict Error for MRI Reconstruction]** [MICCAI 2021]
  - **不确定性表征:** `像素级误差图` (Pixel-wise Error Map)
  - **方法简介:** 不同于传统的偶然不确定性估计，该方法训练了一个独立的“看门狗（Watchdog）”网络，直接预测主 MRI 重建模型的像素级重建误差（L1 差异），从而提供直接的图像质量评估图。
  - [论文链接](https://arxiv.org/abs/2002.05582)

- **[Diagnostic Uncertainty Calibration: Towards Reliable Machine Predictions in Medical Domain]** [AIStats 2021]
  - **不确定性表征:** `期望校准误差` (ECE)
  - **方法简介:** 深入研究了医疗诊断分类中的模型未校准问题。针对医疗数据通常存在的数据不平衡和小样本特性，提出了改进的温度缩放（Temperature Scaling）等校准技术，以确保诊断概率的可靠性。
  - [论文链接](https://arxiv.org/pdf/2007.01659)

- **[Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning]** [MIDL 2020]
  - **不确定性表征:** `高斯方差` ($\sigma^2$)
  - **方法简介:** 解决了医学回归任务（如细胞计数、骨龄估计）中的不确定性问题。通过优化高斯分布的负对数似然（NLL），将标量校准方法扩展到回归任务，确保预测的方差能准确反映误差的幅度。
  - [论文链接](http://proceedings.mlr.press/v121/laves20a.html) - [[PyTorch]](https://github.com/mlaves/well-calibrated-regression-uncertainty)

- **[Uncertainty-aware GAN with Adaptive Loss for Robust MRI Image Enhancement]** [ICCV Workshop 2021]
  - **不确定性表征:** `偶然不确定性图` (Aleatoric Uncertainty Map)
  - **方法简介:** 在生成对抗网络（GAN）中引入不确定性估计用于 MRI 图像增强。利用像素级的不确定性图动态调整重建损失的权重，使模型在训练时能够“容忍”噪声和伪影，从而提高鲁棒性。
  - [论文链接](https://arxiv.org/pdf/2110.03343.pdf)

### 期刊论文 (Journal Papers)

- **[Region-Based Evidential Deep Learning to Quantify Uncertainty and Improve Robustness of Brain Tumor Segmentation]** [NCA 2022]
  - **不确定性表征:** `区域级信度` (Regional Belief Mass)
  - **方法简介:** 将证据学习从像素级扩展到区域级。通过在超像素或解剖区域内聚合证据，为肿瘤边界提供更稳定的不确定性估计，显著提高了对噪声数据的鲁棒性。
  - [论文链接](http://arxiv.org/abs/2208.06038)

- **[Explainable machine learning in image classification models: An uncertainty quantification perspective]** [Knowledge-Based Systems 2022]
  - **不确定性表征:** `可解释性热力图` (Saliency Maps) vs `不确定性图`
  - **方法简介:** 探讨了模型可解释性（Explainability）与预测不确定性之间的关系，旨在提供一个统一的视角：利用不确定性来解释“为什么”医学模型在特定案例上可能会失效。
  - [论文链接](https://www.sciencedirect.com/science/article/pii/S095070512200168X)

### 预印本 (ArXiv Preprints)

- **[Deep evidential fusion with uncertainty quantification and contextual discounting for multimodal medical image segmentation]** [arXiv 2023]
  - **不确定性表征:** `Dempster-Shafer 不确定性` / `上下文折扣因子`
  - **方法简介:** 提出基于 Dempster-Shafer 证据理论的多模态融合框架。根据各模态（如 T1、T2 MRI 序列）的估计不确定性，动态地“折扣”（降低权重）不可靠的模态，解决了临床中常见的模态缺失或伪影问题。
  - [论文链接](https://arxiv.org/abs/2309.05919)

- **[SoftDropConnect (SDC) – Effective and Efficient Quantification of the Network Uncertainty in Deep MR Image Analysis]** [arXiv 2022]
  - **不确定性表征:** `DropConnect 方差`
  - **方法简介:** 提出了 SoftDropConnect，一种在推理阶段随机丢弃权重连接（而非神经元）的方法。相比传统的 MC-Dropout，该方法在密集的 MRI 分析任务中能更高效、更准确地量化结构不确定性。
  - [论文链接](https://arxiv.org/abs/2201.08418)
---
## 🚗 自动驾驶 (Autonomous Driving)

> **概述**: 自动驾驶场景对安全性要求极高，不确定性估计主要用于：1) 避免在检测结果不可靠时做出危险决策；2) 融合多传感器（雷达/相机）数据；3) 检测路面异常物体（如遗落货物）和应对恶劣天气（雨雪雾）。

### 会议论文 (Conference Papers)

- **[Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty]** [ICCV 2019]
  - **不确定性表征:** `高斯方差` (Gaussian Variance, 坐标级)
  - **方法简介:** 针对目标检测任务，将边界框（Bounding Box）的坐标建模为高斯分布，而不仅仅是点估计。模型同时预测坐标值及其不确定性（方差），并利用该不确定性在非极大值抑制（NMS）阶段降低定位不准的框的置信度，从而显著提高检测精度（mAP）。
  - [论文链接](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf) - [[CUDA]](https://github.com/jwchoi384/Gaussian_YOLOv3)

- **[What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?]** [NeurIPS 2017]
  - **不确定性表征:** `偶然不确定性` (异方差) & `认知不确定性` (MC-Dropout)
  - **方法简介:** 计算机视觉不确定性领域的奠基之作。文章提出了一个统一的框架，同时建模**偶然不确定性**（Aleatoric，通过学习输入依赖的损失衰减项来捕捉数据噪声）和**认知不确定性**（Epistemic，通过MC-Dropout捕捉模型知识盲区），并展示了其在语义分割和深度估计中的有效性。
  - [论文链接](https://arxiv.org/abs/1703.04977)

- **[Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation]** [CVPR 2023]
  - **不确定性表征:** `误差分布` (Error Distribution)
  - **方法简介:** 在双目立体匹配（Stereo Matching）任务中，不仅仅输出单一的视差值，而是学习匹配误差的完整概率分布。这使得自动驾驶系统能够获得距离估计的置信区间，对于避障至关重要。
  - [论文链接](https://arxiv.org/abs/2304.00152) - [[PyTorch]](https://github.com/lly00412/sednet)

- **[Hyperdimensional Uncertainty Quantification for Multimodal Uncertainty Fusion in Autonomous Vehicles Perception]** [CVPR 2025]
  - **不确定性表征:** `超维向量算术` (Hyperdimensional Vector Arithmetic)
  - **方法简介:** 提出一种基于超维计算（Hyperdimensional Computing）的新颖方法，用于在感知任务中融合来自不同传感器（如激光雷达和摄像头）的不确定性信息。该方法在处理高维多模态数据时具有计算高效性和强鲁棒性。
  - [论文链接](https://arxiv.org/abs/2503.20011)

- **[PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness]** [CVPR 2024]
  - **不确定性表征:** `体素占用概率` (Voxel Occupancy Probability)
  - **方法简介:** 针对3D全景场景补全（Semantic Scene Completion）任务，引入不确定性感知机制。模型能够推断被遮挡区域（例如车辆后方）的几何结构和语义类别，并给出置信度，帮助规划路径。
  - [论文链接](https://arxiv.org/pdf/2312.02158.pdf) - [[PyTorch]](https://github.com/astra-vision/PaSCo)

- **[Learning Uncertainty For Safety-Oriented Semantic Segmentation In Autonomous Driving]** [ICIP 2022]
  - **不确定性表征:** `预测方差` (Predictive Variance)
  - **方法简介:** 设计了一种面向安全的损失函数，利用预测方差来惩罚模型在高风险区域（如行人、车辆边缘）的错误分类，从而在保持精度的同时提高系统在关键区域的可靠性。
  - [论文链接](https://arxiv.org/abs/2105.13688)

- **[Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection]** [CVPR 2023]
  - **不确定性表征:** `校准误差` (Calibration Error)
  - **方法简介:** 针对目标检测中置信度与定位精度不匹配的问题（即高置信度但框不准，或反之），提出了一种新的训练时损失函数（TCD），强制模型输出的分类置信度能够反映预测框的IoU质量。
  - [论文链接](https://arxiv.org/pdf/2303.14404.pdf)

### Workshop & 期刊论文

- **[Fishyscapes: A Benchmark for Safe Semantic Segmentation in Autonomous Driving]** [ICCV Workshop 2019]
  - **不确定性表征:** `异常分数` (Anomaly Score) / `最大Softmax概率`
  - **方法简介:** 建立了一个专门的基准测试（Benchmark），用于评估语义分割模型检测“路面异常物体”（如遗落的箱子、石头）的能力。该基准主要测试模型能否利用不确定性将OOD物体与背景区分开来。
  - [论文链接](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Blum_Fishyscapes_A_Benchmark_for_Safe_Semantic_Segmentation_in_Autonomous_Driving_ICCVW_2019_paper.html)

- **[Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming]** [NeurIPS Workshop 2019]
  - **不确定性表征:** `mAP下降率` (mAP reduction under corruption)
  - **方法简介:** 这是一个关于鲁棒性的基准研究，系统评估了目标检测模型在面临恶劣天气（雪、雨、雾）和图像腐蚀时的性能下降情况，强调了在不同天气域适应中不确定性的作用。
  - [论文链接](https://arxiv.org/abs/1907.07484) - [[GitHub]](https://github.com/bethgelab/robust-detection-benchmark)

- **[Semantic Foggy Scene Understanding with Synthetic Data]** [IJCV 2018]
  - **不确定性表征:** `透射率不确定性` (Transmission Map Uncertainty)
  - **方法简介:** 专注于雾天驾驶场景。通过合成雾天数据训练模型，并显式建模雾气浓度（透射率）带来的不确定性，从而从模糊的视觉输入中恢复出清晰的语义分割结果。
  - [论文链接](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)

- **[Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles]** [IROS 2016]
  - **不确定性表征:** `视差方差` (Disparity Variance)
  - **方法简介:** 经典的自动驾驶安全论文。利用立体视觉（Stereo Vision）产生的视差不确定性来检测道路上微小的、未分类的障碍物，解决了长尾物体难以检测的问题。
  - [论文链接](https://arxiv.org/abs/1609.04653)

### 数据集与基准 (Benchmarks)

- **[MUAD: Multiple Uncertainties for Autonomous Driving]** [BMVC 2022]
  - **简介:** 一个旨在解耦不确定性来源的合成数据集。它提供了多种挑战场景，用于测试模型是否能区分**数据不确定性**（如恶劣天气、传感器噪声）和**知识不确定性**（如从未见过的物体、白天到黑夜的域偏移）。
  - [论文链接](https://arxiv.org/abs/2203.01437) - [[PyTorch]](https://github.com/ENSTA-U2IS-AI/MUAD-Dataset)

- **[SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation]** [CVPR 2022]
  - **简介:** 具有连续域变化（如云层逐渐变厚、光照连续变化）的大型合成数据集，非常适合评估不确定性估计在域适应（Domain Adaptation）过程中的表现。
  - [论文链接](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_SHIFT_A_Synthetic_Driving_Dataset_for_Continuous_Multi-Task_Domain_Adaptation_CVPR_2022_paper.html)
---
## 🛡️ 军事与安防 (Military & Security)

> **概述**: 此领域关注人工智能在极端对抗环境下的生存能力。主要任务包括防御敌方的**对抗攻击**（Adversarial Attacks）、高精度的**生物特征识别**（如人脸/步态）、以及在未知环境下的**系统级异常检测**（System-level OOD Detection）。

### 会议论文 (Conference Papers)

- **[Understanding Measures of Uncertainty for Adversarial Example Detection]** [UAI 2018]
  - **不确定性表征:** `互信息` (Mutual Information) / `熵` (Entropy)
  - **方法简介:** 这是一个关于防御机制的基础研究。文章深入分析了贝叶斯神经网络（BNN）和 Dropout 在面对**对抗样本**（被恶意扰动以欺骗AI的图像）时的不确定性表现。研究发现，利用不确定性度量可以有效检测出潜在的攻击，防止系统被欺骗。
  - [论文链接](https://arxiv.org/abs/1803.08533)

- **[To Trust Or Not To Trust A Classifier]** [NeurIPS 2018]
  - **不确定性表征:** `信任评分` (Trust Score, 基于 KNN 距离)
  - **方法简介:** 针对高风险决策场景，提出了一种非参数化的“信任评分”。该方法计算测试样本到预测类别的最近邻距离与到次优类别的最近邻距离之比。低信任评分往往意味着样本不仅是 OOD，而且极有可能是对抗攻击生成的，非常适合安全审计。
  - [论文链接](https://arxiv.org/abs/1805.11783) - [[Python]](https://github.com/google/TrustScore)

- **[Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks]** [ICCV 2021]
  - **不确定性表征:** `校准后的 Softmax 概率`
  - **方法简介:** 为了提高安全系统的拒识能力，该方法在训练过程中主动生成位于决策边界附近的对抗性 OOD 样本（Boundary Examples）。这迫使模型收缩决策边界，显著提高了对伪装目标或恶意输入的识别能力。
  - [论文链接](https://arxiv.org/abs/2108.01634) - [[PyTorch]](https://github.com/valeoai/obsnet)

- **[Assessing Uncertainty in Similarity Scoring: Performance & Fairness in Face Recognition]** [ICLR 2024]
  - **不确定性表征:** `相似度方差` (Similarity Variance)
  - **方法简介:** 聚焦于**人脸识别**（监控与安防的核心技术）。文章指出仅有点估计的相似度评分是不够的，提出了一种概率框架来量化两张人脸特征向量相似度的不确定性，这对于高安全等级的门禁和身份验证系统至关重要。
  - [论文链接](https://arxiv.org/abs/2211.07245)

- **[Quantification of Uncertainty with Adversarial Models]** [NeurIPS 2023]
  - **不确定性表征:** `对抗边界距离` (Adversarial Margin)
  - **方法简介:** 提出利用对抗鲁棒性模型来量化不确定性。核心思想是：如果一个样本很容易被攻击（即只需微小扰动就能改变分类），那么模型对该样本的不确定性就应该很高。这为评估战术边缘计算设备的可靠性提供了新思路。
  - [论文链接](https://arxiv.org/abs/2307.03217)

- **[Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness]** [NeurIPS 2019]
  - **不确定性表征:** `Dirichlet 精度` (Dirichlet Precision)
  - **方法简介:** 改进了先验网络（Prior Networks）的训练目标，利用反向 KL 散度，使得模型在面对 OOD 数据或攻击数据时，输出的分布不仅是平坦的（高熵），而且在分布空间上更加分散，从而不仅能检测异常，还能提升对抗鲁棒性。
  - [论文链接](https://proceedings.neurips.cc/paper/2019/hash/7dd2ae7db7d18ee7c9425e38df1af5e2-Abstract.html)

- **[Uncertainty in Gradient Boosting via Ensembles]** [ICLR 2021]
  - **不确定性表征:** `集成方差` (Ensemble Variance)
  - **方法简介:** 虽然是通用方法，但在结构化数据（如雷达信号分类、网络入侵检测日志）中非常重要。文章提出了基于梯度提升树（GBDT）的集成不确定性估计，适用于处理非图像类的安防数据。
  - [论文链接](https://arxiv.org/abs/2006.10562) - [[PyTorch]](https://github.com/yandex-research/GBDT-uncertainty)

### 预印本 (ArXiv Preprints)

- **[A System-Level View on Out-of-Distribution Data in Robotics]** [arXiv 2022]
  - **不确定性表征:** `系统级异常分` (System-level Anomaly Score)
  - **方法简介:** 将视野从单一模型的 OOD 检测扩展到**机器人系统级**。这对于无人作战平台（UGV/UAV）至关重要，因为系统需要区分环境的新颖性（如进入新地形）与传感器故障或敌方干扰导致的异常，从而做出正确的战术决策。
  - [论文链接](https://arxiv.org/abs/2212.14020)

- **[Similarit-Distance-Magnitude Universal Verification]** [arXiv 2025]
  - **不确定性表征:** `SDM 置信度`
  - **方法简介:** 提出一种通用的验证框架，结合相似度、距离和特征幅度来综合评估样本的可靠性，适用于开放世界识别（Open-World Recognition）场景，防止未知敌方目标的误判。
  - [论文链接](https://arxiv.org/abs/2502.20167)
---
## 📦 其他应用与通用理论 (Others: Theory, NLP & General)

> **概述**: 本部分收录了不确定性估计的**基础理论方法**（适用于所有领域）、**自然语言处理（NLP）** 中的最新进展（特别是大模型幻觉检测），以及**工业缺陷检测**相关应用。

### 🧠 基础理论与通用方法 (Foundational Theory & General Methods)

> *这些论文提出了被广泛引用的通用框架，是入门该领域的必读文献。*

- **[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles]** [NeurIPS 2017]
  - **不确定性表征:** `集成方差` (Ensemble Variance)
  - **方法简介:** **黄金基准 (Gold Standard)**。通过训练多个随机初始化（或使用对抗样本训练）的深度神经网络，利用它们预测结果的方差来估计认知不确定性。虽然计算成本高，但效果往往最稳健。
  - [论文链接](https://arxiv.org/abs/1612.01474)

- **[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning]** [ICML 2016]
  - **不确定性表征:** `MC-Dropout 方差`
  - **方法简介:** **理论基石**。从数学上证明了在测试阶段开启 Dropout 等价于对高斯过程（Gaussian Process）的变分近似。通过多次前向传播采样，可以低成本地估计模型的不确定性。
  - [论文链接](https://arxiv.org/abs/1506.02142)

- **[Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation]** [CVPR 2025]
  - **不确定性表征:** `自适应 Dropout 方差`
  - **方法简介:** **最新进展**。不同于固定的 Dropout 率，该方法根据输入的信息量自适应地调整 Dropout 概率，从而在推理阶段提供更精准、更高效的不确定性估计。
  - [论文链接](https://arxiv.org/abs/2412.07169) - [[PyTorch]](https://github.com/code-supplement-25/rate-in)

- **[Deep Evidential Regression]** [NeurIPS 2020]
  - **不确定性表征:** `逆伽马分布参数` (Inverse-Gamma Parameters)
  - **方法简介:** **证据深度学习 (EDL)** 的代表作。网络不再输出单一预测值，而是直接预测高阶分布（Normal-Inverse-Gamma）的参数。单次前向传播即可分离出偶然不确定性（数据噪声）和认知不确定性（模型无知）。
  - [论文链接](https://arxiv.org/abs/1910.02600)

- **[Probabilistic Contrastive Learning Recovers the Correct Aleatoric Uncertainty]** [ICML 2023]
  - **不确定性表征:** `浓度参数` (Concentration Parameter $\kappa$)
  - **方法简介:** 将对比学习（Contrastive Learning）扩展到概率框架。在超球面上对特征进行建模（Von Mises-Fisher 分布），分布的“集中度”直接反映了样本的偶然不确定性（例如模糊图像的分布会更发散）。
  - [论文链接](https://arxiv.org/pdf/2302.02865.pdf)

- **[Spectral-normalized Neural Gaussian Process (SNGP)]** [ICLR 2019/2021]
  - **不确定性表征:** `距离感知不确定性` (Distance-aware Uncertainty)
  - **方法简介:** 通过谱归一化（Spectral Normalization）限制网络 Lipschitz 常数，并结合随机特征映射（Random Fourier Features），使深度网络具备高斯过程的性质，能根据测试样本与训练数据的距离输出高质量的不确定性。
  - [论文链接](https://arxiv.org/abs/1808.05587)

### 💬 自然语言处理与大模型 (NLP & LLMs)

> *关注大语言模型（LLM）的幻觉检测、回答一致性及序列生成的可信度。*

- **[R-U-SURE? Uncertainty-Aware Code Suggestions]** [ICML 2023]
  - **不确定性表征:** `Token 概率` / `序列熵`
  - **方法简介:** 针对代码生成助手（如 Copilot），估计模型生成的代码建议的“效用不确定性”。如果模型对自己生成的代码信心不足，则选择不打扰用户，从而提升用户体验。
  - [论文链接](https://arxiv.org/pdf/2303.00732.pdf)

- **[Strength in Numbers: Estimating Confidence of Large Language Models by Prompt Agreement]** [TrustNLP 2023]
  - **不确定性表征:** `回答一致性` (Consistency Score)
  - **方法简介:** 通过对同一问题进行多次不同 Prompt 的询问（Prompt Augmentation），计算 LLM 回答的一致性。如果模型在不同提问方式下回答一致，则认为其置信度高，反之则可能在“一本正经地胡说”。
  - [论文链接](https://github.com/JHU-CLSP/Confidence-Estimation-TrustNLP2023)

- **[Disentangling Uncertainty in Machine Translation Evaluation]** [EMNLP 2022]
  - **不确定性表征:** `数据不确定性` vs `模型不确定性`
  - **方法简介:** 在机器翻译质量评估（QE）中，尝试分离源文本本身的歧义（数据不确定性）和翻译模型能力的不足（模型不确定性），以提供更公平的评分。
  - [论文链接](https://arxiv.org/abs/2204.06546)

### 🏭 工业异常检测 (Industrial Anomaly Detection)

- **[Towards Total Recall in Industrial Anomaly Detection (PatchCore)]** [CVPR 2022]
  - **不确定性表征:** `核心集距离` (Distance to Coreset)
  - **方法简介:** 工业缺陷检测的 SOTA 方法。利用预训练网络的特征构建一个正常样本的“记忆库（Memory Bank）”，测试样本与记忆库特征的距离即为其异常分数（不确定性），广泛应用于流水线质检。
  - [论文链接](https://arxiv.org/abs/2106.08265) - [[PyTorch]](https://github.com/hcw-00/PatchCore_anomaly_detection)

### 📚 基准库与工具 (Benchmarks & Libraries)

- **[Uncertainty Baselines]** [arXiv 2021]
  - **简介:** Google 推出的标准化基准库，提供了多种任务（图像、文本）下的高质量不确定性基线模型实现。
  - [链接](https://github.com/google/uncertainty-baselines)

- **[TorchUncertainty]**
  - **简介:** 专门基于 PyTorch 的不确定性量化库，旨在快速复现 Deep Ensembles, Packed-Ensembles 等经典方法。
  - [链接](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)
---
## 🧪 纯理论研究与核心算法 (Pure Theoretical Research)

> **概述**: 本部分收录了不确定性估计的**底层方法论**。这些研究致力于改进神经网络的概率解释，涵盖贝叶斯近似、高效集成、确定性不确定性以及分布外检测（OOD）的基础理论。

### 1. 贝叶斯方法与变分推断 (Bayesian Methods & Variational Inference)

- **[Laplace Redux – Effortless Bayesian Deep Learning]** [NeurIPS 2021]
  - **不确定性表征:** `后验高斯近似` (Gaussian Posterior Approximation)
  - **方法简介:** 复兴了经典的**拉普拉斯近似 (Laplace Approximation)**。该方法在标准神经网络训练完成后，通过计算损失函数的曲率（Hessian 矩阵）来拟合权重的后验分布。它几乎不需要改变训练过程，就能低成本地将确定性网络转化为贝叶斯网络。
  - [论文链接](https://arxiv.org/abs/2106.14806) - [[PyTorch]](https://github.com/AlexImmer/Laplace)

- **[Training Bayesian Neural Networks with Sparse Subspace Variational Inference]** [ICLR 2024]
  - **不确定性表征:** `子空间变分参数` (Subspace Variational Parameters)
  - **方法简介:** 为了解决高维参数空间下贝叶斯推断难以收敛的问题，该方法限制变分推断在一个**稀疏子空间**内进行。这使得在超大模型上进行贝叶斯训练成为可能，同时保持了较高的不确定性估计质量。
  - [论文链接](https://arxiv.org/abs/2402.11025)

- **[A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods]** [NeurIPS 2023]
  - **不确定性表征:** `理论界` (Theoretical Bound)
  - **方法简介:** **理论分析**。长期以来，Deep Ensembles 被认为是频率学派的方法，但效果却优于贝叶斯方法。这篇文章从数学上证明了 Deep Ensembles 实际上可以被视为一种特殊的变分贝叶斯方法，填补了两者之间的理论鸿沟。
  - [论文链接](https://arxiv.org/pdf/2305.15027)

### 2. 高效集成与采样 (Efficient Ensembles & Sampling)

- **[BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning]** [ICLR 2020]
  - **不确定性表征:** `秩-1 调制集成方差`
  - **方法简介:** 针对 Deep Ensembles 计算成本过高（需要训练 N 个模型）的问题，提出了一种**参数共享**机制。所有集成成员共享一个主权重矩阵，仅通过一个轻量级的 Rank-1 矩阵进行调制。实现了以接近单个模型的成本获得集成模型的不确定性效果。
  - [论文链接](https://arxiv.org/abs/2002.06715) - [[TorchUncertainty]](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)

- **[Masksembles for Uncertainty Estimation]** [CVPR 2021]
  - **不确定性表征:** `掩码集成方差`
  - **方法简介:** 结合了 Dropout 和 Deep Ensembles 的优点。通过在训练时施加一组固定的二进制掩码（Masks），在单次前向传播中模拟多个子网络的输出，消除了多次推理的开销。
  - [论文链接](https://nikitadurasov.github.io/projects/masksembles/) - [[PyTorch]](https://github.com/nikitadurasov/masksembles)

### 3. 确定性不确定性 (Deterministic Uncertainty Quantification)

> *这类方法不需要多次采样（如 Dropout 或 Ensemble），单次前向传播即可获得不确定性，适合对实时性要求高的场景。*

- **[Deep Deterministic Uncertainty (DDU)]** [CVPR 2023 / 2021]
  - **不确定性表征:** `特征密度` (Feature Density / GMM Likelihood)
  - **方法简介:** 强制神经网络的特征空间服从高斯混合模型（GMM）分布。在测试时，通过计算样本在特征空间中的对数似然（Log-Likelihood）来直接衡量其是否属于分布内数据（In-Distribution），无需进行随机采样。
  - [论文链接](https://arxiv.org/abs/2102.11582) - [[PyTorch]](https://github.com/omegafragger/DDU)

- **[Spectral-normalized Neural Gaussian Process (SNGP)]** [ICLR 2021]
  - **不确定性表征:** `距离感知不确定性` (Distance-aware Uncertainty)
  - **方法简介:** 谷歌提出的经典方法。通过**谱归一化**（Spectral Normalization）限制网络的 Lipschitz 常数，防止特征空间坍塌；结合**随机傅里叶特征**（RFF）作为最后一层，使网络具有高斯过程的性质，能根据输入距离输出闭式解的不确定性。
  - [论文链接](https://arxiv.org/abs/2006.10108) - [[TensorFlow]](https://github.com/google/edward2)

### 4. 证据深度学习 (Evidential Deep Learning)

- **[Evidential Deep Learning to Quantify Classification Uncertainty]** [NeurIPS 2018]
  - **不确定性表征:** `狄利克雷分布参数` (Dirichlet Parameters / Belief Mass)
  - **方法简介:** **EDL 开山之作（分类任务）**。将分类器的输出视为狄利克雷分布的参数，从而对“二阶概率”（概率的概率）进行建模。它能明确区分**数据冲突**（两个类概率相近）和**数据缺乏**（所有类概率都低，即 Vacuity）。
  - [论文链接](https://arxiv.org/abs/1806.01768) - [[PyTorch]](https://github.com/dougbrion/pytorch-classification-uncertainty)

- **[Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts]** [NeurIPS 2020]
  - **不确定性表征:** `后验密度伪计数` (Density-Based Pseudo-Counts)
  - **方法简介:** 结合了标准化流（Normalizing Flows）和贝叶斯更新。利用标准化流估计潜在空间的密度，并将此密度转换为狄利克雷分布的伪计数参数，从而在没有任何 OOD 训练数据的情况下实现稳健的 OOD 检测。
  - [论文链接](https://arxiv.org/abs/2006.09239) - [[PyTorch]](https://github.com/sharpenb/Posterior-Network)

### 5. 校准与 OOD 检测理论 (Calibration & OOD Theory)

- **[On Calibration of Modern Neural Networks]** [ICML 2017]
  - **不确定性表征:** `温度缩放` (Temperature Scaling) / `ECE`
  - **方法简介:** **必读经典**。揭示了现代深度神经网络虽然精度高但往往“过度自信”（Overconfident）。提出了**温度缩放（Temperature Scaling）**，一种简单高效的事后校准方法，通过在 Softmax 层前除以一个标量 $T$ 来优化预测概率的校准度。
  - [论文链接](https://arxiv.org/abs/1706.04599)

- **[Energy-based Out-of-distribution Detection]** [NeurIPS 2020]
  - **不确定性表征:** `自由能` (Free Energy)
  - **方法简介:** 提出使用物理学中的**自由能（Energy Score）**替代传统的 Softmax 置信度来检测 OOD 样本。理论上证明了 Energy Score 与数据的对数似然密度对齐，不需要重新训练模型即可显著提升 OOD 检测性能。
  - [论文链接](https://arxiv.org/abs/2010.03759)

- **[Rethinking Aleatoric and Epistemic Uncertainty]** [ICML 2025]
  - **不确定性表征:** `互信息` (Mutual Information)
  - **方法简介:** **最新理论**。对偶然不确定性和认知不确定性的定义进行了重新审视和严格的数学形式化，指出了现有方法在解耦这两种不确定性时的理论缺陷，并提出了修正方案。
  - [论文链接](https://arxiv.org/abs/2412.20892)
## 🧪 纯理论研究与核心算法 (Pure Theoretical Research)


### 6. 基于采样与 Dropout 的方法 (Sampling & Dropout-based Methods)

> *此类方法通过在推理阶段引入随机性（如随机丢弃神经元或数据增强）来近似贝叶斯积分，是目前最灵活、不需要改变模型结构即可使用的方法。*

- **[Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation]** [CVPR 2025]
  - **不确定性表征:** `自适应 Dropout 方差` (Adaptive Dropout Variance)
  - **方法简介:** **(重点推荐)** 针对传统 MC-Dropout 使用固定丢弃率导致估计不准的问题，提出了一种基于信息论的自适应机制。它根据输入图像的特征信息量动态调整每一层的 Dropout 率，在推理时能更精准地捕捉不确定性，同时显著减少计算开销。
  - [论文链接](https://arxiv.org/abs/2412.07169) - [[PyTorch]](https://github.com/code-supplement-25/rate-in)

- **[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning]** [ICML 2016]
  - **不确定性表征:** `MC-Dropout 方差`
  - **方法简介:** **理论奠基之作**。从数学上严格证明了在神经网络中应用 Dropout 实际上等价于对高斯过程（Gaussian Process）的贝叶斯变分推断。这一发现使得我们可以通过在测试时开启 Dropout 并进行多次前向传播来获得模型不确定性。
  - [论文链接](https://arxiv.org/abs/1506.02142) - [[TorchUncertainty]](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)

- **[Concrete Dropout]** [NeurIPS 2017]
  - **不确定性表征:** `可学习的 Dropout 率` (Learned Dropout Rate)
  - **方法简介:** 传统 Dropout 的丢弃率是超参数（如 0.5）。该文章利用 Concrete 分布（Gumbel-Softmax 的连续松弛）将 Dropout 率变成可微参数，使模型在训练过程中自动学习每一层的最佳丢弃率，从而优化不确定性估计。
  - [论文链接](https://arxiv.org/abs/1705.07832)

- **[Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models]** [CVPR 2024]
  - **不确定性表征:** `后验采样方差`
  - **方法简介:** 提出了一种简单策略，可以将任何预训练的标准神经网络转化为贝叶斯神经网络（BNN），无需从头训练。这对于利用现有的强大预训练模型进行不确定性估计非常有价值。
  - [论文链接](https://arxiv.org/abs/2312.15297)

### 7. 贝叶斯方法与变分推断 (Bayesian Methods & Variational Inference)

- **[Laplace Redux – Effortless Bayesian Deep Learning]** [NeurIPS 2021]
  - **不确定性表征:** `后验高斯近似` (Gaussian Posterior Approximation)
  - **方法简介:** 复兴了经典的**拉普拉斯近似**。在标准网络训练完成后，通过计算损失函数的曲率（Hessian 矩阵）来拟合权重的后验分布，低成本地将确定性网络转化为贝叶斯网络。
  - [论文链接](https://arxiv.org/abs/2106.14806) - [[PyTorch]](https://github.com/AlexImmer/Laplace)

- **[Training Bayesian Neural Networks with Sparse Subspace Variational Inference]** [ICLR 2024]
  - **不确定性表征:** `子空间变分参数`
  - **方法简介:** 为了解决高维参数空间下贝叶斯推断难以收敛的问题，该方法限制变分推断在一个**稀疏子空间**内进行，使得在超大模型上进行贝叶斯训练成为可能。
  - [论文链接](https://arxiv.org/abs/2402.11025)

- **[A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods]** [NeurIPS 2023]
  - **不确定性表征:** `理论界` (Theoretical Bound)
  - **方法简介:** 从数学上证明了 Deep Ensembles 实际上可以被视为一种特殊的变分贝叶斯方法，填补了频率学派集成方法与贝叶斯理论之间的鸿沟。
  - [论文链接](https://arxiv.org/pdf/2305.15027)

### 8. 高效集成方法 (Efficient Ensembles)

> *旨在解决传统 Deep Ensembles 计算和存储成本过高的问题。*

- **[BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning]** [ICLR 2020]
  - **不确定性表征:** `秩-1 调制集成方差`
  - **方法简介:** 通过**权重共享**机制，所有集成成员共享一个主权重矩阵，仅通过一个轻量级的 Rank-1 矩阵进行调制。实现了以接近单个模型的成本获得集成模型的效果。
  - [论文链接](https://arxiv.org/abs/2002.06715)

- **[Masksembles for Uncertainty Estimation]** [CVPR 2021]
  - **不确定性表征:** `掩码集成方差`
  - **方法简介:** 结合 Dropout 和 Deep Ensembles。通过在训练时施加一组固定的二进制掩码（Masks），在单次前向传播中模拟多个子网络的输出。
  - [论文链接](https://nikitadurasov.github.io/projects/masksembles/)

### 9. 确定性不确定性 (Deterministic Uncertainty Quantification)

> *单次前向传播即可获得不确定性，适合实时性要求高的场景。*

- **[Deep Deterministic Uncertainty (DDU)]** [CVPR 2023 / 2021]
  - **不确定性表征:** `特征密度` (Feature Density / GMM Likelihood)
  - **方法简介:** 强制神经网络的特征空间服从高斯混合模型（GMM）分布。测试时通过计算样本的对数似然（Log-Likelihood）来直接衡量其不确定性。
  - [论文链接](https://arxiv.org/abs/2102.11582) - [[PyTorch]](https://github.com/omegafragger/DDU)

- **[Spectral-normalized Neural Gaussian Process (SNGP)]** [ICLR 2021]
  - **不确定性表征:** `距离感知不确定性` (Distance-aware Uncertainty)
  - **方法简介:** 谷歌提出的经典方法。通过**谱归一化**限制网络 Lipschitz 常数，结合**随机傅里叶特征**（RFF），使网络具有高斯过程的性质，能根据输入距离输出闭式解的不确定性。
  - [论文链接](https://arxiv.org/abs/2006.10108)

### 10. 证据深度学习 (Evidential Deep Learning)

- **[Evidential Deep Learning to Quantify Classification Uncertainty]** [NeurIPS 2018]
  - **不确定性表征:** `狄利克雷分布参数` (Belief Mass)
  - **方法简介:** **EDL 开山之作**。将分类器输出视为狄利克雷分布的参数，能明确区分数据冲突（Aleatoric）和数据缺乏（Epistemic/Vacuity）。
  - [论文链接](https://arxiv.org/abs/1806.01768)

- **[Posterior Network: Uncertainty Estimation without OOD Samples]** [NeurIPS 2020]
  - **不确定性表征:** `基于密度的伪计数`
  - **方法简介:** 结合标准化流（Normalizing Flows）估计潜在空间密度，并将其转换为狄利克雷分布的伪计数参数，在无 OOD 数据训练的情况下实现稳健检测。
  - [论文链接](https://arxiv.org/abs/2006.09239)

### 11. 校准与 OOD 理论 (Calibration & OOD Theory)

- **[On Calibration of Modern Neural Networks]** [ICML 2017]
  - **不确定性表征:** `温度缩放` (Temperature Scaling)
  - **方法简介:** **必读经典**。揭示了深度神经网络的“过度自信”问题，并提出了简单高效的**温度缩放**方法进行事后校准。
  - [论文链接](https://arxiv.org/abs/1706.04599)

- **[Rethinking Aleatoric and Epistemic Uncertainty]** [ICML 2025]
  - **不确定性表征:** `互信息` (Mutual Information)
  - **方法简介:** **最新理论**。对偶然不确定性和认知不确定性的定义进行了重新审视，指出了现有方法在解耦这两种不确定性时的理论缺陷。
  - [论文链接](https://arxiv.org/abs/2412.20892)
