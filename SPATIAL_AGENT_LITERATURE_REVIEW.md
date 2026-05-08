# 空间智能体（Spatial Agent）研究全景文献综述

> 检索日期：2026-05-08  
> 检索策略：以核心论文为种子，沿 Related Works 向前追溯，再通过被引网络向后扩展，递归 2 轮后整理主干文献。  
> 范围界定：本文将“空间智能体”定义为能够在 3D、仿真或真实物理空间中进行感知、定位、建图、语言理解、空间推理、规划与行动的智能体。

## 0. 领域总览

- 空间智能体不是单一模型路线，而是由机器人学、Embodied AI、3D 场景理解、视觉语言导航、LLM/VLM 规划和 Vision-Language-Action（VLA）模型共同收敛而成的研究方向。
- 早期主线关注几何空间能力：定位、建图、路径规划和导航。
- 2017 年后，AI2-THOR、Matterport3D、ScanNet、Gibson、Habitat 等环境和数据集使 embodied AI 进入可复现实验阶段。
- 2022 年后，LLM/VLM 进入具身规划和机器人控制，形成“语言理解 + 空间表示 + 可执行动作”的空间智能体框架。
- 2024 年后，研究重心转向开放词汇、主动探索、3D 空间推理、长程记忆和真实机器人泛化。

## 1. 奠基层：空间表征、定位建图与可交互环境

### 1.1 认知地图与机器人空间表征

- **The Cognitive Map in Rats and Men, 1948**：提出“认知地图”概念，为后来的空间记忆、路径规划和 embodied intelligence 提供认知科学基础。
- **Probabilistic Robotics, 2005**：系统化概率定位、地图构建、规划与不确定性处理，成为机器人空间智能的工程底座。
- **ORB-SLAM: A Versatile and Accurate Monocular SLAM System, 2015**：用统一 ORB 特征实现实时单目跟踪、建图、重定位和闭环检测，是视觉 SLAM 的经典基线。
- **Neural SLAM: Learning to Explore with External Memory, 2017**：将 SLAM 式全局地图更新嵌入可微外部记忆，连接传统建图与深度强化学习探索。

### 1.2 大规模 3D 场景数据与仿真平台

- **ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes, 2017**：发布大规模 RGB-D 室内扫描及语义标注，成为 3D 语义理解、3D grounding 与 3D-QA 的核心数据源。
- **Matterport3D: Learning from RGB-D Data in Indoor Environments, 2017**：提供建筑级 RGB-D 全景、位姿、网格与语义标注，支撑真实室内导航与视觉语言导航。
- **AI2-THOR: An Interactive 3D Environment for Visual AI, 2017**：提供可交互室内场景与对象操作接口，推动 embodied visual AI、EQA 和语言任务执行。
- **Gibson Env: Real-World Perception for Embodied Agents, 2018**：用真实扫描空间构建可物理约束的仿真环境，强调 sim-to-real 感知训练。
- **Habitat: A Platform for Embodied AI Research, 2019**：提供高性能可扩展 3D 仿真与任务 API，使大规模导航、EQA、具身学习实验成为常规流程。
- **Replica: A Dataset of Photorealistic 3D Indoor Scene Reconstructions, 2019**：提供高保真室内重建场景，强化真实感渲染与 embodied perception 评测。
- **ProcTHOR: Large-Scale Embodied AI Using Procedural Generation, 2022**：用程序化生成扩展 AI2-THOR 场景规模，缓解 embodied agent 训练数据稀缺问题。

## 2. 任务层：从导航、问答到家庭长程任务

### 2.1 视觉语言导航与目标导航

- **Vision-and-Language Navigation / Room-to-Room, 2018**：首次在真实建筑全景中定义自然语言导航指令执行任务，奠定 VLN 主线。
- **VLN-CE: Vision-and-Language Navigation in Continuous Environments, 2020**：将离散全景图导航推进到连续空间控制，更接近机器人真实部署。
- **LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action, 2022**：组合 GPT-3、CLIP 与视觉导航模型，在无语言轨迹标注条件下实现真实机器人长程语言导航。
- **Visual Language Maps for Robot Navigation, 2022**：将 CLIP/VLM 特征融合进 3D 或 2D 地图，实现开放词汇、空间关系可查询的语言地图。
- **GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation, 2024**：提出“Go to AnyThing”多目标、多模态、长期记忆导航评测，推动通用导航智能体。
- **NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM, 2025**：用 LLM 和 RAG 从 3D 场景树生成更贴近用户需求的导航指令，扩展 VLN 数据与任务语义。

### 2.2 具身问答与环境理解

- **Embodied Question Answering, 2018**：定义智能体必须主动探索环境才能回答问题的 EQA 任务，连接导航、视觉理解、语言和记忆。
- **Neural Modular Control for Embodied Question Answering, 2018**：用层级子策略处理长程 EQA 导航，推动“语义子目标 + 控制策略”的组合式方法。
- **ScanQA: 3D Question Answering for Spatial Scene Understanding, 2022**：将 QA 从 2D 图像推进到完整 3D 扫描场景，强调 3D 空间关系和对象 grounding。
- **OpenEQA: Embodied Question Answering in the Era of Foundation Models, 2024**：面向基础模型时代提出开放词汇 EQA，覆盖 episodic memory 与 active exploration，并显示 GPT-4V 等模型仍显著落后人类。
- **EfficientEQA, 2024**：将高效主动探索、开放词汇回答和检索增强生成结合，用于更实用的 EQA 智能体。
- **Chain-of-View Prompting for Spatial Reasoning, 2026**：将 VLM 转化为主动视角选择器，通过粗到细探索改善 3D EQA 中的多视角空间推理。

### 2.3 家庭活动、交互任务与长程执行

- **VirtualHome: Simulating Household Activities via Programs, 2018**：用程序化动作序列表达家庭活动，为语言到行动脚本和活动理解提供数据。
- **ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks, 2019**：在 AI2-THOR 中定义长程、组合、带状态变化的语言指令任务，成为 household embodied agent 经典基准。
- **BEHAVIOR: Benchmark for Everyday Household Activities, 2021**：用 100 类日常家务任务和谓词逻辑目标定义，推动从导航到真实活动完成的评测。
- **TEACh: Task-driven Embodied Agents that Chat, 2022**：引入人机对话式任务执行，强调智能体在环境中通过语言协作完成目标。

## 3. 3D 语言 Grounding：从对象指代到 3D-LLM

### 3.1 3D 对象定位与指代表达

- **ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language, 2020**：首次大规模研究用自然语言在 3D 扫描中定位目标对象。
- **ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification, 2020**：构建 Sr3D/Nr3D 语料，强调细粒度类别、多实例场景和空间关系指代。
- **3DVG-Transformer, 2021**：用 Transformer 建模 3D 对象与语言关系，提升 3D visual grounding。
- **M3DRef-CLIP / 3D-VisTA, 2022-2023**：将预训练视觉语言模型迁移到 3D grounding，推动从任务特定模型转向预训练范式。

### 3.2 3D 问答、描述与多任务统一

- **ScanQA, 2022**：将 3D 场景问答与目标对象框 grounding 结合，为 3D-language reasoning 提供基准。
- **PointLLM, 2023**：让 LLM 理解点云对象，探索点云-语言对齐的通用接口。
- **3D-LLM: Injecting the 3D World into Large Language Models, 2023**：将 3D 点云和多视图特征注入 LLM，统一 3D captioning、QA、grounding、对话、导航和任务分解。
- **LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, 2023**：通过 3D 指令微调提升复杂 3D 场景理解、推理和规划。
- **Chat-3D / Chat-3D v2, 2023-2024**：将 3D 场景表示接入对话式 LLM，使模型能围绕 3D 场景进行问答与交互。
- **LEO: An Embodied Generalist Agent in 3D World, 2023/2024**：统一 3D 感知、grounding、推理、规划、导航和操作，是 3D 具身通才智能体的重要代表。

## 4. LLM/VLM 规划层：语言模型如何变成空间智能体

### 4.1 LLM 作为高层任务规划器

- **Language Models as Zero-Shot Planners, 2022**：展示 LLM 可从自然语言中抽取行动序列，为具身规划打开早期路径。
- **SayCan / Do As I Can, Not As I Say, 2022**：将 LLM 的高层语义知识与机器人技能可行性 value function 结合，解决“会说但做不了”的 grounding 问题。
- **Inner Monologue, 2022**：让机器人通过语言反馈、环境观察和内部推理循环修正计划，推动闭环具身推理。
- **Code as Policies, 2022**：让代码生成模型输出机器人策略代码，把语言指令转化为可调用感知和控制 API 的程序。
- **LLM+P, 2023**：将 LLM 与经典规划器连接，弥补 LLM 在严格任务规划中的可执行性和正确性不足。
- **Voyager, 2023**：在 Minecraft 中用 LLM、技能库和自动课程学习构建开放式 embodied agent，对长期自主探索和技能积累有重要影响。

### 4.2 语言到 3D 约束、价值场与轨迹

- **VoxPoser: Composable 3D Value Maps for Robotic Manipulation, 2023**：用 LLM/VLM 生成 3D value maps，将语言约束转化为空间可优化的机器人轨迹。
- **Text2Motion, 2023**：将自然语言映射到可行运动计划，连接语义规划和物理约束。
- **Eureka: Human-Level Reward Design via Coding LLMs, 2023**：用 LLM 生成奖励函数，推动语言驱动的机器人技能学习。
- **SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL, 2026**：通过视觉和机器人工具增强 VLM 的度量空间推理，并用于精确真实机器人操作。

## 5. VLA 与机器人基础模型：从空间推理到直接行动

### 5.1 早期大规模机器人策略

- **RT-1: Robotics Transformer for Real-World Control at Scale, 2022**：用大规模真实机器人轨迹训练 Transformer 控制策略，验证数据规模对机器人泛化的重要性。
- **Interactive Language: Talking to Robots in Real Time, 2022**：构建实时语言可控机器人策略，强调自然语言到低层视觉运动技能的端到端映射。
- **VIMA: General Robot Manipulation with Multimodal Prompts, 2022**：用多模态提示统一多类操作任务，推动 prompt-conditioned manipulation。
- **Open X-Embodiment / RT-X, 2023**：聚合多种机器人和多机构数据，证明跨 embodiment 数据能提升通用机器人策略迁移。

### 5.2 多模态基础模型接入机器人

- **PaLM-E: An Embodied Multimodal Language Model, 2023**：将连续传感输入注入 LLM token 空间，统一机器人规划、VQA 和图像字幕等任务。
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 2023**：将网页级 VLM 知识迁移到机器人动作 token，显示语义泛化和简单空间推理能力。
- **OpenVLA: An Open-Source Vision-Language-Action Model, 2024**：发布开放 7B VLA 模型和训练框架，降低通用机器人策略研究门槛。
- **π0: A Vision-Language-Action Flow Model for General Robot Control, 2024**：用 flow matching 架构处理连续动作，提升通用、灵巧、跨平台机器人控制。
- **FAST: Efficient Action Tokenization for Vision-Language-Action Models, 2025**：改进行动 tokenization，降低自回归 VLA 在高频控制中的计算成本。
- **Gemini Robotics, 2025**：基于 Gemini 2.0 构建 VLA 与 embodied reasoning 模型，强调泛化、交互、灵巧操作和安全评测。
- **π0.5: A Vision-Language-Action Model with Open-World Generalization, 2025**：通过异构任务、网络数据、语义子任务与动作协同训练，展示在陌生真实家庭中的长程操作泛化。

## 6. 空间推理专门化：VLM 的 2D/3D 空间能力补课

### 6.1 度量空间与 3D 空间 VQA

- **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities, 2024**：自动生成互联网规模 3D 空间 VQA 数据，使 VLM 获得距离、大小等度量空间估计能力。
- **EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought, 2023**：用 embodied CoT 数据训练多模态模型生成行动计划，强调从视频理解到控制的中间推理。
- **Grounded 3D-LLM, 2024**：用 referent tokens 等机制强化 3D grounding，使语言推理更稳定地绑定到 3D 对象。
- **Loc3R-VLM, 2026**：通过全局布局重建和 egocentric situation modeling，让 2D VLM 从单目视频获得语言定位与 3D 推理能力。
- **The Dual Mechanisms of Spatial Reasoning in VLMs, 2026**：分析 VLM 中空间关系的表征来源，指出视觉编码器布局信号在空间推理中起主导作用。

### 6.2 评测空间推理缺陷的新基准

- **iVISPAR: An Interactive Visual-Spatial Reasoning Benchmark for VLMs, 2025**：用交互式滑块谜题测试 VLM agent 的多步视觉空间推理和规划。
- **REM: Reasoning over Embodied Multi-Frame Trajectories, 2025**：评估 MLLM 在 egocentric 运动中保持对象恒常性、空间关系和数量追踪的能力。
- **IndustryNav, 2025**：将空间推理评测扩展到动态工业导航，补足家庭静态场景之外的复杂场景。
- **ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning, 2026**：将定位与执行拆分为生成式问题，诊断 VLM 从被动空间理解到“推理以行动”的能力。
- **SpatiaLab, 2026**：面向真实复杂图像测试 VLM 野外空间推理，弥补合成或谜题式空间评测的偏差。
- **MultihopSpatial, 2026**：评测多跳组合空间关系与视觉 grounding，针对 VLA 真实部署所需的复杂空间推理能力。

## 7. 场景表示前沿：从显式地图到可执行 3D 世界模型

### 7.1 开放词汇语义地图

- **VLMaps, 2022**：把开放词汇视觉语言特征写入几何地图，使“沙发右侧三米”等语言空间目标可查询。
- **ConceptFusion, 2023**：将开放集视觉语义融合进 3D 场景表示，推动可查询 3D 语义地图。
- **OpenScene, 2023**：用 2D foundation model 特征提升 3D 开放词汇场景理解。
- **LERF: Language Embedded Radiance Fields, 2023**：将 CLIP 式语言特征嵌入 NeRF，使 3D 神经场可用自然语言查询。
- **Feature/Language 3D Gaussian Splatting 系列, 2023-2025**：将开放词汇语义与实时 3DGS 渲染结合，形成可交互、可导航的语言场景表示。

### 7.2 可导航、可物理执行的世界表示

- **SAGE-3D: Towards Physically Executable 3D Gaussian for Embodied Navigation, 2025**：将 3D Gaussian Splatting 加入对象语义与碰撞物理接口，形成可执行 VLN 环境和 SAGE-Bench。
- **D3D-VLP: Dynamic 3D Vision-Language-Planning Model, 2025**：将 3D grounding、动态记忆、规划和导航统一为 3D vision-language-planning 模型。
- **Visual Embodied Brain, 2025**：试图让 MLLM 在空间中同时“看、想、控”，代表 VLM-agent 与控制策略进一步融合的趋势。
- **Chain-of-View Prompting, 2026**：不改变模型参数，通过主动选择视角补足 VLM 对多视角 3D 场景的局部观察缺陷。

## 8. 领域主干演进脉络

### 8.1 2015 年前后：几何空间智能

- 核心问题是“在哪里”和“如何到达”：SLAM、定位、地图、路径规划。
- 代表工作是 ORB-SLAM、概率机器人、早期神经 SLAM。
- 智能体的空间表示以几何地图和位姿图为主，语言与语义能力较弱。

### 8.2 2017-2020：Embodied AI 基准化

- AI2-THOR、Matterport3D、ScanNet、Gibson、Habitat 使可复现实验成为可能。
- R2R、EQA、ALFRED、ScanRefer、ReferIt3D 将导航、问答、对象 grounding 任务标准化。
- 研究重心从“建图”扩展到“在 3D 空间中理解语言并完成任务”。

### 8.3 2021-2023：LLM/VLM 进入具身规划

- SayCan、Inner Monologue、Code as Policies、PaLM-E、RT-2 使 LLM/VLM 成为具身智能体的大脑或策略骨干。
- VLMaps、VoxPoser、3D-LLM 把语言模型与显式 3D 表示结合，形成“语言-空间-行动”的桥梁。
- 关键矛盾变为：LLM 有常识但缺少可执行 grounding，机器人有技能但缺少开放语义泛化。

### 8.4 2024-2026：空间智能体前沿收敛

- OpenEQA、SpatialVLM、GOAT-Bench、OpenVLA 将领域推向开放词汇、通用导航、通用行动和可诊断空间推理。
- π0、π0.5、Gemini Robotics 代表 VLA 从“规划器 + 技能库”转向端到端机器人基础模型。
- Loc3R-VLM、ESPIRE、SpaceTools、SAGE-3D 等表明前沿正在补三类短板：度量空间推理、多视角记忆、物理可执行性。

## 9. 仍未解决的核心问题

- **稳定 3D 世界模型**：当前 VLM/MLLM 仍难在多视角运动中保持对象恒常性、尺度、拓扑和可达性。
- **语言 grounding 到动作的可靠性**：从“知道答案”到“安全执行”之间仍缺少可验证的空间约束和物理约束。
- **长期记忆与在线更新**：空间智能体需要把 episodic memory、语义地图、技能库和用户偏好持续融合。
- **泛化评测不足**：家庭静态环境占比过高，工业、户外、动态人群、多机器人协作仍是薄弱区域。
- **安全与可解释性**：VLA 模型能直接行动后，空间误判会变成物理风险，需要可诊断、可审计、可干预的执行框架。

## 10. 推荐阅读路径

### 10.1 如果关注具身智能基础设施

- AI2-THOR, 2017
- Matterport3D, 2017
- ScanNet, 2017
- Habitat, 2019
- ProcTHOR, 2022

### 10.2 如果关注视觉语言导航和 EQA

- Room-to-Room VLN, 2018
- Embodied Question Answering, 2018
- ALFRED, 2019
- GOAT-Bench, 2024
- OpenEQA, 2024

### 10.3 如果关注 3D 语言 grounding 和 3D-LLM

- ScanRefer, 2020
- ReferIt3D, 2020
- ScanQA, 2022
- 3D-LLM, 2023
- LEO, 2023/2024

### 10.4 如果关注 LLM/VLM 机器人规划

- SayCan, 2022
- Inner Monologue, 2022
- Code as Policies, 2022
- PaLM-E, 2023
- VoxPoser, 2023

### 10.5 如果关注 VLA 和机器人基础模型

- RT-1, 2022
- RT-2, 2023
- Open X-Embodiment / RT-X, 2023
- OpenVLA, 2024
- π0, 2024
- Gemini Robotics, 2025
- π0.5, 2025

### 10.6 如果关注空间推理评测前沿

- SpatialVLM, 2024
- iVISPAR, 2025
- REM, 2025
- ESPIRE, 2026
- Loc3R-VLM, 2026
- SpaceTools, 2026

## 11. 主要参考来源

- AI2-THOR: https://huggingface.co/papers/1712.05474
- Habitat: https://ai.meta.com/research/publications/habitat-a-platform-for-embodied-ai-research/
- Room-to-Room VLN: https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf
- Embodied Question Answering: https://openaccess.thecvf.com/content_cvpr_2018/html/Das_Embodied_Question_Answering_CVPR_2018_paper.html
- ALFRED: https://huggingface.co/papers/1912.01734
- ScanRefer: https://huggingface.co/papers/1912.08830
- ReferIt3D: https://referit3d.github.io/
- ScanQA: https://openaccess.thecvf.com/content/CVPR2022/html/Azuma_ScanQA_3D_Question_Answering_for_Spatial_Scene_Understanding_CVPR_2022_paper.html
- SayCan: https://arxiv.gg/abs/2204.01691
- VLMaps: https://huggingface.co/papers/2210.05714
- PaLM-E: https://huggingface.co/papers/2303.03378
- 3D-LLM: https://research.ibm.com/publications/3d-llm-injecting-the-3d-world-into-large-language-models
- RT-2: https://huggingface.co/papers/2307.15818
- VoxPoser: https://proceedings.mlr.press/v229/huang23b.html
- SpatialVLM: https://huggingface.co/papers/2401.12168
- OpenEQA: https://cvpr.thecvf.com/virtual/2024/poster/29575
- GOAT-Bench: https://openaccess.thecvf.com/content/CVPR2024/html/Khanna_GOAT-Bench_A_Benchmark_for_Multi-Modal_Lifelong_Navigation_CVPR_2024_paper.html
- OpenVLA: https://huggingface.co/papers/2406.09246
- π0: https://huggingface.co/papers/2410.24164
- Gemini Robotics: https://huggingface.co/papers/2503.20020
- Loc3R-VLM: https://www.microsoft.com/en-us/research/publication/loc3r-vlm-language-based-localization-and-3d-reasoning-with-vision-language-models/
- SpaceTools: https://spacetools.github.io/
