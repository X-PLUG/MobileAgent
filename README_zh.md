<div align="center">
<p align="center">
  <img src="assets/logo.png"/>
</p>
</div>

<div align="center">
<h2 style="font-size: 28px;">
  <img src="assets/tongyi.png" width="50px" style="vertical-align: middle; margin-right: 10px;">
  Mobile-Agent: 强大的GUI智能体家族
</h2>

<div align="center">
<p align="center">
  <img src="assets/series.png"/>
</p>
</div>

<p align="center">
<a href="https://trendshift.io/repositories/7423" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7423" alt="MobileAgent | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>
<!-- 
<div align="center">
  <a href="https://www.modelscope.cn/studios/wangjunyang/PC-Agent"><img src="assets/Demo-ModelScope-brightgreen.svg" alt="Demo ModelScope"></a>
  <a href="https://arxiv.org/abs/?"><img src="https://img.shields.io/badge/Arxiv-2502.14282-b31b1b.svg?logo=arXiv" alt=""></a>
</div> -->

</div>
<div align="center">
  <a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>

## 📢新闻
- `[2025.8.20]`🔥 全新 **GUI-Owl** 和 **Mobile-Agent-v3** 即将到来！
  - GUI-Owl 是一个多模态跨平台 GUI 虚拟层模型 (VLM)，具备 GUI 感知、落地和端到端操作能力。
  - Mobile-Agent-v3 是一个基于 GUI-Owl 的跨平台多智能体框架，提供规划、进度管理、反射和内存等功能。
- `[2025.8.14]`🔥 Mobile-Agent-v3 在***第二十四届全国计算语言学大会*** (CCL 2025) 上荣获 **最佳演示奖**。
- `[2025.3.17]` PC-Agent 已被 **ICLR 2025 研讨会** 接收。
- `[2024.9.26]` Mobile-Agent-v2 已被 **第三十八届神经信息处理系统年会 (NeurIPS 2024)** 接收。
- `[2024.7.29]` Mobile-Agent 在***第二十三届全国计算语言学大会*** (CCL 2024) 上荣获 **最佳演示奖**。
- `[2024.3.10]` Mobile-Agent 已被 **ICLR 2024 研讨会** 录用。

## 📊效果

<div align="center">
<p align="center">
  <img src="assets/result.png"/>
</p>
</div>

## 👀特点

<div align="center">
<p align="center">
  <img src="assets/framework.png"/>
</p>
</div>

### GUI-Owl
- 7B以内实现 SOTA 结果。
- 原生端到端多模态代理，旨在作为 GUI 自动化的基础模型。
- 在单一策略网络中统一感知、基础、推理、规划和动作执行。
- 强大的跨平台交互和多轮决策，并具有明确的中间推理功能。
- GUI-Owl 可在 Mobile-Agent-v3 中实例化为不同的专用智能体。

### Mobile-Agent-v3
- 动态任务分解、规划和进度管理。
- 高度集成的操作空间，降低模型的感知和操作频率。
- 丰富的异常处理和反射能力，在弹窗、广告等场景下提供更稳定的性能。
- 关键信息记录能力，支持跨应用任务。

## 📝系列工作

- [**Mobile-Agent-v3**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3) (预印本): 多模态、多平台 GUI 代理。[**[论文]**](https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/MobileAgentV3_Tech.pdf) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3)
- [**GUI-Critic-R1**](https://github.com/X-PLUG/MobileAgent/tree/main/GUI-Critic-R1) (预印本): 一种用于术前错误诊断方法的 GUI-Critic。 [**[论文]**](https://arxiv.org/abs/2506.04614) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/GUI-Critic-R1)
- [**PC-Agent**](https://github.com/X-PLUG/MobileAgent/tree/main/PC-Agent) (ICLR 2025 研讨会): 用于多模态 PC 操作的多智能体。 [**[论文]**](https://arxiv.org/abs/2502.14282) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/PC-Agent)
- [**Mobile-Agent-E**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-E) (预印本): 用于自进化手机操作的多智能体。 [**[论文]**](https://arxiv.org/abs/2501.11733) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-E)
- [**Mobile-Agent-v2**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2) (NeurIPS 2024)：用于多模式手机操作的多智能体。 [**[论文]**](https://arxiv.org/abs/2406.01014) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v2)
- [**Mobile-Agent-v1**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v1) (ICLR 2024 研讨会): 单代理用于多模态手机操作。[**[论文]**](https://arxiv.org/abs/2401.16158) [**[代码]**](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v1)

## 📺Demo

### 💻PC + 🌐Web

<div align="left">
    <h3>在Edge浏览器中搜索阿里巴巴的股价。然后在WPS中新建一个表格，在第一列填入公司名，在第二列填入股价。</h3>
    <video src= "https://github.com/user-attachments/assets/1c810fc1-fe69-4da2-b5f9-19b3cfa72c5e"/>
</div>

### 💻PC

<div align="left">
    <h3>新建一个空白PPT，然后在第一张幻灯片中插入一段艺术字形式的文本，内容为阿里巴巴。</h3>
    <video src= "https://github.com/user-attachments/assets/a978087a-717b-4c8a-9e50-9223dac019dd"/>
</div>

### 🌐Web

<div align="left">
    <h3>Please help me search for flights from Beijing to Paris on Skyscanner departing on September 18th and returning on September 21st.</h3>
    <video src= "https://github.com/user-attachments/assets/fd49a192-f876-4862-b0c3-30aaaf48643a"/>
</div>

<div align="left">
    <h3>进入bilibili，查看雷军的视频，然后点赞第一个视频。</h3>
    <video src= "https://github.com/user-attachments/assets/78702309-0985-4103-ae50-0dec6cc8adf2"/>
</div>

### 📱Phone

<div align="left">
    <h3>帮我在小红书搜一下济南旅游攻略，按照收藏数排序，并收藏第一篇笔记。</h3>
    <video src= "https://github.com/user-attachments/assets/3a405952-953a-4c2a-a26c-d738b6622564"/>
</div>

<div align="left">
    <h3>帮我在携程搜一下济南大明湖景区的详情，包括地址和门票价格等。</h3>
    <video src= "https://github.com/user-attachments/assets/c2572f62-cd78-44c3-8b7d-ae478a168073"/>
</div>

## ⭐Star History
[![Star History Chart](https://api.star-history.com/svg?repos=X-PLUG/MobileAgent&type=Date)](https://star-history.com/#X-PLUG/MobileAgent&Date)

## 📑引用
如果您发现 Mobile-Agent 对您的研究和应用有用，请使用此 BibTeX 进行引用：
```
@article{wanyan2025look,
  title={Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation},
  author={Wanyan, Yuyang and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Ye, Jiabo and Kou, Yutong and Yan, Ming and Huang, Fei and Yang, Xiaoshan and others},
  journal={arXiv preprint arXiv:2506.04614},
  year={2025}
}

@article{liu2025pc,
  title={PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC},
  author={Liu, Haowei and Zhang, Xi and Xu, Haiyang and Wanyan, Yuyang and Wang, Junyang and Yan, Ming and Zhang, Ji and Yuan, Chunfeng and Xu, Changsheng and Hu, Weiming and Huang, Fei},
  journal={arXiv preprint arXiv:2502.14282},
  year={2025}
}

@article{wang2025mobile,
  title={Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks},
  author={Wang, Zhenhailong and Xu, Haiyang and Wang, Junyang and Zhang, Xi and Yan, Ming and Zhang, Ji and Huang, Fei and Ji, Heng},
  journal={arXiv preprint arXiv:2501.11733},
  year={2025}
}

@article{wang2024mobile2,
  title={Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration},
  author={Wang, Junyang and Xu, Haiyang and Jia, Haitao and Zhang, Xi and Yan, Ming and Shen, Weizhou and Zhang, Ji and Huang, Fei and Sang, Jitao},
  journal={arXiv preprint arXiv:2406.01014},
  year={2024}
}

@article{wang2024mobile,
  title={Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception},
  author={Wang, Junyang and Xu, Haiyang and Ye, Jiabo and Yan, Ming and Shen, Weizhou and Zhang, Ji and Huang, Fei and Sang, Jitao},
  journal={arXiv preprint arXiv:2401.16158},
  year={2024}
}
```
