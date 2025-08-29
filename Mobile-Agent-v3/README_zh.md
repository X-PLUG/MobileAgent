
# Mobile-Agent-v3

<div align="center">
<img src=https://youke1.picui.cn/s1/2025/08/18/68a2f82fef3d4.png width="40%"/>
</div>

## 📢新闻
* 🔥🔥[8.29] 我们开源了GUI-Owl和Mobile-Agent-v3在AndroidWorld上的评测代码。
* 🔥[8.10] 我们开源了 [GUI-Owl-7B](https://huggingface.co/mPLUG/GUI-Owl-7B) 和 [GUI-Owl-32B](https://huggingface.co/mPLUG/GUI-Owl-32B)。
* 🔥[8.10] Mobile-Agent-v3的技术报告已经公开 [Mobile-Agent-v3](https://arxiv.org/abs/2508.15144)。

## 📍 TODO
- [x] 开源在AndroidWorld上的评测代码
- [ ] 开源在OSWorld上的评测代码

## 介绍

GUI-Owl是多智能体GUI自动化框架Mobile-Agent-v3的系列基础模型。其在众多GUI自动化评测榜单包括 ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, MMBench-GUI, Android Control, Android World, 和 OSWorld中取得SOTA性能。此外，其也可以扮演Mobile-Agent-v3中的各个智能体进行协同交互，以完成更为复杂的任务。

## 在 AndroidWorld 上进行评估
1. 请按照[官方代码库](https://github.com/google-research/android_world?tab=readme-ov-file#installation)安装 Android 模拟器及必要的依赖项。

2. 安装 qwen 模型所需的依赖项。
```
pip install qwen_agent
pip install qwen_vl_utils
```

3. 在 `run_guiowl.sh` 或 `run_ma3.sh` 脚本中填写您的 vllm 服务信息，包括 api_key、base_url 和 model。

4. 运行评估。
```
cd MobileAgent/Mobile-Agent-v3/androld_world_v3
sh run_guiowl.sh
sh run_ma3.sh
```

5. 我们提供了评估轨迹和日志以供查看。
- GUI-Owl：[轨迹](https://drive.google.com/file/d/1KlSmoSoiVZLrT_Bcd1LtlD7EQO_wipcm/view?usp=sharing)；[日志](https://drive.google.com/file/d/1sihY-Edua5pZ_ZWppjh33QStZ8utJnkA/view?usp=sharing)

- Mobile-Agent-v3：[轨迹](https://drive.google.com/file/d/1lSK4ZtVleZLjauzxBptj22q0EhH6xpcr/view?usp=sharing)；[日志](https://drive.google.com/file/d/1jlFXHed-0y__cNziB3z_0eDazppFwdoj/view?usp=sharing)

## 性能
### ScreenSpot-v2, ScreenSpot-Pro and OSWorld-G
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/screenspot_v2.jpg?raw=true" width="80%"/>
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/screenspot_pro.jpg?raw=true" width="80%"/>
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/osworld_g.jpg?raw=true" width="80%"/>

### MMBench-GUI L1, L2 and Android Control
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/mmbench_gui_l1.jpg?raw=true" width="80%"/>
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/mmbench_gui_l2.jpg?raw=true" width="80%"/>
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/android_control.jpg?raw=true" width="60%"/>

### Android World and OSWorld-Verified
<img src="https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3/assets/online.jpg?raw=true" width="60%"/>

## 使用

请参考我们的cookbook.

## 部署

请参考具体模型的README以获得最佳性能。

## Citation
如果您发现我们的模型和文章对您的研究有益，可以按照下列格式引用.
```
@article{ye2025mobile,
  title={Mobile-Agent-v3: Foundamental Agents for GUI Automation},
  author={Ye, Jiabo and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Zhu, Zhaoqing and Zheng, Ziwei and Gao, Feiyu and Cao, Junjie and Lu, Zhengxi and others},
  journal={arXiv preprint arXiv:2508.15144},
  year={2025}
}
```
