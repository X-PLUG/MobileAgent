
# Mobile-Agent-v3

<div align="center">
<img src=https://youke1.picui.cn/s1/2025/08/18/68a2f82fef3d4.png width="40%"/>
</div>

## 📢News
* 🔥🔥[8.29] We've open-sourced the AndroidWorld benchmark code for GUI-Owl and Mobile-Agent-v3.
* 🔥[8.10] We release [GUI-Owl-7B](https://huggingface.co/mPLUG/GUI-Owl-7B) and [GUI-Owl-32B](https://huggingface.co/mPLUG/GUI-Owl-32B).
* 🔥[8.10] The technical report can be found [here](https://arxiv.org/abs/2508.15144)

## 📍 TODO
- [x] Open source evaluation code on AndroidWorld
- [ ] Open source evaluation code on OSWorld

## Introduction
GUI-Owl is a model series developed as part of the Mobile-Agent-v3 project. It achieves state-of-the-art performance across a range of GUI automation benchmarks, including ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, MMBench-GUI, Android Control, Android World, and OSWorld. Furthermore, it can be instantiated as various specialized agents within the Mobile-Agent-v3 multi-agent framework to accomplish more complex tasks.

## Evaluation on AndroidWorld
1. Please follow the [official code repository](https://github.com/google-research/android_world?tab=readme-ov-file#installation) to install the Android emulator and necessary dependencies.

2. Install the dependencies required by the qwen model.
```
pip install qwen_agent
pip install qwen_vl_utils
```

3. Fill in your vllm service information in the `run_guiowl.sh` or `run_ma3.sh` script, including api_key, base_url, and model.

4. Run the evaluation.
```
cd MobileAgent/Mobile-Agent-v3/androld_world_v3
sh run_guiowl.sh
sh run_ma3.sh
```

5. We provide evaluation trajectory and logs for viewing.
GUI-Owl: [trajectory](https://drive.google.com/file/d/1KlSmoSoiVZLrT_Bcd1LtlD7EQO_wipcm/view?usp=sharing) and [log](https://drive.google.com/file/d/1sihY-Edua5pZ_ZWppjh33QStZ8utJnkA/view?usp=sharing)

Mobile-Agent-v3: [trajectory](https://drive.google.com/file/d/1lSK4ZtVleZLjauzxBptj22q0EhH6xpcr/view?usp=sharing) and [log](https://drive.google.com/file/d/1jlFXHed-0y__cNziB3z_0eDazppFwdoj/view?usp=sharing)

## Performance
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

## Usage

Please refer to our cookbook.

## Deploy

Please refer to the README of model card on HuggingFace for optimized performance.

## Citation
If you find our paper and model useful in your research, feel free to give us a cite.
```
@article{ye2025mobile,
  title={Mobile-Agent-v3: Foundamental Agents for GUI Automation},
  author={Ye, Jiabo and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Zhu, Zhaoqing and Zheng, Ziwei and Gao, Feiyu and Cao, Junjie and Lu, Zhengxi and others},
  journal={arXiv preprint arXiv:2508.15144},
  year={2025}
}
```
