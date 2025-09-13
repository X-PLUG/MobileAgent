
# Mobile-Agent-v3

<div align="center">
<img src=https://youke1.picui.cn/s1/2025/08/18/68a2f82fef3d4.png width="40%"/>
</div>

<div align="center">
<a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>

## 📢新闻
* 🔥🔥[9.10] 我们开源了Mobile-Agent-v3在真实手机场景下的代码。
* 🔥[8.29] 我们开源了GUI-Owl和Mobile-Agent-v3在AndroidWorld上的评测代码。
* 🔥[8.10] 我们开源了 [GUI-Owl-7B](https://huggingface.co/mPLUG/GUI-Owl-7B) 和 [GUI-Owl-32B](https://huggingface.co/mPLUG/GUI-Owl-32B)。
* 🔥[8.10] Mobile-Agent-v3的技术报告已经公开 [Mobile-Agent-v3](https://arxiv.org/abs/2508.15144)。

## 📍 TODO
- [x] 开源在真实手机场景的 Mobile-Agent-v3 代码
- [x] 开源在Android World上评测的代码PC
- [ ] 开源在真实PC场景的 Mobile-Agent-v3 代码
- [ ] 开源在OSWorld上评测的代码PC

## 介绍
GUI-Owl是多智能体GUI自动化框架Mobile-Agent-v3的系列基础模型。其在众多GUI自动化评测榜单包括 ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, MMBench-GUI, Android Control, Android World, 和 OSWorld中取得SOTA性能。此外，其也可以扮演Mobile-Agent-v3中的各个智能体进行协同交互，以完成更为复杂的任务。

## 在你的手机上部署Mobile-Agent-v3
❗目前仅安卓和鸿蒙系统（版本号 <= 4）支持工具调试。其他系统如iOS暂时不支持使用Mobile-Agent。


### 安装 qwen 模型所需的依赖项
```
pip install qwen_agent
pip install qwen_vl_utils
```

### 准备通过ADB连接你的移动设备
1. 下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB）。
2. 在你的移动设备上开启“USB调试”或“ADB调试”，它通常需要打开开发者选项并在其中开启。如果是HyperOS系统需要同时打开 "[USB调试(安全设置)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)"。
3. 通过数据线连接移动设备和电脑，在手机的连接选项中选择“传输文件”。
4. 用下面的命令来测试你的连接是否成功: ```/path/to/adb devices```。如果输出的结果显示你的设备列表不为空，则说明连接成功。
5. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: ```sudo chmod +x /path/to/adb```。
6.  ```/path/to/adb```在Windows电脑上将是```xx/xx/adb.exe```的文件格式，而在MacOS或者Linux则是```xx/xx/adb```的文件格式。

### 在你的移动设备上安装 ADB 键盘
1. 下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)  安装包。
2. 在设备上点击该 apk 来安装。
3. 在系统设置中将默认输入法切换为 “ADB Keyboard”。

### 运行
```
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py \
    --adb_path "Your ADB path" \
    --api_key "Your api key of vllm service" \
    --base_url "Your base url of vllm service" \
    --model "Your model name of vllm service" \
    --instruction "The instruction you want Mobile-Agent-v3 to complete" \
    --add_info "Some supplementary knowledge, can also be empty"
```

### 注意
1. 如果您使用的模型输出 0 到 1000 之间的相对坐标，例如 Seed-VL 或 Qwen-2-VL，请设置：
```
--coor_type "qwen-vl"
```

2. 如果您的指令需要记忆某些页面中内容，请设置：
```
--notetaker True
```

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
- GUI-Owl: [轨迹](https://drive.google.com/file/d/1KlSmoSoiVZLrT_Bcd1LtlD7EQO_wipcm/view?usp=sharing) 和 [日志](https://drive.google.com/file/d/1jlFXHed-0y__cNziB3z_0eDazppFwdoj/view?usp=sharing)

- Mobile-Agent-v3: [轨迹](https://drive.google.com/file/d/1lSK4ZtVleZLjauzxBptj22q0EhH6xpcr/view?usp=sharing) and [日志](https://drive.google.com/file/d/1sihY-Edua5pZ_ZWppjh33QStZ8utJnkA/view?usp=sharing)

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
