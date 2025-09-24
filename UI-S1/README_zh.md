# UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning

<font size=4><div align='center' > [[📖 Paper](https://arxiv.org/abs/2509.11543)] [[🤗 UI-S1-7B](https://huggingface.co/mPLUG/UI-S1-7B)] [[🤗 Daily Paper](https://huggingface.co/papers/2509.11543)]</div></font>

</div>
<div align="center">
  <a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>

## 🔥 概述

我们提出了**半在线强化学习**，这是一种利用离线轨迹模拟在线强化学习的新颖范式，从而能够高效训练基于多轮学习模型 (MLLM) 的 GUI 代理，并增强其多轮交互能力。

<div align="center">
<img src="assets/method_comparison.png" alt="Logo" style="width:80%;">
</div>

我们的 <b>UI-S1-7B</b> 在开源 7B 模型中，在半在线指标 (SOP) 和在线指标 (AndroidWorld) 上均取得了 SOTA 性能。

<div align="center">
<img src="assets/metric.png" alt="Logo" style="width:80%;">
</div>

## 详细结果

<div align="center">
<img src="assets/result.png" alt="Logo" style="width:80%;">
</div>

## 设置

```shell
conda create -n ui-s1 python=3.11
conda activate ui-s1
cd UI-S1
pip install -e .
pip install vllm==0.8.2
pip install flash-attn==2.7.4.post1 --no-build-isolation
# 或从 https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1 安装 wheel
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

```
我们使用 swanlab 进行训练可视化。在 verl/utils/tracking.py 中替换您自己的 swanlab api 密钥和主机名

## 数据

1. 下载 AndroidControl 到 datasets/AndroidControl/images 和 datasets/android_control_train_example.jsonl

## 训练

```shell
bash scripts/train_example.sh
python scripts/model_merger.py merge --local_dir checkpoints/XXX
```

## 推理和评估

```shell
# 1. 启动 vLLM 服务器
vllm serve /checkpoints-7B --served-model-name UI-S1-7B --tensor_parallel_size 1 --trust-remote-code --limit-mm-per-prompt image=2

# 2. 评估 UI-S1-7B 在 SOP 上的表现
python /evaluation/eval_qwenvl.py --model_name UI-S1-7B

# 评估其他模型
python /evaluation/eval_qwenvl.py --model_name Qwen2.5-VL-7B
python /evaluation/eval_agentcpm.py --model_name AgentCPM-GUI-8B
python /evaluation/eval_os-atlas-7b.py --model_name OS-Atlas-7B
python /evaluation/eval_os-genesis-7b.py --model_name OS-Genesis-7B
python /evaluation/eval_ui-tars-7b.py --model_name UI-TARS-7B
```

## 🗞️ 新闻

- **`2025-09-17`**：我们发布 UI-S1 训练和评估代码。
- **`2025-09-16`**：我们发布 UI-S1-7B 模型的 [检查点](https://huggingface.co/mPLUG/UI-S1-7B)。
- **`2025-09-16`**：我们发布了我们的[论文](https://arxiv.org/abs/2509.11543)。


## ⭐️ 引用

如果您觉得这个项目有用，欢迎引用我们。

```bit
@article{lu2025ui,
  title={UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning},
  author={Lu, Zhengxi and Ye, Jiabo and Tang, Fei and Shen, Yongliang and Xu, Haiyang and Zheng, Ziwei and Lu, Weiming and Yan, Ming and Huang, Fei and Xiao, Jun and others},
  journal={arXiv preprint arXiv:2509.11543},
  year={2025}
}
```

## 🤝 致谢

我们真诚感谢 [verl](https://github.com/volcengine/verl) 和 [verl-agent](https://github.com/langfengQ/verl-agent) 项目。
