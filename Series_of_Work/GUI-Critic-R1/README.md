# Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation
This repository contains the official implementation for the paper: [Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation](https://arxiv.org/abs/2506.04614).

<div align="center">
<a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
<hr>
</div>
 
## 📢 News
🔥[2025-06-06] We release the test code to evaluate the performance of GUI-Critic-R1 (will be released on Hugging Face) on the GUI-Critic-Test dataset. 


![](assets/introduction.png)
## 📋 Introduction
Unlike general offline multimodal tasks, GUI automation is executed in online interactive environments, necessitating step-by-step decision-making based on real-time status of the environment. 
This task has a lower tolerance for decision-making errors at each step, as any mistakes may cumulatively disrupt the process and potentially lead to irreversible outcomes like deletions or payments. 
To address these issues, we introduce a pre-operative critic model, **GUI-Criti-R1**, that provides effective feedback prior to the actual execution, by reasoning about the potential outcome and correctness of actions. 
We propose Suggestion-aware Gradient Relative Policy Optimization (S-GRPO) strategy to construct our pre-operative critic model GUI-Critic-R1, incorporating a novel suggestion reward to enhance the reliability of the model's feedback.
Furthermore, we develop a reasoning-bootstrapping based data collection pipeline to create a GUI-Critic-Train and a GUI-Critic-Test, filling existing gaps in GUI critic data.
Static experiments on the GUI-Critic-Test across both mobile and web domains reveal that our GUI-Critic-R1 offers significant advantages in critic accuracy compared to current MLLMs.
Dynamic evaluation on GUI automation benchmark further highlights the effectiveness and superiority of our model, as evidenced by improved success rates and operational efficiency.



## 📍 TODO
- [ ] Publish test data images
- [ ] Release the model checkpoint
- [ ] Publish the GUI-Critic-Train dataset
- [ ] Release the test code that applies GUI-Critic-R1 on the AndroidWorld benchmark


## 💡 Test on GUI-Critic-Test
### 📑 Files

- `test.py`: Main script for running the evaluation on Hugging Face models.
- `statistic.py`: Contains evaluation functions and metrics calculation.
- `test_files/`: Directory containing test files:
  - `gui_i.jsonl`: Test data for GUI-I dataset
  - `gui_s.jsonl`: Test data for GUI-S dataset
  - `gui_web.jsonl`: Test data for GUI-W dataset
  
### 🔧 Use

1. Install the required dependencies in requirement.txt
`pip install -r requirements.txt`
2. Configure the API for Qwen-72B in `statistic.py`. You'll need to set up your API key and endpoint to use the Qwen-72B model for suggestion effectiveness calculation.
3. Run the main evaluation script:

`python test.py --model_dir <model_directory>
--test_file <test_file_path>
--save_dir <output_directory>
--data_dir <dataset_directory>`

Parameter descriptions:
- `--model_dir`: Directory containing the model
- `--test_file`: Path to the test file
- `--save_dir`: Directory to save the results
- `--data_dir`: Directory containing the dataset

## 📑Citation

If you find Mobile-Agent useful for your research and applications, please cite using this BibTeX:
```
@article{wanyan2025look,
  title={Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation},
  author={Wanyan, Yuyang and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Ye, Jiabo and Kou, Yutong and Yan, Ming and Huang, Fei and Yang, Xiaoshan and others},
  journal={arXiv preprint arXiv:2506.04614},
  year={2025}
}
```