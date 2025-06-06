# Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation
This repository contains the official implementation for the paper: [Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation](https://arxiv.org/abs/2506.04614).

## 📢News
🔥[2025-06-06] We release the test code to evaluate the performance of GUI-Critic models (will be released on Hugging Face) on the GUI-Critic-Test dataset. 

## 📍TODO
- [ ] Publish test data images
- [ ] Release the model checkpoint
- [ ] Publish the GUI-Critic-Train dataset
- [ ] Release the test code that applies GUI-Critic-R1 on the AndroidWorld benchmark


## 💡Test on GUI-Critic-Test
### 📑Files

- `test.py`: Main script for running the evaluation on Hugging Face models.
- `statistic.py`: Contains evaluation functions and metrics calculation.
- `test_files/`: Directory containing test files:
  - `gui_i.jsonl`: Test data for GUI-I dataset
  - `gui_s.jsonl`: Test data for GUI-S dataset
  - `gui_web.jsonl`: Test data for GUI-W dataset
  
### 🔧Use

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

