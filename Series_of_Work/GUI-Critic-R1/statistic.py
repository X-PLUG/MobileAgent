from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from dashscope import Generation
from time import sleep
from concurrent.futures import ThreadPoolExecutor

API = 'sk-04507fb92f4249c480095126bc662828'

USER_PROMPT = '''The following two sentences describe actions on a mobile: 
1. {}
2. {}
Determine whether these two sentences describe a similar action? If yes, answer **Yes**, if not **No**, no explanation required.  
'''

Attempt_Num = 3
MAX_WORKERS = 5


def similarity_measure(ground_truth, prediction, answer):
    user_prompt = USER_PROMPT.format(ground_truth, prediction)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': user_prompt}
    ]

    i = 0
    while True:
        try:
            response = Generation.call(
                api_key=API,
                model="qwen2.5-72b-instruct",
                messages=messages,
                result_format="message",
            )
            response = response.output.choices[0].message.content
            break
        except Exception as e:
            i += 1
            print(f"Attempt {i} failed: {e}")
            sleep(10)

            if i > Attempt_Num:
                response = 0
                return response
                break
    if 'Yes' in response:
        return 1
    elif 'No' in response:
        return 0
    else:
        return 0
    return 0


def calculate_metrics_binary(ground_truth, predictions):
    if not set(ground_truth).issubset({0, 1}):
        raise ValueError("ground_truth should only contain 0 and 1")

    valid_predictions = []
    format_errors = 0
    total = len(ground_truth)

    for gt, pred in zip(ground_truth, predictions):
        if pred not in {0, 1}:
            valid_predictions.append(1 - gt)  # 非 0/1 值被视为错误预测
            format_errors += 1
        else:
            valid_predictions.append(pred)

    # 计算格式错误比例
    format_error_rate = format_errors / total

    # 计算总体指标
    correct = sum(g == p for g, p in zip(ground_truth, valid_predictions))
    accuracy = correct / total

    # 计算混淆矩阵
    cm = confusion_matrix(ground_truth, valid_predictions)

    # 计算每个类别的指标
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracy_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "Overall Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy Class 0": accuracy_class_0,
        "Accuracy Class 1": accuracy_class_1,
        "Format Error": format_error_rate
    }


def check_format_output(response):
    try:
        answer = response.split('<score>')[1].split('</score>')[0].sript()
    except:
        return None

    if answer == 'Correct':
        return 1
    elif answer == 'Incorrect':
        return 0
    else:
        return None


def calculate_suggestion_acc(content, sol):
    try:
        gt_answer = sol.split('\n')[0]
        gt_suggestionion = sol.split('\n')[1]
    except:
        return 1
    answer = None
    suggestionion = None
    try:
        answer = content.split("<score>")[-1].split("</score>")[0].strip()
        suggestionion = content.split("<suggestionion>")[-1].split("</suggestionion>")[0].strip()
        if answer == 'Correct':
            answer = True
        elif answer == 'Incorrect':
            answer = False
        else:
            answer = None
    except Exception as e:
        print(e)
        return 0

    if len(suggestionion) == None or answer == None:
        return 0

    return similarity_measure(gt_suggestionion, suggestionion, answer)


def process_item(item):
    if 'suggestion_score' in item.keys():
        return item
    suggestion_score = item['critic_response']
    gt = item['solution']

    suggestion_score = calculate_suggestion_acc(suggestion_score, gt)
    item.update({'suggestion_score': suggestion_score})
    # result_list.append(item)
    return item


def get_result(data_list):
    _ground_truth = []
    _predictions = []

    for item in data_list:
        if 'Incorrect' in item['solution']:
            _ground_truth.append(0)
        else:
            _ground_truth.append(1)

        if 'Incorrect' in item['critic_result']:
            _predictions.append(0)
        elif 'Correct' in item['critic_result']:
            _predictions.append(1)
        else:
            _predictions.append(2)

    metrics = calculate_metrics_binary(_ground_truth, _predictions)

    for metric, value in metrics.items():
        print(f"{metric}: {value * 100:.2f}")
        metrics[metric] = value * 100

    print('Test_Num:', len(_ground_truth))

    suggestion_sum = 0

    # data_list = data_list[:100]
    print(len(data_list))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        processed_results = list(tqdm(executor.map(process_item, data_list),
                                      total=len(data_list),
                                      desc='@wyyy Computing Reward Scores',
                                      unit='item'))

    for item in processed_results:
        suggestion_sum += item['suggestion_score']

    # print(suggestion_sum / len(data_list))
    print("Suggestion score", f"{(suggestion_sum / len(data_list)) * 100:.2f}", end='\n')

    return metrics, data_list

