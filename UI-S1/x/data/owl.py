
def line_llava2owl(line, ds_name='converted_ds', dataset_type='sft'):
    # 把llava格式转化为owl格式
    def build_message(raw_conv):
        messages = []
        for _ in raw_conv:
            if _['from'] == 'human':
                messages.append({"role": "user", "content":_['value'].replace('<image>','<|image|>')})
            elif _['from'] == 'gpt':
                messages.append({"role": "assistant", "content":_['value'].replace('<image>','<|image|>')})
            else:
                assert 1==2
        return messages

    messages = build_message(line['conversations'])
    # dataset_type is passed by the sft_xgpt3.py or pretrain_xgpt3.py, can only be pretrain or sft
    if dataset_type == 'pretrain':
        text = ' '.join([_['content'] for _ in messages])
        text = text.replace('<|image|>','') # Pretrain does not need image, will be add by the processor
        line = {
            'image': line['image'],
            'text': text,
            'task_name': 'pretrain',
            'dataset_name':  line.get('dataset_name', ds_name),
        }
    else:
        line = {
            'image': line['image'],
            'messages':  messages,
            'task_name': line.get('task_name', "mm_sft"),
            'dataset_name': line.get('dataset_name', ds_name),
        }
    return line