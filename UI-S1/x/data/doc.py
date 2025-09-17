import re
def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    # the middle y of bbox_1 belongs to bbox_2 and the middle y of bbox_2 belongs to bbox_1
    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False

def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


def space_layout(texts, boxes):
    """
    copy from https://github.com/WenjinW/LATIN-Prompt/blob/main/utils/space_layout.py
    convert text w/ ocr bboxes to pure-text layout:
    1. organize text by lines
    2. add space between texts according their horizontal distance

    text: list of word
    bboxes: list of [x,y,x,y]
    """

    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    while len(boxes) > 0:
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)

    if max_line_char_num != 0:
        char_width = line_width / max_line_char_num
    else:
        return []

    # print(char_width)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            if j==0:
                # indent
                left_char_num = int(box[0] / char_width)
                space_num = max(4,min(0, left_char_num))
                # print(space_num,line_texts[i][j], box)
            else:
                left_char_width = (line_box[j-1][2]-line_box[j-1][0])/len(line_texts[i][j-1])
                left_char_num = int((box[0]-line_box[j-1][0]) / left_char_width)
                space_num = max(0,left_char_num-len(line_texts[i][j-1]))
                
                # print(left_char_num, len(line_texts[i][j-1]), left_char_width, line_texts[i][j], box)
            # space_num = max(1, left_char_num - len(space_line_text))
            space_line_text += " " * space_num
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts


def build_layout_text(info, bbox_type='xyxy', layout_only=True):
    """
    # raw latin prompt
    PROMPT_DICT = {
        "prompt_task": (
            "You are asked to answer questions asked on a document image.\n"
            "The answers to questions are short text spans taken verbatim from the document. "
            "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
            "Document:\n{document}\n\n"
            "Question: {question}\n\n"
            "Directly extract the answer of the question from the document with as few words as possible.\n\n"
            "Answer:"
        )
    }"""

    PROMPT_DICT = {
        "prompt_task": (
            "You are asked to answer questions asked on a document image.\n"
            "The answers to questions are short text spans taken verbatim from the document. "
            "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
            "Directly extract the answer of the question from the document with as few words as possible.\n\n"
            "Document:\n{document}\n\n"
        )
    }

    if 'text' in info:
        text = info['text']
    else:
        # assert raw_bbox_type in ['xyxy']
        texts = []
        boxes = []
        for ocr_bbox in info['ocr_bboxes']:
            normalize = ocr_bbox['normalize']
            texts.append(ocr_bbox['text'])
            boxes.append(ocr_bbox['bbox'])
        space_line_texts = space_layout(texts, boxes)
        doc = "\n".join(space_line_texts)

        # replace >= 4 spaces with 4 spaces "   "
        doc = re.sub(' {4,}', '    ', doc)

        if layout_only:
            text = doc
        else:
            text = PROMPT_DICT["prompt_task"].format_map({
                "document": doc,
            })
    return text
