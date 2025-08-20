import numpy as np


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1: list or array [x1, y1, x2, y2]
    - box2: list or array [x1, y1, x2, y2]

    Returns:
    - iou: float, IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # print(x2_inter, x1_inter, y2_inter, y1_inter)

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    merged_box = [min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max)]
    return merged_box


def merge_boxes_and_texts(texts, boxes, iou_threshold=0):
    """
    Merge bounding boxes and their corresponding texts based on IoU threshold.

    Parameters:
    - boxes: List of bounding boxes, with each box represented as [x1, y1, x2, y2].
    - texts: List of texts corresponding to each bounding box.
    - iou_threshold: Intersection-over-Union threshold for merging boxes.

    Returns:
    - merged_boxes: List of merged bounding boxes.
    - merged_texts: List of merged texts corresponding to the bounding boxes.
    """
    if len(boxes) == 0:
        return [], []

    # boxes = np.array(boxes)
    merged_boxes = []
    merged_texts = []

    while len(boxes) > 0:
        box = boxes[0]
        text = texts[0]
        boxes = boxes[1:]
        texts = texts[1:]
        to_merge_boxes = [box]
        to_merge_texts = [text]
        keep_boxes = []
        keep_texts = []

        for i, other_box in enumerate(boxes):
            if compute_iou(box, other_box) > iou_threshold:
                to_merge_boxes.append(other_box)
                to_merge_texts.append(texts[i])
            else:
                keep_boxes.append(other_box)
                keep_texts.append(texts[i])

        # Merge the to_merge boxes into a single box
        if len(to_merge_boxes) > 1:
            x1 = min(b[0] for b in to_merge_boxes)
            y1 = min(b[1] for b in to_merge_boxes)
            x2 = max(b[2] for b in to_merge_boxes)
            y2 = max(b[3] for b in to_merge_boxes)
            merged_box = [x1, y1, x2, y2]
            merged_text = " ".join(to_merge_texts)  # You can change the merging strategy here
            merged_boxes.append(merged_box)
            merged_texts.append(merged_text)
        else:
            merged_boxes.extend(to_merge_boxes)
            merged_texts.extend(to_merge_texts)

        # boxes = np.array(keep_boxes)
        boxes = keep_boxes
        texts = keep_texts

    return merged_texts, merged_boxes


def is_contained(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if (x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max):
        return True
    elif (x2_min >= x1_min and y2_min >= y1_min and x2_max <= x1_max and y2_max <= y1_max):
        return True
    return False


def is_overlapping(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
        return True
    return False


def get_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)


def merge_all_icon_boxes(bboxes):
    result_bboxes = []
    while bboxes:
        bbox = bboxes.pop(0)
        to_add = True

        for idx, existing_bbox in enumerate(result_bboxes):
            if is_contained(bbox, existing_bbox):
                if get_area(bbox) > get_area(existing_bbox):
                    result_bboxes[idx] = existing_bbox
                to_add = False
                break
            elif is_overlapping(bbox, existing_bbox):
                if get_area(bbox) < get_area(existing_bbox):
                    result_bboxes[idx] = bbox
                to_add = False
                break

        if to_add:
            result_bboxes.append(bbox)

    return result_bboxes


def merge_all_icon_boxes_new(elements):
    result_elements = []
    while elements:
        ele = elements.pop(0)
        bbox = [ele['position'][0], ele['position'][1], ele['position'][0]+ele['size'][0], ele['position'][1]+ele['size'][1]]
        # bbox = bboxes.pop(0)
        to_add = True

        for idx, existing_ele in enumerate(result_elements):
            existing_bbox = [existing_ele['position'][0], existing_ele['position'][1], existing_ele['position'][0]+existing_ele['size'][0], existing_ele['position'][1]+existing_ele['size'][1]]
            if is_contained(bbox, existing_bbox):
                if get_area(bbox) > get_area(existing_bbox):
                    result_elements[idx] = existing_ele
                to_add = False
                break
            elif is_overlapping(bbox, existing_bbox):
                if get_area(bbox) < get_area(existing_bbox):
                    result_elements[idx] = ele
                to_add = False
                break

        if to_add:
            result_elements.append(ele)

    return result_elements




def merge_bbox_groups(A, B, iou_threshold=0.8):
    i = 0
    while i < len(A):
        box_a = A[i]
        has_merged = False
        for j in range(len(B)):
            box_b = B[j]
            iou = calculate_iou(box_a, box_b)
            if iou > iou_threshold:
                merged_box = merge_boxes(box_a, box_b)
                A[i] = merged_box
                B.pop(j)
                has_merged = True
                break

        if has_merged:
            i -= 1
        i += 1

    return A, B


def bbox_iou(boxA, boxB):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def merge_boxes_and_texts_new(texts, bounding_boxes, iou_threshold=0):
    if not bounding_boxes:
        return [], []

    bounding_boxes = np.array(bounding_boxes)
    merged_boxes = []
    merged_texts = []

    used = np.zeros(len(bounding_boxes), dtype=bool)

    for i, boxA in enumerate(bounding_boxes):
        if used[i]:
            continue
        x_min, y_min, x_max, y_max = boxA
        # text = texts[i]
        text = ''

        overlapping_indices = [i] # []
        for j, boxB in enumerate(bounding_boxes):
            # print(i,j, bbox_iou(boxA, boxB))
            if i != j and not used[j] and bbox_iou(boxA, boxB) > iou_threshold:
                overlapping_indices.append(j)

        # Sort overlapping boxes by vertical position (top to bottom)
        overlapping_indices.sort(key=lambda idx: (bounding_boxes[idx][1] + bounding_boxes[idx][3])/2) # TODO

        for idx in overlapping_indices:
            boxB = bounding_boxes[idx]
            x_min = min(x_min, boxB[0])
            y_min = min(y_min, boxB[1])
            x_max = max(x_max, boxB[2])
            y_max = max(y_max, boxB[3])
            # text += " " + texts[idx]
            text += texts[idx]
            used[idx] = True

        merged_boxes.append([x_min, y_min, x_max, y_max])
        merged_texts.append(text)
        used[i] = True

    return merged_texts, merged_boxes
