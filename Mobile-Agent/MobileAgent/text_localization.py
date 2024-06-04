import cv2
import numpy as np
from MobileAgent.crop import crop_image, calculate_size
from PIL import Image


def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


def longest_common_substring_length(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def ocr(image_path, prompt, ocr_detection, ocr_recognition, x, y):
    text_data = []
    coordinate = []
    image = Image.open(image_path)
    iw, ih = image.size
    
    image_full = cv2.imread(image_path)
    det_result = ocr_detection(image_full)
    det_result = det_result['polygons'] 
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)
        result = ocr_recognition(image_crop)['text'][0]
        
        if result == prompt:
            box = [int(e) for e in list(pts.reshape(-1))]
            box = [box[0], box[1], box[4], box[5]]
            
            if calculate_size(box) > 0.05*iw*ih:
                continue
            
            text_data.append([int(max(0, box[0]-10)*x/iw), int(max(0, box[1]-10)*y/ih), int(min(box[2]+10, iw)*x/iw), int(min(box[3]+10, ih)*y/ih)])
            coordinate.append([int(max(0, box[0]-300)*x/iw), int(max(0, box[1]-400)*y/ih), int(min(box[2]+300, iw)*x/iw), int(min(box[3]+400, ih)*y/ih)])
    
    max_length = 0
    if len(text_data) == 0:
        for i in range(det_result.shape[0]):
            pts = order_point(det_result[i])
            image_crop = crop_image(image_full, pts)
            result = ocr_recognition(image_crop)['text'][0]
            
            if len(result) < 0.3 * len(prompt):
                continue
            
            if result in prompt:
                now_length = len(result)
            else:
                now_length = longest_common_substring_length(result, prompt)
            
            if now_length > max_length:
                max_length = now_length
                box = [int(e) for e in list(pts.reshape(-1))]
                box = [box[0], box[1], box[4], box[5]]
                
                text_data = [[int(max(0, box[0]-10)*x/iw), int(max(0, box[1]-10)*y/ih), int(min(box[2]+10, iw)*x/iw), int(min(box[3]+10, ih)*y/ih)]]
                coordinate = [[int(max(0, box[0]-300)*x/iw), int(max(0, box[1]-400)*y/ih), int(min(box[2]+300, iw)*x/iw), int(min(box[3]+400, ih)*y/ih)]]

        if len(prompt) <= 10:
            if max_length >= 0.8*len(prompt):
                return text_data, coordinate
            else:
                return [], []
        elif (len(prompt) > 10) and (len(prompt) <= 20):
            if max_length >= 0.5*len(prompt):
                return text_data, coordinate
            else:
                return [], []
        else:
            if max_length >= 0.4*len(prompt):
                return text_data, coordinate
            else:
                return [], []
    
    else:
        return text_data, coordinate