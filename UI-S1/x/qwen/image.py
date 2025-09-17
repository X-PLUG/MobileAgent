import base64
import io
import math
from pathlib import Path
from PIL import Image




def round_by_factor(number: int, factor: int) -> int:
    """返回最接近 number 的且能被 factor 整除的整数"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """返回大于等于 number 的且能被 factor 整除的整数"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """返回小于等于 number 的且能被 factor 整除的整数"""
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    """缩放后图片满足以下条件:
    1. 长宽能被 factor 整除
    2. pixels 总数被限制在 [min_pixels, max_pixels] 内
    3. 最长边限制在 max_long_side 内
    4. 保证其长宽比基本不变
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def update_image_size_(image_ele: dict, min_tokens=1, max_tokens=12800, merge_base=2, patch_size=14):
    """根据 min_tokens, max_tokens 更新 image_ele 的尺寸信息

    Args:
        image_ele (dict):
            - image_ele["image"]: str 图片路径
            - image_ele["height"]: int 图片原始高度
            - image_ele["width"]: int 图片原始宽度

    Returns:
        更新后的 image_ele, 新增如下 key-value pair
        dict:
            - image_ele["resized_height"]: int 输入到模型的真实高度
            - image_ele["resized_width"]: int 输入到模型的真实宽度
            - image_ele["seq_len"]: int 输入到模型所占的序列长度
    """
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update(
        {
            "resized_height": resized_height,
            "resized_width": resized_width,
            "seq_len": resized_height * resized_width // pixels_per_token + 2,
        }
    )
    return image_ele

def make_qwen_image_item(img_path: str, image=None):
    if image is not None:
        img = image
    else:
        from x.io.image_io import ImageIO, ImageIO2
        mio = ImageIO2()
        img = mio(img_path)
    if isinstance(img_path, Path):
        img_path = str(img_path.absolute())
    if img_path.startswith("http") or img_path.startswith("oss://"):
        pass
    else:
        img_path = str(Path(img_path).absolute())
    image_ele = {
        "image": img_path,
        "height": img.height,
        "width": img.width,
    }
    image_ele = update_image_size_(image_ele)
    return image_ele

def _convert_bbox_format_from_abs_origin(bbox, image_ele: dict, *, tgt_format: str):
    x1, y1, x2, y2 = bbox
    if tgt_format == "abs_origin":
        new_bbox = [int(x1), int(y1), int(x2), int(y2)]
    elif tgt_format == "abs_resized":
        new_bbox = [
            int(x1 / image_ele["width"] * image_ele["resized_width"]),
            int(y1 / image_ele["height"] * image_ele["resized_height"]),
            int(x2 / image_ele["width"] * image_ele["resized_width"]),
            int(y2 / image_ele["height"] * image_ele["resized_height"]),
        ]
    elif tgt_format == "qwen-vl":
        new_bbox = [
            int(x1 / image_ele["width"] * 999),
            int(y1 / image_ele["height"] * 999),
            int(x2 / image_ele["width"] * 999),
            int(y2 / image_ele["height"] * 999),
        ]
    elif tgt_format == "rel":
        new_bbox = [
            float(x1 / image_ele["width"]),
            float(y1 / image_ele["height"]),
            float(x2 / image_ele["width"]),
            float(y2 / image_ele["height"]),
        ]
    elif tgt_format == "molmo":
        new_bbox = [
            round(x1 / image_ele["width"] * 100, ndigits=1),
            round(y1 / image_ele["height"] * 100, ndigits=1),
            round(x2 / image_ele["width"] * 100, ndigits=1),
            round(y2 / image_ele["height"] * 100, ndigits=1),
        ]
    else:
        assert False, f"Unknown tgt_format: {tgt_format}"
    return new_bbox


def bbox_rep(bbox, image_ele, bbox_style):
    x1, y1, x2, y2 = bbox
    
    if bbox_style == 'qwen-vl':
        new_bbox = [
            int(x1 / image_ele["width"] * 999),
            int(y1 / image_ele["height"] * 999),
            int(x2 / image_ele["width"] * 999),
            int(y2 / image_ele["height"] * 999),
        ]
    elif bbox_style == 'abs_origin':
        new_bbox = [int(x1), int(y1), int(x2), int(y2)]
    else:
        assert 1==2
    return new_bbox

bbox_template = lambda bbox: f"<box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"

def point_rep(point, image_ele, bbox_style):
    x, y = point
    if bbox_style == 'qwen-vl':
        new_point = [
            int(x / image_ele["width"] * 999),
            int(y / image_ele["height"] * 999),
        ]
    elif bbox_style == 'abs_resized':
        new_point = [
            int(x / image_ele["width"] * image_ele["resized_width"]),
            int(y / image_ele["height"] * image_ele["resized_height"]),
        ]
    else:
        assert 1==2
    return new_point




def images_to_pdf(image_list, output_pdf):
    # 打开第一个图像并将其转换为 RGB 模式
    first_image = Image.open(image_list[0]).convert('RGB')
    
    # 打开其余的图像，并且也转换为 RGB 模式
    images = [Image.open(jpg).convert('RGB') for jpg in image_list[1:]]
    
    # 将所有图像保存到 PDF 文件，附加其余的图像
    first_image.save(output_pdf, save_all=True, append_images=images)
    first_image.close()
    for img in images:
        img.close()

def pil_to_data_url(image: Image.Image) -> str:
    # Create a BytesIO buffer to hold the image data
    buffer = io.BytesIO()
    
    # Save the image to the buffer in PNG format
    image.save(buffer, format='PNG')
    
    # Get the byte data from the buffer
    byte_data = buffer.getvalue()
    
    # Encode the byte data to base64
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    
    # Construct the data URL for a PNG image
    data_url = f"data:image/png;base64,{base64_str}"
    
    return data_url

# 坐标转换: 从调整后的坐标到原始坐标
def resize_coordinate(coordinate, source_size, target_size):
    x, y = coordinate
    target_width,  target_height = target_size
    source_width, source_height = source_size
    # 计算比例
    width_ratio = target_width / source_width
    height_ratio =  target_height / source_height
    # 转换坐标
    target_x = x * width_ratio
    target_y = y * height_ratio
    return [target_x, target_y]