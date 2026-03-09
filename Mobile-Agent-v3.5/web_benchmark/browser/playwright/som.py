import asyncio
import base64
import io
import os
import json
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

import random
from typing import Dict, Any

import time
import requests


STRUCTURAL_TAGS = {
    "html", "head", "meta", "link", "style", "script", "base",
    "title", "body",
    "header", "footer", "main", "nav", "section", "article",
    "aside", "summary", "details",
}

NON_INTERACTIVE_ROLES = {
    "presentation", "none",
    "img", "banner", "main", "contentinfo",
    "navigation", "region",
}


def looks_interactive(el: Dict[str, Any]) -> bool:
    tag = (el.get("tag") or "").lower()
    role = (el.get("role") or "").lower()
    typ = (el.get("type") or "").lower()
    text = (el.get("text") or "").strip()
    cls = (el.get("cls") or "").lower()
    id_ = (el.get("id") or "").lower()
    aria_label = (el.get("ariaLabel") or "").strip()

    if tag in {"a", "button", "select", "textarea"}:
        return True
    if tag == "input" and typ not in {"hidden"}:
        return True

    if role in {
        "button", "link", "tab", "menuitem", "option",
        "switch", "checkbox", "radio", "textbox",
    }:
        return True


    if any(x in el for x in ("onclick", "onmousedown", "onmouseup")):
        return True

    if any(k in cls or k in id_ for k in ("btn", "button", "link", "click", "nav", "tab", "menu")):
        return True

    if (text or aria_label) and tag in {"div", "span", "li"}:
        return True

    return False


def is_obviously_non_interactive(el: Dict[str, Any]) -> bool:
    tag = (el.get("tag") or "").lower()
    role = (el.get("role") or "").lower()
    typ = (el.get("type") or "").lower()
    bbox = el.get("bbox") or {}
    w = bbox.get("width") or 0
    h = bbox.get("height") or 0

    if tag in STRUCTURAL_TAGS:
        if not (
            tag == "body"
            and (el.get("isContentEditable")
                 or "ke-content" in (el.get("cls") or ""))
        ):
            return True

    if w * h < 9:
        return True

    if tag in STRUCTURAL_TAGS:
        return True
    if tag in {"video", "audio", "source"}:
        return True

    if tag == "input" and typ == "hidden":
        return True

    if role in NON_INTERACTIVE_ROLES:
        return True

    if not looks_interactive(el):
        return True

    return False


INJECT_CLICK_TRACKER = r"""
(() => {
  if (window.__clickedElementsPatched) return;
  window.__clickedElementsPatched = true;

  const clickedSet = new WeakSet();

  const origAddEventListener = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, listener, options) {
    if (type === 'click') {
      clickedSet.add(this);
    }
    return origAddEventListener.call(this, type, listener, options);
  };


  window.__hasClickListener = function(el) {
    try {
      if (typeof el.onclick === 'function') return true;
    } catch (e) {}
    return clickedSet.has(el);
  };
})();
"""


JS_COLLECT_ALL_RECURSIVE = r"""(args) => {
  const selector = args[0];      // e.g. "*" or "a,button,input"
  const maxDepth = args[1];      // iframe recursion depth
  const maxPerDoc = args[2];     // cap per document
  const minArea = args[3];       // ignore tiny boxes
  const includeIframes = args[4];

  function oneLine(s){ return String(s||"").replace(/\s+/g," ").trim(); }

  function describe(el){
    function oneLine(s){ return String(s||"").replace(/\s+/g," ").trim(); }

    const tagName   = (el.tagName || "").toLowerCase();
    const hrefAttr  = el.getAttribute("href") || "";
    const typeAttr  = (el.getAttribute("type") || "").toLowerCase();
    const roleAttr  = (el.getAttribute("role") || "").toLowerCase();
    const ariaAttr  = el.getAttribute("aria-label") || "";

    let clickable = true;

    if (tagName == "div") clickable = false;

    if (tagName === "a" && hrefAttr) clickable = true;
    if (tagName === "button") clickable = true;
    if (tagName === "input" && typeAttr !== "hidden") clickable = true;

    if (["button","link","tab","menuitem","option","switch","checkbox","radio"].includes(roleAttr)) {
        clickable = true;
    }

    const isContentEditable = el.isContentEditable || el.getAttribute('contenteditable') === 'true';
    if (isContentEditable) {
        clickable = true;
    }

    if (el.onclick === "function") clickable = true;
    try {
        if (!clickable && typeof el.onclick === "function") {
        clickable = true;
        }
    } catch(e) {}

    try {
        if (!clickable && typeof getEventListeners === "function") {
        const ev = getEventListeners(el);
        if (ev && Array.isArray(ev.click) && ev.click.length > 0) {
            clickable = true;
        }
        }
    } catch(e) {}

    const cls = (el.getAttribute("class") || "").toLowerCase();
    if (cls.includes("ke-content") || cls.includes("ke-edit-textarea")) {
        clickable = true;
    }

    return {
        tag: tagName,
        id: oneLine(el.getAttribute("id")),
        cls: oneLine(el.getAttribute("class")),
        role: roleAttr,
        name: oneLine(el.getAttribute("name")),
        type: typeAttr,
        href: oneLine(hrefAttr),
        src: oneLine(el.getAttribute("src")),
        ariaLabel: oneLine(ariaAttr),
        text: oneLine(el.textContent).slice(0, 40),
        clickable: clickable,
        isContentEditable: el.isContentEditable || el.getAttribute('contenteditable') === 'true',
    };
  }

  const out = [];
  let counter = 0;

  function walk(doc, depth, ox, oy, path){
    if (!doc || depth > maxDepth) return;
    const win = doc.defaultView;
    if (!win) return;

    const sx = 0;
    const sy = 0;
    const dpr = win.devicePixelRatio || 1;

    // collect boxes for selector (NO visibility/viewport filtering)
    const nodes = Array.from(doc.querySelectorAll(selector)).slice(0, maxPerDoc);
    for (const el of nodes){
      try {
        const r = el.getBoundingClientRect();
        if (!r) continue;
        const area = r.width * r.height;
        if (area < minArea) continue;

        const vw = win.innerWidth  || doc.documentElement.clientWidth  || 0;
        const vh = win.innerHeight || doc.documentElement.clientHeight || 0;
        if (r.right <= 0 || r.bottom <= 0 || r.left >= vw || r.top >= vh) {
          continue;
        }

        const desc = describe(el);
        if (!desc.clickable) {
            continue;
        }

        if (!el.checkVisibility({ 
        checkOpacity: true, 
        checkVisibilityCSS: true,
        })) continue;


        out.push({
          id: `e_${counter++}`,
          path,
          depth,
          dpr,
          ...describe(el),
          bbox: {
            x: (ox + r.x - sx) * dpr,
            y: (oy + r.y - sy) * dpr,
            width: r.width * dpr,
            height: r.height * dpr
          }
        });
      } catch(e) {}
    }

    if (!includeIframes) return;

    // recurse iframes
    const iframes = Array.from(doc.querySelectorAll("iframe")).slice(0, maxPerDoc);
    for (let i = 0; i < iframes.length; i++){
      const fr = iframes[i];
      try {
        const rr = fr.getBoundingClientRect();
        const iframeOx = ox + rr.x - sx;
        const iframeOy = oy + rr.y - sy;
        const nextPath = path + `/iframe[${i}]`;

        try {
          const childDoc = fr.contentDocument; // same-origin only
          if (!childDoc) {
            out.push({
              id: `iframe_${counter++}`,
              path: nextPath,
              depth,
              dpr,
              tag: "iframe",
              src: oneLine(fr.getAttribute("src")),
              error: "iframe not ready (contentDocument is null)",
              bbox: {x: iframeOx * dpr, y: iframeOy * dpr, width: rr.width * dpr, height: rr.height * dpr},
            });
            continue;
          }
          walk(childDoc, depth + 1, iframeOx, iframeOy, nextPath);
        } catch(e) {
          out.push({
            id: `iframe_${counter++}`,
            path: nextPath,
            depth,
            dpr,
            tag: "iframe",
            src: oneLine(fr.getAttribute("src")),
            error: String(e),
            bbox: {x: iframeOx * dpr, y: iframeOy * dpr, width: rr.width * dpr, height: rr.height * dpr},
          });
        }
      } catch(e) {}
    }
  }

  walk(document, 0, 0, 0, "root");
  return JSON.stringify(out);
}"""


def mark_containing_items_for_removal(items):
    def bbox_contains(a, b):
        ax1 = a["x"]
        ay1 = a["y"]
        ax2 = ax1 + a["width"]
        ay2 = ay1 + a["height"]

        bx1 = b["x"]
        by1 = b["y"]
        bx2 = bx1 + b["width"]
        by2 = by1 + b["height"]

        return (bx1 >= ax1 and by1 >= ay1 and
                bx2 <= ax2 and by2 <= ay2)

    for item in items:
        item["to_remove"] = False

    n = len(items)
    for i in range(n):
        a = items[i]
        for j in range(n):
            if i == j:
                continue
            b = items[j]

            if (a.get("text") or "").strip() == (b.get("text") or "").strip():
                if bbox_contains(a["bbox"], b["bbox"]):
                    a["to_remove"] = True
                    break

    remain_items = []
    for item in items:
        if not item["to_remove"]:
            remain_items.append(item)
    return remain_items


def draw_dashed_line(draw, xy, dash_len=6, gap_len=4, fill=(255, 0, 0, 200), width=2):
    x1, y1, x2, y2 = xy
    # 水平线
    if y1 == y2:
        total_len = abs(x2 - x1)
        step = dash_len + gap_len
        n = max(1, int(total_len // step) + 1)
        direction = 1 if x2 >= x1 else -1
        for i in range(n):
            start = x1 + direction * i * step
            end = start + direction * dash_len
            if direction == 1:
                if start > x2:
                    break
                end = min(end, x2)
            else:
                if start < x2:
                    break
                end = max(end, x2)
            draw.line((start, y1, end, y2), fill=fill, width=width)

    # 垂直线
    elif x1 == x2:
        total_len = abs(y2 - y1)
        step = dash_len + gap_len
        n = max(1, int(total_len // step) + 1)
        direction = 1 if y2 >= y1 else -1
        for i in range(n):
            start = y1 + direction * i * step
            end = start + direction * dash_len
            if direction == 1:
                if start > y2:
                    break
                end = min(end, y2)
            else:
                if start < y2:
                    break
                end = max(end, y2)
            draw.line((x1, start, x2, end), fill=fill, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=fill, width=width)


def draw_dashed_rect(draw, x1, y1, x2, y2, dash_len=6, gap_len=4, fill=(255, 0, 0, 200), width=2):
    draw_dashed_line(draw, (x1, y1, x2, y1), dash_len, gap_len, fill, width)
    draw_dashed_line(draw, (x1, y2, x2, y2), dash_len, gap_len, fill, width)
    draw_dashed_line(draw, (x1, y1, x1, y2), dash_len, gap_len, fill, width)
    draw_dashed_line(draw, (x2, y1, x2, y2), dash_len, gap_len, fill, width)


def screenshot_to_png_bytes(s: Any) -> bytes:
    if isinstance(s, (bytes, bytearray)):
        return bytes(s)
    if isinstance(s, str):
        ss = s.strip()
        if ss.startswith("data:image"):
            ss = ss.split(",", 1)[-1].strip()
        try:
            return base64.b64decode(ss, validate=True)
        except Exception:
            with open(s, "rb") as f:
                return f.read()
    raise TypeError(f"Unsupported screenshot type: {type(s)}")


def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1.get("x"), box1.get("y"), box1.get("width"), box1.get("height")
    x2, y2, w2, h2 = box2.get("x"), box2.get("y"), box2.get("width"), box2.get("height")

    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def mearge_A_with_B(A, B, threshold=0.9):
    keep_result = []
    for b_item in B:
        b_box = b_item['bbox']
        keep = True
        for a_item in A:
            a_box = a_item['bbox']
            if iou_xywh(a_box, b_box) > threshold:
                if a_item.get("text", "") == "":
                    a_item["text"] = b_item.get("text", "")
                keep = False
                break
        if keep:
            keep_result.append(b_item)

    A.extend(keep_result)
    return A


def items_to_text(items_raw):
    format_ele_text = []
    for web_ele_id in range(len(items_raw)):
        item = items_raw[web_ele_id]

        is_menu = item.get('isMenu', False)
        menu_options = item.get('menuOptions', [])

        label_text = item.get('text', "")

        ele_tag_name = item.get("tag", "button")
        ele_type = item.get("type", "")
        ele_aria_label = item.get("ariaLabel", "")

        input_attr_types = ['text', 'search', 'password', 'email', 'tel']

        if is_menu and menu_options:
            trigger_text = label_text.split('\n')[0].strip()
            options_str = ', '.join([f'"{opt}"' for opt in menu_options])

            base_text = f"[{web_ele_id}]: <{ele_tag_name}>"
            if trigger_text:
                base_text += f" \"{trigger_text}\""
            elif ele_aria_label:
                base_text += f" \"{ele_aria_label}\""

            format_ele_text.append(f"{base_text} is a menu with options: [{options_str}];")
            continue

        if not label_text:
            if (ele_tag_name.lower() == 'input' and ele_type in input_attr_types) or ele_tag_name.lower() == 'textarea' or (ele_tag_name.lower() == 'button' and ele_type in ['submit', 'button']):
                if ele_aria_label:
                    format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{ele_aria_label}\";")
                else:
                    format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";" )
        elif label_text and len(label_text) < 200:
            if not ("<img" in label_text and "src=" in label_text):
                if ele_tag_name in ["button", "input", "textarea"]:
                    if ele_aria_label and (ele_aria_label != label_text):
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\", \"{ele_aria_label}\";")
                    else:
                        format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";")
                else:
                    if ele_aria_label and (ele_aria_label != label_text):
                        format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\", \"{ele_aria_label}\";")
                    else:
                        format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\";")

    return '\t'.join(format_ele_text)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_omniparser_stand_res(omniparser_res):
    res = omniparser_res["result"]
    bbox_list = [item["bbox"] for item in res]
    label_list = [item["content"] for item in res]
    return bbox_list, label_list


def call_omnipaser(args, image_path, max_retry=10):
    try:
        authorization = os.environ.get("OMNI_API_KEY")
    except:
        return []
    for _ in range(max_retry):
        try:
            url = args.omni_url
            image_base64 = encode_image(image_path)
            headers = {
                "Authorization": authorization
            }
            data = {
                "image": image_base64
            }

            resp = requests.post(url, json=data, headers=headers).json()["result"]
            _ = resp[0]['bbox']
            _ = resp[0]['content']
            return resp
        except Exception:
            continue

    return []


async def get_element_info_by_xy(page, x, y):
    js = """
    (coords) => {
        const { x, y } = coords;
        const el = document.elementFromPoint(x, y);
        if (!el) return null;

        const oneLine = (v) => {
            if (v == null) return null;
            return String(v).replace(/\\s+/g, ' ').trim();
        };

        const tag = el.tagName.toLowerCase();
        const id = oneLine(el.getAttribute("id"));
        const cls = oneLine(el.getAttribute("class"));

        // role / aria-label
        const roleAttr = oneLine(el.getAttribute("role"));
        const ariaAttr = oneLine(el.getAttribute("aria-label"));

        // name / type / href / src
        const nameAttr = oneLine(el.getAttribute("name"));
        const typeAttr = oneLine(el.getAttribute("type"));
        const hrefAttr = el.getAttribute("href");
        const srcAttr = el.getAttribute("src");

        const clickableTags = ["a", "button", "input"];
        let clickable = clickableTags.includes(tag);
        if (!clickable) {
            const style = window.getComputedStyle(el);
            clickable = (style.cursor === "pointer");
        }

        // contentEditable
        const isContentEditable =
            el.isContentEditable || el.getAttribute("contenteditable") === "true";

        const text = oneLine(el.textContent || "").slice(0, 40);

        return {
            tag: tag,
            id: id,
            cls: cls,
            role: roleAttr,
            name: nameAttr,
            type: typeAttr,
            href: oneLine(hrefAttr),
            src: oneLine(srcAttr),
            ariaLabel: ariaAttr,
            text: text,
            clickable: clickable,
            isContentEditable: isContentEditable,
        };
    }
    """

    return await page.evaluate(js, {"x": x, "y": y})

async def get_omni_som(args, page, img_path_no_box):
    items_omni = call_omnipaser(args, img_path_no_box)
    for i in range(len(items_omni)):
        bbox = items_omni[i]['bbox']
        items_omni[i]['bbox'] = {
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1]
        }
        items_omni[i]['text'] = items_omni[i]['content']

        center_x = items_omni[i]['bbox']["x"] + items_omni[i]['bbox']["width"] / 2
        center_y = items_omni[i]['bbox']["y"] + items_omni[i]['bbox']["height"] / 2

        try:
            for key, value in (await get_element_info_by_xy(page, center_x, center_y)).items():
                if key in items_omni[i]:
                    if items_omni[i][key] == "":
                        items_omni[i][key] = value
                else:
                    items_omni[i][key] = value
        except:
            pass


    return items_omni


def draw_som(items, overlay, max_draw):
    try:
        font = ImageFont.truetype(ImageFont.load_default().path, size=20)
    except Exception:
        font = ImageFont.truetype("browser/playwright/Arial.ttf", size=20)

    placed_label_boxes = []
    draw = ImageDraw.Draw(overlay)
    for idx, it in enumerate(items[:max_draw]):
        b = it.get("bbox") or {}
        x, y, w, h = b.get("x"), b.get("y"), b.get("width"), b.get("height")
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            continue

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b_color = random.randint(0, 255)
        color = (r, g, b_color, 255)

        x1, y1, x2, y2 = x, y, x + w, y + h
        draw_dashed_rect(draw, x1, y1, x2, y2, dash_len=6, gap_len=4, fill=color, width=2)

        if idx < max_draw:
            label = f'{idx}'
            scale = 1

            try:
                tb = draw.textbbox((0, 0), label, font=font)
                base_tw, base_th = tb[2] - tb[0], tb[3] - tb[1]
                tw = int(base_tw * scale)
                th = int(base_th * scale)
            except Exception:
                base_tw, base_th = (len(label) * 6, 12)
                tw = int(base_tw * scale)
                th = int(base_th * scale)

            padding_x = 6
            padding_y = 4

            def rect_intersection_area(a, b):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0
                return (ix2 - ix1) * (iy2 - iy1)

            label_w = tw + padding_x * 2
            label_h = th + padding_y * 2

            candidates = []
            lx_tl = x1
            ly_tl = y1 - label_h - 2
            candidates.append(("top_left", lx_tl, ly_tl))

            lx_tr = x2 - label_w
            ly_tr = y1 - label_h - 2
            candidates.append(("top_right", lx_tr, ly_tr))

            lx_bl = x1
            ly_bl = y2 + 2
            candidates.append(("bottom_left", lx_bl, ly_bl))

            lx_br = x2 - label_w
            ly_br = y2 + 2
            candidates.append(("bottom_right", lx_br, ly_br))

            img_w, img_h = overlay.size
            normalized_candidates = []
            for name, lx_c, ly_c in candidates:
                lx_n = max(0, min(lx_c, img_w - label_w))
                ly_n = max(0, min(ly_c, img_h - label_h))
                normalized_candidates.append((name, lx_n, ly_n))

            best_pos = None
            best_overlap = None

            for name, lx_c, ly_c in normalized_candidates:
                candidate_rect = (lx_c, ly_c, lx_c + label_w, ly_c + label_h)

                total_overlap = 0
                for placed in placed_label_boxes:
                    total_overlap += rect_intersection_area(candidate_rect, placed)

                if total_overlap == 0:
                    best_pos = (lx_c, ly_c)
                    best_overlap = 0
                    break

                if best_overlap is None or total_overlap < best_overlap:
                    best_overlap = total_overlap
                    best_pos = (lx_c, ly_c)

            lx, ly = best_pos
            label_box = (lx, ly, lx + label_w, ly + label_h)
            placed_label_boxes.append(label_box)

            color = list(color)
            color[-1] = 150
            color = tuple(color)
            draw.rectangle(
                [lx, ly, lx + label_w, ly + label_h],
                fill=color
            )

            text_x = lx + padding_x
            text_y = ly + padding_y

            draw.text(
                (text_x, text_y),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )


def is_inside_strict(box_inner: dict, box_outer: dict) -> bool:
    x1, y1, w1, h1 = box_inner.get("x"), box_inner.get("y"), box_inner.get("width"), box_inner.get("height")
    x2, y2, w2, h2 = box_outer.get("x"), box_outer.get("y"), box_outer.get("width"), box_outer.get("height")

    i_left, i_top = x1, y1
    i_right, i_bottom = x1 + w1, y1 + h1
    o_left, o_top = x2, y2
    o_right, o_bottom = x2 + w2, y2 + h2

    return (
        i_left > o_left and
        i_top > o_top and
        i_right < o_right and
        i_bottom < o_bottom
    )


def remove_outer_boxes(A: list) -> list:
    n = len(A)
    remove_flags = [False] * n

    for i in range(n):
        if remove_flags[i]:
            continue
        box_i = A[i]["bbox"]
        for j in range(n):
            if i == j:
                continue
            box_j = A[j]["bbox"]
            if is_inside_strict(box_j, box_i):
                remove_flags[i] = True
                break

    return [box for k, box in enumerate(A) if not remove_flags[k]]


async def get_css_som(
    page,
    selector: str = "*",
    max_depth: int = 16,
    max_per_doc: int = 3000,
    min_area: float = -1.0,
    max_retry: int = 3,
):
    items: List[Dict[str, Any]] = []
    for _ in range(max_retry):
        try:
            items_json = await page.evaluate(
                JS_COLLECT_ALL_RECURSIVE,
                [selector, max_depth, max_per_doc, float(min_area), True],
            )
            items = json.loads(items_json)
            break
        except Exception:
            try:
                await page.wait_for_load_state()
            except Exception:
                pass
            await asyncio.sleep(1)
            continue

    if items:
        items = mark_containing_items_for_removal(items)
    return items


def remove_neg_boxes(A: list) -> list:
    n = len(A)
    remove_flags = [False] * n

    for i in range(n):
        box_i = A[i]["bbox"]
        x, y, w, h = box_i.get("x"), box_i.get("y"), box_i.get("width"), box_i.get("height")
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            remove_flags[i] = True
        if box_i.get("x", 0) < 0 or box_i.get("y", 0) < 0:
            remove_flags[i] = True

    return [box for k, box in enumerate(A) if not remove_flags[k]]


async def get_som(
    page,
    img_path,
    img_path_no_box,
    args,
    selector: str = "*",
    max_depth: int = 16,
    max_per_doc: int = 3000,
    min_area: float = -1.0,   
    max_draw: int = 2000,   
    max_retry: int = 3,
):
    items = []
    if args.use_css_som:
        items = await get_css_som(
            page,
            selector=selector,
            max_depth=max_depth,
            max_per_doc=max_per_doc,
            min_area=min_area,
            max_retry=max_retry,
        )

    shot = await page.screenshot()
    with open(img_path_no_box, "wb") as f:
        f.write(shot)

    if getattr(args, "use_omni_som", False):
        items_omni = await get_omni_som(args, page, img_path_no_box)
        items = mearge_A_with_B(items, items_omni, 0.3)

    items = remove_neg_boxes(items)

    png_bytes = screenshot_to_png_bytes(shot)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_som(items, overlay, max_draw)

    img = Image.alpha_composite(img, overlay)
    img.save(img_path)

    return items, items_to_text(items)
