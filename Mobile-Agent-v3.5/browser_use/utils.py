import base64
import re
import os
import json
import time
import logging
import numpy as np
from PIL import Image
from utils_webarena import fetch_browser_info, fetch_page_accessibility_tree,\
                    parse_accessibility_tree, clean_accesibility_tree


def resize_image(image_path):
    image = Image.open(image_path)
    width, height = image.size

    if min(width, height) < 512:
        return image
    elif width < height:
        new_width = 512
        new_height = int(height * (new_width / width))
    else:
        new_height = 512
        new_width = int(width * (new_height / height))

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_image.save(image_path)
    # return resized_image


# base64 encoding
# Code from OpenAI Document
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# interact with webpage and add rectangles on elements
def get_web_element_rect(browser, fix_color=True, fontsize=12):
    if fix_color:
        selected_function = "getFixedColor"
        # color_you_like = '#5210da'
    else:
        selected_function = "getRandomColor"

    js_script = """
        let labels = [];

        function markPage() {
            var bodyRect = document.body.getBoundingClientRect();

            var items = Array.prototype.slice.call(
                document.querySelectorAll('*')
            ).map(function(element) {
                var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                
                var rects = [...element.getClientRects()].filter(bb => {
                var center_x = bb.left + bb.width / 2;
                var center_y = bb.top + bb.height / 2;
                var elAtCenter = document.elementFromPoint(center_x, center_y);

                return elAtCenter === element || element.contains(elAtCenter) 
                }).map(bb => {
                const rect = {
                    left: Math.max(0, bb.left),
                    top: Math.max(0, bb.top),
                    right: Math.min(vw, bb.right),
                    bottom: Math.min(vh, bb.bottom)
                };
                return {
                    ...rect,
                    width: rect.right - rect.left,
                    height: rect.bottom - rect.top
                }
                });

                var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

                return {
                element: element,
                include: 
                    (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") ||
                    (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") ||
                    (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION")
                ,
                area,
                rects,
                text: element.textContent.trim().replace(/\\s{2,}/g, ' ')
                };
            }).filter(item =>
                item.include && (item.area >= 20)
            );

            // Only keep inner clickable items
            // first delete button inner clickable items
            const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));

            //items = items.filter(x => !buttons.some(y => y.contains(x.element) && !(x.element === y) ));
            items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y) ));
            items = items.filter(x => 
                !(x.element.parentNode && 
                x.element.parentNode.tagName === 'SPAN' && 
                x.element.parentNode.children.length === 1 && 
                x.element.parentNode.getAttribute('role') &&
                items.some(y => y.element === x.element.parentNode)));

            items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)))

            // Function to generate random colors
            function getRandomColor(index) {
                var letters = '0123456789ABCDEF';
                var color = '#';
                for (var i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
                }
                return color;
            }

            function getFixedColor(index) {
                var color = '#000000'
                return color
            }
            //function getFixedColor(index){
            //    var colors = ['#FF0000', '#00FF00', '#0000FF', '#000000']; // Red, Green, Blue, Black
            //    return colors[index % 4];
            //}
            

            // Lets create a floating border on top of these elements that will always be visible
            items.forEach(function(item, index) {
                item.rects.forEach((bbox) => {
                newElement = document.createElement("div");
                var borderColor = COLOR_FUNCTION(index);
                newElement.style.outline = `2px dashed ${borderColor}`;
                newElement.style.position = "fixed";
                newElement.style.left = bbox.left + "px";
                newElement.style.top = bbox.top + "px";
                newElement.style.width = bbox.width + "px";
                newElement.style.height = bbox.height + "px";
                newElement.style.pointerEvents = "none";
                newElement.style.boxSizing = "border-box";
                newElement.style.zIndex = 2147483647;
                // newElement.style.background = `${borderColor}80`;
                
                // Add floating label at the corner
                var label = document.createElement("span");
                label.textContent = index;
                label.style.position = "absolute";
                //label.style.top = "-19px";
                label.style.top = Math.max(-19, -bbox.top) + "px";
                //label.style.left = "0px";
                label.style.left = Math.min(Math.floor(bbox.width / 5), 2) + "px";
                label.style.background = borderColor;
                label.style.color = "white";
                label.style.padding = "2px 4px";
                label.style.fontSize = "FONTSIZE";
                label.style.borderRadius = "2px";
                newElement.appendChild(label);
                
                document.body.appendChild(newElement);
                labels.push(newElement);
                // item.element.setAttribute("-ai-label", label.textContent);
                });
            })

            // For the first way
            // return [labels, items.map(item => ({
            //     rect: item.rects[0] // assuming there's at least one rect
            // }))];

            // For the second way
            return [labels, items]
        }
        return markPage();""".replace("COLOR_FUNCTION", selected_function).replace("FONTSIZE", str(fontsize))
    rects, items_raw = browser.execute_script(js_script)

    # format_ele_text = [f"[{web_ele_id}]: \"{items_raw[web_ele_id]['text']}\";" for web_ele_id in range(len(items_raw)) if items_raw[web_ele_id]['text'] ]
    format_ele_text = []
    for web_ele_id in range(len(items_raw)):
        label_text = items_raw[web_ele_id]['text']
        ele_tag_name = items_raw[web_ele_id]['element'].tag_name
        ele_type = items_raw[web_ele_id]['element'].get_attribute("type")
        ele_aria_label = items_raw[web_ele_id]['element'].get_attribute("aria-label")
        input_attr_types = ['text', 'search', 'password', 'email', 'tel']

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



    format_ele_text = '\t'.join(format_ele_text)
    return rects, [web_ele['element'] for web_ele in items_raw], format_ele_text


js_script_v2 = """// 这是一个立即执行的匿名函数，以避免污染全局作用域
return (function() {
    let labels = [];

    // ========= 新增：递归遍历 Shadow DOM =========
    function getAllElementsIncludingShadowRoots(root) {
        const all = [];
        function traverse(node) {
            if (!(node instanceof Element)) return;
            all.push(node);

            // 进入 open shadowRoot
            if (node.shadowRoot) {
                Array.from(node.shadowRoot.children).forEach(traverse);
            }

            // 遍历子元素
            Array.from(node.children).forEach(traverse);
        }
        traverse(root);
        return all;
    }
    // =============================================

    function getMenuData(element) {
        let isMenu = false;
        let menuOptions = [];

        if (element.tagName === 'SELECT') {
            isMenu = true;
            menuOptions = Array.from(element.querySelectorAll('option'))
                               .map(opt => opt.textContent.trim())
                               .filter(Boolean);
            return { isMenu, menuOptions };
        }

        const hasPopup = element.getAttribute('aria-haspopup');
        const controlsId = element.getAttribute('aria-controls');
        if (controlsId && (hasPopup === 'true' || hasPopup === 'listbox' || hasPopup === 'menu')) {
            const menuContainer = document.getElementById(controlsId);
            if (menuContainer) {
                const options = Array.from(menuContainer.querySelectorAll('[role="menuitem"], [role="option"]'));
                if (options.length > 0) {
                    isMenu = true;
                    menuOptions = options.map(opt => opt.textContent.trim()).filter(Boolean);
                    return { isMenu, menuOptions };
                }
            }
        }
        
        let potentialMenuContainer;
        if (element.parentElement && (element.parentElement.querySelector('ul[class*="menu"], ul[class*="dropdown"], div[role="menu"]'))) {
            potentialMenuContainer = element.parentElement.querySelector('ul[class*="menu"], ul[class*="dropdown"], div[role="menu"]');
        } else if (element.nextElementSibling && (element.nextElementSibling.tagName === 'UL' || element.nextElementSibling.tagName === 'DIV')) {
             potentialMenuContainer = element.nextElementSibling;
        }

        if (potentialMenuContainer) {
            const options = Array.from(potentialMenuContainer.querySelectorAll('li, [role="menuitem"], [role="option"]'));
            if (options.length > 0 && options.length < 50) {
                isMenu = true;
                menuOptions = options.map(opt => opt.textContent.trim()).filter(Boolean);
            }
        }
        return { isMenu, menuOptions };
    }
    
    function checkCollision(rect1, rect2) {
        return !(rect1.right < rect2.left || rect1.left > rect2.right || rect1.bottom < rect2.top || rect1.top > rect2.bottom);
    }

    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16),
            g = parseInt(hex.slice(3, 5), 16),
            b = parseInt(hex.slice(5, 7), 16);
        return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
    }

    // ========= 用新的遍历函数替换 querySelectorAll('*') =========
    var allElements = getAllElementsIncludingShadowRoots(document.documentElement);
    // ==========================================================

    var items = allElements.map(function(element) {
        var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

        var rects = [...element.getClientRects()].filter(bb => {
            if (bb.width === 0 || bb.height === 0) return false; // 过滤掉无大小的元素
            const points = [
                [bb.left + 1, bb.top + 1],
                [bb.right - 1, bb.top + 1],
                [bb.left + 1, bb.bottom - 1],
                [bb.right - 1, bb.bottom - 1]
            ];
            // 只要四个角点中有一个点可见，就认为元素可见
            return points.some(([x, y]) => {
                let elAtPoint = document.elementFromPoint(x, y);
                return elAtPoint &&
                (
                    elAtPoint === element ||
                    element.contains(elAtPoint) ||
                    elAtPoint.contains(element) // 遮罩层/容器盖在 input 上
                );
            });
        }).map(bb => {
            const rect = { left: Math.max(0, bb.left), top: Math.max(0, bb.top), right: Math.min(vw, bb.right), bottom: Math.min(vh, bb.bottom) };
            return { ...rect, width: rect.right - rect.left, height: rect.bottom - rect.top };
        });

        var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);
        var menuData = getMenuData(element);

        // ========= 新增：把 contenteditable 元素视为“输入控件” =========
        const isEditable = element.isContentEditable ||
                           (element.getAttribute && element.getAttribute('contenteditable') === 'true');
        // ============================================================

        return {
            element: element,
            include: 
                // 可编辑区域 (div[contenteditable] 等，如 B站评论框)
                isEditable ||

                // 原有输入控件
                (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") ||

                // 点按类控件
                (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") ||

                // 其它你原来包含的类型
                (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION"),

            area,
            rects,
            text: element.textContent.trim().replace(/\s{2,}/g, ' '),
            isMenu: menuData.isMenu,
            menuOptions: menuData.menuOptions
        };
    }).filter(item => item.include && (item.area >= 20));

    const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));
    // items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y)));
    // items = items.filter(x => !(x.element.parentNode && x.element.parentNode.tagName === 'SPAN' && x.element.parentNode.children.length === 1 && x.element.parentNode.getAttribute('role') && items.some(y => y.element === x.element.parentNode)));
    // items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)));

    function getRandomColor(index) {
        var letters = '0123456789ABCDEF'; var color = '#';
        for (var i = 0; i < 6; i++) { color += letters[Math.floor(Math.random() * 16)]; }
        return color;
    }

    function getFixedColor(index) { return '#000000'; }

    const allBboxes = items.flatMap(item => item.rects);
    const placedLabelRects = [];

    var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

    items.forEach(function(item, index) {
        item.rects.forEach((bbox) => {
            var newElement = document.createElement("div");
            // [关键占位符] COLOR_FUNCTION 会在 Python 侧被替换
            var borderColorHex = COLOR_FUNCTION(index);
            newElement.style.outline = `1px dashed ${borderColorHex}`;
            newElement.style.position = "fixed";
            newElement.style.left = bbox.left + "px";
            newElement.style.top = bbox.top + "px";
            newElement.style.width = bbox.width + "px";
            newElement.style.height = bbox.height + "px";
            newElement.style.pointerEvents = "none";
            newElement.style.boxSizing = "border-box";
            newElement.style.zIndex = 2147483647;

            var label = document.createElement("span");
            label.textContent = index;
            label.style.position = "absolute";
            label.style.background = hexToRgba(borderColorHex, 0.7);
            label.style.color = "white";
            label.style.borderRadius = "2px";
            
            const OUTSIDE_FONT_SIZE = "FONTSIZE", OUTSIDE_PADDING = "1px 3px";
            const INSIDE_FONT_SIZE = "8px", INSIDE_PADDING = "1px 2px";
            const estOuterHeight = 16, estOuterWidth = (String(index).length * 6) + 8, SPACING = 1;
            let labelPlaced = false;
            
            function checkFullCollision(potentialRect) {
                const boxCollision = allBboxes.some(otherBbox => otherBbox !== bbox && checkCollision(potentialRect, otherBbox));
                const labelCollision = placedLabelRects.some(placedRect => checkCollision(potentialRect, placedRect));
                return boxCollision || labelCollision;
            }

            const topPosRect = { left: bbox.left, top: bbox.top - estOuterHeight - SPACING, width: estOuterWidth, height: estOuterHeight };
            topPosRect.right = topPosRect.left + topPosRect.width; topPosRect.bottom = topPosRect.top + topPosRect.height;
            if (bbox.top >= estOuterHeight + SPACING && !checkFullCollision(topPosRect)) {
                label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                label.style.top = `-${estOuterHeight + SPACING}px`; label.style.left = '0px';
                placedLabelRects.push(topPosRect); labelPlaced = true;
            }

            if (!labelPlaced) {
                const leftPosRect = { left: bbox.left - estOuterWidth - SPACING, top: bbox.top, width: estOuterWidth, height: estOuterHeight };
                leftPosRect.right = leftPosRect.left + leftPosRect.width; leftPosRect.bottom = leftPosRect.top + leftPosRect.height;
                if (bbox.left >= estOuterWidth + SPACING && !checkFullCollision(leftPosRect)) {
                   label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                   label.style.top = '0px'; label.style.left = `-${estOuterWidth + SPACING}px`;
                   placedLabelRects.push(leftPosRect); labelPlaced = true;
                }
            }

            if (!labelPlaced) {
                const rightPosRect = { left: bbox.right + SPACING, top: bbox.top, width: estOuterWidth, height: estOuterHeight };
                rightPosRect.right = rightPosRect.left + rightPosRect.width; rightPosRect.bottom = rightPosRect.top + rightPosRect.height;
                if (bbox.right + estOuterWidth + SPACING <= vw && !checkFullCollision(rightPosRect)) {
                    label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                    label.style.top = '0px'; label.style.left = `${bbox.width + SPACING}px`;
                    placedLabelRects.push(rightPosRect); labelPlaced = true;
                }
            }
            
            if (!labelPlaced) {
                const bottomPosRect = { left: bbox.left, top: bbox.bottom + SPACING, width: estOuterWidth, height: estOuterHeight };
                bottomPosRect.right = bottomPosRect.left + bottomPosRect.width; bottomPosRect.bottom = bottomPosRect.top + bottomPosRect.height;
                if (bbox.bottom + estOuterHeight + SPACING <= vh && !checkFullCollision(bottomPosRect)) {
                    label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                    label.style.top = `${bbox.height + SPACING}px`; label.style.left = '0px';
                    placedLabelRects.push(bottomPosRect); labelPlaced = true;
                }
            }

            if (!labelPlaced) {
                label.style.fontSize = INSIDE_FONT_SIZE; label.style.padding = INSIDE_PADDING;
                label.style.top = '0px'; label.style.left = '0px';
                const estInsideWidth = (String(index).length * 5) + 6; const estInsideHeight = 14;
                const insidePosRect = { left: bbox.left, top: bbox.top, width: estInsideWidth, height: estInsideHeight };
                insidePosRect.right = insidePosRect.left + insidePosRect.width; insidePosRect.bottom = insidePosRect.top + insidePosRect.height;
                placedLabelRects.push(insidePosRect);
            }

            newElement.appendChild(label);
            document.body.appendChild(newElement);
            labels.push(newElement);
        });
    });

    return [labels, items];
})();
"""
def get_web_element_rect_with_menu(browser, fix_color=True, fontsize=12):
    if fix_color:
        selected_function = "getFixedColor"
    else:
        selected_function = "getRandomColor"

    # ================= 完整版 JavaScript (菜单检测 + 最终优化版ID定位) =================
    js_script = """
        // 这是一个立即执行的匿名函数，以避免污染全局作用域
        return (function() {
            let labels = [];

            function getMenuData(element) {
                let isMenu = false;
                let menuOptions = [];

                if (element.tagName === 'SELECT') {
                    isMenu = true;
                    menuOptions = Array.from(element.querySelectorAll('option'))
                                       .map(opt => opt.textContent.trim())
                                       .filter(Boolean);
                    return { isMenu, menuOptions };
                }

                const hasPopup = element.getAttribute('aria-haspopup');
                const controlsId = element.getAttribute('aria-controls');
                if (controlsId && (hasPopup === 'true' || hasPopup === 'listbox' || hasPopup === 'menu')) {
                    const menuContainer = document.getElementById(controlsId);
                    if (menuContainer) {
                        const options = Array.from(menuContainer.querySelectorAll('[role="menuitem"], [role="option"]'));
                        if (options.length > 0) {
                            isMenu = true;
                            menuOptions = options.map(opt => opt.textContent.trim()).filter(Boolean);
                            return { isMenu, menuOptions };
                        }
                    }
                }
                
                let potentialMenuContainer;
                if (element.parentElement && (element.parentElement.querySelector('ul[class*="menu"], ul[class*="dropdown"], div[role=\\"menu\\"]'))) {
                    potentialMenuContainer = element.parentElement.querySelector('ul[class*="menu"], ul[class*="dropdown"], div[role=\\"menu\\"]');
                } else if (element.nextElementSibling && (element.nextElementSibling.tagName === 'UL' || element.nextElementSibling.tagName === 'DIV')) {
                     potentialMenuContainer = element.nextElementSibling;
                }

                if (potentialMenuContainer) {
                    const options = Array.from(potentialMenuContainer.querySelectorAll('li, [role="menuitem"], [role="option"]'));
                    if (options.length > 0 && options.length < 50) {
                        isMenu = true;
                        menuOptions = options.map(opt => opt.textContent.trim()).filter(Boolean);
                    }
                }
                return { isMenu, menuOptions };
            }
            
            function checkCollision(rect1, rect2) {
                return !(rect1.right < rect2.left || rect1.left > rect2.right || rect1.bottom < rect2.top || rect1.top > rect2.bottom);
            }

            function hexToRgba(hex, alpha) {
                var r = parseInt(hex.slice(1, 3), 16),
                    g = parseInt(hex.slice(3, 5), 16),
                    b = parseInt(hex.slice(5, 7), 16);
                return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
            }

            var items = Array.prototype.slice.call(
                document.querySelectorAll('*')
            ).map(function(element) {
                var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
                var rects = [...element.getClientRects()].filter(bb => {
                    if (bb.width === 0 || bb.height === 0) return false; // 过滤掉无大小的元素
                    const points = [
                        [bb.left + 1, bb.top + 1],
                        [bb.right - 1, bb.top + 1],
                        [bb.left + 1, bb.bottom - 1],
                        [bb.right - 1, bb.bottom - 1]
                    ];
                    // 只要四个角点中有一个点可见，就认为元素可见
                    return points.some(([x, y]) => {
                        // let elAtPoint = document.elementFromPoint(x, y);
                        // return elAtPoint === element || element.contains(elAtPoint);
                        let elAtPoint = document.elementFromPoint(x, y);
                        return elAtPoint &&
                        (
                            elAtPoint === element ||
                            element.contains(elAtPoint) ||
                            elAtPoint.contains(element) // 关键：遮罩层/容器盖在 input 上
                        );

                    });
                }).map(bb => {
                    const rect = { left: Math.max(0, bb.left), top: Math.max(0, bb.top), right: Math.min(vw, bb.right), bottom: Math.min(vh, bb.bottom) };
                    return { ...rect, width: rect.right - rect.left, height: rect.bottom - rect.top };
                });
                var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);
                var menuData = getMenuData(element);
                return {
                    element: element,
                    include: (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") || (element.tagName === "BUTTON" || element.tagName === "A" || (element.onclick != null) || window.getComputedStyle(element).cursor == "pointer") || (element.tagName === "IFRAME" || element.tagName === "VIDEO" || element.tagName === "LI" || element.tagName === "TD" || element.tagName === "OPTION"),
                    area, rects, text: element.textContent.trim().replace(/\\s{2,}/g, ' '),
                    isMenu: menuData.isMenu, menuOptions: menuData.menuOptions
                };
            }).filter(item => item.include && (item.area >= 20));

            const buttons = Array.from(document.querySelectorAll('button, a, input[type="button"], div[role="button"]'));
            // items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y)));
            // items = items.filter(x => !(x.element.parentNode && x.element.parentNode.tagName === 'SPAN' && x.element.parentNode.children.length === 1 && x.element.parentNode.getAttribute('role') && items.some(y => y.element === x.element.parentNode)));
            // items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)));

            function getRandomColor(index) {
                var letters = '0123456789ABCDEF'; var color = '#';
                for (var i = 0; i < 6; i++) { color += letters[Math.floor(Math.random() * 16)]; }
                return color;
            }

            function getFixedColor(index) { return '#000000'; }

            const allBboxes = items.flatMap(item => item.rects);
            const placedLabelRects = [];

            var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
            var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

            items.forEach(function(item, index) {
                item.rects.forEach((bbox) => {
                    var newElement = document.createElement("div");
                    // [关键占位符]
                    var borderColorHex = COLOR_FUNCTION(index);
                    newElement.style.outline = `1px dashed ${borderColorHex}`;
                    newElement.style.position = "fixed";
                    newElement.style.left = bbox.left + "px";
                    newElement.style.top = bbox.top + "px";
                    newElement.style.width = bbox.width + "px";
                    newElement.style.height = bbox.height + "px";
                    newElement.style.pointerEvents = "none";
                    newElement.style.boxSizing = "border-box";
                    newElement.style.zIndex = 2147483647;

                    var label = document.createElement("span");
                    label.textContent = index;
                    label.style.position = "absolute";
                    label.style.background = hexToRgba(borderColorHex, 0.7);
                    label.style.color = "white";
                    label.style.borderRadius = "2px";
                    
                    const OUTSIDE_FONT_SIZE = "FONTSIZE", OUTSIDE_PADDING = "1px 3px";
                    const INSIDE_FONT_SIZE = "8px", INSIDE_PADDING = "1px 2px";
                    const estOuterHeight = 16, estOuterWidth = (String(index).length * 6) + 8, SPACING = 1;
                    let labelPlaced = false;
                    
                    function checkFullCollision(potentialRect) {
                        const boxCollision = allBboxes.some(otherBbox => otherBbox !== bbox && checkCollision(potentialRect, otherBbox));
                        const labelCollision = placedLabelRects.some(placedRect => checkCollision(potentialRect, placedRect));
                        return boxCollision || labelCollision;
                    }

                    const topPosRect = { left: bbox.left, top: bbox.top - estOuterHeight - SPACING, width: estOuterWidth, height: estOuterHeight };
                    topPosRect.right = topPosRect.left + topPosRect.width; topPosRect.bottom = topPosRect.top + topPosRect.height;
                    if (bbox.top >= estOuterHeight + SPACING && !checkFullCollision(topPosRect)) {
                        label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                        label.style.top = `-${estOuterHeight + SPACING}px`; label.style.left = '0px';
                        placedLabelRects.push(topPosRect); labelPlaced = true;
                    }

                    if (!labelPlaced) {
                        const leftPosRect = { left: bbox.left - estOuterWidth - SPACING, top: bbox.top, width: estOuterWidth, height: estOuterHeight };
                        leftPosRect.right = leftPosRect.left + leftPosRect.width; leftPosRect.bottom = leftPosRect.top + leftPosRect.height;
                        if (bbox.left >= estOuterWidth + SPACING && !checkFullCollision(leftPosRect)) {
                           label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                           label.style.top = '0px'; label.style.left = `-${estOuterWidth + SPACING}px`;
                           placedLabelRects.push(leftPosRect); labelPlaced = true;
                        }
                    }

                    if (!labelPlaced) {
                        const rightPosRect = { left: bbox.right + SPACING, top: bbox.top, width: estOuterWidth, height: estOuterHeight };
                        rightPosRect.right = rightPosRect.left + rightPosRect.width; rightPosRect.bottom = rightPosRect.top + rightPosRect.height;
                        if (bbox.right + estOuterWidth + SPACING <= vw && !checkFullCollision(rightPosRect)) {
                            label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                            label.style.top = '0px'; label.style.left = `${bbox.width + SPACING}px`;
                            placedLabelRects.push(rightPosRect); labelPlaced = true;
                        }
                    }
                    
                    if (!labelPlaced) {
                        const bottomPosRect = { left: bbox.left, top: bbox.bottom + SPACING, width: estOuterWidth, height: estOuterHeight };
                        bottomPosRect.right = bottomPosRect.left + bottomPosRect.width; bottomPosRect.bottom = bottomPosRect.top + bottomPosRect.height;
                        if (bbox.bottom + estOuterHeight + SPACING <= vh && !checkFullCollision(bottomPosRect)) {
                            label.style.fontSize = OUTSIDE_FONT_SIZE; label.style.padding = OUTSIDE_PADDING;
                            label.style.top = `${bbox.height + SPACING}px`; label.style.left = '0px';
                            placedLabelRects.push(bottomPosRect); labelPlaced = true;
                        }
                    }

                    if (!labelPlaced) {
                        label.style.fontSize = INSIDE_FONT_SIZE; label.style.padding = INSIDE_PADDING;
                        label.style.top = '0px'; label.style.left = '0px';
                        const estInsideWidth = (String(index).length * 5) + 6; const estInsideHeight = 14;
                        const insidePosRect = { left: bbox.left, top: bbox.top, width: estInsideWidth, height: estInsideHeight };
                        insidePosRect.right = insidePosRect.left + insidePosRect.width; insidePosRect.bottom = insidePosRect.top + insidePosRect.height;
                        placedLabelRects.push(insidePosRect);
                    }

                    newElement.appendChild(label);
                    document.body.appendChild(newElement);
                    labels.push(newElement);
                });
            });
            return [labels, items];
        })();
    """.replace("FONTSIZE", str(fontsize))

    # 尝试等待网页加载完成，以避免stale element reference: stale element not found in the current frame
    time.sleep(2)
    # [关键修复] 在Python中进行字符串替换
    final_js_script = js_script.replace("COLOR_FUNCTION", selected_function)
    
    # [关键修复] 执行替换后的最终脚本
    rects, items_raw = browser.execute_script(final_js_script)

    # ================= Python 格式化逻辑 (不变) =================
    format_ele_text = []
    for web_ele_id in range(len(items_raw)):
        item = items_raw[web_ele_id]
        
        is_menu = item.get('isMenu', False)
        menu_options = item.get('menuOptions', [])
        
        label_text = item['text']
        
        element_proxy = item['element']
        ele_tag_name = element_proxy.tag_name
        ele_type = element_proxy.get_attribute("type")
        ele_aria_label = element_proxy.get_attribute("aria-label")
        
        input_attr_types = ['text', 'search', 'password', 'email', 'tel']

        if is_menu and menu_options:
            trigger_text = label_text.split('\\n')[0].strip()
            options_str = ', '.join([f'"{opt}"' for opt in menu_options])
            
            base_text = f"[{web_ele_id}]: <{ele_tag_name}>"
            if trigger_text: base_text += f" \"{trigger_text}\""
            elif ele_aria_label: base_text += f" \"{ele_aria_label}\""

            format_ele_text.append(f"{base_text} is a menu with options: [{options_str}];")
            continue

        if not label_text:
            if (ele_tag_name.lower() == 'input' and ele_type in input_attr_types) or ele_tag_name.lower() == 'textarea' or (ele_tag_name.lower() == 'button' and ele_type in ['submit', 'button']):
                if ele_aria_label: format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{ele_aria_label}\";")
                else: format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";" )
        elif label_text and len(label_text) < 200:
            if not ("<img" in label_text and "src=" in label_text):
                if ele_tag_name in ["button", "input", "textarea"]:
                    if ele_aria_label and (ele_aria_label != label_text): format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\", \"{ele_aria_label}\";")
                    else: format_ele_text.append(f"[{web_ele_id}]: <{ele_tag_name}> \"{label_text}\";")
                else:
                    if ele_aria_label and (ele_aria_label != label_text): format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\", \"{ele_aria_label}\";")
                    else: format_ele_text.append(f"[{web_ele_id}]: \"{label_text}\";")

    format_ele_text = '\t'.join(format_ele_text)
    
    web_elements = [item['element'] for item in items_raw]

    return rects, web_elements, format_ele_text



def clip_message(msg, max_img_num):
    clipped_msg = []
    img_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if type(curr_msg['content']) == str:
                clipped_msg = [curr_msg] + clipped_msg
            elif img_num < max_img_num:
                img_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': curr_msg['content'][0]["text"]
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def clip_message_and_obs(msg, max_img_num):
    clipped_msg = []
    img_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if type(curr_msg['content']) == str:
                clipped_msg = [curr_msg] + clipped_msg
            elif img_num < max_img_num:
                img_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                msg_no_pdf = curr_msg['content'][0]["text"].split("Observation:")[0].strip() + "Observation: A screenshot and some texts. (Omitted in context.)"
                msg_pdf    = curr_msg['content'][0]["text"].split("Observation:")[0].strip() + "Observation: A screenshot, a PDF file and some texts. (Omitted in context.)"
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': [{"type":"text", "text":(msg_no_pdf if "You downloaded a PDF file" not in curr_msg['content'][0]["text"] else msg_pdf)}]
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def clip_message_and_obs_text_only(msg, max_tree_num):
    clipped_msg = []
    tree_num = 0
    for idx in range(len(msg)):
        curr_msg = msg[len(msg) - 1 - idx]
        if curr_msg['role'] != 'user':
            clipped_msg = [curr_msg] + clipped_msg
        else:
            if tree_num < max_tree_num:
                tree_num += 1
                clipped_msg = [curr_msg] + clipped_msg
            else:
                msg_no_pdf = curr_msg['content'].split("Observation:")[0].strip() + "Observation: An accessibility tree. (Omitted in context.)"
                msg_pdf = curr_msg['content'].split("Observation:")[0].strip() + "Observation: An accessibility tree and a PDF file. (Omitted in context.)"
                curr_msg_clip = {
                    'role': curr_msg['role'],
                    'content': msg_no_pdf if "You downloaded a PDF file" not in curr_msg['content'] else msg_pdf
                }
                clipped_msg = [curr_msg_clip] + clipped_msg
    return clipped_msg


def print_message(json_object, save_dir=None):
    remove_b64code_obj = []
    for obj in json_object:
        if obj['role'] != 'user':
            # print(obj)
            logging.info(obj)
            remove_b64code_obj.append(obj)
        else:
            if type(obj['content']) == str:
                # print(obj)
                logging.info(obj)
                remove_b64code_obj.append(obj)
            else:
                print_obj = {
                    'role': obj['role'],
                    'content': obj['content']
                }
                for item in print_obj['content']:
                    if item['type'] == 'image':
                        item['image'] =  {"url": "data:image/png;base64,{b64_img}"}
                # print(print_obj)
                logging.info(print_obj)
                remove_b64code_obj.append(print_obj)
    if save_dir:
        with open(os.path.join(save_dir, 'interact_messages.json'), 'w', encoding='utf-8') as fw:
            json.dump(remove_b64code_obj, fw, indent=2)
    # return remove_b64code_obj


def get_webarena_accessibility_tree(browser, save_file=None):
    browser_info = fetch_browser_info(browser)
    accessibility_tree = fetch_page_accessibility_tree(browser_info, browser, current_viewport_only=True)
    content, obs_nodes_info = parse_accessibility_tree(accessibility_tree)
    content = clean_accesibility_tree(content)
    if save_file:
        with open(save_file + '.json', 'w', encoding='utf-8') as fw:
            json.dump(obs_nodes_info, fw, indent=2)
        with open(save_file + '.txt', 'w', encoding='utf-8') as fw:
            fw.write(content)


    return content, obs_nodes_info


def compare_images(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1_array = np.asarray(img1)
    img2_array = np.asarray(img2)

    difference = np.abs(img1_array - img2_array)

    total_difference = np.sum(difference)

    return total_difference


def get_pdf_retrieval_ans_from_assistant(client, pdf_path, task):
    # print("You download a PDF file that will be retrieved using the Assistant API.")
    logging.info("You download a PDF file that will be retrieved using the Assistant API.")
    file = client.files.create(
        file=open(pdf_path, "rb"),
        purpose='assistants'
    )
    # print("Create assistant...")
    logging.info("Create assistant...")
    assistant = client.beta.assistants.create(
        instructions="You are a helpful assistant that can analyze the content of a PDF file and give an answer that matches the given task, or retrieve relevant content that matches the task.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=task,
        file_ids=[file.id]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    while True:
        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            break
        time.sleep(2)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages_text = messages.data[0].content[0].text.value
    file_deletion_status = client.beta.assistants.files.delete(
        assistant_id=assistant.id,
        file_id=file.id
    )
    # print(file_deletion_status)
    logging.info(file_deletion_status)
    assistant_deletion_status = client.beta.assistants.delete(assistant.id)
    # print(assistant_deletion_status)
    logging.info(assistant_deletion_status)
    return messages_text



def extract_information(text):
    text = text.strip()
    
    if text.startswith("ANSWER;"):
        return "answer", {"content": text.split("ANSWER;", 1)[-1].strip()}

    if text.startswith("Call;"):
        return "call", {"content": text.split("Call;", 1)[-1].strip()}

    patterns = {
        "click": r"Click\s*[\(\[]?(\d+)[\)\]]?",
        "type": r"Type\s*[\(\[]?(\d+)[\)\]]?[; ]+[\(\[]?(.*?)[\)\]]?$",
        "select": r"Select\s*[\(\[]?(\d+)[\)\]]?[; ]+[\(\[]?(.*?)[\)\]]?$",
        "scroll": r"Scroll\s*[\(\[]?(\d+|WINDOW)[\)\]]?[; ]+[\(\[]?(up|down)[\)\]]?$",
        "wait": r"^\s*Wait",
        "goback": r"^\s*GoBack",
        "google": r"^\s*Google",
        "wikipedia": r"^\s*Wikipedia"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE) # 添加 re.IGNORECASE 使命令不区分大小写
        if match:
            if key in ["type", "select", "scroll"]:
                return key, {"number": match.group(1), "content": match.group(2)}
            elif key == "click":
                return key, match.groups()
            elif key in ["wait", "goback", "google", "wikipedia"]:
                return key, match.groups()

    return None, None


def format_msg(it, init_msg, pdf_obs, warn_obs, img_url, web_text, web_text_num, task_image=None, max_iter=None):
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Textual information:\n{web_text}"
        # init_msg += ". You must only refer to the elements using the labels explicitly listed above"
        if task_image:
            if isinstance(task_image, list):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': "The follow image will be used in the next task:\n"},
                    ]
                }
                for image in task_image:
                    if "http" in image:
                        init_msg_format["content"].append({"image": image})
                    else:
                        init_msg_format["content"].append({"image": f"data:image/png;base64,{image}"})
                init_msg_format["content"].append({'text': init_msg})
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
            else:
                if "http" in task_image:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": task_image},
                            {'text': init_msg},
                        ]
                    }
                else:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": f"data:image/png;base64,{task_image}"},
                            {'text': init_msg},
                        ]
                    }
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
        else:
            if img_url.startswith("file://"):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": img_url}
                    ]
                }
            else:
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": f"data:image/png;base64,{img_url}"}
                    ]
                }
        return init_msg_format
    else:
        if not pdf_obs:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        else:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        return curr_msg
    

def format_msg_OnlineMind2Web(it, init_msg, pdf_obs, warn_obs, img_url, web_text, task_image=None, max_iter=None):
    if it == 1:
        init_msg += (f" (Remaining Steps: {max_iter-it}). ")
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Textual information:\n{web_text}"
        # init_msg += ". You must only refer to the elements using the labels explicitly listed above"
        if task_image:
            if isinstance(task_image, list):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': "The follow image will be used in the next task:\n"},
                    ]
                }
                for image in task_image:
                    if "http" in image:
                        init_msg_format["content"].append({"image": image})
                    else:
                        init_msg_format["content"].append({"image": f"data:image/png;base64,{image}"})
                init_msg_format["content"].append({'text': init_msg})
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
            else:
                if "http" in task_image:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": {"url": task_image}},
                            {'text': init_msg},
                        ]
                    }
                else:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": f"data:image/png;base64,{task_image}"},
                            {'text': init_msg},
                        ]
                    }
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
        else:
            if img_url.startswith("file://"):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": img_url}
                    ]
                }
            else:
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": f"data:image/png;base64,{img_url}"}
                    ]
                }
        return init_msg_format
    else:
        if not pdf_obs:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        else:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        
        if max_iter == it:
            curr_msg['content'][0]['text'] = (f" (Remaining Steps: {max_iter-it}). Please perform the ANSWER action only, and do not perform any other actions. ") + curr_msg['content'][0]['text']
        else:
            curr_msg['content'][0]['text'] = (f" (Remaining Steps: {max_iter-it}). ") + curr_msg['content'][0]['text']
        return curr_msg


def format_msg_Mind2Web2(it, init_msg, pdf_obs, warn_obs, img_url, web_text, url, task_image=None, max_iter=None):
    if it == 1:
        init_msg += (f" (Remaining Steps: {max_iter-it}). [Current URL: {url}] ")
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Textual information:\n{web_text}"
        # init_msg += ". You must only refer to the elements using the labels explicitly listed above"
        if task_image:
            if isinstance(task_image, list):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': "The follow image will be used in the next task:\n"},
                    ]
                }
                for image in task_image:
                    if "http" in image:
                        init_msg_format["content"].append({"image": image})
                    else:
                        init_msg_format["content"].append({"image": f"data:image/png;base64,{image}"})
                init_msg_format["content"].append({'text': init_msg})
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
            else:
                if "http" in task_image:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": {"url": task_image}},
                            {'text': init_msg},
                        ]
                    }
                else:
                    init_msg_format = {
                        'role': 'user',
                        'content': [
                            {'text': "The follow image will be used in the next task:\n"},
                            {"image": f"data:image/png;base64,{task_image}"},
                            {'text': init_msg},
                        ]
                    }
                if img_url.startswith("file://"):
                    init_msg_format["content"].append({"image": img_url})
                else:
                    init_msg_format["content"].append({"image": f"data:image/png;base64,{img_url}"})
        else:
            if img_url.startswith("file://"):
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": img_url}
                    ]
                }
            else:
                init_msg_format = {
                    'role': 'user',
                    'content': [
                        {'text': init_msg},
                        {"image": f"data:image/png;base64,{img_url}"}
                    ]
                }
        return init_msg_format
    else:
        if not pdf_obs:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        else:
            if img_url.startswith("file://"):
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': img_url
                        }
                    ]
                }
            else:
                curr_msg = {
                    'role': 'user',
                    'content': [
                        {'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                        {
                            
                            'image': f"data:image/png;base64,{img_url}"
                        }
                    ]
                }
        
        curr_msg['content'][0]['text'] = (f" (Remaining Steps: {max_iter-it}). [Current URL: {url}] ") + curr_msg['content'][0]['text']
        return curr_msg
    

def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg