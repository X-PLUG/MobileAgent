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
    items = items.filter(x => !buttons.some(y => items.some(z => z.element === y) && y.contains(x.element) && !(x.element === y)));
    items = items.filter(x => !(x.element.parentNode && x.element.parentNode.tagName === 'SPAN' && x.element.parentNode.children.length === 1 && x.element.parentNode.getAttribute('role') && items.some(y => y.element === x.element.parentNode)));
    items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)));

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