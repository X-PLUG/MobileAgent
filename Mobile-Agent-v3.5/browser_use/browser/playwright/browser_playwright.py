import logging
import termcolor
import time
import os
import sys
import asyncio
from typing import Literal

from playwright_stealth.stealth import Stealth 
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from .som import get_som



PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "shift": "Shift",
    "control": "ControlOrMeta",
    "alt": "Alt",
    "escape": "Escape",
    "space": "Space",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "right": "ArrowRight",
    "down": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "semicolon": ";",
    "equals": "=",
    "multiply": "Multiply",
    "add": "Add",
    "separator": "Separator",
    "subtract": "Subtract",
    "decimal": "Decimal",
    "divide": "Divide",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    "command": "Meta",
}



class PlaywrightComputer:
    """Async Playwright wrapper for your web agent."""

    def __init__(
        self,
        args,
        search_engine_url: str = "https://duckduckgo.com/",
        highlight_mouse: bool = False,
    ):
        initial_url = "https://duckduckgo.com/" if args.web == "" else args.web
        self.args = args
        self._initial_url = initial_url
        self._screen_size = (args.window_width, args.window_height)
        self._search_engine_url = search_engine_url
        self._highlight_mouse = highlight_mouse

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self._storage_state_path = "browser/playwright/storage_state.json" #os.path.join(self.args.task_dir, "storage_state.json")

    async def _handle_new_page(self, new_page: Page):
        """Only keep one tab: redirect new tab url into current page."""
        new_url = new_page.url
        await new_page.close()
        await self._page.goto(new_url)

    async def __aenter__(self):
        print("Creating session...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            args=[
                # "--disable-gpu",
                # "--disable-extensions",
                # "--disable-file-system",
                # "--disable-plugins",
                # "--disable-dev-shm-usage",
                # "--disable-background-networking",
                # "--disable-default-apps",
                # "--disable-sync",
                # "--no-sandbox",
                "--disable-blink-features=AutomationControlled"
            ],
            headless=self.args.headless,
        )

        storage_state = None
        if os.path.exists(self._storage_state_path) and self.args.keep_user_info:
            print("Loadding storage state")
            storage_state = self._storage_state_path

        if sys.platform == "darwin":
            user_agent = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 "
                "Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0 "
            )
            
            self._context = await self._browser.new_context(
                viewport={"width": self._screen_size[0], "height": self._screen_size[1]},
                user_agent=user_agent,
                locale="en-US",
                timezone_id="America/New_York",
                storage_state=storage_state,
            )
        else:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
            self._context = await self._browser.new_context(
                viewport={"width": self._screen_size[0], "height": self._screen_size[1]},
                device_scale_factor=1,
                user_agent=user_agent,
                locale="en-US",
                timezone_id="America/New_York",
                storage_state=storage_state,
            )

        self._page = await self._context.new_page()
        # stealth = Stealth()
        # await stealth.apply_stealth_async(self._page)
        await self._page.goto(self._initial_url, timeout=60000, wait_until="domcontentloaded")

        self._context.on("page", lambda p: asyncio.create_task(self._handle_new_page(p)))

        termcolor.cprint(
            "Started local playwright (async).",
            color="green",
            attrs=["bold"],
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context and self.args.keep_user_info:
            try:
                await self._context.storage_state(path=self._storage_state_path)
            except Exception as e:
                logging.warning(f"Failed to save storage state: {e}")

        if self._context:
            await self._context.close()
        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            if "Browser.close: Connection closed while reading from the driver" in str(e):
                pass
            else:
                raise
        if self._playwright:
            await self._playwright.stop()

    async def open_web_browser(self):
        pass

    async def click_at(self, x: int, y: int):
        await self.highlight_mouse(x, y)
        await self._page.mouse.click(x, y)

        await self._page.wait_for_load_state()

    async def hover_at(self, x: int, y: int):
        await self.highlight_mouse(x, y)
        await self._page.mouse.move(x, y)
        await self._page.wait_for_load_state()

    async def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool = True,
        clear_before_typing: bool = True,
    ):
        await self.highlight_mouse(x, y)
        await self.click_at(x, y)
        await self._page.wait_for_load_state()
        

        if clear_before_typing:
            if sys.platform == "darwin":
                await self.key_combination(["Command", "A"])
            else:
                await self.key_combination(["Control", "A"])
            await asyncio.sleep(0.1)
            await self.key_combination(["Delete"])

            await self._page.wait_for_load_state()

        await self.click_at(x, y)
        await self._page.wait_for_load_state()

        await self._page.keyboard.type(text)

        await self._page.wait_for_load_state()

        if press_enter:
            await self.key_combination(["Enter"])
        await self._page.wait_for_load_state()

    async def _horizontal_document_scroll(
        self, direction: Literal["left", "right"]
    ):
        horizontal_scroll_amount = self.screen_size()[0] // 2
        sign = "-" if direction == "left" else ""
        scroll_argument = f"{sign}{horizontal_scroll_amount}"
        await self._page.evaluate(f"window.scrollBy({scroll_argument}, 0); ")
        await self._page.wait_for_load_state()

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ):
        if direction == "down":
            return await self.key_combination(["PageDown"])
        elif direction == "up":
            return await self.key_combination(["PageUp"])
        elif direction in ("left", "right"):
            return await self._horizontal_document_scroll(direction)
        else:
            raise ValueError("Unsupported direction: ", direction)

    async def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int = 400,
    ):
        await self.highlight_mouse(x, y)

        await self._page.mouse.move(x, y)
        await asyncio.sleep(0.1)

        dx = dy = 0
        if direction == "up":
            dy = -magnitude
        elif direction == "down":
            dy = magnitude
        elif direction == "left":
            dx = -magnitude
        elif direction == "right":
            dx = magnitude
        else:
            raise ValueError("Unsupported direction: ", direction)

        await self._page.mouse.wheel(dx, dy)
        await self._page.wait_for_load_state()

    async def wait_5_seconds(self):
        await asyncio.sleep(5)

    async def go_back(self):
        await self._page.go_back()
        await self._page.wait_for_load_state()

    async def go_forward(self):
        await self._page.go_forward()
        await self._page.wait_for_load_state()

    async def search(self):
        return await self.navigate(self._search_engine_url)

    async def navigate(self, url: str):
        normalized_url = url
        if not normalized_url.startswith(("http://", "https://")):
            normalized_url = "https://" + normalized_url
        await self._page.goto(normalized_url)
        await self._page.wait_for_load_state()

    async def key_combination(self, keys: list[str]):
        keys = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]
        for key in keys[:-1]:
            await self._page.keyboard.down(key)
        await self._page.keyboard.press(keys[-1])
        for key in reversed(keys[:-1]):
            await self._page.keyboard.up(key)

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int
    ):
        await self.highlight_mouse(x, y)
        await self._page.mouse.move(x, y)
        await self._page.wait_for_load_state()
        await self._page.mouse.down()
        await self._page.wait_for_load_state()

        await self.highlight_mouse(destination_x, destination_y)
        await self._page.mouse.move(destination_x, destination_y)
        await self._page.wait_for_load_state()
        await self._page.mouse.up()

    async def current_state(self, it):
        try:
            await self._page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            try:
                await self._page.wait_for_load_state("load", timeout=5000)
            except:
                pass

        await asyncio.sleep(1)


        os.makedirs(os.path.join(self.args.task_dir, "trajectory_som"), exist_ok=True)
        os.makedirs(os.path.join(self.args.task_dir, "trajectory"), exist_ok=True)
        img_path = os.path.join(
            self.args.task_dir, f"trajectory_som/screenshot{it}.png"
        )
        img_path_no_box = os.path.join(
            self.args.task_dir, f"trajectory/{it}_full_screenshot.png"
        )

        SoM_list, format_ele_text = await get_som(
            self._page, img_path, img_path_no_box, self.args
        )

        return {
            "img_path": img_path,
            "img_path_no_box": img_path_no_box,
            "SoM": {
                "SoM_list": SoM_list,
                "format_ele_text": format_ele_text,
            },
            "current_url": self._page.url,
        }

    def screen_size(self) -> tuple[int, int]:
        viewport_size = self._page.viewport_size
        if viewport_size:
            return viewport_size["width"], viewport_size["height"]
        return self._screen_size

    async def highlight_mouse(self, x: int, y: int):
        if not self._highlight_mouse:
            return
        await self._page.evaluate(
            f"""
        () => {{
            const element_id = "playwright-feedback-circle";
            const div = document.createElement('div');
            div.id = element_id;
            div.style.pointerEvents = 'none';
            div.style.border = '4px solid red';
            div.style.borderRadius = '50%';
            div.style.width = '20px';
            div.style.height = '20px';
            div.style.position = 'fixed';
            div.style.zIndex = '9999';
            document.body.appendChild(div);

            div.hidden = false;
            div.style.left = {x} - 10 + 'px';
            div.style.top = {y} - 10 + 'px';

            setTimeout(() => {{
                div.hidden = true;
            }}, 2000);
        }}
    """
        )
        await asyncio.sleep(1)

    async def _scroll(self, info):
        scroll_ele_number = info['number']
        scroll_content = info['content']


        if scroll_ele_number == "WINDOW":

            delta = self.args.window_height * 2 // 3
            if scroll_content != 'down':
                delta = -delta

            await self.page.evaluate(
                "(delta) => { window.scrollBy(0, delta); }",
                delta
            )


        else:
            x, y = info["x"], info["y"]
            await self._page.mouse.click(x, y, button="right")

            await asyncio.sleep(0.1)
            await self._page.mouse.click(x, y, button="right")

            if scroll_content == 'down':
                await self._page.keyboard.down("Alt")
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.up("Alt")
            else:
                await self._page.keyboard.down("Alt")
                await self._page.keyboard.press("ArrowUp")
                await self._page.keyboard.up("Alt")
        
        await asyncio.sleep(0.5)

    async def _select(self, x, y, info):
        try:
            await self._page.mouse.click(x, y)
            target_text = info["tool_call"]["arguments"]["option"]

            handle = await self._page.evaluate_handle(
                "([x,y]) => document.elementFromPoint(x,y)",
                [x, y]
            )

            tag = await handle.evaluate("el => el && el.tagName")
            
            if tag is None:
                raise RuntimeError("No elements found at the specified coordinates")
            
            if tag == "SELECT":
                await handle.evaluate("""(sel, label) => {
                    const opt = [...sel.options].find(o => o.label.trim() === label.trim());
                    if (!opt) throw new Error("Option not found: " + label);
                    sel.value = opt.value;
                    sel.dispatchEvent(new Event('input', {bubbles: true}));
                    sel.dispatchEvent(new Event('change', {bubbles: true}));
                }""", target_text)
            else:
                raise RuntimeError(f"The element at the coordinates is not a SELECT; it is {tag}")

        except Exception as e:
            print(f"[Warning] select operation failed, falling back to type_text: {e}")
            await self.type_text_at(x, y, info["tool_call"]["arguments"]["option"])
