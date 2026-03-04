# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tasks that require interacting with a browser."""

import random
import time
from typing import Any
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils


class BrowserTask(task_eval.TaskEval):
  """Base class for browser tasks."""

  app_names = ['chrome']
  complexity = 2
  schema = {
      'type': 'object',
      'properties': {
          'browser_task_seed': {'type': 'number'},
      },
      'required': ['browser_task_seed'],
  }
  template = ''
  HTML = ''  # Implementation overrides.

  preamble = (
      'Open the file task.html in Downloads in the file manager; when prompted'
      ' open it with Chrome.'
  )

  def initialize_device_time(self, env: interface.AsyncEnv) -> None:
    """Initializes the device time."""
    datetime_utils.toggle_auto_settings(
        env.controller, datetime_utils.Toggle.ON
    )
    time.sleep(1.0)

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    chrome_activity = adb_utils.extract_package_name(
        adb_utils.get_adb_activity('chrome')
    )

    adb_utils.clear_app_data(
        chrome_activity,
        env.controller,
    )
    adb_utils.grant_permissions(
        chrome_activity,
        'android.permission.POST_NOTIFICATIONS',
        env.controller,
    )

    html = self.HTML.replace('%%SEED%%', str(self.params['browser_task_seed']))
    task_html_path = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), 'task.html'
    )
    with open(task_html_path, 'w') as f:
      f.write(html)
    file_utils.copy_data_to_device(
        task_html_path,
        file_utils.convert_to_posix_path(
            device_constants.DOWNLOAD_DATA, 'task.html'
        ),
        env.controller,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)
    adb_utils.clear_app_data(
        adb_utils.extract_package_name(adb_utils.get_adb_activity('chrome')),
        env.controller,
    )
    datetime_utils.toggle_auto_settings(
        env.controller, datetime_utils.Toggle.OFF
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    state = env.get_state()
    package_name = adb_utils.extract_package_name(
        adb_utils.get_current_activity(env.controller)[0]
    )
    if package_name != 'com.android.chrome':
      return 0.0

    for element in state.ui_elements:
      if element.text == 'Success!':
        return 1.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {'browser_task_seed': random.randint(0, 2**32 - 1)}


class BrowserMaze(BrowserTask):
  """Task to create a maze game."""

  @property
  def goal(self) -> str:
    return (
        self.preamble
        + ' Then navigate the X to the bottom-right cell, by using the'
        ' direction buttons.'
    )

  HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Maze Puzzle</title>
  <style>
    .row {
      display: flex;
    }

    .cell {
      width: 110px;
      height: 110px;
      border: 1px solid black;
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: 56px;
    }

    .wall {
      background-color: black;
    }

    .character {
      color: black;
    }

    .goal {
      background-color: green;
    }

    .controls {
      margin-top: 10px;
    }

    .controls button {
      margin-right: 5px;
      padding: 15px 28px;
      font-size: 30px;
    }
  </style>
</head>
<body>

  <div id="maze"></div>

  <div class="controls">
    <button onclick="moveCharacter('up')">Up</button>
    <button onclick="moveCharacter('down')">Down</button>
    <button onclick="moveCharacter('left')">Left</button>
    <button onclick="moveCharacter('right')">Right</button>
  </div>

  <script>
    const mazeSize = 4;
    let mazeLayout = [];
    let characterPosition = { row: 0, col: 0 };

    class SeededRNG {
    constructor(seed) {
        this.seed = seed;
    }

    random() {
        const a = 1664525;
        const c = 1013904223;
        const m = 2 ** 32;

        this.seed = (a * this.seed + c) % m;
        return this.seed / m;
    }
    }

    rng = new SeededRNG(%%SEED%%)
    function generateMaze() {
      mazeLayout = [];
      for (let row = 0; row < mazeSize; row++) {
        const currentRow = [];
        for (let col = 0; col < mazeSize; col++) {
          currentRow.push('#');
        }
        mazeLayout.push(currentRow);
      }

      // Create a path from start to goal
      const stack = [{ row: 0, col: 0 }];
      const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

      while (stack.length > 0) {
        const { row, col } = stack.pop();
        mazeLayout[row][col] = ' ';

        if (row === mazeSize - 1 && col === mazeSize - 1) {
          break;
        }

        // Shuffle the order of directions
        for (let i = directions.length - 1; i > 0; i--) {
          const j = Math.floor(rng.random() * (i + 1));
          [directions[i], directions[j]] = [directions[j], directions[i]];
        }

        for (const [dx, dy] of directions) {
          const newRow = row + dx;
          const newCol = col + dy;
          if (
            newRow >= 0 &&
            newRow < mazeSize &&
            newCol >= 0 &&
            newCol < mazeSize &&
            mazeLayout[newRow][newCol] === '#'
          ) {
            stack.push({ row: newRow, col: newCol });
          }
        }
      }

      mazeLayout[0][0] = ' ';
      mazeLayout[mazeSize - 1][mazeSize - 1] = '$';
      characterPosition = { row: 0, col: 0 };
    }

    function renderMaze() {
      const mazeElement = document.getElementById('maze');
      mazeElement.innerHTML = '';

      for (let row = 0; row < mazeLayout.length; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'row';

        for (let col = 0; col < mazeLayout[row].length; col++) {
          const cellElement = document.createElement('div');
          cellElement.className = 'cell';

          if (mazeLayout[row][col] === '#') {
            cellElement.classList.add('wall');
          } else if (row === characterPosition.row && col === characterPosition.col) {
            cellElement.classList.add('character');
            cellElement.innerHTML = 'X';
          } else if (mazeLayout[row][col] === '$') {
            cellElement.classList.add('goal');
          }

          rowElement.appendChild(cellElement);
        }

        mazeElement.appendChild(rowElement);
      }
    }

    function moveCharacter(direction) {
      const newPosition = { ...characterPosition };

      switch (direction) {
        case 'up':
          newPosition.row--;
          break;
        case 'down':
          newPosition.row++;
          break;
        case 'left':
          newPosition.col--;
          break;
        case 'right':
          newPosition.col++;
          break;
      }

      if (isValidMove(newPosition)) {
        characterPosition = newPosition;
        renderMaze();
        checkGoalReached();
      }
    }

    function isValidMove(position) {
      const { row, col } = position;
      if (
        row < 0 ||
        row >= mazeLayout.length ||
        col < 0 ||
        col >= mazeLayout[row].length ||
        mazeLayout[row][col] === '#'
      ) {
        return false;
      }
      return true;
    }

    function checkGoalReached() {
      const { row, col } = characterPosition;
      if (mazeLayout[row][col] === '$') {
        document.body.innerHTML = '<h1>Success!</h1>';
      }
    }

    generateMaze();
    renderMaze();
  </script>
</body>
</html>"""


class BrowserMultiply(BrowserTask):
  """Task for multiplying multiple numbers together."""

  complexity = 2.2

  @property
  def goal(self) -> str:
    return (
        self.preamble
        + ' Then click the button 5 times, remember the numbers displayed, and'
        ' enter their product in the form.'
    )

  HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Memory Task</title>
  <style>
    .container {
      text-align: center;
      margin-top: 50px;
    }

    .number {
      font-size: 48px;
      margin-bottom: 20px;
    }

    .button {
      padding: 10px 20px;
      font-size: 24px;
      margin-bottom: 20px;
    }

    .form {
      margin-top: 20px;
    }

    .form input {
      padding: 5px;
      font-size: 18px;
    }

    .form button {
      padding: 5px 10px;
      font-size: 18px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="number" id="number"></div>
    <button class="button" id="button" onclick="handleButtonClick()">Click Me</button>
    <div class="form" id="form" style="display: none;">
      <input type="number" id="answer" placeholder="Enter the product">
      <button onclick="checkAnswer()">Submit</button>
    </div>
    <div id="result"></div>
  </div>

  <script>
    class SeededRNG {
      constructor(seed) {
        this.seed = seed;
      }

      random() {
        const a = 1664525;
        const c = 1013904223;
        const m = 2 ** 32;
        this.seed = (a * this.seed + c) % m;
        return this.seed / m;
      }
    }

    const rng = new SeededRNG(%%SEED%%);
    const numbers = [];
    let clickCount = 0;

    function generateNumber() {
      const number = Math.floor(rng.random() * 10) + 1;
      numbers.push(number);
      document.getElementById('number').textContent = number;
    }

    function handleButtonClick() {
      clickCount++;
      if (clickCount < 5) {
        generateNumber();
      } else {
        document.getElementById('button').style.display = 'none';
        document.getElementById('number').style.display = 'none';
        document.getElementById('form').style.display = 'block';
      }
    }

    function checkAnswer() {
      const answer = parseInt(document.getElementById('answer').value);
      const product = numbers.reduce((acc, num) => acc * num, 1);
      const result = document.getElementById('result');
      if (answer === product) {
        result.innerHTML = '<h2>Success!</h2>';
      } else {
        result.innerHTML = '<h2></h2>';
      }
    }

    generateNumber();
  </script>
</body>
</html>"""


class BrowserDraw(BrowserTask):
  """Task for drawing on a canvas."""

  @property
  def goal(self) -> str:
    return (
        self.preamble
        + ' Then create a drawing using the three colors shown at the top'
        ' and hit submit.'
    )

  HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Color Challenge</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      text-align: center;
      font-size: 14px;
    }
    canvas {
      border: 1px solid black;
      touch-action: none;
    }
    .color-button {
      width: 30px;
      height: 30px;
      margin: 3px;
      border: none;
      cursor: pointer;
    }
    #colorPalette {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      max-width: 300px;
      margin: 0 auto;
    }
    #canvasContainer {
      display: flex;
      justify-content: center;
    }
    #taskColors div {
      width: 30px;
      height: 30px;
      margin: 3px;
      display: inline-block;
    }
    button {
      margin: 5px;
      padding: 5px 10px;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div id="taskColors"></div>
  <div id="canvasContainer">
    <canvas id="canvas" width="300" height="300"></canvas>
  </div>
  <br>
  <p>Available Colors:</p>
  <div id="colorPalette"></div>
  <br>
  <button id="clearButton">Clear</button>
  <button id="submitButton">Submit</button>
  <p id="result"></p>
  <script>
    class SeededRNG {
      constructor(seed) {
        this.seed = seed;
      }

      random() {
        const a = 1664525;
        const c = 1013904223;
        const m = 2 ** 32;
        this.seed = (a * this.seed + c) % m;
        return this.seed / m;
      }
    }

    const rng = new SeededRNG(%%SEED%%);

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const taskColorsElement = document.getElementById('taskColors');
    const colorPalette = document.getElementById('colorPalette');
    const clearButton = document.getElementById('clearButton');
    const submitButton = document.getElementById('submitButton');
    const resultElement = document.getElementById('result');

    let taskColors = [];

    const availableColors = [
      '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
      '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
      '#ffa500', '#ff1493', '#9932cc', '#20b2aa', '#4b0082', '#00ff7f',
      '#ff6347', '#00ced1', '#9400d3', '#f0e68c', '#ff8c00', '#228b22',
    ];

    function clearCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function generateRandomColors(count) {
      const colors = [];
      const remainingColors = [...availableColors];

      for (let i = 0; i < count; i++) {
        if (remainingColors.length === 0) {
          break;
        }

        const randomIndex = Math.floor(rng.random() * remainingColors.length);
        const selectedColor = remainingColors[randomIndex];
        colors.push(selectedColor);
        remainingColors.splice(randomIndex, 1);
      }

      return colors;
    }

    function displayTaskColors() {
      taskColorsElement.innerHTML = '';
      taskColors.forEach(color => {
        const div = document.createElement('div');
        div.style.backgroundColor = color;
        div.style.width = '50px';
        div.style.height = '50px';
        div.style.display = 'inline-block';
        div.style.margin = '5px';
        taskColorsElement.appendChild(div);
      });
    }

    function createColorPalette() {
      colorPalette.innerHTML = '';
      availableColors.forEach(color => {
        const button = document.createElement('button');
        button.style.backgroundColor = color;
        button.classList.add('color-button');
        button.addEventListener('click', () => {
          ctx.strokeStyle = color;
        });
        colorPalette.appendChild(button);
      });
    }

    function submitTask() {
      submitButton.disabled = true;
      evaluateTask();
      submitButton.disabled = false;
    }

    function evaluateTask() {
      const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const usedColors = new Set();
      for (let i = 0; i < pixelData.length; i += 4) {
        const r = pixelData[i];
        const g = pixelData[i + 1];
        const b = pixelData[i + 2];
        const color = rgbToHex(r, g, b);
        usedColors.add(color);
      }
      const success = taskColors.every(color => usedColors.has(color));
      showResult(success);
    }

    function rgbToHex(r, g, b) {
      const componentToHex = (c) => {
        const hex = c.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
      };
      return '#' + componentToHex(r) + componentToHex(g) + componentToHex(b);
    }

    function showResult(success) {
      if (success) {
        resultElement.textContent = 'Success!';
      } else {
        resultElement.textContent = '';
      }
    }

    function init() {
      taskColors = generateRandomColors(3);
      displayTaskColors();
      createColorPalette();
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function startDrawing(e) {
      isDrawing = true;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX || e.touches[0].clientX) - rect.left;
      const y = (e.clientY || e.touches[0].clientY) - rect.top;
      lastX = x * scaleX;
      lastY = y * scaleY;
    }

    function draw(e) {
      if (!isDrawing) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX || e.touches[0].clientX) - rect.left;
      const y = (e.clientY || e.touches[0].clientY) - rect.top;
      const currentX = x * scaleX;
      const currentY = y * scaleY;
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(currentX, currentY);
      ctx.stroke();
      [lastX, lastY] = [currentX, currentY];
    }
    function stopDrawing() {
      isDrawing = false;
    }

    init();
    clearButton.addEventListener('click', clearCanvas);
    submitButton.addEventListener('click', submitTask);
  </script>
</body>
</html>
"""
