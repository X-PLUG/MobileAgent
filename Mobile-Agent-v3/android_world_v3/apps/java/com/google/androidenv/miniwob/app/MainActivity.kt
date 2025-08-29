// Copyright 2024 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.androidenv.miniwob.app

import android.annotation.SuppressLint
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.view.Window
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import com.google.androidenv.AndroidLogger
import com.google.androidenv.AppConfiguration
import com.google.androidenv.Logger
import java.lang.Runnable
import java.util.Timer
import java.util.TimerTask

private const val TAG: String = "AndroidRLTask"

private const val DEFAULT_TASK = "index"

private const val INDEX_PATH = "file:///android_asset/index.html"

private const val GET_REWARD_ACTION = "com.google.androidenv.miniwob.app.GET_REWARD_ACTION"
private const val GET_UTTERANCE_ACTION = "com.google.androidenv.miniwob.app.GET_UTTERANCE_ACTION"

// This javascript function returns a string of a pair of floats where the
// numbers represent respectively the max time and the remaining time for this task.
private const val GET_TIMERS =
  """
function () {
  const timer = document.getElementById('timer-countdown').innerText;
  if (timer === '-') return "0,0";

  const regex = /([0-9]+) \/ ([0-9]+)sec/;
  const matches = timer.match(regex);
  if (matches === null || matches.length !== 3) {
    console.error("Invalid timer: " + timer);
    return "0,0";
  }
  const timeLeft = matches[1];
  const maxTime = matches[2];
  return maxTime + "," + timeLeft;
}()
"""

// Logs Extras at the given rate if `shouldPeriodicallyLogExtras` is true.
class PeriodicExtrasLogger(private val context: MainActivity?, private val rate_ms: Long) :
  TimerTask() {
  init {
    var timer = Timer()
    timer.schedule(this, 0, rate_ms)
  }

  override fun run() {
    if (context == null || context.isFinishing()) {
      // Activity killed
      cancel()
      return
    }

    context.runOnUiThread(
      object : Runnable {
        override fun run() {
          if (!context.isPaused && context.shouldPeriodicallyLogExtras) {
            context.logExtras()
          }
        }
      }
    )
  }
}

class MainActivity(private val logger: Logger = AndroidLogger()) : Activity() {

  private var appConfiguration: AppConfiguration = AppConfiguration()
  private lateinit var webView: WebView
  private var isFirstStep: Boolean = true
  private var inSetterSolverMode: Boolean = true
  private var taskSamplingDone: Boolean = false
  private var periodicExtrasLogger: PeriodicExtrasLogger = PeriodicExtrasLogger(this, 100)
  internal var shouldPeriodicallyLogExtras: Boolean = false
  internal var isPaused: Boolean = false
  private var episodeReward: String = ""
  private lateinit var rewardReceiver: RewardBroadcastReceiver
  private var episodeUtterance: String = ""
  private lateinit var taskInfoReceiver: RewardBroadcastReceiver

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    taskInfoReceiver = RewardBroadcastReceiver()
    requestWindowFeature(Window.FEATURE_NO_TITLE)
    setContentView(R.layout.activity_main)

    webView = findViewById(R.id.webView)
    webView.settings.javaScriptEnabled = true
    webView.settings.domStorageEnabled = true // Allows window.localStorage
    webView.setInitialScale(500)

    webView.webChromeClient =
      object : WebChromeClient() {
        override fun onConsoleMessage(consoleMessage: ConsoleMessage): Boolean {
          var message = consoleMessage.message()
          if (message.startsWith("reward")) {
            // The expected message is of the form "reward: 0.5292310719 (raw: 0.690902182)"
            // We want the raw (undiscounted) reward.
            setEpisodeReward(message.split(" ")[3].dropLast(1).toString())
            logger.i(TAG, "reward: " + getEpisodeReward())
            logger.i(TAG, "episode_end")
          } else if (message.startsWith("utterance")) {
            setEpisodeUtterance(message.split("utterance: ")[1])
          } else {
            logger.d(TAG, message)
          }
          return true
        }
      }
    webView.webViewClient =
      object : WebViewClient() {
        override fun onPageFinished(view: WebView, url: String) {
          if (inSetterSolverMode && url == INDEX_PATH && !taskSamplingDone) {
            // We sample 10 tasks and then trigger the function that draws the table of tasks.
            webView.loadUrl(
              """javascript:
tasks.selectTasksForEpisode(10, /*excludedTasks=*/BROKEN_ANDROID_TASKS);
window.onload();
"""
            )
            taskSamplingDone = true
            logger.i(TAG, "Tasks selected.")
          }
          super.onPageFinished(view, url)
        }

        override fun shouldOverrideUrlLoading(
          view: WebView?,
          request: WebResourceRequest?,
        ): Boolean {
          logger.i(TAG, "shouldOverrideUrlLoading")
          val url = request?.url.toString()
          view?.loadUrl(url)
          return super.shouldOverrideUrlLoading(view, request)
        }
      }
    appConfiguration = AppConfiguration.fromIntent(intent) ?: return
    loadTask()
  }

  override fun onPause() {
    isPaused = true
    super.onPause()
    unregisterReceiver(taskInfoReceiver)
  }

  // This is for a research environment, running on emulators. We need to query this using adb
  // hence it is cross-app and cross-process.
  @SuppressLint("UnprotectedReceiver")
  override fun onResume() {
    isPaused = false
    super.onResume()
    val filter =
      IntentFilter().apply {
        addAction(GET_REWARD_ACTION)
        addAction(GET_UTTERANCE_ACTION)
      }
    registerReceiver(taskInfoReceiver, filter)
  }

  override fun onBackPressed() {
    if (inSetterSolverMode && webView.canGoBack()) {
      webView.goBack()
    }
  }

  override fun onNewIntent(intent: Intent?) {
    super.onNewIntent(intent)
    intent ?: return
    val extras: Bundle? = intent.getExtras()
    extras ?: return
    for (key: String in extras.keySet()) {
      when (key) {
        "reset" -> reset()
        "step" -> step()
        "RL_TASK_APP_CONFIG" -> {
          appConfiguration = AppConfiguration.fromIntent(intent) ?: return
          for (bundleKey in extras.keySet()) {
            val value = extras.getString(bundleKey)
            logger.i(TAG, "Intent bundle: key='$bundleKey', value='$value'")
          }
          loadTask()
        }
        else -> logger.w(TAG, "Ignoring intent extra: " + key)
      }
    }
  }

  // This method is expected to be called once the reset_steps are finished, and only in
  // task mode ([!inSetterSolverMode]).
  private fun reset() {
    // Skip start screen and pause timers.
    webView.loadUrl(
      """javascript:
core.startEpisodeReal();
clearTimeout(core.EP_TIMER);
clearInterval(core.CD_TIMER);
core.EP_TIMER = null;
core.CD_TIMER = null;
core.hideDisplay();
"""
    )
    logExtras()
    isFirstStep = true
    setEpisodeReward("")
    setEpisodeUtterance("")
  }

  // This method is expected to be called on the first step, and only in
  // task mode ([!inSetterSolverMode]).
  private fun step() {
    if (isFirstStep) {
      isFirstStep = false
      shouldPeriodicallyLogExtras = true
    }
  }

  internal fun logExtras() {
    webView.loadUrl(
      """javascript:
console.log("extra: utterance '" + core.getUtterance() + "'");
console.log("extra: task_name '" + document.URL.substring(document.URL.lastIndexOf('/') + 1).replace(/\.html.*/, '') + "'");
console.log("extra: timer " + $GET_TIMERS);
"""
    )
  }

  // This method is expected to be called at every reset, before the [reset] intent.
  fun loadTask() {
    shouldPeriodicallyLogExtras = false
    val taskName: String = appConfiguration.get("task", DEFAULT_TASK)
    logger.d(TAG, "taskName: " + taskName)
    val path: String
    if (taskName == "index") {
      path = INDEX_PATH
      inSetterSolverMode = true
      taskSamplingDone = false
    } else {
      path = "file:///android_asset/html/miniwob/" + taskName + ".html"
      inSetterSolverMode = false
    }
    webView.loadUrl(path)
  }

  inner class RewardBroadcastReceiver : BroadcastReceiver() {
    // Expose reward, so it can be queried with adb.
    override fun onReceive(context: Context, intent: Intent) {
      if (intent.action == GET_REWARD_ACTION) {
        setResultCode(Activity.RESULT_OK)
        setResultData(getEpisodeReward())
      }
      if (intent.action == GET_UTTERANCE_ACTION) {
        setResultCode(Activity.RESULT_OK)
        setResultData(getEpisodeUtterance())
      }
    }
  }

  fun setEpisodeReward(rewardValue: String) {
    episodeReward = rewardValue
  }

  fun getEpisodeReward(): String {
    return episodeReward
  }

  fun setEpisodeUtterance(utteranceValue: String) {
    episodeUtterance = utteranceValue
  }

  fun getEpisodeUtterance(): String {
    return episodeUtterance
  }

  fun getWebView(): WebView {
    return webView
  }
}
