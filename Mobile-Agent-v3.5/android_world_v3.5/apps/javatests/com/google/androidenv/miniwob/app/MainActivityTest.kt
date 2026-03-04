package com.google.androidenv.miniwob.app

import android.app.Activity
import android.app.Application
import android.content.BroadcastReceiver
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.rules.ActivityScenarioRule
import java.lang.reflect.Method
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.shadows.ShadowApplication.Wrapper

private const val TAG: String = "AndroidRLTask"
private const val GET_REWARD_ACTION = "com.google.androidenv.miniwob.app.GET_REWARD_ACTION"
private const val GET_UTTERANCE_ACTION = "com.google.androidenv.miniwob.app.GET_UTTERANCE_ACTION"

@RunWith(RobolectricTestRunner::class)
class MainActivityTest {
  private lateinit var intent: Intent
  private val context = ApplicationProvider.getApplicationContext<Application>()

  @get:Rule var activityScenarioRule = ActivityScenarioRule(MainActivity::class.java)

  private fun getBroadcastReceiver(wrappers: List<Wrapper>): BroadcastReceiver {
    for (wrapper in wrappers) {
      if (
        wrapper.intentFilter.hasAction(GET_REWARD_ACTION) &&
          wrapper.intentFilter.hasAction(GET_UTTERANCE_ACTION)
      ) {
        return wrapper.broadcastReceiver
      }
    }
    throw AssertionError("BroadcastReceiver was not registered")
  }

  @Test
  fun testIntent_isDelivered() {
    // Create a new Intent with test data
    val newIntent = Intent().putExtra("RL_TASK_APP_CONFIG", "{\"task\":\"bisect-angle\"}")

    activityScenarioRule.scenario.onActivity { activity ->
      activity.startActivity(newIntent)
      // Get a reference to ShadowActivity using Robolectric
      val shadowActivity = org.robolectric.Shadows.shadowOf(activity)
      val deliveredIntent = shadowActivity.nextStartedActivity
      // Assert that the intent was delivered
      assert(
        deliveredIntent.extras!!.getString("RL_TASK_APP_CONFIG") == "{\"task\":\"bisect-angle\"}"
      )
    }
  }

  @Test
  fun testOnNewIntent_loadsTaskOnWebview() {
    // Create a new Intent with test data
    val newIntent = Intent().putExtra("RL_TASK_APP_CONFIG", "{\"task\":\"bisect-angle\"}")
    // Find the onNewIntent method using reflection
    val onNewIntentMethod: Method =
      MainActivity::class.java.getDeclaredMethod("onNewIntent", Intent::class.java)
    // Enable access to protected method
    onNewIntentMethod.isAccessible = true

    activityScenarioRule.scenario.onActivity { activity ->
      // Invoke the onNewIntent method using reflection
      onNewIntentMethod.invoke(activity, newIntent)
      // Make assertions that WebView loaded the expected URL for the task intent
      val expectedUrl = "file:///android_asset/html/miniwob/bisect-angle.html"
      val actualUrl = activity.getWebView().getUrl()
      assertEquals(expectedUrl, actualUrl)
    }
  }

  @Test
  fun testBroadcast_getRewardAction_setsResult() {
    val expectedResultCode = Activity.RESULT_OK
    val expectedResultData = "0.590902182"
    val intent = Intent(GET_REWARD_ACTION)

    activityScenarioRule.scenario.onActivity { activity ->
      // Mock a call to set the episode reward
      activity.setEpisodeReward(expectedResultData)
      val broadcastReceiver = getBroadcastReceiver(shadowOf(context).getRegisteredReceivers())
      // Send broadcast with GET_REWARD_ACTION action to RewardBroadcastReceiver
      context.sendOrderedBroadcast(intent, null)
      broadcastReceiver.onReceive(context, intent)
      // Assert that the result code and data have been set by onReceive method
      assertEquals(expectedResultCode, broadcastReceiver.getResultCode())
      assertEquals(expectedResultData, broadcastReceiver.getResultData())
    }
  }

  @Test
  fun testBroadcast_getUtteranceAction_setsResult() {
    val expectedResultCode = Activity.RESULT_OK
    val expectedResultData = "A fake utterance"
    val intent = Intent(GET_UTTERANCE_ACTION)

    activityScenarioRule.scenario.onActivity { activity ->
      // Mock a call to set the episode utterance
      activity.setEpisodeUtterance(expectedResultData)
      val broadcastReceiver = getBroadcastReceiver(shadowOf(context).getRegisteredReceivers())
      // Send broadcast with GET_UTTERANCE_ACTION action to RewardBroadcastReceiver
      context.sendOrderedBroadcast(intent, null)
      broadcastReceiver.onReceive(context, intent)
      // Assert that the result code and data have been set by onReceive method
      assertEquals(expectedResultCode, broadcastReceiver.getResultCode())
      assertEquals(expectedResultData, broadcastReceiver.getResultData())
    }
  }
}
