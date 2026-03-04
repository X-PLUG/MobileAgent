package com.google.androidenv.accessibilityforwarder

import android.content.Intent
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FlagsBroadcastReceiverTest {

  @Test
  fun onReceive_nullIntent_shouldNotLogAnything() {
    // Arrange.
    LogFlags.logAccessibilityTree = false
    val receiver = FlagsBroadcastReceiver()

    // Act.
    receiver.onReceive(context = null, intent = null)

    // Assert.
    assertThat(LogFlags.logAccessibilityTree).isFalse()
  }

  @Test
  fun onReceive_nullIntent_actionShouldNotLogAnything() {
    // Arrange.
    LogFlags.logAccessibilityTree = false
    val receiver = FlagsBroadcastReceiver()
    val intent = Intent()

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.logAccessibilityTree).isFalse()
  }

  @Test
  fun onReceive_unknownIntent_actionShouldIssueWarning() {
    // Arrange.
    LogFlags.logAccessibilityTree = false
    val receiver = FlagsBroadcastReceiver()
    val intent = Intent("SOME_WEIRD_ACTION")

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.logAccessibilityTree).isFalse()
  }

  @Test
  fun onReceive_intentWithDisableAction_shouldDisableTreeLogging() {
    // Arrange.
    LogFlags.logAccessibilityTree = true
    val receiver = FlagsBroadcastReceiver()
    val intent = Intent("accessibility_forwarder.intent.action.DISABLE_ACCESSIBILITY_TREE_LOGS")

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.logAccessibilityTree).isFalse()
  }

  @Test
  fun onReceive_intentWithEnableAction_shouldEnableTreeLogging() {
    // Arrange.
    LogFlags.logAccessibilityTree = false
    val receiver = FlagsBroadcastReceiver()
    val intent = Intent("accessibility_forwarder.intent.action.ENABLE_ACCESSIBILITY_TREE_LOGS")

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.logAccessibilityTree).isTrue()
  }

  @Test
  fun onReceive_intentWithSetGrpcActionNoArgs_shouldDefaultToEmuIpAndPortZero() {
    // Arrange.
    LogFlags.grpcHost = "some_host"
    LogFlags.grpcPort = 9999
    val receiver = FlagsBroadcastReceiver()
    val intent = Intent("accessibility_forwarder.intent.action.SET_GRPC")

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.grpcHost).isEqualTo("10.0.2.2")
    assertThat(LogFlags.grpcPort).isEqualTo(0)
  }

  @Test
  fun onReceive_intentWithSetGrpcActionWithHostNoPort_shouldDefaultPortToZero() {
    // Arrange.
    LogFlags.grpcHost = "some_host"
    LogFlags.grpcPort = 9999
    val receiver = FlagsBroadcastReceiver()
    val intent =
      Intent("accessibility_forwarder.intent.action.SET_GRPC").apply {
        putExtra("host", "awesome.server.ca")
      }

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.grpcHost).isEqualTo("awesome.server.ca")
    assertThat(LogFlags.grpcPort).isEqualTo(0)
  }

  @Test
  fun onReceive_intentWithSetGrpcActionWithPortNoHost_shouldDefaultHostToEmuIp() {
    // Arrange.
    LogFlags.grpcHost = "some_host"
    LogFlags.grpcPort = 9999
    val receiver = FlagsBroadcastReceiver()
    val intent =
      Intent("accessibility_forwarder.intent.action.SET_GRPC").apply { putExtra("port", 54321) }

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.grpcHost).isEqualTo("10.0.2.2")
    assertThat(LogFlags.grpcPort).isEqualTo(54321)
  }

  @Test
  fun onReceive_intentWithSetGrpcActionWithHostAndPort_shouldSetBoth() {
    // Arrange.
    LogFlags.grpcHost = "some_host"
    LogFlags.grpcPort = 9999
    val receiver = FlagsBroadcastReceiver()
    val intent =
      Intent("accessibility_forwarder.intent.action.SET_GRPC").apply {
        putExtra("host", "grpc.ca")
        putExtra("port", 54321)
      }

    // Act.
    receiver.onReceive(context = null, intent = intent)

    // Assert.
    assertThat(LogFlags.grpcHost).isEqualTo("grpc.ca")
    assertThat(LogFlags.grpcPort).isEqualTo(54321)
  }
}
