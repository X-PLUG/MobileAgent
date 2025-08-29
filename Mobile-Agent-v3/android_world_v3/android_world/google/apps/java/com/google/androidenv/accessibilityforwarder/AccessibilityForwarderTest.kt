package com.google.androidenv.accessibilityforwarder

import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import com.google.common.truth.Truth.assertThat
import io.grpc.Status
import io.grpc.StatusException
import io.grpc.inprocess.InProcessChannelBuilder
import io.grpc.inprocess.InProcessServerBuilder
import io.grpc.testing.GrpcCleanupRule
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestParameterInjector
import org.robolectric.Shadows.shadowOf

@RunWith(RobolectricTestParameterInjector::class)
class AccessibilityForwarderTest {

  @get:Rule(order = 1) val cleanupRule = GrpcCleanupRule()

  class FakeAccessibilityService : A11yServiceGrpcKt.A11yServiceCoroutineImplBase() {
    var sendForestChecker: (AndroidAccessibilityForest) -> String = { _ -> "" }
    var sendEventChecker: (EventRequest) -> String = { _ -> "" }

    override suspend fun sendForest(request: AndroidAccessibilityForest) = forestResponse {
      error = sendForestChecker(request)
    }

    override suspend fun sendEvent(request: EventRequest) = eventResponse {
      error = sendEventChecker(request)
    }
  }

  protected lateinit var forwarder: AccessibilityForwarder
  protected val fakeA11yService = FakeAccessibilityService()
  protected val channel by lazy {
    val serverName: String = InProcessServerBuilder.generateName()
    cleanupRule.register(
      InProcessServerBuilder.forName(serverName)
        .directExecutor()
        .addService(fakeA11yService)
        .build()
        .start()
    )
    cleanupRule.register(InProcessChannelBuilder.forName(serverName).directExecutor().build())
  }

  /** Initializes [forwarder] and [LogFlags] from the given args. */
  fun createForwarder(
    logAccessibilityTree: Boolean = false,
    a11yTreePeriodMs: Long = 0,
    grpcHost: String = "10.0.2.2",
    grpcPort: Int = 0,
    a11yWindows: MutableList<AccessibilityWindowInfo>? = null,
  ) {
    LogFlags.logAccessibilityTree = logAccessibilityTree
    LogFlags.a11yTreePeriodMs = a11yTreePeriodMs
    LogFlags.grpcHost = grpcHost
    LogFlags.grpcPort = grpcPort
    LogFlags.logUsingInternalStorage = false
    forwarder = AccessibilityForwarder({ _, _ -> channel })
    if (a11yWindows == null) {
      shadowOf(forwarder).setWindows(mutableListOf(AccessibilityWindowInfo.obtain()))
    } else {
      shadowOf(forwarder).setWindows(a11yWindows)
    }
  }

  @Test
  fun onInterrupt_doesNotCrash() {
    // Arrange.
    createForwarder(logAccessibilityTree = false)
    fakeA11yService.sendEventChecker = { _: EventRequest ->
      assertFalse(true) // This should not be called.
      "" // This should be unreachable
    }

    // Act.
    forwarder.onInterrupt()

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_nullEventShouldBeIgnored() {
    // Arrange.
    createForwarder(logAccessibilityTree = false)
    fakeA11yService.sendEventChecker = { _: EventRequest ->
      assertFalse(true) // This should not be called.
      "" // This should be unreachable
    }

    // Act.
    forwarder.onAccessibilityEvent(null)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_knownEventWithNoInformationShouldNotBeEmitted() {
    // Arrange.
    createForwarder(logAccessibilityTree = false)
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("")
    var event = AccessibilityEvent()
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { _: EventRequest ->
      assertFalse(true) // This should not be called.
      "" // This should be unreachable
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_typeViewClicked_sendEventViaGrpc() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, grpcPort = 1234)
    forwarder = AccessibilityForwarder({ _, _ -> channel })
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_CLICKED)
    event.getText().add("Some text!")
    event.setPackageName("some.loooong.package.name")
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { request: EventRequest ->
      // Check that all fields are consistent with how they were set above.
      assertThat(request.eventMap.get("event_type")).isEqualTo("TYPE_VIEW_CLICKED")
      assertThat(request.eventMap.get("event_package_name")).isEqualTo("some.loooong.package.name")
      assertThat(request.eventMap.get("source_content_description"))
        .isEqualTo("My Content Description")
      assertThat(request.eventMap.get("source_text")).isEqualTo("My Source Text")
      assertThat(request.eventMap.get("source_class_name")).isEqualTo("AwesomeClass")
      assertThat(request.eventMap.get("event_text")).isEqualTo("Some text!")
      assertThat(request.eventMap.get("event_timestamp_ms")).isEqualTo("1357924680")
      // No error message
      ""
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_typeViewTextChanged_ensureAllFieldsForwarded() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, grpcPort = 1234)
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED)
    event.getText().add("Some text!")
    event.fromIndex = 7
    event.beforeText = "Old words"
    event.addedCount = 12
    event.removedCount = 9
    event.setPackageName("some.loooong.package.name")
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { request: EventRequest ->
      // Check that all fields are consistent with how they were set above.
      assertThat(request.eventMap.get("event_type")).isEqualTo("TYPE_VIEW_TEXT_CHANGED")
      assertThat(request.eventMap.get("event_package_name")).isEqualTo("some.loooong.package.name")
      assertThat(request.eventMap.get("source_content_description"))
        .isEqualTo("My Content Description")
      assertThat(request.eventMap.get("source_text")).isEqualTo("My Source Text")
      assertThat(request.eventMap.get("source_class_name")).isEqualTo("AwesomeClass")
      assertThat(request.eventMap.get("event_text")).isEqualTo("Some text!")
      assertThat(request.eventMap.get("event_timestamp_ms")).isEqualTo("1357924680")
      assertThat(request.eventMap.get("from_index")).isEqualTo("7")
      assertThat(request.eventMap.get("before_text")).isEqualTo("Old words")
      assertThat(request.eventMap.get("added_count")).isEqualTo("12")
      assertThat(request.eventMap.get("removed_count")).isEqualTo("9")
      assertFalse(request.eventMap.containsKey("to_index"))
      assertFalse(request.eventMap.containsKey("view_id"))
      assertFalse(request.eventMap.containsKey("action"))
      assertFalse(request.eventMap.containsKey("movement_granularity"))
      assertFalse(request.eventMap.containsKey("scroll_delta_x"))
      assertFalse(request.eventMap.containsKey("scroll_delta_y"))
      // No error message
      ""
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_typeViewScrolled_ensureAllFieldsForwarded() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, grpcPort = 1234)
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_SCROLLED)
    event.getText().add("Some text!")
    event.scrollDeltaX = 13
    event.scrollDeltaY = 27
    event.setPackageName("some.loooong.package.name")
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { request: EventRequest ->
      // Check that all fields are consistent with how they were set above.
      assertThat(request.eventMap.get("event_type")).isEqualTo("TYPE_VIEW_SCROLLED")
      assertThat(request.eventMap.get("event_package_name")).isEqualTo("some.loooong.package.name")
      assertThat(request.eventMap.get("source_content_description"))
        .isEqualTo("My Content Description")
      assertThat(request.eventMap.get("source_text")).isEqualTo("My Source Text")
      assertThat(request.eventMap.get("source_class_name")).isEqualTo("AwesomeClass")
      assertThat(request.eventMap.get("event_text")).isEqualTo("Some text!")
      assertThat(request.eventMap.get("event_timestamp_ms")).isEqualTo("1357924680")
      assertThat(request.eventMap.get("scroll_delta_x")).isEqualTo("13")
      assertThat(request.eventMap.get("scroll_delta_y")).isEqualTo("27")
      assertFalse(request.eventMap.containsKey("from_index"))
      assertFalse(request.eventMap.containsKey("to_index"))
      assertFalse(request.eventMap.containsKey("before_text"))
      assertFalse(request.eventMap.containsKey("added_count"))
      assertFalse(request.eventMap.containsKey("removed_count"))
      // No error message
      ""
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_typeViewTextTraversedAtMovementGranularity_ensureAllFieldsForwarded() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, grpcPort = 1234)
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    nodeInfo.viewIdResourceName = "this.big.old.view.id"
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_TEXT_TRAVERSED_AT_MOVEMENT_GRANULARITY)
    event.getText().add("Some text!")
    event.setPackageName("some.loooong.package.name")
    event.movementGranularity = 5
    event.fromIndex = 6
    event.toIndex = 8
    event.action = 23
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { request: EventRequest ->
      // Check that all fields are consistent with how they were set above.
      assertThat(request.eventMap.get("event_type"))
        .isEqualTo("TYPE_VIEW_TEXT_TRAVERSED_AT_MOVEMENT_GRANULARITY")
      assertThat(request.eventMap.get("event_package_name")).isEqualTo("some.loooong.package.name")
      assertThat(request.eventMap.get("source_content_description"))
        .isEqualTo("My Content Description")
      assertThat(request.eventMap.get("source_text")).isEqualTo("My Source Text")
      assertThat(request.eventMap.get("source_class_name")).isEqualTo("AwesomeClass")
      assertThat(request.eventMap.get("event_text")).isEqualTo("Some text!")
      assertThat(request.eventMap.get("event_timestamp_ms")).isEqualTo("1357924680")
      assertThat(request.eventMap.get("movement_granularity")).isEqualTo("5")
      assertThat(request.eventMap.get("from_index")).isEqualTo("6")
      assertThat(request.eventMap.get("to_index")).isEqualTo("8")
      assertThat(request.eventMap.get("view_id")).isEqualTo("this.big.old.view.id")
      assertThat(request.eventMap.get("action")).isEqualTo("23")
      // No error message
      ""
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_sendingevent_grpcTimeout() {
    // Arrange.
    createForwarder(
      logAccessibilityTree = false,
      a11yTreePeriodMs = 0,
      grpcHost = "amazing.host",
      grpcPort = 4321,
    )
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_CLICKED)
    event.getText().add("Some text!")
    event.setPackageName("some.loooong.package.name")
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { _ ->
      // Delay the request to prompt a timeout
      Thread.sleep(1500L)
      "" // Return no error.
    }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Run a second request to ensure that the channel gets rebuilt.
    fakeA11yService.sendEventChecker = { _ -> "" }
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun onAccessibilityEvent_sendingevent_grpcStatusException() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, grpcHost = "amazing.host", grpcPort = 4321)
    var nodeInfo = AccessibilityNodeInfo()
    nodeInfo.setContentDescription("My Content Description")
    nodeInfo.setText("My Source Text")
    nodeInfo.setClassName("AwesomeClass")
    var event = AccessibilityEvent()
    event.setEventTime(1357924680)
    event.setEventType(AccessibilityEvent.TYPE_VIEW_CLICKED)
    event.getText().add("Some text!")
    event.setPackageName("some.loooong.package.name")
    shadowOf(event).setSourceNode(nodeInfo)
    fakeA11yService.sendEventChecker = { _ -> throw StatusException(Status.UNAVAILABLE) }

    // Act.
    forwarder.onAccessibilityEvent(event)

    // Run a second request to ensure that the channel gets rebuilt.
    fakeA11yService.sendEventChecker = { _ -> "" }
    forwarder.onAccessibilityEvent(event)

    // Assert.
    // See `sendEventChecker` above.
  }

  @Test
  fun logAccessibilityTreeFalse_doesNotLogAccessibilityTree() {
    // Arrange.
    createForwarder(logAccessibilityTree = false, a11yTreePeriodMs = 10, grpcPort = 13579)
    fakeA11yService.sendForestChecker = { _: AndroidAccessibilityForest ->
      assertFalse(true) // This should not be called.
      "" // This should be unreachable
    }

    // Act.
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }

  @Test
  fun grpcPortZero_doesNotSendTree() {
    // Arrange.
    createForwarder(logAccessibilityTree = true, a11yTreePeriodMs = 10, grpcPort = 0)
    fakeA11yService.sendForestChecker = { _: AndroidAccessibilityForest ->
      assertFalse(true) // This should not be called.
      "" // This should be unreachable
    }

    // Act.
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }

  @Test
  fun grpcPortPositive_shouldSendTreeViaGrpc() {
    // Arrange.
    val window = AccessibilityWindowInfo()
    shadowOf(window).setType(AccessibilityWindowInfo.TYPE_SYSTEM)
    createForwarder(
      logAccessibilityTree = true,
      a11yTreePeriodMs = 10,
      grpcPort = 1234,
      a11yWindows = mutableListOf(window),
    )
    fakeA11yService.sendForestChecker = { request: AndroidAccessibilityForest ->
      // Check that we get only a single window.
      assertThat(request.windowsList.size).isEqualTo(1)
      // And that its type is what we set above.
      assertThat(request.windowsList[0].windowType)
        .isEqualTo(AndroidAccessibilityWindowInfo.WindowType.TYPE_SYSTEM)
      // The error message
      "Something went wrong!"
    }

    // Act.
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }

  @Test
  fun grpcPortPositiveAndHost_shouldSendTreeViaGrpc() {
    // Arrange.
    fakeA11yService.sendForestChecker = { request: AndroidAccessibilityForest ->
      // Check that we get only a single window.
      assertThat(request.windowsList.size).isEqualTo(1)
      // And that its type is what we set above.
      assertThat(request.windowsList[0].windowType)
        .isEqualTo(AndroidAccessibilityWindowInfo.WindowType.TYPE_ACCESSIBILITY_OVERLAY)
      "" // Return no error.
    }
    val window = AccessibilityWindowInfo()
    shadowOf(window).setType(AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY)
    createForwarder(
      logAccessibilityTree = true,
      a11yTreePeriodMs = 500,
      grpcHost = "amazing.host",
      grpcPort = 4321,
      a11yWindows = mutableListOf(window),
    )

    // Act.
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }

  @Test
  fun sendingForest_grpcTimeout() {
    // Arrange.
    fakeA11yService.sendForestChecker = { _ ->
      // Delay the request to prompt a timeout
      Thread.sleep(1500L)
      "" // Return no error.
    }
    val window = AccessibilityWindowInfo()
    shadowOf(window).setType(AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY)
    createForwarder(
      logAccessibilityTree = true,
      a11yTreePeriodMs = 10,
      grpcHost = "amazing.host",
      grpcPort = 4321,
      a11yWindows = mutableListOf(window),
    )

    // Act.
    Thread.sleep(2000) // Sleep a bit to give time to trigger the tree logging function.

    // Run a second request to ensure that the channel gets rebuilt.
    fakeA11yService.sendForestChecker = { _ -> "" }
    Thread.sleep(2000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }

  @Test
  fun sendingForest_grpcStatusException() {
    // Arrange.
    val window = AccessibilityWindowInfo()
    shadowOf(window).setType(AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY)
    createForwarder(
      logAccessibilityTree = true,
      a11yTreePeriodMs = 10,
      grpcHost = "amazing.host",
      grpcPort = 4321,
      a11yWindows = mutableListOf(window),
    )
    fakeA11yService.sendForestChecker = { _ -> throw StatusException(Status.UNAVAILABLE) }

    // Act.
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Run a second request to ensure that the channel gets rebuilt.
    fakeA11yService.sendForestChecker = { _ -> "" }
    Thread.sleep(1000) // Sleep a bit to give time to trigger the tree logging function.

    // Assert.
    // See `sendForestChecker` above.
  }
}
