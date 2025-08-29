package com.google.androidenv.accessibilityforwarder

import android.accessibilityservice.AccessibilityService
import android.content.Context
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import com.google.androidenv.accessibilityforwarder.A11yServiceGrpcKt.A11yServiceCoroutineStub
import com.google.gson.Gson
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.ProxyDetector
import io.grpc.StatusException
import java.io.File
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout

/**
 * An Android service that listens to accessibility events and forwards them via gRPC.
 *
 * This service also logs the accessibility tree if [LogFlags.logAccessibilityTree] is set and if
 * [LogFlags.grpcPort] is positive.
 *
 * Please see
 * https://developer.android.com/reference/android/view/accessibility/AccessibilityEvent#getEventType()
 * for a comprehensive list of events emitted by Android.
 */
class AccessibilityForwarder(
  private val channelFactory: (host: String, port: Int) -> ManagedChannel = { host, port ->
    ManagedChannelBuilder.forAddress(host, port)
      .proxyDetector(ProxyDetector { _ -> null })
      .usePlaintext()
      .build()
  }
) : AccessibilityService() {

  init {
    // Spawn long-running thread for periodically logging the tree.
    Thread(
        Runnable {
          while (LogFlags.a11yTreePeriodMs > 0) {
            try {
              val windows = this.windows
              logAccessibilityTree(windows)
            } catch (e: ConcurrentModificationException) {
              continue
            }

            Thread.sleep(/* millis= */ LogFlags.a11yTreePeriodMs)
          }
        }
      )
      .start()
  }

  // grpcStub has a backing property that can be reset to null.
  private var _grpcStub: A11yServiceCoroutineStub? = null
  val grpcStub: A11yServiceCoroutineStub
    get() {
      if (_grpcStub == null) {
        Log.i(TAG, "Building channel on ${LogFlags.grpcHost}:${LogFlags.grpcPort}.")
        _grpcStub = A11yServiceCoroutineStub(channelFactory(LogFlags.grpcHost, LogFlags.grpcPort))
      }
      return _grpcStub!!
    }

  private fun resetGrpcStub() {
    _grpcStub = null
  }

  override fun onInterrupt() {
    LogFlags.a11yTreePeriodMs = 0 // Turn off periodic tree forwarding.
  }

  override fun onAccessibilityEvent(event: AccessibilityEvent?) {
    if (event == null) {
      Log.i(TAG, "`event` is null.")
      return
    }

    logExtrasForEvent(event)
    val eventType = event.eventType
    val eventTypeStr: String = AccessibilityEvent.eventTypeToString(eventType)
    if (eventTypeStr.isNotEmpty()) {
      Log.i(TAG, eventTypeStr)
    }
  }

  private fun serializeMapToJson(map: MutableMap<String, String>): String {
    val gson = Gson()
    return gson.toJson(map)
  }

  private fun writeToInternalStorage(data: Any, fileName: String, context: Context) {
    val tmpFile = File(context.filesDir, "$fileName.tmp")

    when (data) {
      is ByteArray -> {
        tmpFile.writeBytes(data)
      }
      is String -> {
        tmpFile.writeText(data)
      }
      else -> {
        Log.e(TAG, "Unsupported data type for writing to internal storage.")
        return
      }
    }

    // Rename the file to the final name to prevent partial writes.
    val finalFile = File(context.filesDir, fileName)
    if (tmpFile.renameTo(finalFile)) {
      Log.i(TAG, "File written successfully and renamed to $fileName")
    } else {
      Log.i(TAG, "File renaming failed")
    }
  }

  private fun logAccessibilityTree(windows: List<AccessibilityWindowInfo>) {
    if (!LogFlags.logAccessibilityTree) {
      Log.i(TAG, "Not logging accessibility tree")
      return
    }

    // Check gRPC port before actually building the forest.
    if (LogFlags.grpcPort <= 0) {
      Log.w(TAG, "Can't log accessibility tree because gRPC port has not been set.")
      return
    }

    val forest = creator.buildForest(windows)

    if (LogFlags.logUsingInternalStorage) {
      writeToInternalStorage(forest.toByteArray(), "accessibility_tree.pb", this)
    }

    if (LogFlags.logUsingGRPC) {
      try {
        val grpcTimeoutMillis = 1000L
        val response: ForestResponse =
          with(grpcStub) {
            Log.i(TAG, "sending (blocking) gRPC request for tree.")
            runBlocking { withTimeout(grpcTimeoutMillis) { sendForest(forest) } }
          }
        if (response.error.isNotEmpty()) {
          Log.w(TAG, "gRPC response.error: ${response.error}")
        } else {
          Log.i(TAG, "gRPC request for tree succeeded.")
        }
      } catch (e: StatusException) {
        Log.w(TAG, "gRPC StatusException; are you sure networking is turned on?")
        Log.i(TAG, "extra: exception ['$e']")
        resetGrpcStub()
      } catch (e: TimeoutCancellationException) {
        Log.w(TAG, "gRPC TimeoutCancellationException; are you sure networking is turned on?")
        Log.i(TAG, "extra: exception ['$e']")
        resetGrpcStub()
      }
    }
  }

  /** Logs extras for all event types. */
  private fun logExtrasForEvent(event: AccessibilityEvent) {

    val events: MutableMap<String, String> = mutableMapOf()

    val sourceDescription = event.source?.contentDescription()
    if (!sourceDescription.isNullOrEmpty()) {
      events.put("source_content_description", sourceDescription)
    }

    // Output the event text.
    val eventText = event.text.joinToString(", ")
    if (eventText.isNotEmpty()) {
      events.put("event_text", eventText)
    }

    // Output the source text.
    val sourceText = event.source?.text?.toString()
    if (!sourceText.isNullOrEmpty()) {
      events.put("source_text", sourceText)
    }

    val eventTypeStr: String = AccessibilityEvent.eventTypeToString(event.eventType)
    if (eventTypeStr.isNotEmpty()) {
      events.put("event_type", eventTypeStr)
    }

    val className = event.source?.className?.toString()
    if (!className.isNullOrEmpty()) {
      events.put("source_class_name", className)
    }

    val packageName = event.packageName?.toString()
    if (!packageName.isNullOrEmpty()) {
      events.put("event_package_name", packageName)
    }

    // Text editing properties.
    val beforeText = event.beforeText?.toString()
    if (!beforeText.isNullOrEmpty()) {
      events.put("before_text", beforeText)
    }

    val fromIndex = event.fromIndex
    if (fromIndex != -1) {
      events.put("from_index", fromIndex.toString())
    }

    val toIndex = event.toIndex
    if (toIndex != -1) {
      events.put("to_index", toIndex.toString())
    }

    val addedCount = event.addedCount
    if (addedCount != -1) {
      events.put("added_count", addedCount.toString())
    }

    val removedCount = event.removedCount
    if (removedCount != -1) {
      events.put("removed_count", removedCount.toString())
    }

    //  Text traversal properties
    val movementGranularity = event.movementGranularity
    if (movementGranularity != 0) {
      events.put("movement_granularity", movementGranularity.toString())
    }

    val action = event.action
    if (action != 0) {
      events.put("action", action.toString())
    }

    // Scrolling properties.
    if (eventTypeStr == "TYPE_VIEW_SCROLLED") {
      events.put("scroll_delta_x", event.scrollDeltaX.toString())
      events.put("scroll_delta_y", event.scrollDeltaY.toString())
    }

    // Report viewID so we know exactly where the event came from.
    val viewId = event.source?.viewIdResourceName?.toString()
    if (!viewId.isNullOrEmpty()) {
      events.put("view_id", viewId)
    }

    if (LogFlags.logUsingInternalStorage) {
      writeToInternalStorage(serializeMapToJson(events), "event.json", this)
    }

    // Format [events] as a Python dict.
    if (events.isNotEmpty()) {
      events.put("event_timestamp_ms", event.eventTime.toString(10))
      // Check if we want to use gRPC.
      if (LogFlags.grpcPort > 0) {
        try {
          val grpcTimeoutMillis = 1000L
          val request = eventRequest { this.event.putAll(events) }
          val response: EventResponse =
            with(grpcStub) {
              Log.i(TAG, "sending (blocking) gRPC request for event.")
              runBlocking { withTimeout(grpcTimeoutMillis) { sendEvent(request) } }
            }
          if (response.error.isNotEmpty()) {
            Log.w(TAG, "gRPC response.error: ${response.error}")
          } else {
            Log.i(TAG, "gRPC request for event succeeded.")
          }
        } catch (e: StatusException) {
          Log.w(TAG, "gRPC StatusException; are you sure networking is turned on?")
          Log.i(TAG, "extra: exception ['$e']")
          resetGrpcStub()
        } catch (e: TimeoutCancellationException) {
          Log.w(TAG, "gRPC TimeoutCancellationException; are you sure networking is turned on?")
          Log.i(TAG, "extra: exception ['$e']")
          resetGrpcStub()
        }
      } else {
        Log.w(TAG, "Can't log accessibility event because gRPC port has not been set.")
      }
    }
  }

  /** Recursively climbs the accessibility tree until the root, collecting descriptions. */
  private fun AccessibilityNodeInfo?.contentDescription(): String {
    if (this == null) {
      return ""
    }

    val descriptions = mutableListOf<String>()
    var current: AccessibilityNodeInfo? = this
    while (current != null) {
      val description = current.contentDescription
      if (description != null) {
        descriptions.add(description.toString())
      }

      current = current.parent
    }
    return descriptions.joinToString(", ")
  }

  companion object {
    private const val TAG = "AndroidRLTask"
    private val creator = AccessibilityTreeCreator()
  }
}
