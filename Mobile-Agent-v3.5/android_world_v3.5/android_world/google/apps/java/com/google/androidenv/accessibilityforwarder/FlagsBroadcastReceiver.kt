package com.google.androidenv.accessibilityforwarder

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/** Broadcast receiver responsible for enabling or disabling flags. */
class FlagsBroadcastReceiver() : BroadcastReceiver() {

  override fun onReceive(context: Context?, intent: Intent?) {
    val action = intent?.action
    Log.i(TAG, "Received broadcast intent with action: " + action)
    when (action) {
      ACTION_ENABLE_ACCESSIBILITY_TREE_LOGS -> {
        Log.i(TAG, "Enabling Accessibility Tree logging.")
        LogFlags.logAccessibilityTree = true
      }
      ACTION_DISABLE_ACCESSIBILITY_TREE_LOGS -> {
        Log.i(TAG, "Disabling Accessibility Tree logging.")
        LogFlags.logAccessibilityTree = false
      }
      ACTION_SET_GRPC -> {
        // The Android Emulator uses 10.0.2.2 as a redirect to the workstation's IP. Most often the
        // gRPC server will be running locally so it makes sense to use this as the default value.
        // See https://developer.android.com/studio/run/emulator-networking#networkaddresses.
        val host = intent.getStringExtra("host") ?: "10.0.2.2"
        // The TCP port to connect. If <=0 gRPC is disabled.
        val port = intent.getIntExtra("port", 0)
        Log.i(TAG, "Setting gRPC endpoint to ${host}:${port}.")
        LogFlags.grpcHost = host
        LogFlags.grpcPort = port
      }
      ACTION_ENABLE_GRPC -> {
        Log.i(TAG, "Enabling gRPC logging.")
        LogFlags.logUsingGRPC = true
      }
      ACTION_DISABLE_GRPC -> {
        Log.i(TAG, "Disabling gRPC logging.")
        LogFlags.logUsingGRPC = false
      }
      ACTION_ENABLE_INTERNAL_STORAGE_LOGGING -> {
        Log.i(TAG, "Enabling internal storage logging.")
        LogFlags.logUsingInternalStorage = true
      }
      ACTION_DISABLE_INTERNAL_STORAGE_LOGGING -> {
        Log.i(TAG, "Disabling internal storage logging.")
        LogFlags.logUsingInternalStorage = false
      }
      else -> Log.w(TAG, "Unknown action: ${action}")
    }
  }

  companion object {
    private const val TAG = "FlagsBroadcastReceiver"
    private const val ACTION_ENABLE_ACCESSIBILITY_TREE_LOGS =
      "accessibility_forwarder.intent.action.ENABLE_ACCESSIBILITY_TREE_LOGS"
    private const val ACTION_DISABLE_ACCESSIBILITY_TREE_LOGS =
      "accessibility_forwarder.intent.action.DISABLE_ACCESSIBILITY_TREE_LOGS"
    private const val ACTION_ENABLE_GRPC = "accessibility_forwarder.intent.action.ENABLE_GRPC"
    private const val ACTION_DISABLE_GRPC = "accessibility_forwarder.intent.action.DISABLE_GRPC"
    private const val ACTION_ENABLE_INTERNAL_STORAGE_LOGGING =
      "accessibility_forwarder.intent.action.ENABLE_INTERNAL_STORAGE_LOGGING"
    private const val ACTION_DISABLE_INTERNAL_STORAGE_LOGGING =
      "accessibility_forwarder.intent.action.DISABLE_INTERNAL_STORAGE_LOGGING"
    private const val ACTION_SET_GRPC = "accessibility_forwarder.intent.action.SET_GRPC"
  }
}
