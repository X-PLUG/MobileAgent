package com.google.androidenv.accessibilityforwarder

/**
 * Controls global settings in AccessibilityForwarder.
 *
 * Please note that this class is not thread safe.
 */
object LogFlags {
  var logUsingGRPC: Boolean = false
  var logUsingInternalStorage: Boolean = true

  // Whether to log the accessibility tree.
  var logAccessibilityTree: Boolean = false
  // How frequent to emit a11y trees (in milliseconds).
  var a11yTreePeriodMs: Long = 100

  // The gRPC server to connect to. (Only available if grpcPort>0).
  var grpcHost: String = ""
  // If >0 this represents the gRPC port number to connect to.
  var grpcPort: Int = 0
}
