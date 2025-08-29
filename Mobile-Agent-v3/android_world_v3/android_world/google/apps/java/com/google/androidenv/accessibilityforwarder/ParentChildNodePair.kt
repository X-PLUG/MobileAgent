package com.google.androidenv.accessibilityforwarder

import android.view.accessibility.AccessibilityNodeInfo
import com.google.auto.value.AutoValue

/** Parent and child [AccessibilityNodeInfo] relationship. */
@AutoValue
internal abstract class ParentChildNodePair {
  abstract fun parent(): AccessibilityNodeInfo?

  abstract fun child(): AccessibilityNodeInfo

  /** [ParentChildNodePair] builder. */
  @AutoValue.Builder
  abstract class Builder {
    abstract fun parent(parent: AccessibilityNodeInfo?): Builder

    abstract fun child(child: AccessibilityNodeInfo): Builder

    abstract fun build(): ParentChildNodePair
  }

  companion object {
    @JvmStatic fun builder(): Builder = AutoValue_ParentChildNodePair.Builder()
  }
}
