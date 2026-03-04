package com.google.androidenv.accessibilityforwarder

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.function.Function

/** Thread-safe helper class for assigning a unique ID to an object. */
internal class UniqueIdsGenerator<A : Any> {
  private val nextId = AtomicInteger(0)
  private val uniqueIdsByNode = ConcurrentHashMap<A, Int>()

  fun getUniqueId(a: A): Int {
    return uniqueIdsByNode.computeIfAbsent(a, Function { _: A -> nextId.getAndIncrement() })
  }
}
