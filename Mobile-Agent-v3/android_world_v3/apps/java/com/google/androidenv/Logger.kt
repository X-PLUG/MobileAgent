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

package com.google.androidenv

import android.util.Log
import com.google.errorprone.annotations.CanIgnoreReturnValue

/**
 * A class that outputs messages to a log stream.
 *
 * This is used as an interface to Android's [Log] class which only provides static methods. This
 * can then be used to inject dependencies by replacing this implementation. Notice that only the
 * relevant methods to AndroidEnv have been replaced. For more details, please see
 * https://developer.android.com/reference/android/util/Log.
 */
interface Logger {
  /** Sends a DEBUG log message. */
  @CanIgnoreReturnValue fun d(tag: String, message: String): Int

  /** Sends an ERROR log message. */
  @CanIgnoreReturnValue fun e(tag: String, message: String): Int

  /** Sends an INFO log message. */
  @CanIgnoreReturnValue fun i(tag: String, message: String): Int

  /** Sends a VERBOSE log message. */
  @CanIgnoreReturnValue fun v(tag: String, message: String): Int

  /** Sends a WARN log message. */
  @CanIgnoreReturnValue fun w(tag: String, message: String): Int

  /** Sends a "What a Terrible Failure" log message. */
  @CanIgnoreReturnValue fun wtf(tag: String, message: String): Int
}

/** The default Logger implementation which uses `android.util.Log`. */
public class AndroidLogger : Logger {
  override fun d(tag: String, message: String): Int = Log.d(tag, message)

  override fun e(tag: String, message: String): Int = Log.e(tag, message)

  override fun i(tag: String, message: String): Int = Log.i(tag, message)

  override fun v(tag: String, message: String): Int = Log.v(tag, message)

  override fun w(tag: String, message: String): Int = Log.w(tag, message)

  override fun wtf(tag: String, message: String): Int = Log.wtf(tag, message)
}
