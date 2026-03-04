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
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LoggerTest {

  @Test
  fun androidLogger_defaultImplementationIsLog() {
    // Arrange.
    val logger = AndroidLogger()
    val tag = "MY_TAG"
    val msg = "My message."

    // Act & Assert.
    // Notice that the default implementation of Android's [Log()] is not mockable so we can't
    // feasibly verify that these messages actually end up in logs without spawning a real system
    // and querying the logs with `logcat`. So here we merely verify that the return values of the
    // default [logger] are the same as what [Log()` returns.
    assertThat(logger.d(tag, msg)).isEqualTo(Log.d(tag, msg))
    assertThat(logger.e(tag, msg)).isEqualTo(Log.e(tag, msg))
    assertThat(logger.i(tag, msg)).isEqualTo(Log.i(tag, msg))
    assertThat(logger.v(tag, msg)).isEqualTo(Log.v(tag, msg))
    assertThat(logger.w(tag, msg)).isEqualTo(Log.w(tag, msg))
    assertThat(logger.wtf(tag, msg)).isEqualTo(Log.wtf(tag, msg))
  }
}
