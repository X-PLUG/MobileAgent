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

import android.content.Intent
import android.os.Bundle
import com.google.gson.Gson
import com.google.gson.JsonParseException
import com.google.gson.JsonParser

/**
 * A simple way to configure apps using JSON strings.
 *
 * Note: Only JSON numbers and strings are supported. Numbers always return as Doubles.
 *
 * Note: Only "flat" (1-level deep) json objects are supported. Nested values are ignored.
 */
public class AppConfiguration {

  // Supported value types.
  private var doubles = mutableMapOf<String, Double>()
  private var strings = mutableMapOf<String, String>()

  private val gson = Gson()

  /** Returns the value of the given `key` as a `Double`, `default` if not available. */
  fun get(key: String, default: Double): Double {
    return doubles.get(key) ?: default
  }

  /** Returns the value of the given `key` as a `String`, `default` if not available. */
  fun get(key: String, default: String): String {
    return strings.get(key) ?: default
  }

  /** Inserts the (`key`, `value`) pair into the internal state. */
  private fun set(key: String, value: Double) {
    doubles.put(key, value)
  }

  /** Inserts the (`key`, `value`) pair into the internal state. */
  private fun set(key: String, value: String) {
    strings.put(key, value)
  }

  companion object {
    /** Constructs an AppConfiguration from a JSON string. */
    fun fromJson(jsonString: String): AppConfiguration {
      val output = AppConfiguration()
      val json =
        try {
          JsonParser.parseString(jsonString).asJsonObject
        } catch (exception: Exception) {
          when (exception) {
            is JsonParseException -> {
              print("*** Malformed JSON string: $jsonString\n")
              return output
            }
            else -> return output
          }
        }

      for ((key, value) in json.entrySet()) {
        val primitive = if (value.isJsonPrimitive()) value.asJsonPrimitive else continue
        when {
          primitive.isNumber() -> output.set(key, primitive.getAsDouble())
          primitive.isString() -> output.set(key, primitive.getAsString())
        }
      }

      return output
    }

    const val EXTRAS_CONFIG = "RL_TASK_APP_CONFIG" // This particular value is for legacy reasons.

    /** Constructs an [AppConfiguration] from an Android [Intent]. */
    fun fromIntent(intent: Intent?): AppConfiguration? {
      intent ?: return null
      val extras: Bundle = intent.getExtras() ?: return null
      val jsonString: String = extras.getString(EXTRAS_CONFIG) ?: return null
      return AppConfiguration.fromJson(jsonString)
    }
  }
}
