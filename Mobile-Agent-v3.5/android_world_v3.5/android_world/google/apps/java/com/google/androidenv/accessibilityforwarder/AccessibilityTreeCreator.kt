package com.google.androidenv.accessibilityforwarder

import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import com.google.androidenv.accessibilityforwarder.AndroidAccessibilityWindowInfo.WindowType
import java.util.concurrent.ConcurrentHashMap
import java.util.stream.Collectors
import kotlin.collections.mutableListOf
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking

/** Helper methods for creating the android accessibility info extra. */
class AccessibilityTreeCreator() {

  /** Creates an accessibility forest proto. */
  fun buildForest(windowInfos: List<AccessibilityWindowInfo>): AndroidAccessibilityForest {
    val sourcesMap: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo> =
      ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>()
    val windows: List<AndroidAccessibilityWindowInfo> =
      processWindowsAndBlock(windowInfos, sourcesMap)
    return androidAccessibilityForest { this.windows += windows }
  }

  private fun processWindowsAndBlock(
    windowInfos: List<AccessibilityWindowInfo>,
    sourcesMap: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>,
  ): List<AndroidAccessibilityWindowInfo> {
    val windows: List<AndroidAccessibilityWindowInfo>
    runBlocking { windows = processWindows(windowInfos, sourcesMap) }
    return windows
  }

  private suspend fun processWindows(
    windowInfos: List<AccessibilityWindowInfo>,
    sourcesMap: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>,
  ): List<AndroidAccessibilityWindowInfo> {
    var windowInfoProtos = mutableListOf<AndroidAccessibilityWindowInfo>()
    for (i in windowInfos.size - 1 downTo 0) {
      val windowInfoProto = processWindow(windowInfos.get(i), sourcesMap)
      windowInfoProto?.let { windowInfoProtos.add(windowInfoProto) }
    }
    return windowInfoProtos.toList()
  }

  private suspend fun processWindow(
    windowInfo: AccessibilityWindowInfo,
    sources: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>,
  ): AndroidAccessibilityWindowInfo? {
    val bounds = Rect()
    windowInfo.getBoundsInScreen(bounds)
    val root: AccessibilityNodeInfo? = windowInfo.root
    if (root == null) {
      Log.i(TAG, "window root is null")
      return androidAccessibilityWindowInfo {
        this.tree = androidAccessibilityTree {}
        this.isActive = windowInfo.isActive
        this.id = windowInfo.id
        this.layer = windowInfo.layer
        this.isAccessibilityFocused = windowInfo.isAccessibilityFocused
        this.isFocused = windowInfo.isFocused
        this.boundsInScreen = convertToRectProto(bounds)
        this.windowType = toWindowType(windowInfo.type)
      }
    }
    val treeDeferred: Deferred<AndroidAccessibilityTree>
    @Suppress("SuspendBlocks")
    runBlocking { treeDeferred = async { processNodesInWindow(root, sources) } }
    return androidAccessibilityWindowInfo {
      this.tree = treeDeferred.await()
      this.isActive = windowInfo.isActive
      this.id = windowInfo.id
      this.layer = windowInfo.layer
      this.isAccessibilityFocused = windowInfo.isAccessibilityFocused
      this.isFocused = windowInfo.isFocused
      this.boundsInScreen = convertToRectProto(bounds)
      this.windowType = toWindowType(windowInfo.type)
    }
  }

  private suspend fun processNodesInWindow(
    root: AccessibilityNodeInfo,
    sources: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>,
  ): AndroidAccessibilityTree {
    Log.d(TAG, "processNodesInWindow()")
    val traversalQueue = ArrayDeque<ParentChildNodePair>()
    traversalQueue.add(ParentChildNodePair.builder().child(root).build())
    val uniqueIdsCache: UniqueIdsGenerator<AccessibilityNodeInfo> = UniqueIdsGenerator()
    var currentDepth = 0
    val nodesDeferred = mutableListOf<Deferred<AndroidAccessibilityNodeInfo>>()
    val seenNodes: HashSet<AccessibilityNodeInfo> = HashSet()
    seenNodes.add(root)
    @Suppress("SuspendBlocks")
    runBlocking {
      while (!traversalQueue.isEmpty()) {
        // Traverse the tree layer-by-layer.
        // The first layer has only the root and depth 0.
        // The second layer has all the root's children and depth 1.
        for (nodesAtCurrentDepth in traversalQueue.size downTo 1) {
          val nodePair: ParentChildNodePair = traversalQueue.removeFirst()
          for (i in 0 until nodePair.child().childCount) {
            val childNode: AccessibilityNodeInfo? = nodePair.child().getChild(i)
            if (childNode != null && !seenNodes.contains(childNode)) {
              traversalQueue.add(
                ParentChildNodePair.builder().child(childNode).parent(nodePair.child()).build()
              )
              seenNodes.add(childNode)
            }
          }
          val thisDepth = currentDepth
          var deferred = async { processNode(nodePair, sources, uniqueIdsCache, thisDepth) }
          nodesDeferred.add(deferred)
        }
        currentDepth++
      }
    }
    return androidAccessibilityTree { this.nodes += nodesDeferred.awaitAll() }
  }

  companion object {
    private const val TAG = "AndroidRLTask"
  }
}

private fun processNode(
  nodePair: ParentChildNodePair,
  sourceBuilder: ConcurrentHashMap<AndroidAccessibilityNodeInfo, AccessibilityNodeInfo>,
  uniqueIdsCache: UniqueIdsGenerator<AccessibilityNodeInfo>,
  nodeDepth: Int,
): AndroidAccessibilityNodeInfo {
  val node: AccessibilityNodeInfo = nodePair.child()
  val immutableNode: AndroidAccessibilityNodeInfo =
    createAndroidAccessibilityNode(
      node,
      uniqueIdsCache.getUniqueId(node),
      nodeDepth,
      getChildUniqueIds(node, uniqueIdsCache),
    )
  sourceBuilder.put(immutableNode, node)
  return immutableNode
}

private fun createAndroidAccessibilityNode(
  node: AccessibilityNodeInfo,
  nodeId: Int,
  depth: Int,
  childIds: List<Int>,
): AndroidAccessibilityNodeInfo {
  val bounds = Rect()
  node.getBoundsInScreen(bounds)
  val actions = node.getActionList().stream().map(::createAction).collect(Collectors.toList())
  return androidAccessibilityNodeInfo {
    this.actions += actions
    this.boundsInScreen = convertToRectProto(bounds)
    this.isCheckable = node.isCheckable
    this.isChecked = node.isChecked
    this.className = stringFromNullableCharSequence(node.getClassName())
    this.isClickable = node.isClickable
    this.contentDescription = stringFromNullableCharSequence(node.getContentDescription())
    this.isEditable = node.isEditable
    this.isEnabled = node.isEnabled
    this.isFocusable = node.isFocusable
    this.hintText = stringFromNullableCharSequence(node.getHintText())
    this.isLongClickable = node.isLongClickable
    this.packageName = stringFromNullableCharSequence(node.getPackageName())
    this.isPassword = node.isPassword
    this.isScrollable = node.isScrollable
    this.isSelected = node.isSelected
    this.text = stringFromNullableCharSequence(node.getText())
    this.textSelectionEnd = node.getTextSelectionEnd().toLong()
    this.textSelectionStart = node.getTextSelectionStart().toLong()
    this.viewIdResourceName = node.getViewIdResourceName() ?: ""
    this.isVisibleToUser = node.isVisibleToUser
    this.windowId = node.windowId
    this.uniqueId = nodeId
    this.childIds += childIds
    this.drawingOrder = node.drawingOrder
    this.tooltipText = stringFromNullableCharSequence(node.getTooltipText())
    this.depth = depth
  }
}

private fun createAction(
  action: AccessibilityNodeInfo.AccessibilityAction
): AndroidAccessibilityAction =
  AndroidAccessibilityAction.newBuilder()
    .setId(action.id)
    .setLabel(stringFromNullableCharSequence(action.label))
    .build()

private fun getChildUniqueIds(
  node: AccessibilityNodeInfo,
  uniqueIdsCache: UniqueIdsGenerator<AccessibilityNodeInfo>,
): List<Int> {
  val ids = mutableListOf<Int>()
  for (childId in 0 until node.getChildCount()) {
    val child: AccessibilityNodeInfo = node.getChild(childId) ?: continue
    ids.add(uniqueIdsCache.getUniqueId(child))
  }
  return ids.toList()
}

fun stringFromNullableCharSequence(cs: CharSequence?): String = cs?.toString() ?: ""

fun convertToRectProto(rect: Rect) = protoRect {
  left = rect.left
  top = rect.top
  right = rect.right
  bottom = rect.bottom
}

private fun toWindowType(type: Int): WindowType =
  when (type) {
    AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY -> WindowType.TYPE_ACCESSIBILITY_OVERLAY
    AccessibilityWindowInfo.TYPE_APPLICATION -> WindowType.TYPE_APPLICATION
    AccessibilityWindowInfo.TYPE_INPUT_METHOD -> WindowType.TYPE_INPUT_METHOD
    AccessibilityWindowInfo.TYPE_SYSTEM -> WindowType.TYPE_SYSTEM
    AccessibilityWindowInfo.TYPE_SPLIT_SCREEN_DIVIDER -> WindowType.TYPE_SPLIT_SCREEN_DIVIDER
    else -> WindowType.UNKNOWN_TYPE
  }
