// ################################################
// Record demonstrations

/* POST submit format

* utterance
* states: array of objects with the following keys:
  - time: time elapsed
  - dom: DOM structure
  - action: action performed at that moment
* reward

*/

var recorder = {};
recorder.SERVER_DEFAULT = 'http://localhost:8032';
recorder.DISPLAY_HTML = `
  <div class="info">
    <label>Server URL:</label>
    <span id='server-name'>-</span>
  </div>
  <div class="info">
    <label>Server reply:</label>
    <span id='server-reply'>-</span>
  </div>
`;

// Add event listeners
recorder.LISTENERS = [
  'click',
  'dblclick',
  'mousedown',
  'mouseup',
  'keypress',
  'keydown',
  'keyup',
  'scroll',
];
recorder.setup =
    function() {
  if (recorder.isSetup) return;
  document.getElementById('reward-display').innerHTML += recorder.DISPLAY_HTML;
  recorder.LISTENERS.forEach(function(name) {
    document.addEventListener(name, recorder['on' + name], true);
    document.addEventListener(name, recorder['on' + name], false);
  });
  recorder.server =
      (core.QueryString.server || recorder.SERVER_DEFAULT) + '/record';
  document.getElementById('server-name').innerHTML = recorder.server;
  var url = window.location.pathname;
  recorder.taskName =
      url.substr(url.lastIndexOf('/') + 1).replace(/\.html/, '');
  recorder.isSetup = true;
}

    // Start recording the episode
    recorder.startRecording =
        function() {
  recorder.data = {};
  recorder.data.taskName = recorder.taskName;
  var utterance = core.getUtterance();
  if (typeof utterance === 'string') {
    recorder.data.utterance = utterance;
  } else {
    recorder.data.utterance = utterance.utterance;
    recorder.data.fields = utterance.fields;
  }
  recorder.data.states = [];
  recorder.isRecording = true;
  recorder.addState(null, null);
}

        // Add a state to the recording data
        recorder.addState =
            function(event, action) {
  if (!recorder.isRecording) return;
  if (event && action) action.timing = event.eventPhase;
  console.log('Adding state', action);
  var state = {
    'time': new Date().getTime() - core.ept0,
    'action': action,
  };
  if (event) event.target.dataset.recording_target = true;
  state.dom = core.getDOMInfo();
  if (event) delete event.target.dataset.recording_target;
  recorder.data.states.push(state);
}

            // Actions
            recorder.ondblclick =
                function(event) {
  if (event.target === core.cover_div || event.pageX >= 160 ||
      event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'dblclick',
    'x': event.pageX,
    'y': event.pageY,
  });
} recorder.onclick =
                    function(event) {
  if (event.target === core.cover_div || event.pageX >= 160 ||
      event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'click',
    'x': event.pageX,
    'y': event.pageY,
  });
} recorder.onmousedown =
                        function(event) {
  if (event.target === core.cover_div || event.pageX >= 160 ||
      event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'mousedown',
    'x': event.pageX,
    'y': event.pageY,
  });
} recorder.onmouseup =
                            function(event) {
  if (event.target === core.cover_div || event.pageX >= 160 ||
      event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'mouseup',
    'x': event.pageX,
    'y': event.pageY,
  });
}

                            recorder.onkeypress =
                                function(event) {
  recorder.addState(event, {
    'type': 'keypress',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
} recorder.onkeydown =
                                    function(event) {
  recorder.addState(event, {
    'type': 'keydown',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
} recorder.onkeyup =
                                        function(event) {
  recorder.addState(event, {
    'type': 'keyup',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
}

                                        recorder.onscroll =
                                            function(event) {
  // Scroll is super redundant; only keep the first one
  if (recorder.data.states.length) {
    var lastState = recorder.data.states[recorder.data.states.length - 1];
    if (lastState.action && lastState.action.type === 'scroll') return;
    // recorder.data.states.pop();     // <-- use this for keeping the last one
  }
  recorder.addState(event, {
    'type': 'scroll',
  });
}

                                            // End recording the episode
                                            recorder.endRecording =
                                                function() {
  recorder.data.reward = WOB_REWARD_GLOBAL;
  recorder.data.rawReward = WOB_RAW_REWARD_GLOBAL;
  // Send the data to the server
  recorder.isRecording = false;
  var data = recorder.data;
  recorder.data = {};  // Prevent future addition
  console.log(data);
  var req = new XMLHttpRequest();
  req.open('POST', recorder.server);
  req.setRequestHeader('Content-type', 'text/plain');
  req.onreadystatechange = function() {
    if (req.readyState === XMLHttpRequest.DONE) {
      var msg = document.getElementById('server-reply');
      if (req.status === 200) {
        msg.setAttribute('style', 'color:green');
        msg.textContent = 'OK: ' + req.responseText;
      } else {
        msg.setAttribute('style', 'color:red');
        msg.textContent = 'ERROR: ' + req.statusText;
      }
    }
  } req.send(JSON.stringify(data));
  // Make it ready for the next episode
  core.cover_div.classList.remove('transparent');
}

                                                // ################################
                                                // Wrappers

                                                // Wrap startEpisodeReal
                                                core.startEpisodeReal = (function(
                                                    startEpisodeReal) {
  return function() {
    if (core.cover_div.classList.contains('transparent')) return;
    recorder.setup();
    startEpisodeReal();
    recorder.startRecording();
  }
})(core.startEpisodeReal);

// Wrap endEpisode
core.endEpisode = (function(endEpisode) {
  return function(reward, time_proportional, reason) {
    if (core.EP_TIMER === null) return;
    core.cover_div.classList.add('transparent');
    endEpisode(reward, time_proportional, reason);
    // Delay to allow the last action to be recorded
    setTimeout(recorder.endRecording, 500);
  }
})(core.endEpisode);
