(() => {
  const chatList = document.getElementById("chatList");
  const systemWaveCanvas = document.getElementById("systemWave");
  const userWaveCanvas = document.getElementById("userWave");
  const wsStatus = document.getElementById("wsStatus");
  const audioStatus = document.getElementById("audioStatus");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnReset = document.getElementById("btnReset");
  const btnDone = document.getElementById("btnDone");
  const companionNow = document.getElementById("companionNow");
  const companionLog = document.getElementById("companionLog");

  const showCompanionState = (text) => {
    companionNow.textContent = text;
    const line = document.createElement("div");
    line.className = "state-line";
    line.textContent = text;
    companionLog.prepend(line);
    while (companionLog.children.length > 30) {
      companionLog.removeChild(companionLog.lastChild);
    }
  };
  const btnMute = document.getElementById("btnMute");

  const nowTs = () => {
    const d = new Date();
    return d.toLocaleTimeString("en-US", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
  };

  const roleLabels = {
    assistant: "System",
    user: "User",
  };

  const BACKCHANNEL_TOKEN = "<|assistant_backchannel|>";
  const USER_FINISH_TOKEN = "<|user finish speaking|>";
  let lastUserFullText = "";
  let userBaselineText = "";
  let lastSpecial = null;

  let lastUserBubble = null;
  let lastAssistantBubble = null;
  let backchannelHold = false;

  const scrollChat = () => {
    if (!chatList) return;
    chatList.scrollTop = chatList.scrollHeight;
  };

  const createMessage = (role) => {
    const wrap = document.createElement("div");
    wrap.className = `chat-msg ${role}`;
    wrap.dataset.role = role;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    const meta = document.createElement("div");
    meta.className = "chat-meta";

    const body = document.createElement("div");
    body.className = "chat-text";

    bubble.appendChild(meta);
    bubble.appendChild(body);
    wrap.appendChild(bubble);
    return { wrap, meta, body };
  };

  const updateActiveClass = () => {
    if (!chatList) return;
    const nodes = chatList.querySelectorAll(".chat-msg");
    nodes.forEach((node) => node.classList.remove("active"));
    const last = chatList.lastElementChild;
    if (last) last.classList.add("active");
  };

  const appendBubble = (role) => {
    const bubble = createMessage(role);
    if (chatList) {
      chatList.appendChild(bubble.wrap);
      updateActiveClass();
      scrollChat();
    }
    return bubble;
  };

  const insertBubbleBeforeLast = (role) => {
    const bubble = createMessage(role);
    if (chatList) {
      const last = chatList.lastElementChild;
      if (last) {
        chatList.insertBefore(bubble.wrap, last);
      } else {
        chatList.appendChild(bubble.wrap);
      }
      updateActiveClass();
      scrollChat();
    }
    return bubble;
  };

  const getLastRole = () => {
    if (!chatList || !chatList.lastElementChild) return null;
    return chatList.lastElementChild.dataset.role || null;
  };

  const findLastBubbleByRole = (role) => {
    if (!chatList) return null;
    const nodes = chatList.querySelectorAll(`.chat-msg.${role}`);
    if (!nodes.length) return null;
    const wrap = nodes[nodes.length - 1];
    return {
      wrap,
      meta: wrap.querySelector(".chat-meta"),
      body: wrap.querySelector(".chat-text"),
    };
  };

  const ensureFirstUserBubble = () => {
    if (!chatList || chatList.children.length > 0) return;
    lastUserBubble = appendBubble("user");
  };

  const getUserSegment = (fullText) => {
    if (fullText.length < lastUserFullText.length) {
      userBaselineText = "";
    }
    if (userBaselineText && fullText.startsWith(userBaselineText)) {
      let seg = fullText.slice(userBaselineText.length);
      seg = seg.replace(/^\s+/, "");
      return seg;
    }
    return fullText;
  };

  const updateUserText = (text) => {
    if (typeof text !== "string") return;
    // simple chat history: one bubble per user turn; incremental ASR text
    // replaces the CURRENT user bubble until the turn is committed
    if (!lastUserBubble) {
      lastUserBubble = appendBubble("user");
    }
    const msg = lastUserBubble;
    if (!msg) return;
    msg.meta.textContent = roleLabels.user;
    msg.body.textContent = text;
    scrollChat();
  };

  const commitUserTurn = () => {
    // the answer started: the next user_asr opens a NEW user bubble
    lastUserBubble = null;
    lastUserFullText = "";
    userBaselineText = "";
  };

  const startAssistantTurn = () => {
    lastAssistantBubble = appendBubble("assistant");
    if (lastAssistantBubble) {
      lastAssistantBubble.meta.textContent = roleLabels.assistant;
      lastAssistantBubble.body.textContent = "";
    }
    scrollChat();
  };

  const appendAssistantText = (text) => {
    if (typeof text !== "string" || !text.length) return;
    if (!lastAssistantBubble) startAssistantTurn();
    const msg = lastAssistantBubble;
    if (!msg) return;
    msg.meta.textContent = roleLabels.assistant;
    msg.body.appendChild(document.createTextNode(text));
    scrollChat();
  };

  const commitAssistantTurn = () => {
    // the answer finished: the next assistant_start opens a NEW bubble
    lastAssistantBubble = null;
  };

  const appendAssistantBackchannel = (label) => {
    const msg = appendBubble("assistant");
    lastAssistantBubble = msg;
    msg.meta.textContent = roleLabels.assistant;
    // Use a span element for backchannel text with light blue color
    msg.body.innerHTML = "";
    const backchannelSpan = document.createElement("span");
    backchannelSpan.textContent = label;
    backchannelSpan.style.color = "#60a5fa"; // Light blue color for backchannel
    msg.body.appendChild(backchannelSpan);
    scrollChat();
  };

  const getBackchannelLabel = (msg) => {
    let name = "";
    if (msg && typeof msg.filename === "string" && msg.filename.trim()) {
      // Extract filename without extension (e.g., "certainly.wav" -> "certainly")
      const filename = msg.filename.trim();
      name = filename.replace(/\.[^/.]+$/, "") || filename;
    } else if (msg && typeof msg.text === "string") {
      const match = msg.text.match(/([\\w.-]+)\\.wav/i);
      if (match) name = match[1];
    }
    if (!name) name = "assistant_backchannel";
    // Replace underscores with spaces, capitalize first letter, add period and space
    name = name.replace(/_/g, " ");
    return name.charAt(0).toUpperCase() + name.slice(1) + ". ";
  };

  const createWaveform = (canvas, options) => {
    if (!canvas) return { setLevel: () => {} };
    const ctx = canvas.getContext("2d");
    if (!ctx) return { setLevel: () => {} };

    const color = options?.color || "#7aa2ff";
    const background = options?.background || "#0a1226";
    const bars = options?.bars || 26;
    let target = 0;
    let level = 0;
    let phase = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { w: rect.width, h: rect.height };
    };

    const draw = () => {
      const { w, h } = resize();
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, w, h);

      level += (target - level) * 0.18;
      target *= 0.92;
      phase += 0.08 + level * 0.1;

      const gap = Math.max(2, w / (bars * 4));
      const barWidth = Math.max(2, (w - gap * (bars - 1)) / bars);
      const maxH = h * 0.65;

      for (let i = 0; i < bars; i++) {
        const wave = 0.5 + 0.5 * Math.sin(phase + i * 0.35);
        const amp = Math.max(0.08, level) * wave;
        const barH = Math.max(2, amp * maxH);
        const x = i * (barWidth + gap);
        const y = (h - barH) / 2;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, barWidth, barH);
      }
      requestAnimationFrame(draw);
    };

    draw();

    return {
      setLevel: (value) => {
        const v = Math.max(0, Math.min(1, value || 0));
        if (v > target) target = v;
      },
    };
  };

  const systemWave = createWaveform(systemWaveCanvas, {
    color: "#2563eb",
    background: "#f8fafc",
    bars: 28,
  });
  const userWave = createWaveform(userWaveCanvas, {
    color: "#3b82f6",
    background: "#f8fafc",
    bars: 28,
  });

  const calcRms = (buf) => {
    if (!buf || buf.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i];
      sum += v * v;
    }
    return Math.sqrt(sum / buf.length);
  };

  const loc = window.location;
  const wsProto = loc.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProto}//${loc.host}`;

  const TARGET_SR = 24000;
  const FRAME_SAMPLES = 1920; // 80ms @ 24k

  let ws = null;

  // Mic
  let micCtx = null;
  let mediaStream = null;
  let micProcessor = null;
  let pending = new Float32Array(0);

  // Playback
  let muted = false;
  let playCtx = null;
  let playProcessor = null;
  let playQueue = [];
  let playOffset = 0;

  const setWsBadge = (text, ok) => {
    wsStatus.textContent = `WS: ${text}`;
    wsStatus.style.background = ok ? "#1f5d3b" : "#2a3b70";
    wsStatus.style.color = ok ? "#d8ffe9" : "#aab6e8";
  };
  const setAudioBadge = (text, ok) => {
    audioStatus.textContent = `Audio: ${text}`;
    audioStatus.style.background = ok ? "#1f5d3b" : "#2a3b70";
    audioStatus.style.color = ok ? "#d8ffe9" : "#aab6e8";
  };

  const concatF32 = (a, b) => {
    if (!a || a.length === 0) return b;
    if (!b || b.length === 0) return a;
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  };

  // Simple downsample/upsample by averaging buckets (good enough for demo).
  const resampleToTarget = (buffer, inSampleRate, outSampleRate) => {
    if (inSampleRate === outSampleRate) return buffer;
    const ratio = inSampleRate / outSampleRate;
    const newLen = Math.max(1, Math.round(buffer.length / ratio));
    const result = new Float32Array(newLen);
    let offset = 0;
    for (let i = 0; i < newLen; i++) {
      const nextOffset = Math.round((i + 1) * ratio);
      let sum = 0;
      let count = 0;
      for (let j = offset; j < nextOffset && j < buffer.length; j++) {
        sum += buffer[j];
        count++;
      }
      result[i] = count ? sum / count : 0;
      offset = nextOffset;
    }
    return result;
  };

  const ensurePlayback = async () => {
    if (playCtx) return;
    try {
      playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
    } catch {
      playCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    // browsers suspend autoplay-created AudioContexts; resume here (and in
    // startMic's user gesture) so assistant audio actually plays
    if (playCtx.state !== "running") {
      try {
        await playCtx.resume();
      } catch {}
    }

    playProcessor = playCtx.createScriptProcessor(4096, 0, 1);
    playProcessor.onaudioprocess = (e) => {
      const out = e.outputBuffer.getChannelData(0);
      out.fill(0);
      let i = 0;
      while (i < out.length && playQueue.length > 0) {
        const cur = playQueue[0];
        const remain = cur.length - playOffset;
        const n = Math.min(remain, out.length - i);
        out.set(cur.subarray(playOffset, playOffset + n), i);
        playOffset += n;
        i += n;
        if (playOffset >= cur.length) {
          playQueue.shift();
          playOffset = 0;
        }
      }
    };
    playProcessor.connect(playCtx.destination);
  };

  const connect = () => {
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    setWsBadge("Connecting…", false);
    setAudioBadge("Idle", false);

    ws.onopen = () => setWsBadge("Connected", true);
    ws.onclose = () => {
      setWsBadge("Disconnected", false);
      // auto-reconnect: the bridge restarts on every deploy
      setTimeout(connect, 2000);
    };
    ws.onerror = () => setWsBadge("Error", false);

    ws.onmessage = async (evt) => {
      // binary audio
      if (evt.data instanceof ArrayBuffer) {
        const f32 = new Float32Array(evt.data);
        if (f32.length) {
          systemWave.setLevel(Math.min(1, calcRms(f32) * 6));
        }
        if (muted) return;
        await ensurePlayback();
        if (f32.length) {
          playQueue.push(f32);
          setAudioBadge("Playing", true);
        }
        return;
      }

      // JSON text
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }

      if (msg.type === "companion_state") {
        showCompanionState(msg.text);
        return;
      }
      if (msg.type === "user_asr") {
        if (typeof msg.text === "string") {
          updateUserText(msg.text);
        }
        return;
      }

      if (msg.type === "user_commit") {
        commitUserTurn();
        return;
      }

      if (msg.type === "assistant_start") {
        startAssistantTurn();
        return;
      }

      if (msg.type === "assistant_text") {
        if (typeof msg.text === "string") {
          appendAssistantText(msg.text);
        }
        return;
      }

      if (msg.type === "assistant_commit") {
        commitAssistantTurn();
        return;
      }

      if (msg.type === "audio_control") {
        if (msg.action === "stop") {
          // Stop playback immediately: drop queued audio.
          playQueue = [];
          playOffset = 0;
          setAudioBadge("Stopped", false);
        }
        return;
      }
    };
  };

  const startMic = async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      alert("WebSocket is not connected.");
      return;
    }
    if (micCtx) return;

    // getUserMedia is only available in secure contexts (HTTPS or localhost).
    // If the page is served over plain HTTP from a remote host, browsers may expose
    // navigator.mediaDevices as undefined.
    const hasGum =
      typeof navigator !== "undefined" &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function";
    if (!hasGum) {
      const loc = window.location;
      const isLocalhost =
        loc.hostname === "localhost" ||
        loc.hostname === "127.0.0.1" ||
        loc.hostname === "::1";
      const hint =
        loc.protocol !== "https:" && !isLocalhost
          ? `Current origin is ${loc.origin}, which is not a secure context; please use HTTPS, or port-forward and open via http://localhost.`
          : `Please check browser microphone permissions/policies.`;
      throw new Error(
        `Microphone capture is unavailable: navigator.mediaDevices.getUserMedia is not available. ${hint}`
      );
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micCtx = new (window.AudioContext || window.webkitAudioContext)();

    // create the playback AudioContext inside this user gesture so the
    // autoplay policy lets assistant audio play (ensurePlayback resumes later)
    await ensurePlayback();

    const source = micCtx.createMediaStreamSource(mediaStream);
    const bufferSize = 4096;
    micProcessor = micCtx.createScriptProcessor(bufferSize, 1, 1);

    pending = new Float32Array(0);

    micProcessor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const rms = calcRms(input);
      userWave.setLevel(Math.min(1, rms * 6));
      const down = resampleToTarget(input, micCtx.sampleRate, TARGET_SR);
      pending = concatF32(pending, down);
      while (pending.length >= FRAME_SAMPLES) {
        const chunk = pending.slice(0, FRAME_SAMPLES);
        ws.send(chunk.buffer);
        pending = pending.slice(FRAME_SAMPLES);
      }
    };

    source.connect(micProcessor);
    // Connect the mic processor to a ZERO-GAIN node at the destination so
    // onaudioprocess keeps firing (a ScriptProcessor must be downstream-connected
    // to run), WITHOUT routing the user's own voice audibly back to the speakers
    // (which would echo and feed back into ASR).
    const micGain = micCtx.createGain();
    micGain.gain.value = 0;
    micProcessor.connect(micGain);
    micGain.connect(micCtx.destination);

    btnStart.disabled = true;
    btnStop.disabled = false;
  };

  const stopMic = async () => {
    if (!micCtx) return;
    // flush any audio still buffered in `pending` so the tail of the last
    // utterance isn't silently dropped before the STT finalizes
    if (pending && pending.length >= FRAME_SAMPLES) {
      try {
        while (pending.length >= FRAME_SAMPLES) {
          const chunk = pending.slice(0, FRAME_SAMPLES);
          ws.send(chunk.buffer);
          pending = pending.slice(FRAME_SAMPLES);
        }
      } catch {}
    }
    try {
      if (micProcessor) micProcessor.disconnect();
      micProcessor = null;
      await micCtx.close();
    } catch {}
    micCtx = null;

    if (mediaStream) {
      for (const t of mediaStream.getTracks()) t.stop();
    }
    mediaStream = null;

    btnStart.disabled = false;
    btnStop.disabled = true;
  };

  btnStart.addEventListener("click", () => {
    startMic().catch((e) => alert(`Failed to start microphone: ${e?.message || e}`));
  });
  btnStop.addEventListener("click", () => stopMic());

  btnMute.addEventListener("click", () => {
    muted = !muted;
    btnMute.textContent = `Mute: ${muted ? "On" : "Off"}`;
    if (muted) {
      playQueue = [];
      playOffset = 0;
      setAudioBadge("Muted", false);
    }
  });

  btnReset.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send("Reset");
    if (chatList) chatList.innerHTML = "";
    lastUserBubble = null;
    lastAssistantBubble = null;
    lastUserFullText = "";
    userBaselineText = "";
    lastSpecial = null;
    backchannelHold = false;
    ensureFirstUserBubble();
    playQueue = [];
    playOffset = 0;
    setAudioBadge("Idle", false);
  });

  btnDone.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send("Done");
    stopMic();
  });

  ensureFirstUserBubble();
  connect();
})();

