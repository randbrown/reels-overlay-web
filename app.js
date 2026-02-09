import { PoseLandmarker, FilesetResolver } from
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Body skeleton connections
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],
  [24,26],[26,28],[28,30],[30,32]
];

// Face "V" like the Python look (nose -> eyes)
const FACE_CONNECTIONS = [
  [0, 1],
  [0, 2],
];

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const restartBtn = document.getElementById("restartBtn");
const exportBtn = document.getElementById("exportBtn");

// Disable playback controls until a video is loaded
playBtn.disabled = true;
pauseBtn.disabled = true;
restartBtn.disabled = true;
if (exportBtn) exportBtn.disabled = true;

function setStatus(message, isError = false) {
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

if (exportBtn) {
  exportBtn.addEventListener("click", () => {
    exportWithOverlay();
  });
}

function describeMediaError(error) {
  if (!error) return "Unknown media error";
  switch (error.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return "Video loading was aborted.";
    case MediaError.MEDIA_ERR_NETWORK:
      return "Network error while loading the video.";
    case MediaError.MEDIA_ERR_DECODE:
      return "Decode error (unsupported codec or corrupted file).";
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return "Video format not supported by the browser.";
    default:
      return "Unknown media error.";
  }
}

function pickRecorderMimeType() {
  const preferred = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  for (const type of preferred) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function getAudioTrackFromVideo(video) {
  try {
    const stream = video.captureStream();
    const tracks = stream.getAudioTracks();
    return tracks.length ? tracks[0] : null;
  } catch {
    return null;
  }
}

async function exportWithOverlay() {
  if (!currentVideo || exportInProgress) return;
  exportInProgress = true;
  if (exportBtn) exportBtn.disabled = true;

  const baseName = currentFile?.name?.replace(/\.[^/.]+$/, "") || "overlay";
  setStatus("Exporting…", false);

  const prevMuted = currentVideo.muted;
  const prevVolume = currentVideo.volume;
  currentVideo.muted = false;
  currentVideo.volume = 0;

  currentVideo.currentTime = 0;

  try {
    await currentVideo.play();
  } catch (err) {
    setStatus("Playback blocked by the browser. Click Play once, then Export.", true);
    currentVideo.muted = prevMuted;
    currentVideo.volume = prevVolume;
    exportInProgress = false;
    if (exportBtn) exportBtn.disabled = false;
    return;
  }

  const canvasStream = canvas.captureStream(30);
  const audioTrack = getAudioTrackFromVideo(currentVideo);
  const tracks = [...canvasStream.getVideoTracks()];
  if (audioTrack) tracks.push(audioTrack);
  const combinedStream = new MediaStream(tracks);

  recordingChunks = [];
  const mimeType = pickRecorderMimeType();
  recorder = new MediaRecorder(combinedStream, mimeType ? { mimeType } : undefined);

  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordingChunks.push(e.data);
  };

  recorder.onerror = () => {
    setStatus("Export failed during recording.", true);
  };

  recorder.onstop = () => {
    const finalType = recorder.mimeType || "video/webm";
    const blob = new Blob(recordingChunks, { type: finalType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const ext = finalType.includes("mp4") ? "mp4" : "webm";
    a.href = url;
    a.download = `${baseName}-overlay.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    currentVideo.muted = prevMuted;
    currentVideo.volume = prevVolume;

    exportInProgress = false;
    if (exportBtn) exportBtn.disabled = false;
    setStatus("Export complete", false);
  };

  recorder.start(200);

  currentVideo.addEventListener("ended", () => {
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, { once: true });
}

const opts = {
  trail: document.getElementById("trail"),
  trailFade: document.getElementById("trailFade"),
  trailDrawAlpha: document.getElementById("trailDrawAlpha"),
  smoothing: document.getElementById("smoothing"),
  drawIds: document.getElementById("drawIds"),
  idSize: document.getElementById("idSize"),
  velocityColor: document.getElementById("velocityColor"),
  scanlines: document.getElementById("scanlines"),
  scanStrength: document.getElementById("scanStrength"),
  detConf: document.getElementById("detConf"),
  trkConf: document.getElementById("trkConf"),
  numPoses: document.getElementById("numPoses"),
  codeOverlay: document.getElementById("codeOverlay"),
};

function bindRangeValue(input) {
  const valueEl = document.querySelector(`.value[data-for="${input.id}"]`);
  if (!valueEl) return;
  const update = () => {
    valueEl.textContent = input.value;
  };
  update();
  input.addEventListener("input", update);
  input.addEventListener("change", update);
}

document.querySelectorAll('input[type="range"]').forEach(bindRangeValue);

let vision = null;
let pose = null;

// Offscreen trail buffer (like your OpenCV trail_buf)
const trailCanvas = document.createElement("canvas");
const trailCtx = trailCanvas.getContext("2d");

// Previous landmarks per pose index (for smoothing/velocity + occlusion memory)
let prevPoses = [];

// We'll keep the current "fit" transform each frame for drawing + landmark mapping
let lastFit = null;

async function initVision() {
  vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
}

async function createPose() {
  if (!vision) await initVision();

  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    },
    runningMode: "VIDEO",
    numPoses: parseInt(opts.numPoses.value, 10),
    minPoseDetectionConfidence: parseFloat(opts.detConf.value),
    minTrackingConfidence: parseFloat(opts.trkConf.value),

    // Helps stability (especially lower body) on some footage
    enablePoseWorldLandmarks: true,
  });

  prevPoses = [];
}

// Debounced rebuild for sliders that require model recreation
let rebuildTimer = null;
function requestRebuild() {
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => createPose(), 250);
}
opts.detConf.addEventListener("input", requestRebuild);
opts.trkConf.addEventListener("input", requestRebuild);
opts.numPoses.addEventListener("input", requestRebuild);

await createPose();

/**
 * Compute crop (sx,sy,sw,sh) that we use to draw 9:16 "fit to reels"
 * and return it so we can apply the same transform to landmarks.
 */
function computeFit(video, outW, outH) {
  const targetAspect = outW / outH;
  const srcAspect = video.videoWidth / video.videoHeight;

  let sx, sy, sw, sh;

  if (srcAspect > targetAspect) {
    // crop left/right
    sw = video.videoHeight * targetAspect;
    sh = video.videoHeight;
    sx = (video.videoWidth - sw) / 2;
    sy = 0;
  } else {
    // crop top/bottom
    sw = video.videoWidth;
    sh = video.videoWidth / targetAspect;
    sx = 0;
    sy = (video.videoHeight - sh) / 2;
  }

  return { sx, sy, sw, sh, outW, outH };
}

function drawFitted(video) {
  const w = canvas.width;
  const h = canvas.height;

  const fit = computeFit(video, w, h);
  lastFit = fit;

  ctx.drawImage(video, fit.sx, fit.sy, fit.sw, fit.sh, 0, 0, w, h);
}

/**
 * Map a normalized landmark (lm.x/lm.y in [0..1] relative to original video)
 * into our cropped+scaled canvas coordinates.
 */
function lmToCanvas(lm) {
  if (!lastFit) return { x: lm.x * canvas.width, y: lm.y * canvas.height };

  const { sx, sy, sw, sh, outW, outH } = lastFit;

  const px = lm.x * currentVideo.videoWidth;
  const py = lm.y * currentVideo.videoHeight;

  const nx = (px - sx) / sw;
  const ny = (py - sy) / sh;

  return { x: nx * outW, y: ny * outH };
}

// Keep feet alive under occlusion
const FOOT_POINTS = new Set([25,26,27,28,29,30,31,32]);

function isValid(lm, index) {
  if (lm.visibility === undefined) return false;
  // extremely lenient for feet/ankles
  if (FOOT_POINTS.has(index)) return lm.visibility > 0.05;
  return lm.visibility > 0.4;
}

function smoothLandmarks(current, prev) {
  if (!prev) return current;

  const base = parseFloat(opts.smoothing.value);

  return current.map((lm, i) => {
    // If occluded/low confidence: hold previous (decay visibility)
    if (!isValid(lm, i)) {
      return {
        x: prev[i].x,
        y: prev[i].y,
        visibility: (prev[i].visibility ?? 1) * 0.95,
      };
    }

    // Extra smoothing for lower body for stability
    const extra = FOOT_POINTS.has(i) ? 0.15 : 0.0;
    const alpha = Math.min(0.96, base + extra);

    return {
      x: alpha * prev[i].x + (1 - alpha) * lm.x,
      y: alpha * prev[i].y + (1 - alpha) * lm.y,
      visibility: lm.visibility,
    };
  });
}

function velocityToColor(v) {
  // Map motion into ramp similar to your python (blue->green->red).
  // The scaling here is more "phone-friendly".
  let t = Math.max(0, Math.min(1, v / 120));
  if (t < 0.5) {
    const a = t / 0.5;
    return `rgb(${Math.round(255*(1-a))}, ${Math.round(255*a)}, 0)`;
  } else {
    const a = (t - 0.5) / 0.5;
    return `rgb(0, ${Math.round(255*(1-a))}, ${Math.round(255*a)})`;
  }
}

function drawScanlines() {
  const strength = parseFloat(opts.scanStrength.value);
  ctx.fillStyle = `rgba(0,0,0,${strength})`;

  // Thicker scanlines
  for (let y = 0; y < canvas.height; y += 4) {
    ctx.fillRect(0, y, canvas.width, 2);
  }
}

// Current video element (created dynamically)
let currentVideo = null;
let lastDetectedPoses = null;
let lastTimestamp = null;
let currentFile = null;
let recorder = null;
let recordingChunks = [];
let exportInProgress = false;

// Function to process and draw a single frame
function processFrame(video, timestamp) {
  // 1) Draw video frame (also computes fit transform for landmark mapping)
  drawFitted(video);

  // 2) Pose detection
  const results = pose.detectForVideo(video, timestamp);
  const poses = results.landmarks || [];

  // Store for potential redraw when paused
  lastDetectedPoses = poses;
  lastTimestamp = timestamp;

  // Resize prevPoses list to match number of returned poses
  while (prevPoses.length < poses.length) prevPoses.push(null);
  if (prevPoses.length > poses.length) prevPoses = prevPoses.slice(0, poses.length);

  // 3) Fade trail buffer
  if (opts.trail.checked) {
    const fade = parseFloat(opts.trailFade.value);
    trailCtx.save();
    trailCtx.globalCompositeOperation = "destination-in";
    trailCtx.fillStyle = `rgba(0,0,0,${fade})`;
    trailCtx.fillRect(0, 0, trailCanvas.width, trailCanvas.height);
    trailCtx.restore();

    trailCtx.globalAlpha = parseFloat(opts.trailDrawAlpha.value);
  } else {
    trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
    trailCtx.globalAlpha = 1.0;
  }

  // 4) Draw each pose
  for (let p = 0; p < poses.length; p++) {
    const cur = poses[p];
    const prev = prevPoses[p];
    const smoothed = smoothLandmarks(cur, prev);
    prevPoses[p] = smoothed;

    // connections
    for (const [a, b] of POSE_CONNECTIONS) {
      const la = smoothed[a];
      const lb = smoothed[b];
      if (!isValid(la, a) || !isValid(lb, b)) continue;

      const A = lmToCanvas(la);
      const B = lmToCanvas(lb);

      let color = "white";
      if (opts.velocityColor.checked && prev) {
        const pa = prev[a];
        const pb = prev[b];
        const Ap = lmToCanvas(pa);
        const Bp = lmToCanvas(pb);
        const v = Math.max(
          Math.hypot(A.x - Ap.x, A.y - Ap.y),
          Math.hypot(B.x - Bp.x, B.y - Bp.y)
        );
        color = velocityToColor(v);
      }

      trailCtx.strokeStyle = color;
      trailCtx.lineWidth = 2;
      trailCtx.beginPath();
      trailCtx.moveTo(A.x, A.y);
      trailCtx.lineTo(B.x, B.y);
      trailCtx.stroke();
    }

    // face V-shape
    for (const [a, b] of FACE_CONNECTIONS) {
      const la = smoothed[a];
      const lb = smoothed[b];
      if (!isValid(la, a) || !isValid(lb, b)) continue;

      const A = lmToCanvas(la);
      const B = lmToCanvas(lb);

      trailCtx.strokeStyle = "white";
      trailCtx.lineWidth = 2;
      trailCtx.beginPath();
      trailCtx.moveTo(A.x, A.y);
      trailCtx.lineTo(B.x, B.y);
      trailCtx.stroke();
    }

    // joints + IDs
    for (let i = 0; i < smoothed.length; i++) {
      const lm = smoothed[i];
      if (!isValid(lm, i)) continue;

      const P = lmToCanvas(lm);

      let color = "white";
      if (opts.velocityColor.checked && prev) {
        const pp = lmToCanvas(prev[i]);
        const v = Math.hypot(P.x - pp.x, P.y - pp.y);
        color = velocityToColor(v);
      }

      trailCtx.fillStyle = color;
      trailCtx.beginPath();
      trailCtx.arc(P.x, P.y, 3, 0, Math.PI * 2);
      trailCtx.fill();

      if (opts.drawIds.checked) {
        ctx.fillStyle = color;
        ctx.font = `${opts.idSize.value}px system-ui, sans-serif`;
        ctx.fillText(String(i), P.x + 8, P.y - 8);
      }
    }
  }

  // 5) Composite trails
  ctx.drawImage(trailCanvas, 0, 0);

  // 6) Scanlines
  if (opts.scanlines.checked) drawScanlines();
}

// Function to redraw current frame (used when paused)
function redrawCurrentFrame() {
  if (!currentVideo || !lastDetectedPoses || lastTimestamp === null) return;
  
  // Redraw video frame
  drawFitted(currentVideo);

  // Use stored poses but re-apply current visual settings
  const poses = lastDetectedPoses;

  // Reapply trail fade
  if (opts.trail.checked) {
    const fade = parseFloat(opts.trailFade.value);
    trailCtx.save();
    trailCtx.globalCompositeOperation = "destination-in";
    trailCtx.fillStyle = `rgba(0,0,0,${fade})`;
    trailCtx.fillRect(0, 0, trailCanvas.width, trailCanvas.height);
    trailCtx.restore();

    trailCtx.globalAlpha = parseFloat(opts.trailDrawAlpha.value);
  } else {
    trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
    trailCtx.globalAlpha = 1.0;
  }

  // Redraw each pose with current settings
  for (let p = 0; p < poses.length; p++) {
    const cur = poses[p];
    const prev = prevPoses[p];
    const smoothed = smoothLandmarks(cur, prev);
    prevPoses[p] = smoothed;

    // connections
    for (const [a, b] of POSE_CONNECTIONS) {
      const la = smoothed[a];
      const lb = smoothed[b];
      if (!isValid(la, a) || !isValid(lb, b)) continue;

      const A = lmToCanvas(la);
      const B = lmToCanvas(lb);

      let color = "white";
      if (opts.velocityColor.checked && prev) {
        const pa = prev[a];
        const pb = prev[b];
        const Ap = lmToCanvas(pa);
        const Bp = lmToCanvas(pb);
        const v = Math.max(
          Math.hypot(A.x - Ap.x, A.y - Ap.y),
          Math.hypot(B.x - Bp.x, B.y - Bp.y)
        );
        color = velocityToColor(v);
      }

      trailCtx.strokeStyle = color;
      trailCtx.lineWidth = 2;
      trailCtx.beginPath();
      trailCtx.moveTo(A.x, A.y);
      trailCtx.lineTo(B.x, B.y);
      trailCtx.stroke();
    }

    // face V-shape
    for (const [a, b] of FACE_CONNECTIONS) {
      const la = smoothed[a];
      const lb = smoothed[b];
      if (!isValid(la, a) || !isValid(lb, b)) continue;

      const A = lmToCanvas(la);
      const B = lmToCanvas(lb);

      trailCtx.strokeStyle = "white";
      trailCtx.lineWidth = 2;
      trailCtx.beginPath();
      trailCtx.moveTo(A.x, A.y);
      trailCtx.lineTo(B.x, B.y);
      trailCtx.stroke();
    }

    // joints + IDs
    for (let i = 0; i < smoothed.length; i++) {
      const lm = smoothed[i];
      if (!isValid(lm, i)) continue;

      const P = lmToCanvas(lm);

      let color = "white";
      if (opts.velocityColor.checked && prev) {
        const pp = lmToCanvas(prev[i]);
        const v = Math.hypot(P.x - pp.x, P.y - pp.y);
        color = velocityToColor(v);
      }

      trailCtx.fillStyle = color;
      trailCtx.beginPath();
      trailCtx.arc(P.x, P.y, 3, 0, Math.PI * 2);
      trailCtx.fill();

      if (opts.drawIds.checked) {
        ctx.fillStyle = color;
        ctx.font = `${opts.idSize.value}px system-ui, sans-serif`;
        ctx.fillText(String(i), P.x + 8, P.y - 8);
      }
    }
  }

  // Composite trails
  ctx.drawImage(trailCanvas, 0, 0);

  // Scanlines
  if (opts.scanlines.checked) drawScanlines();
}

fileInput.onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  currentFile = file;
  if (exportBtn) exportBtn.disabled = true;

  setStatus(`Loading: ${file.name} (${Math.round(file.size / 1024 / 1024)} MB)`);

  // Create a fresh video element each time
  const video = document.createElement("video");
  currentVideo = video;

  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;
  video.preload = "auto";

  video.src = URL.createObjectURL(file);
  video.load();

  const canPlay = video.canPlayType(file.type);
  if (!canPlay) {
    setStatus(
      `This file may use a codec your browser can't play (${file.type || "unknown type"}). ` +
      "Try re-encoding to H.264 (AVC) + AAC.",
      true
    );
  }

  const metaLoaded = await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve(true);
    video.onerror = () => resolve(false);
  });

  if (!metaLoaded || video.videoWidth === 0 || video.videoHeight === 0) {
    const msg = describeMediaError(video.error);
    setStatus(`Failed to read video metadata. ${msg}`, true);
    return;
  }

  const canPlayReady = await new Promise((resolve) => {
    video.oncanplay = () => resolve(true);
    video.onerror = () => resolve(false);
  });

  if (!canPlayReady) {
    const msg = describeMediaError(video.error);
    setStatus(`Video can't play in this browser. ${msg}`, true);
    return;
  }

  try {
    await video.play();
  } catch (err) {
    setStatus("Playback blocked by the browser. Click Play to start.", true);
  }

  const duration = Number.isFinite(video.duration) ? video.duration.toFixed(2) : "?";
  setStatus(`Ready (${video.videoWidth}x${video.videoHeight}, ${duration}s)`, false);

  // Enable playback controls
  playBtn.disabled = false;
  pauseBtn.disabled = false;
  restartBtn.disabled = false;
  if (exportBtn) exportBtn.disabled = false;

  // Setup button handlers
  playBtn.onclick = () => {
    if (video.paused) video.play();
  };

  pauseBtn.onclick = () => {
    video.pause();
  };

  restartBtn.onclick = () => {
    video.currentTime = 0;
    video.play();
  };

  // Output size fixed to reels
  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = 1080;
  trailCanvas.height = 1920;

  // Reset buffers
  trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
  prevPoses = [];
  lastFit = null;

  // Main loop
  video.requestVideoFrameCallback(function tick(now) {
    processFrame(video, now);

    // Next frame
    video.requestVideoFrameCallback(tick);
  });

  // Add listeners for controls to redraw when paused
  const visualControls = [
    opts.trail, opts.trailFade, opts.trailDrawAlpha, opts.smoothing,
    opts.drawIds, opts.idSize, opts.velocityColor, opts.scanlines, opts.scanStrength
  ];

  visualControls.forEach(control => {
    control.addEventListener('input', () => {
      if (currentVideo && currentVideo.paused) {
        redrawCurrentFrame();
      }
    });
    control.addEventListener('change', () => {
      if (currentVideo && currentVideo.paused) {
        redrawCurrentFrame();
      }
    });
  });
};