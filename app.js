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
  await currentVideo.play();

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

  recorder.onstop = () => {
    const finalType = recorder.mimeType || "video/webm";
    const blob = new Blob(recordingChunks, { type: finalType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${baseName}-overlay.webm`;
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
};

let vision = null;
let pose = null;

const trailCanvas = document.createElement("canvas");
const trailCtx = trailCanvas.getContext("2d");

let prevPoses = [];
let prevPrevPoses = [];
let missingCounts = [];

const LOWER_BODY = new Set([23,24,25,26,27,28,29,30,31,32]);
const FEET_ONLY  = new Set([27,28,29,30,31,32]);

const OCCLUSION_GRACE_FRAMES = 10;
const PREDICT_WHEN_OCCLUDED = true;

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
    enablePoseWorldLandmarks: true,
  });

  prevPoses = [];
  prevPrevPoses = [];
  missingCounts = [];
}

await createPose();

function computeFit(video, outW, outH, biasY = 0.12) {
  const targetAspect = outW / outH;
  const srcAspect = video.videoWidth / video.videoHeight;

  let sx, sy, sw, sh;

  if (srcAspect > targetAspect) {
    sw = video.videoHeight * targetAspect;
    sh = video.videoHeight;
    sx = (video.videoWidth - sw) / 2;
    sy = 0;
  } else {
    sw = video.videoWidth;
    sh = video.videoWidth / targetAspect;
    sx = 0;
    const maxSy = Math.max(0, video.videoHeight - sh);
    sy = maxSy * biasY;
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

let lastFit = null;

function lmToCanvas(lm) {
  if (!lastFit) return { x: lm.x * canvas.width, y: lm.y * canvas.height };

  const { sx, sy, sw, sh, outW, outH } = lastFit;

  const px = lm.x * currentVideo.videoWidth;
  const py = lm.y * currentVideo.videoHeight;

  const nx = (px - sx) / sw;
  const ny = (py - sy) / sh;

  return { x: nx * outW, y: ny * outH };
}

function clampCanvasPoint(P) {
  return {
    x: Math.max(-40, Math.min(canvas.width + 40, P.x)),
    y: Math.max(-40, Math.min(canvas.height + 40, P.y)),
  };
}

function isValid(lm, index) {
  if (lm.visibility === undefined) return false;

  if (FEET_ONLY.has(index)) return lm.visibility > 0.08;
  if (LOWER_BODY.has(index)) return lm.visibility > 0.25;

  return lm.visibility > 0.4;
}

function smoothLandmarks(current, prev, prevPrev, poseIndex) {
  if (!prev) return current;

  const base = parseFloat(opts.smoothing.value);

  const upperAlpha = Math.min(0.96, base);
  const lowerAlpha = Math.min(0.80, base * 0.55);

  if (!missingCounts[poseIndex]) {
    missingCounts[poseIndex] = new Array(current.length).fill(0);
  }

  return current.map((lm, i) => {
    const alpha = LOWER_BODY.has(i) ? lowerAlpha : upperAlpha;

    const ok = isValid(lm, i);
    if (ok) {
      missingCounts[poseIndex][i] = 0;
      return {
        x: alpha * prev[i].x + (1 - alpha) * lm.x,
        y: alpha * prev[i].y + (1 - alpha) * lm.y,
        visibility: lm.visibility,
      };
    }

    const miss = ++missingCounts[poseIndex][i];

    if (miss > OCCLUSION_GRACE_FRAMES) {
      return {
        x: prev[i].x,
        y: prev[i].y,
        visibility: (prev[i].visibility ?? 1) * 0.85,
      };
    }

    if (PREDICT_WHEN_OCCLUDED && prevPrev && FEET_ONLY.has(i)) {
      const vx = prev[i].x - prevPrev[i].x;
      const vy = prev[i].y - prevPrev[i].y;

      return {
        x: prev[i].x + vx * 0.7,
        y: prev[i].y + vy * 0.7,
        visibility: (prev[i].visibility ?? 1) * 0.95,
      };
    }

    return prev[i];
  });
}

let currentVideo = null;
let currentFile = null;
let recorder = null;
let recordingChunks = [];
let exportInProgress = false;

function processFrame(video, timestamp) {
  drawFitted(video);

  const results = pose.detectForVideo(video, timestamp);
  const poses = results.landmarks || [];

  while (prevPoses.length < poses.length) prevPoses.push(null);
  while (prevPrevPoses.length < poses.length) prevPrevPoses.push(null);
  while (missingCounts.length < poses.length) missingCounts.push(null);

  for (let p = 0; p < poses.length; p++) {
    const cur = poses[p];
    const prev = prevPoses[p];
    const prevPrev = prevPrevPoses[p];

    const smoothed = smoothLandmarks(cur, prev, prevPrev, p);

    prevPrevPoses[p] = prevPoses[p];
    prevPoses[p] = smoothed;

    for (const [a, b] of POSE_CONNECTIONS) {
      const la = smoothed[a];
      const lb = smoothed[b];
      if (!isValid(la, a) || !isValid(lb, b)) continue;

      const A = clampCanvasPoint(lmToCanvas(la));
      const B = clampCanvasPoint(lmToCanvas(lb));

      ctx.strokeStyle = "white";
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    }
  }
}

fileInput.onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  currentFile = file;
  setStatus(`Loading ${file.name}`);

  const video = document.createElement("video");
  currentVideo = video;

  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;

  video.src = URL.createObjectURL(file);
  await video.play();

  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = 1080;
  trailCanvas.height = 1920;

  prevPoses = [];
  prevPrevPoses = [];
  missingCounts = [];

  video.requestVideoFrameCallback(function tick(now) {
    processFrame(video, now);
    video.requestVideoFrameCallback(tick);
  });

  playBtn.disabled = false;
  pauseBtn.disabled = false;
  restartBtn.disabled = false;
  if (exportBtn) exportBtn.disabled = false;
};