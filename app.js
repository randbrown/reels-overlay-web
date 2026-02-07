import { PoseLandmarker, FilesetResolver } from
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Core body connections
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],
  [24,26],[26,28],[28,30],[30,32]
];

// Classic face V-shape like old Python drawing utils
const FACE_CONNECTIONS = [
  [0, 1], // nose → left eye
  [0, 2]  // nose → right eye
];

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");

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
  codeOverlay: document.getElementById("codeOverlay")
};

let vision = null;
let pose = null;

// Trail buffer
const trailCanvas = document.createElement("canvas");
const trailCtx = trailCanvas.getContext("2d");

// Keep previous landmarks for smoothing + velocity
let prevPoses = [];

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
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    },
    runningMode: "VIDEO",

    numPoses: parseInt(opts.numPoses.value, 10),

    minPoseDetectionConfidence: parseFloat(opts.detConf.value),
    minTrackingConfidence: parseFloat(opts.trkConf.value),

    // Important for better legs/feet
    enablePoseWorldLandmarks: true
  });

  prevPoses = [];
}

// Rebuild pose model when relevant sliders change
let rebuildTimer = null;
function requestRebuild() {
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => createPose(), 250);
}

opts.detConf.addEventListener("input", requestRebuild);
opts.trkConf.addEventListener("input", requestRebuild);
opts.numPoses.addEventListener("input", requestRebuild);

await createPose();

// Fit video to 9:16 canvas
function drawFitted(video) {
  const w = canvas.width;
  const h = canvas.height;

  const targetAspect = w / h;
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
    sy = (video.videoHeight - sh) / 2;
  }

  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h);
}

fileInput.onchange = e => startProcessing(e.target.files[0]);

// Smart visibility rules – lenient for feet
function isValid(lm, index) {
  if (lm.visibility === undefined) return false;

  const FOOT_POINTS = [25,26,27,28,29,30,31,32];
  const threshold = FOOT_POINTS.includes(index) ? 0.2 : 0.5;

  return lm.visibility > threshold;
}

// Smoothing with extra stability for legs/feet
function smoothLandmarks(current, prev) {
  if (!prev) return current;

  const base = parseFloat(opts.smoothing.value);

  return current.map((lm, i) => {
    const EXTRA = [25,26,27,28,29,30,31,32];
    const alpha = EXTRA.includes(i)
      ? Math.min(0.92, base + 0.1)
      : base;

    return {
      x: alpha * prev[i].x + (1 - alpha) * lm.x,
      y: alpha * prev[i].y + (1 - alpha) * lm.y,
      visibility: lm.visibility
    };
  });
}

// Velocity → color ramp
function velocityToColor(v) {
  let t = Math.max(0, Math.min(1, v / 200));

  if (t < 0.5) {
    const a = t / 0.5;
    return `rgb(${255*(1-a)}, ${255*a}, 0)`;
  } else {
    const a = (t - 0.5) / 0.5;
    return `rgb(0, ${255*(1-a)}, ${255*a})`;
  }
}

async function startProcessing(file) {
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;

  video.src = URL.createObjectURL(file);

  await new Promise(resolve => video.onloadeddata = resolve);
  await video.play();

  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = 1080;
  trailCanvas.height = 1920;

  trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
  prevPoses = [];

  video.requestVideoFrameCallback(function process(now) {
    drawFitted(video);

    const results = pose.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      drawOverlays(results.landmarks);
    }

    video.requestVideoFrameCallback(process);
  });
}

function drawOverlays(posesNow) {
  const w = canvas.width;
  const h = canvas.height;

  // ---- Proper trail fade (key fix) ----
  if (opts.trail.checked) {
    const fade = parseFloat(opts.trailFade.value);

    trailCtx.save();
    trailCtx.globalCompositeOperation = "destination-in";
    trailCtx.fillStyle = `rgba(0,0,0,${fade})`;
    trailCtx.fillRect(0, 0, w, h);
    trailCtx.restore();

    trailCtx.globalAlpha = parseFloat(opts.trailDrawAlpha.value);
  } else {
    trailCtx.clearRect(0, 0, w, h);
    trailCtx.globalAlpha = 1.0;
  }

  const n = posesNow.length;

  while (prevPoses.length < n) prevPoses.push(null);
  if (prevPoses.length > n) prevPoses = prevPoses.slice(0, n);

  for (let p = 0; p < n; p++) {
    const cur = posesNow[p];
    const prev = prevPoses[p];

    const smoothed = smoothLandmarks(cur, prev);
    prevPoses[p] = smoothed;

    // Body skeleton
    for (const [a, b] of POSE_CONNECTIONS) {
      const pa = smoothed[a];
      const pb = smoothed[b];

      if (!isValid(pa, a) || !isValid(pb, b)) continue;

      const x1 = pa.x * w, y1 = pa.y * h;
      const x2 = pb.x * w, y2 = pb.y * h;

      let color = "white";

      if (opts.velocityColor.checked && prev) {
        let v = Math.hypot(
          x1 - prev[a].x * w,
          y1 - prev[a].y * h
        );
        v = Math.min(200, v * 0.7);
        color = velocityToColor(v);
      }

      trailCtx.strokeStyle = color;
      trailCtx.lineWidth = 2;

      trailCtx.beginPath();
      trailCtx.moveTo(x1, y1);
      trailCtx.lineTo(x2, y2);
      trailCtx.stroke();
    }

    // Face V-shape
    for (const [a, b] of FACE_CONNECTIONS) {
      const pa = smoothed[a];
      const pb = smoothed[b];

      if (!isValid(pa, a) || !isValid(pb, b)) continue;

      const x1 = pa.x * w, y1 = pa.y * h;
      const x2 = pb.x * w, y2 = pb.y * h;

      trailCtx.strokeStyle = "white";
      trailCtx.lineWidth = 2;

      trailCtx.beginPath();
      trailCtx.moveTo(x1, y1);
      trailCtx.lineTo(x2, y2);
      trailCtx.stroke();
    }

    // Joints + IDs
    for (let i = 0; i < smoothed.length; i++) {
      const lm = smoothed[i];
      if (!isValid(lm, i)) continue;

      const x = lm.x * w;
      const y = lm.y * h;

      let color = "white";

      if (opts.velocityColor.checked && prev) {
        let v = Math.hypot(
          x - prev[i].x * w,
          y - prev[i].y * h
        );
        v = Math.min(200, v * 0.7);
        color = velocityToColor(v);
      }

      trailCtx.fillStyle = color;
      trailCtx.beginPath();
      trailCtx.arc(x, y, 3, 0, Math.PI * 2);
      trailCtx.fill();

      if (opts.drawIds.checked) {
        ctx.fillStyle = color;
        ctx.font = `${opts.idSize.value}px system-ui, sans-serif`;
        ctx.fillText(String(i), x + 8, y - 8);
      }
    }
  }

  ctx.drawImage(trailCanvas, 0, 0);

  if (opts.scanlines.checked) drawScanlines();
}

function drawScanlines() {
  const strength = parseFloat(opts.scanStrength.value);

  ctx.fillStyle = `rgba(0,0,0,${strength})`;

  for (let y = 0; y < canvas.height; y += 4) {
    ctx.fillRect(0, y, canvas.width, 2);
  }
}