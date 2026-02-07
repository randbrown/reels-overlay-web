import { PoseLandmarker, FilesetResolver } from 
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");

const opts = {
  trail: document.getElementById("trail"),
  trailAlpha: document.getElementById("trailAlpha"),
  drawIds: document.getElementById("drawIds"),
  velocityColor: document.getElementById("velocityColor"),
  scanlines: document.getElementById("scanlines"),
  codeOverlay: document.getElementById("codeOverlay")
};

// ---- Hardcoded pose connections (same as mediapipe) ----
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],
  [24,26],[26,28],[28,30],[30,32]
];

let pose;

let trailCanvas = document.createElement("canvas");
let trailCtx = trailCanvas.getContext("2d");

async function initPose() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    },
    runningMode: "VIDEO"
  });
}

await initPose();

fileInput.onchange = e => {
  const file = e.target.files[0];
  startProcessing(file);
};

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

async function startProcessing(file) {

  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.playsInline = true;

  await video.play();

  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = 1080;
  trailCanvas.height = 1920;

  let prevLandmarks = null;

  video.requestVideoFrameCallback(function process(now) {

    drawFitted(video);

    const results = pose.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      drawOverlays(results.landmarks[0], prevLandmarks);
      prevLandmarks = results.landmarks[0];
    }

    video.requestVideoFrameCallback(process);
  });
}

function velocityToColor(v) {
  const t = Math.min(1, v / 200);

  if (t < 0.5) {
    const a = t / 0.5;
    return `rgb(${255*(1-a)}, ${255*a}, 0)`;
  } else {
    const a = (t - 0.5) / 0.5;
    return `rgb(0, ${255*(1-a)}, ${255*a})`;
  }
}

function drawOverlays(landmarks, prev) {

  const w = canvas.width;
  const h = canvas.height;

  if (opts.trail.checked) {
    trailCtx.globalAlpha = parseFloat(opts.trailAlpha.value);
    trailCtx.drawImage(trailCanvas, 0, 0);
  } else {
    trailCtx.clearRect(0, 0, w, h);
  }

  // draw skeleton lines
  for (const [a, b] of POSE_CONNECTIONS) {

    const pa = landmarks[a];
    const pb = landmarks[b];

    const x1 = pa.x * w;
    const y1 = pa.y * h;
    const x2 = pb.x * w;
    const y2 = pb.y * h;

    let color = "white";

    if (opts.velocityColor.checked && prev) {
      const v = Math.hypot(
        x1 - prev[a].x * w,
        y1 - prev[a].y * h
      );
      color = velocityToColor(v);
    }

    trailCtx.strokeStyle = color;
    trailCtx.lineWidth = 2;

    trailCtx.beginPath();
    trailCtx.moveTo(x1, y1);
    trailCtx.lineTo(x2, y2);
    trailCtx.stroke();
  }

  // draw joints
  for (let i = 0; i < landmarks.length; i++) {

    const lm = landmarks[i];

    const x = lm.x * w;
    const y = lm.y * h;

    let color = "white";

    if (opts.velocityColor.checked && prev) {
      const v = Math.hypot(
        x - prev[i].x * w,
        y - prev[i].y * h
      );
      color = velocityToColor(v);
    }

    trailCtx.fillStyle = color;
    trailCtx.beginPath();
    trailCtx.arc(x, y, 3, 0, Math.PI * 2);
    trailCtx.fill();

    if (opts.drawIds.checked) {
      ctx.fillStyle = color;
      ctx.font = "12px sans-serif";
      ctx.fillText(i, x + 4, y - 4);
    }
  }

  ctx.drawImage(trailCanvas, 0, 0);

  if (opts.scanlines.checked) {
    drawScanlines();
  }
}

function drawScanlines() {
  ctx.fillStyle = "rgba(0,0,0,0.06)";
  for (let y = 0; y < canvas.height; y += 2) {
    ctx.fillRect(0, y, canvas.width, 1);
  }
}