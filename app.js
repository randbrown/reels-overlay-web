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

async function startProcessing(file) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  await video.play();

  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = 1080;
  trailCanvas.height = 1920;

  let prevLandmarks = null;

  video.requestVideoFrameCallback(function process(now) {

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const results = pose.detectForVideo(video, now);

    if (results.landmarks?.length) {
      drawOverlays(results.landmarks[0], prevLandmarks);
      prevLandmarks = results.landmarks[0];
    }

    video.requestVideoFrameCallback(process);
  });
}

function drawOverlays(landmarks, prev) {

  const w = canvas.width;
  const h = canvas.height;

  if (opts.trail.checked) {
    trailCtx.globalAlpha = opts.trailAlpha.value;
    trailCtx.drawImage(trailCanvas, 0, 0);
  } else {
    trailCtx.clearRect(0, 0, w, h);
  }

  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];

    const x = lm.x * w;
    const y = lm.y * h;

    let color = "white";

    if (opts.velocityColor.checked && prev) {
      const px = prev[i].x * w;
      const py = prev[i].y * h;

      const v = Math.hypot(x - px, y - py);

      color = velocityToColor(v);
    }

    trailCtx.fillStyle = color;
    trailCtx.beginPath();
    trailCtx.arc(x, y, 3, 0, Math.PI * 2);
    trailCtx.fill();

    if (opts.drawIds.checked) {
      ctx.fillStyle = color;
      ctx.fillText(i, x + 5, y - 5);
    }
  }

  ctx.drawImage(trailCanvas, 0, 0);

  if (opts.scanlines.checked) {
    drawScanlines();
  }
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

function drawScanlines() {
  ctx.fillStyle = "rgba(0,0,0,0.06)";
  for (let y = 0; y < canvas.height; y += 2) {
    ctx.fillRect(0, y, canvas.width, 1);
  }
}
