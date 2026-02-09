import { PoseLandmarker, FilesetResolver } from
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

setStatus("NEW SCRIPT LOADED v4");

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");

const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const restartBtn = document.getElementById("restartBtn");

const opts = {
  smoothing: document.getElementById("smoothing"),
  detConf: document.getElementById("detConf"),
  trkConf: document.getElementById("trkConf"),
  drawIds: document.getElementById("drawIds"),
};

function setStatus(msg, err=false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", err);
}

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],
  [24,26],[26,28],[28,30],[30,32]
];

let vision = null;
let pose = null;

let prevPoses = [];
let prevPrevPoses = [];
let missingCounts = [];

const LOWER_BODY = new Set([23,24,25,26,27,28,29,30,31,32]);
const FEET_ONLY  = new Set([27,28,29,30,31,32]);

const OCCLUSION_GRACE_FRAMES = 12;
const PREDICT_WHEN_OCCLUDED = true;

async function init() {
  vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  await createPose();
}

async function createPose() {
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    },
    runningMode: "VIDEO",

    minPoseDetectionConfidence: parseFloat(opts.detConf.value),
    minTrackingConfidence: parseFloat(opts.trkConf.value),
    minPosePresenceConfidence: 0.25,

    enablePoseWorldLandmarks: false,
  });

  prevPoses = [];
  prevPrevPoses = [];
  missingCounts = [];

  setStatus("Model ready");
}

await init();

function computeFit(video, outW, outH, biasY = 0.1) {
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

  return { sx, sy, sw, sh };
}

let lastFit = null;

function drawFitted(video) {
  const fit = computeFit(video, canvas.width, canvas.height);
  lastFit = fit;

  ctx.drawImage(
    video,
    fit.sx, fit.sy, fit.sw, fit.sh,
    0, 0, canvas.width, canvas.height
  );
}

function lmToCanvas(lm, video) {
  const { sx, sy, sw, sh } = lastFit;

  const px = lm.x * video.videoWidth;
  const py = lm.y * video.videoHeight;

  const nx = (px - sx) / sw;
  const ny = (py - sy) / sh;

  return {
    x: nx * canvas.width,
    y: ny * canvas.height
  };
}

function isValid(lm, i) {
  if (!lm) return false;

  if (FEET_ONLY.has(i)) return lm.visibility > 0.05;
  if (LOWER_BODY.has(i)) return lm.visibility > 0.20;

  return lm.visibility > 0.35;
}

function smoothLandmarks(current, prev, prevPrev, poseIndex) {
  if (!prev) return current;

  const base = parseFloat(opts.smoothing.value);

  const upperAlpha = Math.min(0.96, base);
  const lowerAlpha = Math.min(0.75, base * 0.5);

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
      return prev[i];
    }

    if (PREDICT_WHEN_OCCLUDED && prevPrev && FEET_ONLY.has(i)) {
      const vx = prev[i].x - prevPrev[i].x;
      const vy = prev[i].y - prevPrev[i].y;

      return {
        x: prev[i].x + vx * 0.8,
        y: prev[i].y + vy * 0.8,
        visibility: prev[i].visibility,
      };
    }

    return prev[i];
  });
}

let currentVideo = null;

fileInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus("Loading video...");

  const video = document.createElement("video");
  currentVideo = video;

  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.playsInline = true;

  await video.play();

  setStatus("Playing");

  prevPoses = [];
  prevPrevPoses = [];
  missingCounts = [];

  video.requestVideoFrameCallback(function tick(now) {

    drawFitted(video);

    const results = pose.detectForVideo(video, now);
    const poses = results.landmarks || [];

    // DEBUG: raw ankle detections
    for (const pose of poses) {
      [27,28,29,30,31,32].forEach(i => {
        const lm = pose[i];
        if (lm && lm.visibility > 0.05) {
          const P = lmToCanvas(lm, video);
          ctx.fillStyle = "red";
          ctx.fillRect(P.x-4, P.y-4, 8, 8);
        }
      });
    }

    while (prevPoses.length < poses.length) prevPoses.push(null);
    while (prevPrevPoses.length < poses.length) prevPrevPoses.push(null);

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

        const A = lmToCanvas(la, video);
        const B = lmToCanvas(lb, video);

        ctx.strokeStyle = "white";
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        ctx.stroke();
      }
    }

    video.requestVideoFrameCallback(tick);
  });

  playBtn.onclick = () => video.play();
  pauseBtn.onclick = () => video.pause();
  restartBtn.onclick = () => { video.currentTime = 0; video.play(); };
};