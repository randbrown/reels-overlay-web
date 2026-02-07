import { PoseLandmarker, FilesetResolver } from
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],
  [24,26],[26,28],[28,30],[30,32]
];

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");

const opts = {
  trail: document.getElementById("trail"),
  trailFade: document.getElementById("trailFade"),           // fades existing trail buffer
  trailDrawAlpha: document.getElementById("trailDrawAlpha"), // opacity for newly drawn lines/joints
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

// Single shared trail buffer (we draw all people into it)
const trailCanvas = document.createElement("canvas");
const trailCtx = trailCanvas.getContext("2d");

// For smoothing + velocity we keep previous landmarks per pose index
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
    minPosePresenceConfidence: 0.5
  });

  // Reset previous poses whenever we rebuild the model
  prevPoses = [];
}

// Rebuild when these sliders change (debounced a bit)
let rebuildTimer = null;
function requestRebuild() {
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => createPose(), 250);
}
opts.detConf.addEventListener("input", requestRebuild);
opts.trkConf.addEventListener("input", requestRebuild);
opts.numPoses.addEventListener("input", requestRebuild);

await createPose();

// --- aspect ratio fit (like fit_to_reels) ---
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

function smoothLandmarks(current, prev) {
  if (!prev) return current;
  const a = parseFloat(opts.smoothing.value);
  return current.map((lm, i) => ({
    x: a * prev[i].x + (1 - a) * lm.x,
    y: a * prev[i].y + (1 - a) * lm.y,
    visibility: lm.visibility
  }));
}

function isValid(lm) {
  return lm.visibility !== undefined && lm.visibility > 0.5;
}

function velocityToColor(v) {
  // Clamp and map to a "blue->green->red" ramp like your Python version
  let t = Math.max(0, Math.min(1, v / 200));
  if (t < 0.5) {
    const a = t / 0.5;
    return `rgb(${255*(1-a)}, ${255*a}, 0)`;
  } else {
    const a = (t - 0.5) / 0.5;
    return `rgb(0, ${255*(1-a)}, ${255*a})`;
  }
}

fileInput.onchange = e => startProcessing(e.target.files[0]);

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

  // Clear buffers when starting a new video
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

  // ---- TRAIL BUFFER FADE (this is the key fix) ----
  if (opts.trail.checked) {
    const fade = parseFloat(opts.trailFade.value); // 0.75..0.995

    // Multiply existing trail alpha by "fade" each frame:
    trailCtx.save();
    trailCtx.globalCompositeOperation = "destination-in";
    trailCtx.fillStyle = `rgba(0,0,0,${fade})`;
    trailCtx.fillRect(0, 0, w, h);
    trailCtx.restore();

    // New strokes opacity:
    trailCtx.globalAlpha = parseFloat(opts.trailDrawAlpha.value);
  } else {
    trailCtx.clearRect(0, 0, w, h);
    trailCtx.globalAlpha = 1.0;
  }

  // Ensure prevPoses array matches num poses
  const n = posesNow.length;
  while (prevPoses.length < n) prevPoses.push(null);
  if (prevPoses.length > n) prevPoses = prevPoses.slice(0, n);

  // Draw each pose into the trail buffer
  for (let p = 0; p < n; p++) {
    const cur = posesNow[p];
    const prev = prevPoses[p];

    const smoothed = smoothLandmarks(cur, prev);
    prevPoses[p] = smoothed;

    // connections
    for (const [a, b] of POSE_CONNECTIONS) {
      const pa = smoothed[a], pb = smoothed[b];
      if (!isValid(pa) || !isValid(pb)) continue;

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

    // joints + IDs
    for (let i = 0; i < smoothed.length; i++) {
      const lm = smoothed[i];
      if (!isValid(lm)) continue;

      const x = lm.x * w, y = lm.y * h;

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

  // Composite trail buffer onto main frame
  ctx.drawImage(trailCanvas, 0, 0);

  if (opts.scanlines.checked) drawScanlines();
}

function drawScanlines() {
  const strength = parseFloat(opts.scanStrength.value);
  ctx.fillStyle = `rgba(0,0,0,${strength})`;

  // thicker CRT-ish scanlines
  for (let y = 0; y < canvas.height; y += 4) {
    ctx.fillRect(0, y, canvas.width, 2);
  }
}