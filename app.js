import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let pose = null;
let vision = null;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const trailCanvas = document.createElement("canvas");
const trailCtx = trailCanvas.getContext("2d");

let prevPoses = [];

const opts = {
  trails: document.getElementById("trails"),
  trailAlpha: document.getElementById("trailAlpha"),
  drawIds: document.getElementById("drawIds"),
  velocityColor: document.getElementById("velocityColor"),
  scanlines: document.getElementById("scanlines"),
  scanStrength: document.getElementById("scanStrength"),
  codeOverlay: document.getElementById("codeOverlay"),
  detConf: document.getElementById("detConf"),
  trkConf: document.getElementById("trkConf"),
  numPoses: document.getElementById("numPoses"),
  smoothing: document.getElementById("smoothing")
};

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
  [27,29],[29,31],[28,30],[30,32]
];

const FACE_CONNECTIONS = [
  [0,1],
  [0,2]
];

async function initVision() {
  vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
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

    enablePoseWorldLandmarks: true
  });

  prevPoses = [];
}

function velocityToColor(v) {
  const max = 800;
  let t = Math.min(1, Math.max(0, v / max));

  if (t < 0.5) {
    const a = t / 0.5;
    return `rgb(${255 * (1 - a)},${255 * a},0)`;
  } else {
    const a = (t - 0.5) / 0.5;
    return `rgb(0,${255 * (1 - a)},${255 * a})`;
  }
}

function drawFitted(src, dstCtx) {
  const w = dstCtx.canvas.width;
  const h = dstCtx.canvas.height;

  const targetAspect = w / h;
  const srcAspect = src.videoWidth / src.videoHeight;

  let sx, sy, sw, sh;

  if (srcAspect > targetAspect) {
    sw = src.videoHeight * targetAspect;
    sh = src.videoHeight;
    sx = (src.videoWidth - sw) / 2;
    sy = 0;
  } else {
    sw = src.videoWidth;
    sh = src.videoWidth / targetAspect;
    sx = 0;
    sy = (src.videoHeight - sh) / 2;
  }

  dstCtx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
}

function transformPoint(lm, video) {
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

  const px = lm.x * video.videoWidth;
  const py = lm.y * video.videoHeight;

  const nx = (px - sx) / sw;
  const ny = (py - sy) / sh;

  return {
    x: nx * w,
    y: ny * h
  };
}

function isValid(lm, index) {
  if (lm.visibility === undefined) return false;

  const FOOT_POINTS = [25,26,27,28,29,30,31,32];

  if (FOOT_POINTS.includes(index)) {
    return lm.visibility > 0.05;
  }

  return lm.visibility > 0.4;
}

function smoothLandmarks(current, prev) {
  if (!prev) return current;

  const base = parseFloat(opts.smoothing.value);

  return current.map((lm, i) => {

    if (!isValid(lm, i)) {
      return {
        x: prev[i].x,
        y: prev[i].y,
        visibility: prev[i].visibility * 0.95
      };
    }

    const EXTRA = [25,26,27,28,29,30,31,32];
    const alpha = EXTRA.includes(i)
      ? Math.min(0.96, base + 0.15)
      : base;

    return {
      x: alpha * prev[i].x + (1 - alpha) * lm.x,
      y: alpha * prev[i].y + (1 - alpha) * lm.y,
      visibility: lm.visibility
    };
  });
}

function applyScanlines() {
  const strength = parseFloat(opts.scanStrength.value);

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = img.data;

  for (let y = 0; y < canvas.height; y += 2) {
    for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      data[i] *= (1 - strength);
      data[i+1] *= (1 - strength);
      data[i+2] *= (1 - strength);
    }
  }

  ctx.putImageData(img, 0, 0);
}

function drawLandmarkId(x, y, id) {
  ctx.font = "18px monospace";
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillText(id.toString(), x + 5, y - 5);
}

function drawOverlays(poses) {
  const w = canvas.width;
  const h = canvas.height;

  if (opts.trails.checked) {
    trailCtx.globalAlpha = parseFloat(opts.trailAlpha.value);
    trailCtx.drawImage(trailCanvas, 0, 0);
  } else {
    trailCtx.clearRect(0, 0, w, h);
  }

  poses.forEach((poseData, pi) => {
    const raw = poseData.landmarks[0];

    if (!raw) return;

    const smoothed = smoothLandmarks(
      raw,
      prevPoses[pi]
    );

    prevPoses[pi] = smoothed;

    const drawCtx = opts.trails.checked ? trailCtx : ctx;

    POSE_CONNECTIONS.forEach(([a, b]) => {
      const pa = smoothed[a];
      const pb = smoothed[b];

      if (!isValid(pa, a) || !isValid(pb, b)) return;

      const { x: x1, y: y1 } = transformPoint(pa, video);
      const { x: x2, y: y2 } = transformPoint(pb, video);

      const vel = Math.hypot(
        pa.x - pb.x,
        pa.y - pb.y
      ) * 1000;

      drawCtx.strokeStyle = opts.velocityColor.checked
        ? velocityToColor(vel)
        : "red";

      drawCtx.lineWidth = 2;

      drawCtx.beginPath();
      drawCtx.moveTo(x1, y1);
      drawCtx.lineTo(x2, y2);
      drawCtx.stroke();
    });

    FACE_CONNECTIONS.forEach(([a, b]) => {
      const pa = smoothed[a];
      const pb = smoothed[b];

      if (!isValid(pa, a) || !isValid(pb, b)) return;

      const { x: x1, y: y1 } = transformPoint(pa, video);
      const { x: x2, y: y2 } = transformPoint(pb, video);

      trailCtx.strokeStyle = "white";
      trailCtx.lineWidth = 2;

      trailCtx.beginPath();
      trailCtx.moveTo(x1, y1);
      trailCtx.lineTo(x2, y2);
      trailCtx.stroke();
    });

    if (opts.drawIds.checked) {
      smoothed.forEach((lm, i) => {
        if (!isValid(lm, i)) return;

        const { x, y } = transformPoint(lm, video);
        drawLandmarkId(x, y, i);
      });
    }
  });

  if (opts.trails.checked) {
    ctx.drawImage(trailCanvas, 0, 0);
  }
}

async function process() {
  if (!pose) await createPose();

  canvas.width = 1080;
  canvas.height = 1920;

  trailCanvas.width = canvas.width;
  trailCanvas.height = canvas.height;

  function step() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFitted(video, ctx);

    const now = performance.now();

    const result = pose.detectForVideo(video, now);

    if (result.landmarks) {
      drawOverlays(result.landmarks.map(l => ({ landmarks: [l] })));
    }

    if (opts.scanlines.checked) {
      applyScanlines();
    }

    requestAnimationFrame(step);
  }

  step();
}

document.getElementById("fileInput").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  video.src = url;
  video.play();

  video.onloadeddata = () => process();
});

[opts.detConf, opts.trkConf, opts.numPoses].forEach(el => {
  el.addEventListener("input", () => createPose());
});