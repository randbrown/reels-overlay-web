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

fileInput.onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  // Create a fresh video element each time
  const video = document.createElement("video");
  currentVideo = video;

  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;

  video.src = URL.createObjectURL(file);

  await new Promise((resolve) => (video.onloadeddata = resolve));
  await video.play();

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
    // 1) Draw video frame (also computes fit transform for landmark mapping)
    drawFitted(video);

    // 2) Pose detection
    const results = pose.detectForVideo(video, now);
    const poses = results.landmarks || [];

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

    // Next frame
    video.requestVideoFrameCallback(tick);
  });
};