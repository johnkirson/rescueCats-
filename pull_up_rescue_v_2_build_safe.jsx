import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";

/**
 * Pull‑Up Rescue — camera mini‑game (iPhone‑ready, build‑safe)
 * Defaults per user confirmation:
 *  - Keypoint: nose crosses threshold
 *  - minAboveMs = 200 ms, minIntervalMs = 500 ms
 *
 * Notes:
 *  - No shadcn/ui or alias imports. Plain DOM+Tailwind classes.
 *  - Defensive guards for MediaRecorder + captureStream; Record disabled if unsupported.
 *  - Composite recording via hidden canvas so overlay is included.
 *  - Tiny console test suite validates the rep counter.
 */

// ---------------- Helper drawing functions ----------------
function drawKeypoints(ctx, keypoints, scale = 1) {
  keypoints.forEach((kp) => {
    if (kp.score && kp.score > 0.3) {
      const x = kp.x * scale;
      const y = kp.y * scale;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fill();
    }
  });
}

function drawSkeleton(ctx, keypoints, edges, scale = 1) {
  ctx.strokeStyle = "rgba(255,255,255,0.4)";
  ctx.lineWidth = 2;
  Object.values(edges).forEach(([i, j]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    if (kp1 && kp2 && kp1.score > 0.3 && kp2.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp1.x * scale, kp1.y * scale);
      ctx.lineTo(kp2.x * scale, kp2.y * scale);
      ctx.stroke();
    }
  });
}

// MoveNet skeleton map
const MOVENET_EDGES = {
  0: [0, 1], 1: [1, 3], 2: [0, 2], 3: [2, 4],
  4: [5, 7], 5: [7, 9], 6: [6, 8], 7: [8, 10],
  8: [5, 6], 9: [5, 11], 10: [6, 12], 11: [11, 12],
  12: [11, 13], 13: [13, 15], 14: [12, 14], 15: [14, 16]
};

// -------------- Rep‑counter state machine --------------
function updateRepState(prev, isAbove, now, minAboveMs = 200, minIntervalMs = 500) {
  const next = { ...prev };
  let counted = 0;
  if (prev.phase === "down" && isAbove) {
    next.phase = "up";
    next.lastAbove = now;
  } else if (prev.phase === "up" && !isAbove) {
    if (now - prev.lastAbove > minAboveMs && now - prev.lastRepAt > minIntervalMs) {
      next.phase = "down";
      next.lastRepAt = now;
      counted = 1;
    } else {
      next.phase = "down";
    }
  }
  return { next, counted };
}

export default function PullUpRescueV2() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const recCanvasRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const detectorRef = useRef(null);
  const rafRef = useRef(null);

  const [detecting, setDetecting] = useState(false);
  const [recording, setRecording] = useState(false);
  const [cats, setCats] = useState(0);
  const [barY, setBarY] = useState(null);
  const [sensitivity, setSensitivity] = useState(24);
  const [msg, setMsg] = useState("Tap the bar location to calibrate");
  const [canRecord, setCanRecord] = useState(false);
  const [status, setStatus] = useState({ camera: false, tf: false, detector: false });

  const repStateRef = useRef({ phase: "down", lastAbove: 0, lastRepAt: 0 });
  const [rescues, setRescues] = useState([]);

  // Feature detect recording early
  useEffect(() => {
    const tmpCanvas = document.createElement("canvas");
    const hasCapture = typeof tmpCanvas.captureStream === "function";
    const hasRecorder = typeof window !== "undefined" && "MediaRecorder" in window;
    setCanRecord(hasCapture && hasRecorder);
  }, []);

  // Init camera + TF + detector
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          setMsg("Camera API not available.");
          return;
        }
        await tf.setBackend("webgl");
        await tf.ready();
        setStatus((s) => ({ ...s, tf: true }));

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) return;
        const video = videoRef.current;
        video.srcObject = stream;
        await video.play();
        setStatus((s) => ({ ...s, camera: true }));

        detectorRef.current = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: "Lightning" }
        );
        if (!cancelled) setStatus((s) => ({ ...s, detector: true }));
        if (!cancelled) setMsg("Tap the bar on screen, then Start");
      } catch (e) {
        console.error(e);
        if (!cancelled) setMsg("Init failed. Check permissions.");
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current || 0);
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((t) => t.stop());
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const tick = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const recCanvas = recCanvasRef.current;
    if (!video || !canvas || !overlay) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw; canvas.height = vh;
      overlay.width = vw; overlay.height = vh;
      if (recCanvas) { recCanvas.width = vw; recCanvas.height = vh; }
    }

    const ctx = canvas.getContext("2d");
    const octx = overlay.getContext("2d");

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -vw, 0, vw, vh);
    ctx.restore();

    if (detecting && detectorRef.current) {
      const poses = await detectorRef.current.estimatePoses(video, { maxPoses: 1, flipHorizontal: true });
      if (poses[0]) {
        const kps = poses[0].keypoints.map((kp) => ({ ...kp }));
        drawSkeleton(ctx, kps, MOVENET_EDGES, 1);
        drawKeypoints(ctx, kps, 1);

        const nose = kps[0];
        if (nose?.score > 0.4 && barY !== null) {
          const threshold = barY - sensitivity;
          const now = performance.now();
          const above = nose.y <= threshold;
          const { next, counted } = updateRepState(repStateRef.current, above, now);
          repStateRef.current = next;
          if (counted) { setCats((c) => c + 1); spawnRescue(); }

          ctx.strokeStyle = "#00ff88";
          ctx.lineWidth = 3;
          ctx.setLineDash([10, 8]);
          ctx.beginPath();
          ctx.moveTo(0, threshold);
          ctx.lineTo(vw, threshold);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    octx.clearRect(0, 0, overlay.width, overlay.height);
    drawHouse(octx, overlay, cats);
    rescues.forEach((r) => { octx.font = "48px system-ui"; octx.textAlign = "center"; octx.fillText("🐱", r.x, r.y); });

    if (recording && recCanvas) {
      const rctx = recCanvas.getContext("2d");
      rctx.clearRect(0, 0, recCanvas.width, recCanvas.height);
      rctx.drawImage(canvas, 0, 0);
      rctx.drawImage(overlay, 0, 0);
    }

    rafRef.current = requestAnimationFrame(tick);
  };

  useEffect(() => {
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current || 0);
  });

  const onCanvasClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const y = e.clientY - rect.top;
    setBarY(y);
    setMsg("Bar set. Press Start to detect.");
  };

  const startDetect = () => { setDetecting(true); setMsg("Detecting pull-ups…"); };
  const stopDetect = () => { setDetecting(false); setMsg("Paused"); };

  const startRecording = () => {
    try {
      const recCanvas = recCanvasRef.current;
      if (!recCanvas) throw new Error("Recorder canvas missing");
      if (!("captureStream" in recCanvas)) throw new Error("captureStream unsupported");
      if (!("MediaRecorder" in window)) throw new Error("MediaRecorder unsupported");

      const mime = getBestMime();
      if (!mime) throw new Error("No supported MIME for MediaRecorder");

      recordedChunksRef.current = [];
      const stream = recCanvas.captureStream(30);
      const mr = new MediaRecorder(stream, { mimeType: mime });
      mr.ondataavailable = (e) => { if (e.data.size > 0) recordedChunksRef.current.push(e.data); };
      mr.onstop = () => {
        const type = mr.mimeType || mime;
        const blob = new Blob(recordedChunksRef.current, { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        const ext = type.includes("webm") ? "webm" : type.includes("mp4") ? "mp4" : "webm";
        a.download = `pullup-rescue-${ts}.${ext}`;
        a.click();
        setMsg("Recording saved");
      };
      mediaRecorderRef.current = mr;
      mr.start();
      setRecording(true);
      setMsg("Recording…");
    } catch (e) {
      console.error(e);
      setMsg(e.message || "Recording not supported on this device/browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  function getBestMime() {
    const options = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4", // last‑resort on iOS, if available
    ];
    for (const opt of options) {
      try { if (window.MediaRecorder && MediaRecorder.isTypeSupported(opt)) return opt; } catch {}
    }
    return "";
  }

  function drawHouse(ctx, overlay, cats) {
    const w = overlay.width;
    const h = overlay.height;
    const houseW = Math.min(180, w * 0.22);
    const houseH = houseW * 0.75;
    const x = 20;
    const y = h - houseH - 20;
    ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(x, y, houseW, houseH);
    ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + houseW / 2, y - 40); ctx.lineTo(x + houseW + 10, y); ctx.closePath();
    ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillRect(x + houseW * 0.65, y + houseH * 0.4, houseW * 0.2, houseH * 0.45);
    ctx.font = "28px system-ui"; ctx.fillText("🐾", x + houseW * 0.75, y + houseH * 0.38);
    const label = `${cats} saved`;
    ctx.font = "20px system-ui";
    const tw = ctx.measureText(label).width + 24;
    ctx.fillStyle = "rgba(0,0,0,0.65)"; ctx.fillRect(x, y - 36, tw, 28);
    ctx.fillStyle = "white"; ctx.fillText(label, x + 12, y - 16);
  }

  function spawnRescue() {
    const overlay = overlayRef.current;
    const w = overlay.width; const h = overlay.height;
    const startX = 20 + Math.min(180, w * 0.22) * 0.75;
    const startY = h - Math.min(180, w * 0.22) * 0.75 - 20 + Math.min(180, w * 0.22) * 0.4;
    const id = Math.random().toString(36).slice(2);
    const created = { id, x: startX, y: startY };
    setRescues((arr) => [...arr, created]);
    const start = performance.now(); const dur = 900;
    const step = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const nx = startX + (w * 0.35) * p;
      const ny = startY - (h * 0.4) * p;
      setRescues((arr) => arr.map((r) => (r.id === id ? { ...r, x: nx, y: ny } : r)));
      if (p < 1) requestAnimationFrame(step);
      else setTimeout(() => setRescues((arr) => arr.filter((r) => r.id !== id)), 200);
    };
    requestAnimationFrame(step);
  }

  // -------- Tiny test suite (console) --------
  useEffect(() => {
    function simulate(seq) {
      let state = { phase: "down", lastAbove: 0, lastRepAt: 0 };
      let count = 0;
      for (const [t, above] of seq) {
        const out = updateRepState(state, above, t);
        state = out.next; count += out.counted;
      }
      return count;
    }
    const log = [];
    const seq1 = [[0,false],[100,false],[120,true],[380,true],[400,false],[900,false]]; // 1 rep
    const seq2 = [[0,false],[100,true],[250,false],[500,false]]; // 0
    const seq3 = [[0,false],[50,true],[300,false],[900,false],[950,true],[1300,false],[2000,false]]; // 2
    const seq4 = [[0,false],[50,true],[300,false],[650,true],[800,false]]; // 1 (debounce)
    console.assert(simulate(seq1) === 1, "Test1 failed"); log.push("Test1 ok");
    console.assert(simulate(seq2) === 0, "Test2 failed"); log.push("Test2 ok");
    console.assert(simulate(seq3) === 2, "Test3 failed"); log.push("Test3 ok");
    console.assert(simulate(seq4) === 1, "Test4 failed"); log.push("Test4 ok");
    console.log("RepCounter tests:", log.join(", "));
  }, []);

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="max-w-5xl mx-auto p-4 grid gap-4">
        <h1 className="text-2xl sm:text-3xl font-semibold">Pull‑Up Rescue 🐱 — v2</h1>
        <p className="text-sm opacity-80">Nose-based counting, 200 ms hold, 500 ms debounce.</p>

        <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-3 sm:p-4">
          <div className="relative w-full">
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} onClick={onCanvasClick} className="w-full rounded-2xl shadow-lg" />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 w-full h-full" />
            <canvas ref={recCanvasRef} className="hidden" />
            {barY !== null && (
              <div className="absolute left-0 right-0 border-t-2 border-emerald-400 pointer-events-none" style={{ top: barY - sensitivity }} />
            )}
          </div>

          <div className="mt-3 grid grid-cols-2 sm:grid-cols-5 gap-2 items-center">
            {!detecting ? (
              <button onClick={startDetect} className="rounded-2xl bg-white/10 hover:bg-white/20 px-4 py-2">Start</button>
            ) : (
              <button onClick={stopDetect} className="rounded-2xl bg-white/10 hover:bg-white/20 px-4 py-2">Pause</button>
            )}

            {!recording ? (
              <button onClick={startRecording} disabled={!canRecord} className={`rounded-2xl px-4 py-2 ${canRecord ? "bg-white/10 hover:bg-white/20" : "bg-white/5 cursor-not-allowed"}`} title={canRecord ? "Start recording" : "Recording not supported on this device/browser"}>Record</button>
            ) : (
              <button onClick={stopRecording} className="rounded-2xl bg-red-500/80 hover:bg-red-500 px-4 py-2">Stop & Save</button>
            )}

            <button onClick={() => { setCats(0); repStateRef.current = { phase: "down", lastAbove: 0, lastRepAt: 0 }; }} className="rounded-2xl bg-white/10 hover:bg-white/20 px-4 py-2">Reset</button>
            <button onClick={() => { setBarY(null); setMsg("Tap the bar location to calibrate"); }} className="rounded-2xl bg-white/0 hover:bg-white/10 px-4 py-2 border border-white/10">Re‑set bar</button>

            <div className="text-xs opacity-70">
              <div>Camera: {String(status.camera)}</div>
              <div>TF: {String(status.tf)}</div>
              <div>Detector: {String(status.detector)}</div>
            </div>
          </div>

          <div className="mt-3">
            <label className="text-xs uppercase tracking-wide opacity-70">Sensitivity (pixels above bar)</label>
            <div className="flex items-center gap-3">
              <input type="range" min={8} max={60} step={1} value={sensitivity} onChange={(e) => setSensitivity(parseInt(e.target.value, 10))} className="w-full" />
              <div className="w-12 text-right tabular-nums">{sensitivity} px</div>
            </div>
          </div>

          <p className="mt-2 text-sm opacity-80">{msg}</p>
          <ul className="mt-2 text-xs opacity-70 list-disc pl-5 space-y-1">
            <li>Tap the bar on the video to set threshold; nose must rise above, then drop below.</li>
            <li>Adjust sensitivity if false positives occur.</li>
            <li>Recording is saved locally if supported.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
