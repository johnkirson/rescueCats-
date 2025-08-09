import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";

/**
 * v4.9 — POSE ALIGNMENT FIX (iOS/front/back)
 *
 * Root cause:
 *  - We ran MoveNet on a downscaled offscreen canvas (inferCanvas),
 *    drawing the video with letterboxing (scale=s, offsets ix/iy).
 *  - The returned keypoints are in the inferCanvas coordinate space.
 *  - We previously multiplied keypoints by the display scale directly,
 *    ignoring ix/iy and s — hence the pose drifted.
 *
 * Fix:
 *  - Convert keypoints from inferCanvas → video pixel space:
 *        x_v = (x_infer - ix) / s
 *        y_v = (y_infer - iy) / s
 *  - Then convert video → display canvas:
 *        x_d = dx + (mirrored ? (vw - x_v) : x_v) * displayScale
 *        y_d = dy + y_v * displayScale
 *  - Store infer mapping (s, ix, iy) per frame and apply to all kps.
 *
 * Also kept calibration knobs at the top, with comments.
 */

// ===================== CALIBRATION =====================
const ROPE_BASELINE_FROM_BOTTOM = 0; // where the rope baseline is inside fire.png (0..1 from bottom)
const CAT_BASELINE_ABOVE_ROPE_PX = 0;  // cat feet above the rope baseline (device px)
const DEFAULT_SENSITIVITY = 24;         // threshold above rope (device px). Slider changes this.
const INFER_EVERY_MS = 70;              // pose estimation cadence (~14Hz)
// =======================================================

const SPRITES = {
  rope: "/assets/fire.png",
  cat_idle: "/assets/1.png",
  cat_attached: "/assets/2.png",
  cat_jump: "/assets/3.png",
  cat_seated: "/assets/4.png",
};

const MOVENET_EDGES = { 0:[0,1],1:[1,3],2:[0,2],3:[2,4],4:[5,7],5:[7,9],6:[6,8],7:[8,10],8:[5,6],9:[5,11],10:[6,12],11:[11,12],12:[11,13],13:[13,15],14:[12,14],15:[14,16] };
function updateRepState(prev, isAbove, now, minAboveMs = 160, minIntervalMs = 420){ const next={...prev}; let counted=0; if(prev.phase==='down'&&isAbove){next.phase='up'; next.lastAbove=now;} else if(prev.phase==='up'&&!isAbove){ if(now-prev.lastAbove>minAboveMs && now-prev.lastRepAt>minIntervalMs){ next.phase='down'; next.lastRepAt=now; counted=1;} else { next.phase='down'; } } return {next, counted}; }

export default function PullUpRescueV49(){
  const videoRef = useRef(null); const baseRef=useRef(null); const uiRef=useRef(null); const recRef=useRef(null);
  const detectorRef=useRef(null); const rafRef=useRef(null); const streamRef=useRef(null);
  const inferCanvasRef=useRef(document.createElement('canvas'));
  const inferMapRef=useRef({s:1, ix:0, iy:0}); // <— NEW: remembers inferCanvas transform

  // UI state
  const [camReady,setCamReady]=useState(false);
  const [bucketMap,setBucketMap]=useState({ultra:null,wide:null,front:null});
  const [bucketChoice,setBucketChoice]=useState('ultra');
  const [msg,setMsg]=useState("Drag the rope to the bar height"); const [debug,setDebug]=useState("");
  const [recording,setRecording]=useState(false); const recordingRef=useRef(false);
  const [canRecord,setCanRecord]=useState(false);
  const [barY,setBarY]=useState(null); const barYRef=useRef(null);
  const [sensitivity,setSensitivity]=useState(DEFAULT_SENSITIVITY); const sensitivityRef=useRef(DEFAULT_SENSITIVITY);
  const [showPose,setShowPose]=useState(true);

  const [saved,setSaved]=useState(0);
  const repRef=useRef({phase:'down',lastAbove:0,lastRepAt:0});

  const geomRef = useRef({ W:0,H:0,vw:0,vh:0,scale:1,dx:0,dy:0, mirrored:false });

  const ropeRef = useRef({ y:null });
  const catRef = useRef({ mode:'idle', x:0, y:0, vx:0, vy:0, lastT:0 });
  const seatedCatsRef = useRef([]);

  const lastInferRef = useRef(0); const lastPoseRef = useRef(null);

  const [imgs,setImgs]=useState({});
  useEffect(()=>{ const names=Object.entries(SPRITES); const loaded={}; let left=names.length; names.forEach(([k,src])=>{ const im=new Image(); im.onload=()=>{ loaded[k]=im; left--; if(left===0) setImgs(loaded); }; im.src=src; }); },[]);

  const isFrontLabel = (label='') => /front|user|face/i.test(label);
  const isUltraLabel = (label='') => /ultra\s*wide|0\.5x|ultra/i.test(label);
  const isTeleLabel  = (label='') => /tele|2x|3x|zoom/i.test(label);

  useEffect(()=>{ barYRef.current = barY; ropeRef.current.y = barY; },[barY]);
  useEffect(()=>{ sensitivityRef.current = sensitivity; },[sensitivity]);

  useEffect(()=>{ const c=document.createElement('canvas'); setCanRecord(typeof c.captureStream==='function' && 'MediaRecorder' in window); },[]);
  useEffect(()=>{ const resize=()=>{ const dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1)); const W=window.innerWidth,H=window.innerHeight; [baseRef.current,uiRef.current,recRef.current].forEach(cv=>{ if(!cv) return; cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=Math.floor(W*dpr); cv.height=Math.floor(H*dpr);}); if (uiRef.current && barYRef.current==null) { const mid=Math.floor(uiRef.current.height*0.5); setBarY(mid); } updateGeom(); if(!catRef.current.lastT) spawnCatCentered(); }; resize(); window.addEventListener('resize',resize); return()=>window.removeEventListener('resize',resize); },[]);

  function updateGeom(){ const video=videoRef.current, base=baseRef.current; if(!video||!base) return; const W=base.width,H=base.height; const vw=video.videoWidth||1280, vh=video.videoHeight||720; const scale=Math.max(W/vw,H/vh); const dw=vw*scale, dh=vh*scale; const dx=(W-dw)/2, dy=(H-dh)/2; geomRef.current={...geomRef.current, W,H,vw,vh,scale,dx,dy}; const inW=320; const inH=Math.round(inW*vh/vw)||240; const c=inferCanvasRef.current; c.width=inW; c.height=inH; }
  function setMirrorFromStream(stream){ try{ const track=stream.getVideoTracks?.()[0]; const s=track?.getSettings?.()||{}; const label=track?.label||''; const mirrored = s.facingMode ? /user|front/i.test(s.facingMode) : isFrontLabel(label); geomRef.current={...geomRef.current, mirrored}; }catch{} }

  async function createMoveNetDetector(){ await tf.setBackend('webgl'); await tf.ready(); const opts=[ {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING}, {modelType:poseDetection.movenet.modelType.SINGLEPOSE_THUNDER} ]; let lastErr; for(const o of opts){ try{ return await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,o);}catch(e){ lastErr=e; } } throw lastErr; }
  async function ensureDetector(){ if(!detectorRef.current){ detectorRef.current = await createMoveNetDetector(); } }

  function toBuckets(devices){ const vids=devices.filter(d=>d.kind==='videoinput'); const fronts=vids.filter(d=>isFrontLabel(d.label||'')); const backs = vids.filter(d=>!isFrontLabel(d.label||'')); const ultra = vids.find(d=>isUltraLabel(d.label||'')); const wide  = backs.find(d=>!isTeleLabel(d.label||'')) || backs[0] || null; return { ultra: ultra?.deviceId||null, wide: wide?.deviceId||null, front: fronts[0]?.deviceId||null }; }

  async function enableCamera(){ setDebug(""); try{ await ensureDetector(); const pre=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); const devices=await navigator.mediaDevices.enumerateDevices(); pre.getTracks().forEach(t=>t.stop()); const buckets=toBuckets(devices); setBucketMap(buckets); const preferred = buckets.ultra ? 'ultra' : (buckets.wide ? 'wide' : 'front'); await switchToBucket(preferred); setCamReady(true); cancelAnimationFrame(rafRef.current||0); rafRef.current=requestAnimationFrame(tick); }catch(e){ console.error(e); setMsg('Camera init failed.'); setDebug(`${e.name||'Error'}: ${e.message||e}`); } }

  async function switchToBucket(bucket){ try{ const deviceId=bucketMap[bucket]; const baseConstraints = { width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:30, max:60} };
      let stream; if(deviceId){ try{ stream=await navigator.mediaDevices.getUserMedia({ video:{ deviceId:{ exact: deviceId }, ...baseConstraints }, audio:false }); }catch(e){ setDebug(`Exact ${bucket} failed: ${e.name}`); } }
      if(!stream){ try{ stream=await navigator.mediaDevices.getUserMedia({ video: baseConstraints, audio:false }); }catch{ stream=await navigator.mediaDevices.getUserMedia({ video:true, audio:false }); } }
      if(streamRef.current) streamRef.current.getTracks().forEach(t=>t.stop()); const v=videoRef.current; v.srcObject=stream; await v.play(); streamRef.current=stream; setMirrorFromStream(stream); updateGeom(); const label = stream.getVideoTracks?.()[0]?.label || ''; const finalBucket = /front|user|face/i.test(label) ? 'front' : (/ultra\s*wide|0\.5x|ultra/i.test(label) ? 'ultra' : 'wide'); setBucketChoice(finalBucket); return finalBucket; }catch(e){ setDebug(`Switch failed: ${e.name}`); return bucket; } }

  const tick = async ()=>{ const video=videoRef.current, base=baseRef.current, ui=uiRef.current; if(!video||!base||!ui) return; const b=base.getContext('2d'); const u=ui.getContext('2d'); const {W,H,vw,vh,scale,dx,dy,mirrored}=geomRef.current; const inC=inferCanvasRef.current; if(!W) updateGeom();
    // BASE: camera
    b.clearRect(0,0,W,H); const dw=vw*scale, dh=vh*scale; if(mirrored){ b.save(); b.translate(W,0); b.scale(-1,1); b.drawImage(video, -dx - dw + W, dy, dw, dh); b.restore(); } else { b.drawImage(video, dx, dy, dw, dh); }

    // UI: clear
    u.clearRect(0,0,W,H);

    // Pose inference while recording
    const now=performance.now(); if(recordingRef.current && detectorRef.current){ if(now - lastInferRef.current >= INFER_EVERY_MS){ lastInferRef.current=now; const ic=inC.getContext('2d'); const s=Math.max(inC.width/vw, inC.height/vh); const iw=vw*s, ih=vh*s; const ix=(inC.width-iw)/2, iy=(inC.height-ih)/2; inferMapRef.current={s,ix,iy}; ic.clearRect(0,0,inC.width,inC.height); ic.drawImage(video, ix, iy, iw, ih); try{ const poses=await detectorRef.current.estimatePoses(inC,{maxPoses:1,flipHorizontal:false}); lastPoseRef.current = poses && poses[0] ? poses[0] : null; }catch(e){} }
      const pose = lastPoseRef.current; if(pose){ const kps=pose.keypoints; // map each kp from infer→video→display
        const {s,ix,iy}=inferMapRef.current; const mapped=kps.map((kp)=>{ const xv=(kp.x-ix)/s; const yv=(kp.y-iy)/s; const xd = dx + (mirrored ? (vw - xv) : xv) * scale; const yd = dy + yv * scale; return {X:xd,Y:yd,score:kp.score, raw:{xv,yv}}; });
        if(showPose){ u.save(); u.strokeStyle='rgba(255,255,255,.9)'; u.lineWidth=2; drawPoseMapped(u,mapped); u.restore(); }
        const nose = mapped[0]; const by=barYRef.current; const sens=sensitivityRef.current; if(nose?.score>0.4 && by!==null){ const thr=by - sens; const above = nose.Y <= thr; const {next}=updateRepState(repRef.current,above,now); const wasAttached = catRef.current.mode==='attached'; repRef.current=next; if(above && catRef.current.mode==='idle'){ catRef.current.mode='attached'; }
          if(!above && wasAttached && next.phase==='down' && catRef.current.mode==='attached'){ startCatFall(); }
          if(catRef.current.mode==='attached'){ const p=window.devicePixelRatio||1; catRef.current.x = nose.X; catRef.current.y = by - CAT_BASELINE_ABOVE_ROPE_PX*p; }
        }
      }
    }

    // overlays
    drawRopeSprite(u,W,H,barYRef.current,imgs.rope);
    drawThreshold(u,W,H,barYRef.current,sensitivityRef.current);
    drawSeatedCats(u,imgs);
    drawActiveCat(u,imgs);
    drawSavedCounter(u,W,H,saved);

    rafRef.current=requestAnimationFrame(tick);
  };

  function drawPoseMapped(ctx,kps){ const edges = MOVENET_EDGES; for(const [i,j] of Object.values(edges)){ const a=kps[i],b=kps[j]; if(a?.score>0.3&&b?.score>0.3){ ctx.beginPath(); ctx.moveTo(a.X,a.Y); ctx.lineTo(b.X,b.Y); ctx.stroke(); } } for(const k of kps){ if(k.score>0.3){ ctx.beginPath(); ctx.arc(k.X,k.Y,3,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.95)'; ctx.fill(); } } }

  function drawRopeSprite(ctx,W,H,y,img){ if(!img||y==null) return; const scaleW = W / img.width; const renderW = W; const renderH = img.height * scaleW; const baseline = renderH * (1 - ROPE_BASELINE_FROM_BOTTOM); const yTop = Math.round(y - baseline); ctx.drawImage(img, 0, yTop, renderW, renderH); }

  function spawnCatCentered(){ const u=uiRef.current; if(!u) return; const p=window.devicePixelRatio||1; const W=u.width; const y=barYRef.current ?? Math.floor(u.height*0.5); catRef.current={ mode:'idle', x:Math.floor(W/2), y:y - CAT_BASELINE_ABOVE_ROPE_PX*p, vx:0, vy:0, lastT:performance.now() }; }
  function startCatFall(){ const now=performance.now(); const c=catRef.current; c.mode='falling'; c.vx=(Math.random()*2-1)*24; c.vy=0; c.lastT=now; }
  function stepActiveCat(){ const u=uiRef.current; if(!u) return; const p=window.devicePixelRatio||1; const H=u.height; const groundY=H-28*p; const c=catRef.current; const now=performance.now(); const dt=Math.min(0.05,(now-c.lastT)/1000); c.lastT=now; if(c.mode==='falling'){ const g=1200*p; c.vy += g*dt; c.y += c.vy*dt; c.x += c.vx*dt; if(c.y >= groundY){ c.y=groundY; c.mode='seated'; const seat = placeSeatedCat(u); seatedCatsRef.current.push(seat); setSaved(v=>v+1); setTimeout(()=>{ spawnCatCentered(); }, 250); } } }
  function placeSeatedCat(u){ const p=window.devicePixelRatio||1; const W=u.width; const H=u.height; const margin=20*p; const spacing=56*p; const baseY=H-28*p; const count=seatedCatsRef.current.length; const maxPerRow=Math.floor((W-2*margin)/spacing); const row=Math.floor(count/maxPerRow); const col=count%maxPerRow; const x=margin + col*spacing; const y=baseY - row*spacing*0.75; return {x,y}; }
  function drawSeatedCats(ctx,imgs){ ctx.save(); const p=window.devicePixelRatio||1; const im = imgs.cat_seated; if(!im){ ctx.restore(); return; } const w=48*p, h=w*(im.height/im.width); for(const s of seatedCatsRef.current){ ctx.drawImage(im, Math.round(s.x - w/2), Math.round(s.y - h), w, h); } ctx.restore(); }
  function drawActiveCat(ctx,imgs){ stepActiveCat(); const c=catRef.current; if(!c) return; const p=window.devicePixelRatio||1; const im = c.mode==='attached' ? imgs.cat_attached : (c.mode==='falling' ? imgs.cat_jump : imgs.cat_idle); const w=64*p, h=w*(im.height/im.width); ctx.drawImage(im, Math.round(c.x - w/2), Math.round(c.y - h), w, h); }

  function drawSavedCounter(ctx,W,H,val){ const p=window.devicePixelRatio||1; const pad=14*p; const boxW=140*p, boxH=56*p; const x=W - boxW - pad, y=pad; ctx.fillStyle='rgba(0,0,0,.35)'; ctx.fillRect(x,y,boxW,boxH); ctx.font=`${14*p}px system-ui`; ctx.fillStyle='#fff'; ctx.fillText('Saved', x+12*p, y+20*p); ctx.font=`${26*p}px system-ui`; ctx.fillText(`${val}`, x+12*p, y+44*p); }
  function drawThreshold(ctx,W,H,barY,sensitivity){ if(barY==null) return; const p=window.devicePixelRatio||1; const thr=barY - sensitivity; ctx.save(); ctx.setLineDash([16*p, 10*p]); ctx.lineWidth=4*p; ctx.strokeStyle='#00ff88'; ctx.beginPath(); ctx.moveTo(0,thr); ctx.lineTo(W,thr); ctx.stroke(); ctx.setLineDash([]); ctx.beginPath(); ctx.arc(W-40*p,thr,10*p,0,Math.PI*2); ctx.fillStyle='#00ff88'; ctx.fill(); ctx.restore(); }

  // Interactions
  const draggingRef=useRef(false);
  function onPointerDown(e){ const y=getY(e); if(y==null) return; setBarY(y); if(catRef.current.mode==='idle') alignCatToBar(); draggingRef.current=true; }
  function onPointerMove(e){ if(!draggingRef.current) return; const y=getY(e); if(y==null) return; setBarY(y); if(catRef.current.mode==='idle') alignCatToBar(); }
  function onPointerUp(){ draggingRef.current=false; }
  function alignCatToBar(){ const p=window.devicePixelRatio||1; catRef.current.y = (barYRef.current ?? 0) - CAT_BASELINE_ABOVE_ROPE_PX*p; }
  function onTouchStart(e){ e.preventDefault(); onPointerDown(e); }
  function onTouchMove(e){ e.preventDefault(); onPointerMove(e); }
  function onTouchEnd(e){ e.preventDefault(); onPointerUp(); }
  function getY(e){ const rect=uiRef.current.getBoundingClientRect(); const dpr=uiRef.current.width/rect.width; if(e.touches&&e.touches[0]) return (e.touches[0].clientY-rect.top)*dpr; if(typeof e.clientY==='number') return (e.clientY-rect.top)*dpr; return null; }

  // Recording
  const mediaRecorderRef=useRef(null); const recordedChunksRef=useRef([]);
  function isiOSSafari(){ return /iP(hone|ad|od)/.test(navigator.userAgent) && /Safari\//.test(navigator.userAgent) && !/CriOS|FxiOS/.test(navigator.userAgent); }
  function pickMime(){ const prefer = isiOSSafari() ? ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4'] : []; const fall=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm']; const opts=[...prefer,...fall]; for(const o of opts){ try{ if(window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o; }catch{} } return ''; }
  function startRecording(){ if(!canRecord) return; const rec=recRef.current, base=baseRef.current, ui=uiRef.current; if(!rec||!base||!ui) return; const r=rec.getContext('2d'); rec.width=base.width; rec.height=base.height; const mime=pickMime(); if(!mime) return; recordedChunksRef.current=[]; setRecording(true); recordingRef.current=true; const targetFps=30; const compose=()=>{ if(!recordingRef.current) return; r.clearRect(0,0,rec.width,rec.height); r.drawImage(base,0,0); r.drawImage(ui,0,0); requestAnimationFrame(compose); }; requestAnimationFrame(compose); const stream=rec.captureStream(targetFps); let mr; try{ mr=new MediaRecorder(stream,{mimeType:mime, videoBitsPerSecond: 6_000_000}); }catch{ mr=new MediaRecorder(stream,{mimeType:mime}); } mr.ondataavailable=(e)=>{ if(e.data && e.data.size>0) recordedChunksRef.current.push(e.data); }; mr.onstop=()=>{ const type=mr.mimeType||mime; const blob=new Blob(recordedChunksRef.current,{type}); if(!blob || blob.size<150000){ setMsg('Recording tiny — iOS codec limited. Try again or use iOS Screen Recording.'); return; } const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; const ts=new Date().toISOString().replace(/[:.]/g,'-'); a.download=`pullup-rescue-${ts}.${type.includes('mp4')?'mp4':'webm'}`; a.click(); setMsg('Recording saved'); }; mediaRecorderRef.current=mr; try{ mr.start(500); }catch{ mr.start(); } }
  function stopRecording(){ if(mediaRecorderRef.current && mediaRecorderRef.current.state!=='inactive') mediaRecorderRef.current.stop(); setRecording(false); recordingRef.current=false; }

  return (
    <div style={{position:'fixed',inset:0,background:'#000',color:'#fff',overflow:'hidden'}}>
      <video ref={videoRef} playsInline muted style={{display:'none'}} onLoadedMetadata={()=>{ updateGeom(); if(streamRef.current) setMirrorFromStream(streamRef.current); }} />
      <canvas ref={baseRef}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
        onTouchStart={onTouchStart} onTouchMove={onTouchMove} onTouchEnd={onTouchEnd}
        style={{position:'absolute',inset:0,touchAction:'none'}} />
      <canvas ref={uiRef}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
        onTouchStart={onTouchStart} onTouchMove={onTouchMove} onTouchEnd={onTouchEnd}
        style={{position:'absolute',inset:0,touchAction:'none'}} />
      <canvas ref={recRef} style={{display:'none'}} />

      {/* Top bar */}
      <div style={{position:'absolute',top:0,left:0,right:0,padding:'10px env(safe-area-inset-right) 10px env(safe-area-inset-left)',display:'flex',justifyContent:'space-between',alignItems:'center',gap:8}}>
        <div style={{fontSize:14,opacity:.9}}>Pull‑Up Rescue</div>
        <div style={{display:'flex',gap:8,alignItems:'center'}}>
          {camReady && (
            <select value={bucketChoice} onChange={async(e)=>{ const b=e.target.value; await switchToBucket(b); }} style={{background:'rgba(255,255,255,.12)',color:'#fff',border:0,borderRadius:10,padding:'6px'}}>
              {bucketMap.ultra && <option value="ultra" style={{color:'#000'}}>Back — Ultra‑Wide (0.5×)</option>}
              {bucketMap.wide && <option value="wide" style={{color:'#000'}}>Back — Wide (1×)</option>}
              {bucketMap.front && <option value="front" style={{color:'#000'}}>Front</option>}
            </select>
          )}
        </div>
      </div>

      {/* Bottom controls */}
      <div style={{position:'absolute',left:0,right:0,bottom:0,padding:'10px env(safe-area-inset-right) 14px env(safe-area-inset-left)',display:'grid',gap:8}}>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
          {!camReady ? (
            <button onClick={enableCamera} style={btn(1,'#22c55e')}>Enable Camera</button>
          ) : !recording ? (
            <button onClick={startRecording} disabled={!canRecord} style={btn(canRecord?1:.5)}>Start Recording</button>
          ) : (
            <button onClick={stopRecording} style={btn(1,'#ef4444')}>Stop</button>
          )}
          <button onClick={()=>{ setSaved(0); repRef.current={phase:'down',lastAbove:0,lastRepAt:0}; seatedCatsRef.current=[]; spawnCatCentered(); }} style={btn()}>Reset</button>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto',alignItems:'center',gap:8}}>
          <Labeled label={`Sensitivity (px above rope): ${sensitivity}`}>
            <input type="range" min={8} max={80} step={1} value={sensitivity} onChange={(e)=>{ const v=parseInt(e.target.value,10); setSensitivity(v); }} style={{width:'100%'}} />
          </Labeled>
          <label style={{display:'flex',gap:6,alignItems:'center',fontSize:12,opacity:.85}}>
            <input type="checkbox" checked={showPose} onChange={(e)=>setShowPose(e.target.checked)} /> Show pose
          </label>
        </div>
        <div style={{fontSize:12,opacity:.85,textAlign:'center'}}>{msg}</div>
        {debug && (<div style={{fontSize:10,opacity:.6,textAlign:'center',userSelect:'all'}}>{debug}</div>)}
      </div>
    </div>
  );
}

function btn(opacity=1,bg){ return {border:0,borderRadius:14,padding:'10px 12px',background:bg||'rgba(255,255,255,.12)',color:'#fff',opacity,backdropFilter:'saturate(120%) blur(6px)'}; }
function Labeled({label,children}){ return (<div><div style={{fontSize:11,opacity:.75,marginBottom:4}}>{label}</div>{children}</div>); }




