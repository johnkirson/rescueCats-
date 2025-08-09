import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";

/**
 * v4.1 — Camera picker fix + cats sit at bottom
 * - Fixed: picker now correctly selects bucket, mirrors only for Front
 * - Fixed: switching cameras reliably (uses exact deviceId; updates geometry & mirror)
 * - Default camera prefers Ultra‑Wide > Wide > Front
 * - Cats: after falling, they "sit" along the bottom row and look up. New rescues join the row.
 */

const MOVENET_EDGES = { 0:[0,1],1:[1,3],2:[0,2],3:[2,4],4:[5,7],5:[7,9],6:[6,8],7:[8,10],8:[5,6],9:[5,11],10:[6,12],11:[11,12],12:[11,13],13:[13,15],14:[12,14],15:[14,16] };
function updateRepState(prev, isAbove, now, minAboveMs = 200, minIntervalMs = 500){ const next={...prev}; let counted=0; if(prev.phase==='down'&&isAbove){next.phase='up'; next.lastAbove=now;} else if(prev.phase==='up'&&!isAbove){ if(now-prev.lastAbove>minAboveMs && now-prev.lastRepAt>minIntervalMs){ next.phase='down'; next.lastRepAt=now; counted=1;} else { next.phase='down'; } } return {next, counted}; }

export default function PullUpRescueV41(){
  const videoRef = useRef(null); const baseRef=useRef(null); const uiRef=useRef(null); const recRef=useRef(null);
  const detectorRef=useRef(null); const rafRef=useRef(null); const streamRef=useRef(null);

  // UI state
  const [camReady,setCamReady]=useState(false);
  const [bucketMap,setBucketMap]=useState({ultra:null,wide:null,front:null});
  const [bucketChoice,setBucketChoice]=useState('ultra');
  const [msg,setMsg]=useState("Drag the rope to the bar height, then Start"); const [debug,setDebug]=useState("");
  const [detecting,setDetecting]=useState(false); const detectingRef=useRef(false);
  const [recording,setRecording]=useState(false); const recordingRef=useRef(false);
  const [canRecord,setCanRecord]=useState(false);
  const [barY,setBarY]=useState(null); const barYRef=useRef(null);
  const [sensitivity,setSensitivity]=useState(24); const sensitivityRef=useRef(24);
  const [showPose,setShowPose]=useState(true);

  // counters
  const [saved,setSaved]=useState(0); const savedRef=useRef(0);
  const repRef=useRef({phase:'down',lastAbove:0,lastRepAt:0});

  // geometry
  const geomRef = useRef({ W:0,H:0,vw:0,vh:0,scale:1,dx:0,dy:0, mirrored:false });

  // rope & cats
  const ropeRef = useRef({ y:null });
  const catRef = useRef({ mode:'idle', x:0, y:0, vx:0, vy:0, lastT:0 });
  const seatedCatsRef = useRef([]); // [{x,y}]

  // sync refs
  useEffect(()=>{ savedRef.current=saved; },[saved]);
  useEffect(()=>{ detectingRef.current=detecting; },[detecting]);
  useEffect(()=>{ recordingRef.current=recording; },[recording]);
  useEffect(()=>{ barYRef.current=barY; ropeRef.current.y=barY; },[barY]);
  useEffect(()=>{ sensitivityRef.current=sensitivity; },[sensitivity]);

  useEffect(()=>{ const c=document.createElement('canvas'); setCanRecord(typeof c.captureStream==='function' && 'MediaRecorder' in window); },[]);
  useEffect(()=>{ const resize=()=>{ const dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1)); const W=window.innerWidth,H=window.innerHeight; [baseRef.current,uiRef.current,recRef.current].forEach(cv=>{ if(!cv) return; cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=Math.floor(W*dpr); cv.height=Math.floor(H*dpr);}); if (uiRef.current && barYRef.current==null) { const mid=Math.floor(uiRef.current.height*0.5); setBarY(mid); barYRef.current=mid; } updateGeom(); if(!catRef.current.lastT) spawnCatCentered(); }; resize(); window.addEventListener('resize',resize); return()=>window.removeEventListener('resize',resize); },[]);

  function updateGeom(){ const video=videoRef.current, base=baseRef.current; if(!video||!base) return; const W=base.width,H=base.height; const vw=video.videoWidth||1280, vh=video.videoHeight||720; const scale=Math.max(W/vw,H/vh); const dw=vw*scale, dh=vh*scale; const dx=(W-dw)/2, dy=(H-dh)/2; const mirrored = bucketChoice==='front'; geomRef.current={W,H,vw,vh,scale,dx,dy,mirrored}; }

  async function createMoveNetDetector(){ await tf.setBackend('webgl'); await tf.ready(); const opts=[ {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING}, {modelType:poseDetection.movenet.modelType.SINGLEPOSE_THUNDER}, {modelType:poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING} ]; let lastErr; for(const o of opts){ try{ return await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,o);}catch(e){ lastErr=e; } } throw lastErr; }
  async function ensureDetector(){ if(!detectorRef.current){ detectorRef.current = await createMoveNetDetector(); } }

  function toBuckets(devices){ const vids=devices.filter(d=>d.kind==='videoinput'); const LB=(d)=>(d.label||'').toLowerCase(); const find=(rx)=>vids.find(d=>rx.test(LB(d))); const ultra = find(/ultra\s*wide|0\.5x|ultra/); const front = find(/front|user|face/); const wide = vids.find(d=>!/front|user|face/i.test(d.label||'')); return { ultra: ultra?.deviceId||null, wide: wide?.deviceId||null, front: front?.deviceId||null }; }

  async function enableCamera(){ setDebug(""); try{ await ensureDetector(); const pre=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); const v=videoRef.current; v.setAttribute('playsinline',''); v.srcObject=pre; await v.play(); const devices=await navigator.mediaDevices.enumerateDevices(); const buckets=toBuckets(devices); setBucketMap(buckets); const choice = buckets.ultra ? 'ultra' : (buckets.wide ? 'wide' : 'front'); setBucketChoice(choice); await switchToBucket(choice, pre); setCamReady(true); cancelAnimationFrame(rafRef.current||0); rafRef.current=requestAnimationFrame(tick); if (uiRef.current && barYRef.current==null) { const mid=Math.floor(uiRef.current.height*0.5); setBarY(mid); } }catch(e){ console.error(e); setMsg('Camera init failed. Check permissions in Settings > Safari > Camera.'); setDebug(`${e.name||'Error'}: ${e.message||e}`); } }

  async function switchToBucket(bucket, preStream=null){ try{ const deviceId=bucketMap[bucket]; let stream=preStream; const baseConstraints = { width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:30, max:60} };
      if(deviceId){ try{ stream=await navigator.mediaDevices.getUserMedia({ video:{ deviceId:{ exact: deviceId }, ...baseConstraints }, audio:false }); }catch(e){ setDebug(`Exact deviceId (${bucket}) failed: ${e.name}`); try{ stream=await navigator.mediaDevices.getUserMedia({ video: baseConstraints, audio:false }); }catch{ stream=await navigator.mediaDevices.getUserMedia({ video:true, audio:false }); } } }
      if(!stream){ stream=await navigator.mediaDevices.getUserMedia({ video: baseConstraints, audio:false }); }
      if(streamRef.current) streamRef.current.getTracks().forEach(t=>t.stop()); const v=videoRef.current; v.srcObject=stream; await v.play(); setBucketChoice(bucket); updateGeom(); }
    catch(e){ setDebug(`Switch bucket failed: ${e.name}`); } }

  const tick = async ()=>{ const video=videoRef.current, base=baseRef.current, ui=uiRef.current; if(!video||!base||!ui) return; const b=base.getContext('2d'); const u=ui.getContext('2d'); const {W,H,vw,vh,scale,dx,dy,mirrored}=geomRef.current; if(!W) updateGeom(); b.clearRect(0,0,W,H); const dw=vw*scale, dh=vh*scale; if(mirrored){ b.save(); b.translate(W,0); b.scale(-1,1); b.drawImage(video, -dx - dw + W, dy, dw, dh); b.restore(); } else { b.drawImage(video, dx, dy, dw, dh); }
    if(detectingRef.current && detectorRef.current){ const poses=await detectorRef.current.estimatePoses(video,{maxPoses:1,flipHorizontal: mirrored}); if(poses[0]){ const kps=poses[0].keypoints.map(k=>({...k})); if(showPose){ b.save(); b.globalAlpha=0.6; const mapped=kps.map((kp)=>({ ...kp, X: mirrored ? (dx + dw - kp.x*scale) : (dx + kp.x*scale), Y: dy + kp.y*scale })); drawPoseMapped(b,mapped); b.restore(); }
        const nose=kps[0]; const by=barYRef.current; const sens=sensitivityRef.current; if(nose?.score>0.4 && by!==null){ const thr=by - sens; const ny = dy + nose.y*scale; const above = ny <= thr; const now=performance.now(); const {next,counted}=updateRepState(repRef.current,above,now); repRef.current=next; if(above && catRef.current.mode==='idle'){ catRef.current.mode='attached'; }
          if(!above && repRef.current.phase==='down' && catRef.current.mode==='attached'){ startCatFall(); }
          if(catRef.current.mode==='attached'){ catRef.current.x = mirrored ? (dx + dw - nose.x*scale) : (dx + nose.x*scale); catRef.current.y = ny + 24*(window.devicePixelRatio||1); }
        }
      } }
    // UI layer
    u.clearRect(0,0,W,H);
    drawRope(u,W,H,ropeRef.current.y,true);
    drawThreshold(u,W,H,barYRef.current,sensitivityRef.current);
    drawSeatedCats(u);
    drawActiveCat(u);
    drawSavedCounter(u,W,H,savedRef.current);
    rafRef.current=requestAnimationFrame(tick);
  };

  function drawPoseMapped(ctx,kps){ ctx.strokeStyle='rgba(255,255,255,.85)'; ctx.lineWidth=2; Object.values(MOVENET_EDGES).forEach(([i,j])=>{ const a=kps[i],b=kps[j]; if(a?.score>0.3&&b?.score>0.3){ ctx.beginPath(); ctx.moveTo(a.X,a.Y); ctx.lineTo(b.X,b.Y); ctx.stroke(); } }); kps.forEach(k=>{ if(k.score>0.3){ ctx.beginPath(); ctx.arc(k.X,k.Y,4,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.95)'; ctx.fill(); } }); }

  // --- Rope & Cat visuals ---
  function drawRope(ctx,W,H,y,flames){ if(y==null) return; const p=window.devicePixelRatio||1; const x1=16*p, x2=W-16*p; ctx.save(); ctx.strokeStyle='rgba(0,0,0,.6)'; ctx.lineWidth=10*p; ctx.beginPath(); ctx.moveTo(x1,y+2*p); ctx.lineTo(x2,y+2*p); ctx.stroke(); const grad=ctx.createLinearGradient(x1,y,x2,y); grad.addColorStop(0,'#c8a76a'); grad.addColorStop(1,'#b08a4a'); ctx.strokeStyle=grad; ctx.lineWidth=6*p; ctx.setLineDash([8*p,6*p]); ctx.beginPath(); ctx.moveTo(x1,y); ctx.lineTo(x2,y); ctx.stroke(); ctx.setLineDash([]); if(flames){ drawFlame(ctx,x1,y,p); drawFlame(ctx,x2,y,p);} ctx.restore(); }
  function drawFlame(ctx,x,y,p){ const t=performance.now()/1000; const r=12*p+4*p*Math.sin(t*10); const g=ctx.createRadialGradient(x,y,r*0.1,x,y,r); g.addColorStop(0,'rgba(255,240,180,.9)'); g.addColorStop(0.6,'rgba(255,140,0,.8)'); g.addColorStop(1,'rgba(255,80,0,.0)'); ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); }

  function spawnCatCentered(){ const u=uiRef.current; if(!u) return; const p=window.devicePixelRatio||1; const W=u.width; const y=barYRef.current ?? Math.floor(u.height*0.5); catRef.current={ mode:'idle', x:Math.floor(W/2), y:y-12*p, vx:0, vy:0, lastT:performance.now() }; }
  function startCatFall(){ const u=uiRef.current; if(!u) return; const now=performance.now(); const c=catRef.current; c.mode='falling'; c.vx=(Math.random()*2-1)*30; c.vy=0; c.lastT=now; }
  function stepActiveCat(){ const u=uiRef.current; if(!u) return; const p=window.devicePixelRatio||1; const H=u.height; const groundY=H-28*p; const c=catRef.current; const now=performance.now(); const dt=Math.min(0.05,(now-c.lastT)/1000); c.lastT=now; if(c.mode==='falling'){ const g=1200*p; c.vy += g*dt; c.y += c.vy*dt; c.x += c.vx*dt; if(c.y >= groundY){ c.y=groundY; c.mode='seated'; // add to seated row
        const seat = placeSeatedCat(u); seatedCatsRef.current.push(seat); setSaved(v=>{ const nx=v+1; savedRef.current=nx; return nx; }); setTimeout(()=>{ spawnCatCentered(); }, 350); }
    } }
  function placeSeatedCat(u){ const p=window.devicePixelRatio||1; const W=u.width; const H=u.height; const margin=20*p; const spacing=42*p; const baseY=H-28*p; const count=seatedCatsRef.current.length; const maxPerRow=Math.floor((W-2*margin)/spacing); const row=Math.floor(count/maxPerRow); const col=count%maxPerRow; const x=margin + col*spacing; const y=baseY - row*spacing*0.7; return {x,y}; }
  function drawSeatedCats(ctx){ ctx.save(); const p=window.devicePixelRatio||1; ctx.font=`${28*p}px system-ui`; const emoji='🐱'; seatedCatsRef.current.forEach(s=>{ const w=ctx.measureText(emoji).width; ctx.fillText(emoji, s.x - w/2, s.y); }); ctx.restore(); }
  function drawActiveCat(ctx){ stepActiveCat(); const c=catRef.current; if(!c) return; const p=window.devicePixelRatio||1; ctx.save(); ctx.font=`${32*p}px system-ui`; const emoji=c.mode==='seated'?'🐱':'🐱'; const w=ctx.measureText(emoji).width; ctx.fillText(emoji, c.x - w/2, c.y); // eyes looking up (simple)
    if(c.mode==='seated'){ ctx.beginPath(); ctx.arc(c.x-6*p, c.y-20*p, 2*p, 0, Math.PI*2); ctx.arc(c.x+6*p, c.y-20*p, 2*p, 0, Math.PI*2); ctx.fillStyle='#fff'; ctx.fill(); ctx.beginPath(); ctx.arc(c.x-6*p, c.y-20*p-1*p, 1*p, 0, Math.PI*2); ctx.arc(c.x+6*p, c.y-20*p-1*p, 1*p, 0, Math.PI*2); ctx.fillStyle='#000'; ctx.fill(); }
    ctx.restore(); }

  function drawSavedCounter(ctx,W,H,val){ const p=window.devicePixelRatio||1; const pad=14*p; const boxW=140*p, boxH=56*p; const x=W - boxW - pad, y=pad; ctx.fillStyle='rgba(0,0,0,.35)'; ctx.fillRect(x,y,boxW,boxH); ctx.font=`${14*p}px system-ui`; ctx.fillStyle='#fff'; ctx.fillText('Saved', x+12*p, y+20*p); ctx.font=`${26*p}px system-ui`; ctx.fillText(`${val}`, x+12*p, y+44*p); }
  function drawThreshold(ctx,W,H,barY,sensitivity){ if(barY==null) return; const p=window.devicePixelRatio||1; const thr=barY - sensitivity; ctx.save(); ctx.setLineDash([16*p, 10*p]); ctx.lineWidth=4*p; ctx.strokeStyle='#00ff88'; ctx.beginPath(); ctx.moveTo(0,thr); ctx.lineTo(W,thr); ctx.stroke(); ctx.setLineDash([]); ctx.beginPath(); ctx.arc(W-40*p,thr,10*p,0,Math.PI*2); ctx.fillStyle='#00ff88'; ctx.shadowColor='rgba(0,0,0,.6)'; ctx.shadowBlur=6*p; ctx.fill(); ctx.restore(); }

  // Dragging moves the rope height
  function onPointerDown(e){ const y=getY(e); if(y==null) return; setBarY(y); ropeRef.current.y=y; if(catRef.current.mode==='idle') catRef.current.y=y-12*(window.devicePixelRatio||1); draggingRef.current=true; }
  function onPointerMove(e){ if(!draggingRef.current) return; const y=getY(e); if(y==null) return; setBarY(y); ropeRef.current.y=y; if(catRef.current.mode==='idle') catRef.current.y=y-12*(window.devicePixelRatio||1); }
  function onPointerUp(){ draggingRef.current=false; }
  function onTouchStart(e){ e.preventDefault(); onPointerDown(e); }
  function onTouchMove(e){ e.preventDefault(); onPointerMove(e); }
  function onTouchEnd(e){ e.preventDefault(); onPointerUp(); }
  function getY(e){ const rect=uiRef.current.getBoundingClientRect(); const dpr=uiRef.current.width/rect.width; if(e.touches&&e.touches[0]) return (e.touches[0].clientY-rect.top)*dpr; if(typeof e.clientY==='number') return (e.clientY-rect.top)*dpr; return null; }

  // Recording (same as v3.10)
  const mediaRecorderRef=useRef(null); const recordedChunksRef=useRef([]);
  function isiOSSafari(){ return /iP(hone|ad|od)/.test(navigator.userAgent) && /Safari\//.test(navigator.userAgent) && !/CriOS|FxiOS/.test(navigator.userAgent); }
  function pickMime(){ const prefer = isiOSSafari() ? ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4'] : []; const fall=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm']; const opts=[...prefer,...fall]; for(const o of opts){ try{ if(window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o; }catch{} } return ''; }
  function startRecording(){ const rec=recRef.current, base=baseRef.current, ui=uiRef.current; if(!rec||!base||!ui||!canRecord) return; const r=rec.getContext('2d'); rec.width=base.width; rec.height=base.height; const mime=pickMime(); if(!mime){ setMsg('Recording not supported on this device/browser'); return; } recordedChunksRef.current=[]; setRecording(true); recordingRef.current=true; const compose=()=>{ if(!recordingRef.current) return; r.clearRect(0,0,rec.width,rec.height); r.drawImage(base,0,0); r.drawImage(ui,0,0); requestAnimationFrame(compose); }; requestAnimationFrame(compose); const stream=rec.captureStream(60); let mr; try{ mr=new MediaRecorder(stream,{mimeType:mime, videoBitsPerSecond: 8_000_000}); }catch{ mr=new MediaRecorder(stream,{mimeType:mime}); } mr.ondataavailable=(e)=>{ if(e.data && e.data.size>0) recordedChunksRef.current.push(e.data); }; mr.onstop=()=>{ const type=mr.mimeType||mime; const blob=new Blob(recordedChunksRef.current,{type}); if(!blob || blob.size<150000){ setMsg('Recording tiny — Safari limited codec. Try again or use iOS Screen Recording.'); return; } const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; const ts=new Date().toISOString().replace(/[:.]/g,'-'); a.download=`pullup-rescue-${ts}.${type.includes('mp4')?'mp4':'webm'}`; a.click(); setMsg('Recording saved'); }; mediaRecorderRef.current=mr; try{ mr.start(500); }catch{ mr.start(); } }
  function stopRecording(){ if(mediaRecorderRef.current && mediaRecorderRef.current.state!=='inactive') mediaRecorderRef.current.stop(); setRecording(false); recordingRef.current=false; }

  return (
    <div style={{position:'fixed',inset:0,background:'#000',color:'#fff',overflow:'hidden'}}>
      <video ref={videoRef} playsInline muted style={{display:'none'}} onLoadedMetadata={updateGeom} />
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
          {!detecting ? (
            <button onClick={()=>{ if(!camReady) return setMsg('Enable camera first'); setDetecting(true); detectingRef.current=true; setMsg('Saving cats…'); }} style={btn(camReady?1:.5)}>Start</button>
          ) : (
            <button onClick={()=>{ setDetecting(false); detectingRef.current=false; setMsg('Paused'); }} style={btn()}>Pause</button>
          )}
        </div>
      </div>

      {/* Bottom controls */}
      <div style={{position:'absolute',left:0,right:0,bottom:0,padding:'10px env(safe-area-inset-right) 14px env(safe-area-inset-left)',display:'grid',gap:8}}>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
          {!camReady ? (
            <button onClick={enableCamera} style={btn(1,'#22c55e')}>Enable Camera</button>
          ) : !recording ? (
            <button onClick={startRecording} disabled={!canRecord} style={btn(canRecord?1:.5)}>Record</button>
          ) : (
            <button onClick={stopRecording} style={btn(1,'#ef4444')}>Stop</button>
          )}
          <button onClick={()=>{ setSaved(0); savedRef.current=0; repRef.current={phase:'down',lastAbove:0,lastRepAt:0}; seatedCatsRef.current=[]; spawnCatCentered(); }} style={btn()}>Reset</button>
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
