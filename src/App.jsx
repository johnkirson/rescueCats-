import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";

/**
 * v3.2 — Robust iOS camera bring‑up
 * 1) Requires user gesture (Enable Camera)
 * 2) First asks for { video: true } to unlock labels → enumerateDevices →
 *    pick deviceId by label (back/front). This is more reliable on iOS than facingMode.
 * 3) If selected device fails, falls back across all available video devices.
 * 4) Shows debug error text (tap to copy) if init fails.
 * 5) Keeps one‑screen minimalist UI from v3.
 */

const MOVENET_EDGES = { 0:[0,1],1:[1,3],2:[0,2],3:[2,4],4:[5,7],5:[7,9],6:[6,8],7:[8,10],8:[5,6],9:[5,11],10:[6,12],11:[11,12],12:[11,13],13:[13,15],14:[12,14],15:[14,16] };
function updateRepState(prev, isAbove, now, minAboveMs = 200, minIntervalMs = 500){ const next={...prev}; let counted=0; if(prev.phase==='down'&&isAbove){next.phase='up'; next.lastAbove=now;} else if(prev.phase==='up'&&!isAbove){ if(now-prev.lastAbove>minAboveMs && now-prev.lastRepAt>minIntervalMs){ next.phase='down'; next.lastRepAt=now; counted=1;} else { next.phase='down'; } } return {next, counted}; }

export default function PullUpRescueV32(){
  const videoRef = useRef(null); const baseRef=useRef(null); const uiRef=useRef(null); const recRef=useRef(null);
  const detectorRef=useRef(null); const rafRef=useRef(null); const streamRef=useRef(null);

  const [camReady,setCamReady]=useState(false); const [useBack,setUseBack]=useState(true);
  const [msg,setMsg]=useState("Tap ‘Enable Camera’, then Allow"); const [debug,setDebug]=useState("");
  const [cats,setCats]=useState(0); const [detecting,setDetecting]=useState(false); const [recording,setRecording]=useState(false); const [canRecord,setCanRecord]=useState(false);
  const [barY,setBarY]=useState(null); const [sensitivity,setSensitivity]=useState(24); const [lineWidth,setLineWidth]=useState(6); const [lineColor,setLineColor]=useState('#00ff88'); const draggingRef=useRef(false);
  const repRef=useRef({phase:'down',lastAbove:0,lastRepAt:0}); const [rescues,setRescues]=useState([]);

  useEffect(()=>{ const c=document.createElement('canvas'); setCanRecord(typeof c.captureStream==='function' && 'MediaRecorder' in window); },[]);
  useEffect(()=>{ const resize=()=>{ const dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1)); const W=window.innerWidth,H=window.innerHeight; [baseRef.current,uiRef.current,recRef.current].forEach(cv=>{ if(!cv) return; cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=Math.floor(W*dpr); cv.height=Math.floor(H*dpr);}); }; resize(); window.addEventListener('resize',resize); return()=>window.removeEventListener('resize',resize); },[]);

  async function ensureDetector(){ if(!detectorRef.current){ await tf.setBackend('webgl'); await tf.ready(); detectorRef.current = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,{modelType:'Lightning'}); } }

  async function enableCamera(){
    setDebug("");
    try{
      await ensureDetector();
      // Step 1: ask for generic camera to unlock labels
      const pre = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      // Attach temporarily to satisfy autoplay policies
      const v=videoRef.current; v.setAttribute('playsinline',''); v.srcObject=pre; await v.play();
      // Step 2: enumerate devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videos = devices.filter(d=>d.kind==='videoinput');
      // Pick back/front by label
      const isBack = (d)=>/back|rear|environment/i.test(d.label);
      const isFront = (d)=>/front|user|face/i.test(d.label);
      let target; if(useBack){ target = videos.find(isBack) || videos.find(d=>!isFront(d)) || videos[0]; } else { target = videos.find(isFront) || videos[0]; }
      // Step 3: if we found a deviceId different from pre, switch to it
      let stream = pre;
      if(target && target.deviceId){ try{ stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: target.deviceId } }, audio:false }); } catch(e){ setDebug(`Exact deviceId failed: ${e.name}`); }}
      // If still not ok (black frames sometimes), fallback cycling through videos
      if(!stream){ for(const d of videos){ try{ stream = await navigator.mediaDevices.getUserMedia({ video:{ deviceId:{ exact:d.deviceId }}, audio:false}); break; } catch(e){ setDebug(prev=> prev+` | try ${d.label}: ${e.name}`); } } }
      // Bind final stream
      if(streamRef.current) streamRef.current.getTracks().forEach(t=>t.stop());
      streamRef.current = stream; v.srcObject=stream; await v.play();
      setCamReady(true); setMsg('Drag the bar to the pull‑up bar height, then Start');
      cancelAnimationFrame(rafRef.current||0); rafRef.current = requestAnimationFrame(tick);
    }catch(e){ console.error(e); setMsg('Camera init failed. Check permissions in Settings > Safari > Camera.'); setDebug(`${e.name||'Error'}: ${e.message||e}`); }
  }

  function flipCamera(){ setUseBack(v=>!v); setCamReady(false); setMsg('Tap Enable Camera after flipping'); if(streamRef.current){ streamRef.current.getTracks().forEach(t=>t.stop()); } }

  const tick = async ()=>{
    const video=videoRef.current, base=baseRef.current, ui=uiRef.current; if(!video||!base||!ui) return; const b=base.getContext('2d'); const u=ui.getContext('2d'); const W=base.width,H=base.height;
    const vw=video.videoWidth||1280, vh=video.videoHeight||720; const s=Math.max(W/vw,H/vh); const dw=vw*s, dh=vh*s; const dx=(W-dw)/2, dy=(H-dh)/2;
    b.clearRect(0,0,W,H);
    // Mirror only for front camera UX
    if(!useBack){ b.save(); b.translate(W,0); b.scale(-1,1); b.drawImage(video, -dx - dw + W, dy, dw, dh); b.restore(); } else { b.drawImage(video, dx, dy, dw, dh); }

    if(detecting && detectorRef.current){ const poses=await detectorRef.current.estimatePoses(video,{maxPoses:1, flipHorizontal: !useBack}); if(poses[0]){ const kps=poses[0].keypoints.map(k=>({...k})); drawPose(b,kps); const nose=kps[0]; if(nose?.score>0.4 && barY!==null){ const thr=barY - sensitivity; const now=performance.now(); const above = nose.y <= thr; const {next,counted}=updateRepState(repRef.current,above,now); repRef.current=next; if(counted){ setCats(c=>c+1); spawnRescue(); } } } }
    u.clearRect(0,0,W,H); drawHouse(u,W,H,cats); drawBar(u,W,H,barY,sensitivity,lineWidth,lineColor); drawRescues(u,rescues);
    rafRef.current = requestAnimationFrame(tick);
  };

  function drawPose(ctx,kps){ ctx.strokeStyle='rgba(255,255,255,.4)'; ctx.lineWidth=2; Object.values(MOVENET_EDGES).forEach(([i,j])=>{ const a=kps[i],b=kps[j]; if(a?.score>0.3&&b?.score>0.3){ ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); } }); kps.forEach(k=>{ if(k.score>0.3){ ctx.beginPath(); ctx.arc(k.x,k.y,4,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill(); } }); }
  function drawHouse(ctx,W,H,cats){ const p=window.devicePixelRatio||1; const w=Math.min(200*p, W*.22), h=w*.75; const x=16*p, y=H-h-16*p; ctx.fillStyle='rgba(0,0,0,.35)'; ctx.fillRect(x,y,w,h); ctx.beginPath(); ctx.moveTo(x-10,y); ctx.lineTo(x+w/2,y-40); ctx.lineTo(x+w+10,y); ctx.closePath(); ctx.fillStyle='rgba(0,0,0,.45)'; ctx.fill(); ctx.font=`${20*p}px system-ui`; ctx.fillStyle='#fff'; ctx.fillText(`${cats} saved`, x, y-12); }
  function drawBar(ctx,W,H,barY,sensitivity,width,color){ if(barY==null) return; const thr=barY - sensitivity; const p=window.devicePixelRatio||1; ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=Math.max(2,width*p); ctx.setLineDash([16*p,10*p]); ctx.beginPath(); ctx.moveTo(0,thr); ctx.lineTo(W,thr); ctx.stroke(); ctx.setLineDash([]); ctx.beginPath(); ctx.arc(W-40*p,thr,10*p,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); ctx.restore(); }
  function drawRescues(ctx,items){ const p=window.devicePixelRatio||1; items.forEach(r=>{ ctx.font=`${48*p}px system-ui`; ctx.textAlign='center'; ctx.fillText('🐱', r.x, r.y); }); }
  function spawnRescue(){ const p=window.devicePixelRatio||1; const W=uiRef.current.width,H=uiRef.current.height; const sx=60*p, sy=H-120*p; const id=Math.random().toString(36).slice(2); const created={id,x:sx,y:sy}; setRescues(a=>[...a,created]); const t0=performance.now(), dur=900; const step=(t)=>{ const p2=Math.min(1,(t-t0)/dur); const nx=sx+(W*.35)*p2, ny=sy-(H*.35)*p2; setRescues(a=>a.map(r=>r.id===id?{...r,x:nx,y:ny}:r)); if(p2<1) requestAnimationFrame(step); else setTimeout(()=> setRescues(a=>a.filter(r=>r.id!==id)),150); }; requestAnimationFrame(step); }

  function onPointerDown(e){ const y=getY(e); if(y==null) return; setBarY(y); draggingRef.current=true; }
  function onPointerMove(e){ if(!draggingRef.current) return; const y=getY(e); if(y==null) return; setBarY(y); }
  function onPointerUp(){ draggingRef.current=false; }
  function getY(e){ const rect=uiRef.current.getBoundingClientRect(); const dpr=uiRef.current.width/rect.width; if(e.touches&&e.touches[0]) return (e.touches[0].clientY-rect.top)*dpr; if(typeof e.clientY==='number') return (e.clientY-rect.top)*dpr; return null; }

  // Recording (same as before)
  const mediaRecorderRef=useRef(null); const recordedChunksRef=useRef([]);
  function getBestMime(){ const opts=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm','video/mp4']; for(const o of opts){ try{ if(window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o; }catch{} } return ''; }
  function startRecording(){ if(!canRecord) return; const rec=recRef.current, base=baseRef.current, ui=uiRef.current; if(!rec||!base||!ui) return; const r=rec.getContext('2d'); const mime=getBestMime(); if(!mime){ setMsg('No supported recording format'); return;} recordedChunksRef.current=[]; const stream=rec.captureStream(30); const mr=new MediaRecorder(stream,{mimeType:mime}); mr.ondataavailable=(e)=>e.data.size>0&&recordedChunksRef.current.push(e.data); mr.onstop=()=>{ const type=mr.mimeType||mime; const blob=new Blob(recordedChunksRef.current,{type}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; const ts=new Date().toISOString().replace(/[:.]/g,'-'); a.download=`pullup-rescue-${ts}.${type.includes('webm')?'webm':'mp4'}`; a.click(); setMsg('Recording saved'); }; mediaRecorderRef.current=mr; setRecording(true); const comp=()=>{ if(!recording) return; r.clearRect(0,0,rec.width,rec.height); r.drawImage(base,0,0); r.drawImage(ui,0,0); requestAnimationFrame(comp); }; requestAnimationFrame(comp); mr.start(); }
  function stopRecording(){ if(mediaRecorderRef.current && mediaRecorderRef.current.state!=='inactive') mediaRecorderRef.current.stop(); setRecording(false); }

  return (
    <div style={{position:'fixed',inset:0,background:'#000',color:'#fff',overflow:'hidden'}}>
      <video ref={videoRef} playsInline muted style={{display:'none'}} />
      <canvas ref={baseRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onTouchEnd={onPointerUp} style={{position:'absolute',inset:0}} />
      <canvas ref={uiRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onTouchEnd={onPointerUp} style={{position:'absolute',inset:0}} />
      <canvas ref={recRef} style={{display:'none'}} />

      <div style={{position:'absolute',top:0,left:0,right:0,padding:'10px env(safe-area-inset-right) 10px env(safe-area-inset-left)',display:'flex',justifyContent:'space-between',alignItems:'center',gap:8}}>
        <div style={{fontSize:14,opacity:.9}}>Pull‑Up Rescue</div>
        <div style={{display:'flex',gap:8}}>
          <button onClick={()=>{setUseBack(v=>!v); flipCamera();}} style={btn()}>Flip</button>
          {!detecting ? (
            <button onClick={()=>{ if(!camReady) return setMsg('Enable camera first'); setDetecting(true); setMsg('Detecting pull‑ups…'); }} style={btn(camReady?1:.5)}>Start</button>
          ) : (
            <button onClick={()=>{ setDetecting(false); setMsg('Paused'); }} style={btn()}>Pause</button>
          )}
        </div>
      </div>

      <div style={{position:'absolute',left:0,right:0,bottom:0,padding:'10px env(safe-area-inset-right) 14px env(safe-area-inset-left)',display:'grid',gap:8}}>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:8}}>
          {!camReady ? (
            <button onClick={enableCamera} style={btn(1,'#22c55e')}>Enable Camera</button>
          ) : !recording ? (
            <button onClick={startRecording} disabled={!canRecord} style={btn(canRecord?1:.5)}>Record</button>
          ) : (
            <button onClick={stopRecording} style={btn(1,'#ef4444')}>Stop</button>
          )}
          <button onClick={()=>{ setCats(0); repRef.current={phase:'down',lastAbove:0,lastRepAt:0}; }} style={btn()}>Reset</button>
          <button onClick={()=> setBarY(null)} style={btn()}>Re‑set bar</button>
        </div>
        <div style={{display:'grid',gap:6}}>
          <Labeled label={`Sensitivity: ${sensitivity}px`}><input type="range" min={8} max={80} step={1} value={sensitivity} onChange={(e)=>setSensitivity(parseInt(e.target.value,10))} style={{width:'100%'}} /></Labeled>
          <Labeled label={`Line width: ${lineWidth}px`}><input type="range" min={2} max={16} step={1} value={lineWidth} onChange={(e)=>setLineWidth(parseInt(e.target.value,10))} style={{width:'100%'}} /></Labeled>
          <Labeled label="Line color"><select value={lineColor} onChange={(e)=>setLineColor(e.target.value)} style={select()}><option value="#00ff88">Green</option><option value="#ffd60a">Yellow</option><option value="#ff4d4f">Red</option><option value="#60a5fa">Blue</option><option value="#ffffff">White</option></select></Labeled>
        </div>
        <div style={{fontSize:12,opacity:.85,textAlign:'center'}}>{msg}</div>
        {debug && (<div style={{fontSize:10,opacity:.6,textAlign:'center',userSelect:'all'}}>{debug}</div>)}
      </div>
    </div>
  );
}

function btn(opacity=1,bg){ return {border:0,borderRadius:14,padding:'10px 12px',background:bg||'rgba(255,255,255,.12)',color:'#fff',opacity,backdropFilter:'saturate(120%) blur(6px)'}; }
function select(){ return {width:'100%',borderRadius:10,background:'rgba(255,255,255,.12)',color:'#fff',border:'0',padding:'8px'}; }
function Labeled({label,children}){ return (<div><div style={{fontSize:11,opacity:.75,marginBottom:4}}>{label}</div>{children}</div>); }
