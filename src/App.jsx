import React, { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";

// Register TensorFlow.js backend
tf.setBackend('webgl');

/**
 * v6.3 — Complete, de-frozen build (previous working)
 * --------------------------------------------------
 * - Полный, самодостаточный App.jsx без пропусков/обрывов.
 * - Исправлен возможный фриз rAF (try/finally + watchdog + перезапуск при reset/enable).
 * - Камеры переключаются через актуальную mapRef.
 * - Сохранены механики v5.8 и v6.x: трос, кот, delayed drop, landing(7)->seated без idle‑флика,
 *   счётчик, запись, масштаб троса X/Y, milestones.
 */

// ===================== CALIBRATION =====================
// ——— Visual alignment ———
export const ROPE_BASELINE_FROM_BOTTOM = 0.30; // 0..1 (от низа огня до базовой линии троса)
export const CAT_BASELINE_ABOVE_ROPE_PX = 0;   // px (лапки кота над базовой линией троса)
export const DEFAULT_SENSITIVITY = 27;         // px (порог срабатывания от уровня троса)
export const INFER_EVERY_MS = 70;              // частота инференса позы (~14 Гц)

// ——— Sprite sizing ———
export const CAT_BASE_WIDTH_PX = 64;    // базовая логическая ширина спрайтов кота
export const CAT_GLOBAL_SCALE = 1.5;    // глобальный масштаб всех спрайтов кота
export const CAT_PER_STATE_SCALE = { idle:1.00, attached:1.00, falling:1.00, landing:1.00, seated:0.90 };
export const CAT_Y_NUDGE_PX     = { idle:0,    attached:0,    falling:0,    landing:0,    seated:0 };

// ——— Rope scaling (X/Y) ———
export const ROPE_SCALE_X = 1.2;  // ширина fire.png относительно ширины экрана
export const ROPE_SCALE_Y = 1.0;  // вертикальное растяжение fire.png

// ——— Delayed drop ———
export const DROP_TRAVEL_BELOW_PX = 22; // на сколько «пронести» ниже порога
export const DROP_MIN_TIME_MS     = 500; // сколько времени держать ниже порога

// ——— Landing ———
export const CAT_LAND_DURATION_MS = 220; // длительность спрайта приземления (7.png)

// ——— Milestones (декларативно) ———
/** actions:
 *  - seatedSwap: временная анимация сидящих котов (dance/chant)
 *  - fireworks: простые частицы фейерверка сверху
 *  - confetti: простые частицы конфетти сверху
 *  - groupOverlay: плоский оверлей (можно заменить PNG-групповую сцену)
 */
export const MILESTONES = [
  { at: 5,  action: "seatedSwap", style: "dance",  durationMs: 4000 },
  { at: 10, action: "seatedSwap", style: "chant",  durationMs: 5000 },
  { at: 15, action: "fireworks",  durationMs: 2500, count: 18 },
  { at: 20, action: "groupOverlay", label: "METAL BAND", durationMs: 3000 },
  { at: 30, action: "confetti",   durationMs: 3000, density: "high" },
];
// =======================================================

// ===================== SPRITES =========================
const SPRITES = {
  rope: "/assets/fire.png",
  cat_idle: "/assets/1.png",
  cat_attached: "/assets/2.png",
  cat_jump: "/assets/3.png",
  cat_seated: "/assets/4.png",
  cat_land: "/assets/7.png", // landing sprite
};
// =======================================================

// ===================== POSE EDGES ======================
const MOVENET_EDGES = {
  0:[0,1],1:[1,3],2:[0,2],3:[2,4],4:[5,7],5:[7,9],6:[6,8],7:[8,10],8:[5,6],9:[5,11],10:[6,12],11:[11,12],12:[11,13],13:[13,15],14:[12,14],15:[14,16]
};
// =======================================================

// Simple rep FSM (used mainly for stability; counting is by cats)
function updateRepState(prev, isAbove, now, minAboveMs = 160, minIntervalMs = 420){
  const next={...prev}; let counted=0;
  if(prev.phase==='down'&&isAbove){ next.phase='up'; next.lastAbove=now; }
  else if(prev.phase==='up'&&!isAbove){
    if(now-prev.lastAbove>minAboveMs && now-prev.lastRepAt>minIntervalMs){ next.phase='down'; next.lastRepAt=now; counted=1; }
    else { next.phase='down'; }
  }
  return {next, counted};
}

export default function PullUpRescueV63(){
  // ===== Refs & state =====
  const videoRef = useRef(null); const baseRef=useRef(null); const uiRef=useRef(null); const recRef=useRef(null);
  const detectorRef=useRef(null); const rafRef=useRef(0); const streamRef=useRef(null);
  const inferCanvasRef=useRef(document.createElement('canvas')); const inferMapRef=useRef({s:1, ix:0, iy:0});

  const [camReady,setCamReady]=useState(false);
  const [bucketMap,setBucketMap]=useState({ultra:null,wide:null,front:null});
  const bucketMapRef = useRef({ultra:null,wide:null,front:null});
  useEffect(()=>{ bucketMapRef.current = bucketMap; }, [bucketMap]);
  const [bucketChoice,setBucketChoice]=useState('ultra');
  const [msg,setMsg]=useState('Drag the rope to the bar height'); const [debug,setDebug]=useState('');
  const [recording,setRecording]=useState(false); const recordingRef=useRef(false);
  const [canRecord,setCanRecord]=useState(false);
  const [barY,setBarY]=useState(null); const barYRef=useRef(null);
  const [sensitivity,setSensitivity]=useState(DEFAULT_SENSITIVITY); const sensitivityRef=useRef(DEFAULT_SENSITIVITY);
  const [showPose,setShowPose]=useState(true);

  const [saved,setSaved]=useState(0); const savedRef=useRef(0); 
  useEffect(()=>{ savedRef.current=saved; },[saved]);
  const repRef=useRef({phase:'down',lastAbove:0,lastRepAt:0});
  const geomRef=useRef({ W:0,H:0,vw:0,vh:0,scale:1,dx:0,dy:0, mirrored:false });

  const catRef=useRef({ mode:'idle', x:0, y:0, vx:0, vy:0, lastT:0, attachDy:0, belowStart:0, maxDepthBelow:0, landUntil:0 });
  const seatedCatsRef=useRef([]);

  const lastInferRef=useRef(0); const lastPoseRef=useRef(null);

  // ===== Load sprites =====
  const [imgs,setImgs]=useState({});
  useEffect(()=>{
    const entries=Object.entries(SPRITES); const loaded={}; let left=entries.length;
    entries.forEach(([k,src])=>{ const im=new Image(); im.onload=()=>{ loaded[k]=im; left--; if(left===0) setImgs(loaded); }; im.src=src; });
  },[]);

  // ===== Helpers for labels =====
  const isFrontLabel=(label='')=>/front|user|face/i.test(label);
  const isUltraLabel=(label='')=>/ultra\s*wide|0\.5x|ultra/i.test(label);
  const isTeleLabel =(label='')=>/tele|2x|3x|zoom/i.test(label);

  // ===== Mirrors: keep refs updated =====
  useEffect(()=>{ barYRef.current=barY; },[barY]);
  useEffect(()=>{ sensitivityRef.current=sensitivity; },[sensitivity]);

  // ===== Recording capability detection =====
  useEffect(()=>{ 
    const c=document.createElement('canvas'); 
    setCanRecord(typeof c.captureStream==='function' && 'MediaRecorder' in window); 
  },[]);

  // ===== Cleanup on unmount =====
  useEffect(() => {
    return () => {
      // Clean up camera stream
      if(streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      
      // Clean up RAF
      if(rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      
      // Clean up media recorder
      if(mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop();
        } catch(e) {
          console.warn('Error stopping recorder on cleanup:', e);
        }
      }
    };
  }, []);

  // ===== Resize handling =====
  useEffect(()=>{
    const resize=()=>{
      const dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1));
      const W=window.innerWidth, H=window.innerHeight;
      [baseRef.current,uiRef.current,recRef.current].forEach(cv=>{
        if(!cv) return; cv.style.width=W+'px'; cv.style.height=H+'px'; cv.width=Math.floor(W*dpr); cv.height=Math.floor(H*dpr);
      });
      if (uiRef.current && barYRef.current==null) {
        const mid=Math.floor(uiRef.current.height*0.5); setBarY(mid);
      }
      updateGeom(); if(!catRef.current.lastT) spawnCatCentered();
    };
    resize(); 
    window.addEventListener('resize',resize);
    return()=>window.removeEventListener('resize',resize);
  },[]);

  function updateGeom(){
    const video=videoRef.current, base=baseRef.current; 
    if(!video||!base) return;
    
    const W=base.width,H=base.height; 
    const vw=video.videoWidth||1280, vh=video.videoHeight||720;
    
    if(vw <= 0 || vh <= 0) {
      console.warn('Invalid video dimensions:', {vw, vh});
      return;
    }
    
    const scale=Math.max(W/vw,H/vh); 
    const dw=vw*scale, dh=vh*scale; 
    const dx=(W-dw)/2, dy=(H-dh)/2;
    
    geomRef.current={...geomRef.current, W,H,vw,vh,scale,dx,dy};
    
    const inW=320; 
    const inH=Math.round(inW*vh/vw)||240; 
    const c=inferCanvasRef.current; 
    if(c) {
      c.width=inW; 
      c.height=inH;
    }
  }
  function setMirrorFromStream(stream){
    try{ 
      const track=stream.getVideoTracks?.()[0]; 
      if(!track) return;
      
      const s=track?.getSettings?.()||{}; 
      const label=track?.label||'';
      const mirrored = s.facingMode ? /user|front/i.test(s.facingMode) : isFrontLabel(label);
      geomRef.current={...geomRef.current, mirrored};
    }catch(e){
      console.warn('Failed to detect mirror setting:', e);
      // Default to not mirrored
      geomRef.current={...geomRef.current, mirrored: false};
    }
  }

  // ===== Detector =====
  async function createMoveNetDetector(){
    await tf.setBackend('webgl'); await tf.ready();
    const opts=[ {modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING}, {modelType:poseDetection.movenet.modelType.SINGLEPOSE_THUNDER} ];
    let lastErr; for(const o of opts){ try{ return await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,o);}catch(e){ lastErr=e; } }
    throw lastErr;
  }
  async function ensureDetector(){ if(!detectorRef.current) detectorRef.current=await createMoveNetDetector(); }

  // ===== Camera enumerate & switch =====
  function toBuckets(devices){
    const vids=devices.filter(d=>d.kind==='videoinput');
    const fronts=vids.filter(d=>isFrontLabel(d.label||''));
    const backs = vids.filter(d=>!isFrontLabel(d.label||''));
    const ultra = vids.find(d=>isUltraLabel(d.label||''));
    const wide  = backs.find(d=>!isTeleLabel(d.label||'')) || backs[0] || null;
    return { ultra: ultra?.deviceId||null, wide: wide?.deviceId||null, front: fronts[0]?.deviceId||null };
  }

  async function enableCamera(){
    setDebug('');
    try{
      await ensureDetector();
      const pre=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      const devices=await navigator.mediaDevices.enumerateDevices();
      pre.getTracks().forEach(t=>t.stop());
      const buckets=toBuckets(devices); 
      setBucketMap(buckets); 
      bucketMapRef.current=buckets;
      const preferred=buckets.ultra?'ultra':(buckets.wide?'wide':'front');
      await switchToBucket(preferred, buckets);
      setCamReady(true);
      restartRAF();
    }catch(e){ 
      console.error('Camera initialization failed:', e); 
      setMsg('Camera init failed.'); 
      setDebug(`${e.name||'Error'}: ${e.message||e}`); 
    }
  }

  async function switchToBucket(bucket, mapOverride){
    try{
      const map = mapOverride || bucketMapRef.current || {};
      const deviceId=map[bucket];
      const baseConstraints={ width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:30, max:60} };
      let stream;
      
      if(deviceId){ 
        try{ 
          stream=await navigator.mediaDevices.getUserMedia({ 
            video:{ deviceId:{ exact: deviceId }, ...baseConstraints }, 
            audio:false 
          }); 
        }catch(e){ 
          setDebug(`Exact ${bucket} failed: ${e.name}`); 
        } 
      }
      
      if(!stream){ 
        try{ 
          stream=await navigator.mediaDevices.getUserMedia({ video: baseConstraints, audio:false }); 
        }catch(e){ 
          try {
            stream=await navigator.mediaDevices.getUserMedia({ video:true, audio:false }); 
          } catch(e2) {
            throw new Error(`Failed to get camera stream: ${e2.message}`);
          }
        } 
      }
      
      if(streamRef.current) {
        streamRef.current.getTracks().forEach(t=>t.stop());
      }
      
      const v=videoRef.current; 
      if(!v) throw new Error('Video element not found');
      
      v.srcObject=stream; 
      try{ 
        await v.play(); 
      }catch(e){
        console.warn('Video play failed:', e);
      }
      
      streamRef.current=stream; 
      setMirrorFromStream(stream); 
      updateGeom();
      
      const label = stream.getVideoTracks?.()[0]?.label || '';
      const finalBucket = /front|user|face/i.test(label) ? 'front' : (/ultra\s*wide|0\.5x|ultra/i.test(label) ? 'ultra' : 'wide');
      setBucketChoice(finalBucket);
      return finalBucket;
    }catch(e){ 
      console.error('Camera switch failed:', e);
      setDebug(`Switch failed: ${e.name || 'Unknown error'}`); 
      return bucket; 
    }
  }

  // ===== Milestones state/engine =====
  const firedMilestonesRef = useRef(new Set());
  const effectsRef = useRef({
    fireworks: [],
    confetti: [],
    overlay: { active:false, label:"", until:0 },
    seatedStyle: { mode: "default", until:0 },
  });
  function triggerMilestones(score){
    try {
      for(const rule of MILESTONES){
        if(score>=rule.at && !firedMilestonesRef.current.has(rule.at)){
          firedMilestonesRef.current.add(rule.at);
          runMilestone(rule);
        }
      }
    } catch(e) {
      console.error('Trigger milestones failed:', e);
    }
  }
  
  function runMilestone(rule){
    try {
      const now=performance.now(); 
      const fx=effectsRef.current;
      switch(rule.action){
        case 'seatedSwap': 
          fx.seatedStyle.mode = rule.style||'dance'; 
          fx.seatedStyle.until = now + (rule.durationMs||3000); 
          break;
        case 'fireworks' : 
          startFireworks(rule.count||14, rule.durationMs||2500); 
          break;
        case 'confetti'  : 
          startConfetti(rule.density==='high'? 220:120, rule.durationMs||3000); 
          break;
        case 'groupOverlay': 
          fx.overlay.active=true; 
          fx.overlay.label=rule.label||'GROUP'; 
          fx.overlay.until= now + (rule.durationMs||2500); 
          break;
        default: 
          break;
      }
    } catch(e) {
      console.error('Run milestone failed:', e);
    }
  }

  // ===== Particles =====
  function startFireworks(count,durationMs){
    try {
      const now=performance.now(); 
      const arr=effectsRef.current.fireworks; 
      const u=uiRef.current; 
      if(!u) return;
      
      const W=u.width,H=u.height;
      for(let i=0;i<count;i++){
        const cx=Math.random()*W; 
        const cy=Math.random()*H*0.35; 
        const spokes=36;
        arr.push({type:'fw', cx, cy, spokes, born:now, ttl:durationMs});
      }
    } catch(e) {
      console.error('Start fireworks failed:', e);
    }
  }
  
  function startConfetti(count,durationMs){
    try {
      const now=performance.now(); 
      const arr=effectsRef.current.confetti; 
      const u=uiRef.current; 
      if(!u) return;
      
      const W=u.width,H=u.height;
      for(let i=0;i<count;i++){
        const x=Math.random()*W; 
        const vy=(H/(durationMs/1000))*(0.6+Math.random()*0.6); 
        const rot=(Math.random()*Math.PI); 
        const size=6+Math.random()*10;
        arr.push({type:'cf', x, y:-20, vy, rot, born:now, ttl:durationMs, size});
      }
    } catch(e) {
      console.error('Start confetti failed:', e);
    }
  }
  function stepAndDrawEffects(ctx){
    try {
      const now=performance.now(); 
      const fx=effectsRef.current; 
      const u=uiRef.current; 
      if(!u) return; 
      
      const W=u.width,H=u.height;
      ctx.save();
      
      // fireworks
      for(let i=fx.fireworks.length-1;i>=0;i--){
        const p=fx.fireworks[i]; 
        const t=(now-p.born)/p.ttl; 
        if(t>=1){ 
          fx.fireworks.splice(i,1); 
          continue; 
        }
        const R=80*(0.3+t)*window.devicePixelRatio; 
        ctx.globalAlpha=1-Math.min(1,t);
        for(let k=0;k<p.spokes;k++){
          const a=(k/p.spokes)*Math.PI*2; 
          const x=p.cx+Math.cos(a)*R; 
          const y=p.cy+Math.sin(a)*R;
          ctx.beginPath(); 
          ctx.arc(x,y,2,0,Math.PI*2); 
          ctx.fillStyle='rgba(255,200,80,0.9)'; 
          ctx.fill();
        }
      }
      
      // confetti
      for(let i=fx.confetti.length-1;i>=0;i--){
        const c=fx.confetti[i]; 
        const t=(now-c.born)/c.ttl; 
        if(t>=1 || c.y>H+40){ 
          fx.confetti.splice(i,1); 
          continue; 
        }
        c.y+=c.vy*(1/60); 
        c.rot+=0.2; 
        ctx.save(); 
        ctx.translate(c.x,c.y); 
        ctx.rotate(c.rot);
        ctx.fillStyle='rgba(255,255,255,.9)'; 
        ctx.fillRect(-c.size/2,-c.size/4,c.size,c.size/2);
        ctx.restore();
      }
      
      // overlay
      if(fx.overlay.active){
        if(now>=fx.overlay.until){ 
          fx.overlay.active=false; 
        }
        else {
          const pad=20*(window.devicePixelRatio||1);
          ctx.fillStyle='rgba(0,0,0,.35)'; 
          ctx.fillRect(pad,pad,W-2*pad, 80*(window.devicePixelRatio||1));
          ctx.font=`${36*(window.devicePixelRatio||1)}px system-ui`; 
          ctx.fillStyle='#fff';
          ctx.fillText(fx.overlay.label, pad+20*(window.devicePixelRatio||1), pad+56*(window.devicePixelRatio||1));
        }
      }
      ctx.restore();
    } catch(e) {
      console.error('Step and draw effects failed:', e);
    }
  }

  // ===== RAF watchdog =====
  const heartbeatRef = useRef(0);
  function restartRAF(){ 
    cancelAnimationFrame(rafRef.current||0); 
    rafRef.current = requestAnimationFrame(tick); 
  }
  
  useEffect(()=>{
    const id=setInterval(()=>{
      const last=heartbeatRef.current||0; 
      if(performance.now()-last>700){ restartRAF(); }
    }, 800);
    return ()=>clearInterval(id);
  },[]);
  
  useEffect(()=>{
    const onVis=()=>{ 
      if(!document.hidden && camReady){ restartRAF(); } 
    };
    document.addEventListener('visibilitychange', onVis);
    return ()=>document.removeEventListener('visibilitychange', onVis);
  },[camReady]);

  // ===== Main tick =====
  const tick = async ()=>{
    heartbeatRef.current = performance.now();
    try{
      const video=videoRef.current, base=baseRef.current, ui=uiRef.current; 
      if(!video||!base||!ui) return;
      
      const b=base.getContext('2d'); 
      const u=ui.getContext('2d');
      if(!b || !u) return;
      
      const {W,H,vw,vh,scale,dx,dy,mirrored}=geomRef.current; 
      const inC=inferCanvasRef.current; 
      if(!W) updateGeom();

      // camera → base
      b.clearRect(0,0,W,H); const dw=vw*scale, dh=vh*scale;
      if(mirrored){ b.save(); b.translate(W,0); b.scale(-1,1); b.drawImage(video, -dx - dw + W, dy, dw, dh); b.restore(); }
      else { b.drawImage(video, dx, dy, dw, dh); }

      // UI clear
      u.clearRect(0,0,W,H);

      // inference only while recording (экономим ресурсы)
      const now=performance.now();
      if(recordingRef.current && detectorRef.current){
        if(now - lastInferRef.current >= INFER_EVERY_MS){
          lastInferRef.current=now; const ic=inC.getContext('2d');
          const s=Math.max(inC.width/vw, inC.height/vh); const iw=vw*s, ih=vh*s; const ix=(inC.width-iw)/2, iy=(inC.height-ih)/2;
          inferMapRef.current={s,ix,iy}; ic.clearRect(0,0,inC.width,inC.height); ic.drawImage(video, ix, iy, iw, ih);
          try{ 
            const poses=await detectorRef.current.estimatePoses(inC,{maxPoses:1,flipHorizontal:false}); 
            lastPoseRef.current = (poses && poses[0]) ? poses[0] : null; 
          }catch(e){
            console.warn('Pose detection failed:', e);
            lastPoseRef.current = null;
          }
        }
        const pose = lastPoseRef.current;
        if(pose){
          const kps=pose.keypoints; const {s,ix,iy}=inferMapRef.current;
          const mapped=kps.map((kp)=>{ const xv=(kp.x-ix)/s; const yv=(kp.y-iy)/s; const xd = dx + (mirrored ? (vw - xv) : xv) * scale; const yd = dy + yv * scale; return {X:xd,Y:yd,score:kp.score, raw:{xv,yv}}; });
          if(showPose){ u.save(); u.strokeStyle='rgba(255,255,255,.9)'; u.lineWidth=2; drawPoseMapped(u,mapped); u.restore(); }

          const nose=mapped[0]; const by=barYRef.current; const sens=sensitivityRef.current;
          if(nose?.score>0.4 && by!==null){
            const thr=by - sens; const above = nose.Y <= thr;
            const {next}=updateRepState(repRef.current,above,now); repRef.current=next;

            if(catRef.current.mode==='idle' && above){
              catRef.current.mode='attached';
              catRef.current.attachDy = (by - CAT_BASELINE_ABOVE_ROPE_PX*(window.devicePixelRatio||1)) - nose.Y;
            }
            if(catRef.current.mode==='attached'){
              const targetY = nose.Y + catRef.current.attachDy; catRef.current.y += (targetY - catRef.current.y) * 0.45; catRef.current.x = nose.X;
              if(above){ catRef.current.belowStart=0; catRef.current.maxDepthBelow=0; }
              else {
                if(!catRef.current.belowStart){ catRef.current.belowStart=now; catRef.current.maxDepthBelow=0; }
                const depth=Math.max(0, nose.Y - thr); if(depth>catRef.current.maxDepthBelow) catRef.current.maxDepthBelow=depth;
                const timeEnough=(now-catRef.current.belowStart)>=DROP_MIN_TIME_MS; const travelEnough=catRef.current.maxDepthBelow>=DROP_TRAVEL_BELOW_PX;
                if(timeEnough && travelEnough){ startCatFall(); }
              }
            }
          }
        }
      }

      // overlays
      drawRopeSprite(u,W,H,barYRef.current,imgs.rope);
      drawThreshold(u,W,H,barYRef.current,sensitivityRef.current);
      drawSeatedCats(u,imgs);
      drawActiveCat(u,imgs);

      // effects layer
      drawEffectsLayer(u);

      // HUD
      drawSavedCounter(u,W,H,savedRef.current);
    }catch(e){
      console.error(e); setDebug(`Tick error: ${e?.message||e}`);
    }finally{
      rafRef.current = requestAnimationFrame(tick);
    }
  };

  function drawEffectsLayer(ctx){
    try {
      const fx=effectsRef.current; 
      const now=performance.now();
      if(fx.seatedStyle.until && now>=fx.seatedStyle.until){ 
        fx.seatedStyle.mode='default'; 
        fx.seatedStyle.until=0; 
      }
      stepAndDrawEffects(ctx);
    } catch(e) {
      console.error('Draw effects layer failed:', e);
    }
  }

  function drawPoseMapped(ctx,kps){
    const edges = MOVENET_EDGES;
    for(const [i,j] of Object.values(edges)){
      const a=kps[i],b=kps[j]; if(a?.score>0.3&&b?.score>0.3){ ctx.beginPath(); ctx.moveTo(a.X,a.Y); ctx.lineTo(b.X,b.Y); ctx.stroke(); }
    }
    for(const k of kps){ if(k.score>0.3){ ctx.beginPath(); ctx.arc(k.X,k.Y,3,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.95)'; ctx.fill(); } }
  }

  function drawRopeSprite(ctx,W,H,y,img){
    if(!img||y==null) return;
    const fullW=W*ROPE_SCALE_X; const scaleW=fullW/img.width; const renderW=fullW;
    const renderH0=img.height*scaleW; const renderH=renderH0*ROPE_SCALE_Y;
    const baseline=renderH*(1-ROPE_BASELINE_FROM_BOTTOM);
    const yTop=Math.round(y - baseline); const xLeft=Math.round((W - renderW)/2);
    ctx.drawImage(img, xLeft, yTop, renderW, renderH);
  }

  function catWidthPx(state){ const dpr=window.devicePixelRatio||1; const local=(CAT_PER_STATE_SCALE[state] ?? 1)*CAT_GLOBAL_SCALE; return CAT_BASE_WIDTH_PX * local * dpr; }
  function catHeightFor(img, w){ return w * (img.height/img.width); }

  function spawnCatCentered(){
    try {
      const u=uiRef.current; 
      if(!u) return; 
      
      const p=window.devicePixelRatio||1; 
      const W=u.width; 
      const y=barYRef.current ?? Math.floor(u.height*0.5);
      
      catRef.current={ 
        mode:'idle', 
        x:Math.floor(W/2), 
        y:y - CAT_BASELINE_ABOVE_ROPE_PX*p, 
        vx:0, 
        vy:0, 
        lastT:performance.now(), 
        attachDy:0, 
        belowStart:0, 
        maxDepthBelow:0, 
        landUntil:0 
      };
    } catch(e) {
      console.error('Spawn cat failed:', e);
    }
  }
  
  function startCatFall(){ 
    try {
      const now=performance.now(); 
      const c=catRef.current; 
      c.mode='falling'; 
      c.vx=(Math.random() * 2 - 1) * 24; 
      c.vy=0; 
      c.lastT=now; 
      c.belowStart=0; 
      c.maxDepthBelow=0; 
    } catch(e) {
      console.error('Start cat fall failed:', e);
    }
  }
  function stepActiveCat(){
    try {
      const u=uiRef.current; 
      if(!u) return; 
      
      const p=window.devicePixelRatio||1; 
      const H=u.height; 
      const groundY=H-28*p;
      
      const c=catRef.current; 
      const now=performance.now(); 
      const dt=Math.min(0.05,(now-c.lastT)/1000); 
      c.lastT=now;
      
      if(c.mode==='falling'){
        const g=1200*p; 
        c.vy += g*dt; 
        c.y += c.vy*dt; 
        c.x += c.vx*dt;
        if(c.y >= groundY){ 
          c.y=groundY; 
          c.mode='landing'; 
          c.landUntil = now + CAT_LAND_DURATION_MS; 
        }
      } else if(c.mode==='landing'){
        if(now >= c.landUntil){
          c.mode='seated'; 
          const seat = placeSeatedCat(u); 
          seatedCatsRef.current.push(seat);
          savedRef.current = savedRef.current + 1; 
          setSaved(v=>v+1); 
          triggerMilestones(savedRef.current);
          setTimeout(()=>{ 
            try {
              spawnCatCentered(); 
            } catch(e) {
              console.error('Spawn cat in timeout failed:', e);
            }
          }, 250);
        }
      }
    } catch(e) {
      console.error('Step active cat failed:', e);
    }
  }

  function placeSeatedCat(u){
    try {
      const p=window.devicePixelRatio||1; 
      const W=u.width; 
      const H=u.height; 
      const margin=20*p; 
      const spacing=56*p; 
      const baseY=H-28*p;
      
      const count=seatedCatsRef.current.length; 
      const maxPerRow=Math.floor((W-2*margin)/spacing);
      const row=Math.floor(count/maxPerRow); 
      const col=count%maxPerRow;
      
      const x=margin + col*spacing; 
      const y=baseY - row*spacing*0.75; 
      
      return {x,y};
    } catch(e) {
      console.error('Place seated cat failed:', e);
      // Return a safe default position
      return {x: 100, y: 100};
    }
  }

  function drawSeatedCats(ctx,imgs){
    const im = imgs.cat_seated; if(!im) return; const w = catWidthPx('seated'); const h = catHeightFor(im,w);
    const fx=effectsRef.current; const mode=fx.seatedStyle.mode; const t=performance.now()/1000;
    for(const s of seatedCatsRef.current){
      let yN=0; if(mode==='dance'){ yN = Math.sin(t*6 + s.x*0.01) * 6 * (window.devicePixelRatio||1); } else if(mode==='chant'){ yN = Math.sin(t*3 + s.x*0.02) * 2 * (window.devicePixelRatio||1); }
      const yNudge=(CAT_Y_NUDGE_PX.seated||0)*(window.devicePixelRatio||1) + yN;
      ctx.drawImage(im, Math.round(s.x - w/2), Math.round(s.y - h + yNudge), w, h);
    }
  }

  function drawActiveCat(ctx,imgs){
    const c=catRef.current; stepActiveCat(); if(!c) return; if(c.mode==='seated') return; // не рисуем seated на активном слое (исключает idle‑моргание)
    const state = c.mode==='attached' ? 'attached' : (c.mode==='falling' ? 'falling' : (c.mode==='landing' ? 'landing' : 'idle'));
    const im = state==='attached' ? imgs.cat_attached : (state==='falling' ? imgs.cat_jump : (state==='landing' ? imgs.cat_land : imgs.cat_idle)); if(!im) return;
    const w = catWidthPx(state); const h = catHeightFor(im,w); const yNudge=(CAT_Y_NUDGE_PX[state]||0)*(window.devicePixelRatio||1);
    ctx.drawImage(im, Math.round(c.x - w/2), Math.round(c.y - h + yNudge), w, h);
  }

  function drawSavedCounter(ctx,W,H,val){
    const p=window.devicePixelRatio||1; const pad=10*p; const boxW=180*p, boxH=56*p; const x=(W-boxW)/2, y=pad;
    ctx.save(); ctx.fillStyle='rgba(0,0,0,.35)'; ctx.beginPath(); const r=12*p; roundRect(ctx,x,y,boxW,boxH,r); ctx.fill();
    ctx.font=`${14*p}px system-ui`; ctx.fillStyle='#fff'; ctx.fillText('Saved', x+14*p, y+20*p);
    ctx.font=`${28*p}px system-ui`; ctx.fillText(`${val}`, x+14*p, y+44*p); ctx.restore();
  }
  function roundRect(ctx,x,y,w,h,r){ ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); }

  function drawThreshold(ctx,W,H,barY,sensitivity){
    if(barY==null) return; const p=window.devicePixelRatio||1; const thr=barY - sensitivity;
    ctx.save(); ctx.setLineDash([16*p, 10*p]); ctx.lineWidth=4*p; ctx.strokeStyle='#00ff88';
    ctx.beginPath(); ctx.moveTo(0,thr); ctx.lineTo(W,thr); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(W-40*p,thr,10*p,0,Math.PI*2); ctx.fillStyle='#00ff88'; ctx.fill(); ctx.restore();
  }

  // ===== Interactions (drag rope) =====
  const draggingRef=useRef(false);
  function onPointerDown(e){ 
    try {
      const y=getY(e); 
      if(y==null) return; 
      setBarY(y); 
      if(catRef.current.mode==='idle') alignCatToBar(); 
      draggingRef.current=true; 
    } catch(e) {
      console.error('Pointer down failed:', e);
    }
  }
  
  function onPointerMove(e){ 
    try {
      if(!draggingRef.current) return; 
      const y=getY(e); 
      if(y==null) return; 
      setBarY(y); 
      if(catRef.current.mode==='idle') alignCatToBar(); 
    } catch(e) {
      console.error('Pointer move failed:', e);
    }
  }
  
  function onPointerUp(){ 
    try {
      draggingRef.current=false; 
    } catch(e) {
      console.error('Pointer up failed:', e);
    }
  }
  
  function alignCatToBar(){ 
    try {
      const p=window.devicePixelRatio||1; 
      catRef.current.y = (barYRef.current ?? 0) - CAT_BASELINE_ABOVE_ROPE_PX*p; 
    } catch(e) {
      console.error('Align cat failed:', e);
    }
  }
  
  function onTouchStart(e){ 
    try {
      e.preventDefault(); 
      onPointerDown(e); 
    } catch(e) {
      console.error('Touch start failed:', e);
    }
  }
  
  function onTouchMove(e){ 
    try {
      e.preventDefault(); 
      onPointerMove(e); 
    } catch(e) {
      console.error('Touch move failed:', e);
    }
  }
  
  function onTouchEnd(e){ 
    try {
      e.preventDefault(); 
      onPointerUp(); 
    } catch(e) {
      console.error('Touch end failed:', e);
    }
  }
  
  function getY(e){ 
    try {
      const rect=uiRef.current.getBoundingClientRect(); 
      const dpr=uiRef.current.width/rect.width; 
      if(e.touches&&e.touches[0]) return (e.touches[0].clientY-rect.top)*dpr; 
      if(typeof e.clientY==='number') return (e.clientY-rect.top)*dpr; 
      return null; 
    } catch(e) {
      console.error('Get Y failed:', e);
      return null;
    }
  }

  // ===== Recording =====
  const mediaRecorderRef=useRef(null); const recordedChunksRef=useRef([]);
  function isiOSSafari(){ return /iP(hone|ad|od)/.test(navigator.userAgent) && /Safari\//.test(navigator.userAgent) && !/CriOS|FxiOS/.test(navigator.userAgent); }
  function pickMime(){
    const prefer = isiOSSafari() ? ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4'] : [];
    const fall=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
    const opts=[...prefer,...fall];
    for(const o of opts){ try{ if(window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o; }catch{} }
    return '';
  }
  function startRecording(){
    if(!canRecord) return; 
    
    const rec=recRef.current, base=baseRef.current, ui=uiRef.current; 
    if(!rec||!base||!ui) return;
    
    const r=rec.getContext('2d'); 
    if(!r) return;
    
    rec.width=base.width; 
    rec.height=base.height; 
    
    const mime=pickMime(); 
    if(!mime) {
      setMsg('No supported video format found');
      return;
    }
    
    recordedChunksRef.current=[]; 
    setRecording(true); 
    recordingRef.current=true;
    
    const targetFps=30; 
    const compose=()=>{ 
      if(!recordingRef.current) return; 
      r.clearRect(0,0,rec.width,rec.height); 
      r.drawImage(base,0,0); 
      r.drawImage(ui,0,0); 
      requestAnimationFrame(compose); 
    }; 
    requestAnimationFrame(compose);
    
    const stream=rec.captureStream(targetFps); 
    let mr; 
    try{ 
      mr=new MediaRecorder(stream,{mimeType:mime, videoBitsPerSecond: 6_000_000}); 
    }catch(e){ 
      console.warn('High bitrate failed, using default:', e);
      mr=new MediaRecorder(stream,{mimeType:mime}); 
    }
    
    mr.ondataavailable=(e)=>{ 
      if(e.data && e.data.size>0) recordedChunksRef.current.push(e.data); 
    };
    
    mr.onstop=()=>{ 
      const type=mr.mimeType||mime; 
      const blob=new Blob(recordedChunksRef.current,{type}); 
      if(!blob || blob.size<150000){ 
        setMsg('Recording tiny — iOS codec limited. Try again or use iOS Screen Recording.'); 
        return; 
      } 
      const url=URL.createObjectURL(blob); 
      const a=document.createElement('a'); 
      a.href=url; 
      const ts=new Date().toISOString().replace(/[:.]/g,'-'); 
      a.download=`pullup-rescue-${ts}.${type.includes('mp4')?'mp4':'webm'}`; 
      a.click(); 
      setMsg('Recording saved'); 
    };
    
    mediaRecorderRef.current=mr; 
    try{ 
      mr.start(500); 
    }catch(e){
      console.warn('500ms timeslice failed, using default:', e);
      mr.start(); 
    }
  }
  function stopRecording(){ 
    if(mediaRecorderRef.current && mediaRecorderRef.current.state!=='inactive') {
      try {
        mediaRecorderRef.current.stop(); 
      } catch(e) {
        console.error('Error stopping recording:', e);
      }
    }
    setRecording(false); 
    recordingRef.current=false; 
  }

  // ===== Render =====
  return (
    <div style={{position:'fixed',inset:0,background:'#000',color:'#fff',overflow:'hidden'}}>
      <video ref={videoRef} playsInline muted style={{display:'none'}} onLoadedMetadata={()=>{ 
        try {
          updateGeom(); 
          if(streamRef.current) setMirrorFromStream(streamRef.current); 
        } catch(e) {
          console.error('Video metadata error:', e);
        }
      }} />

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
            <select value={bucketChoice} onChange={async(e)=>{ 
              try {
                const b=e.target.value; 
                await switchToBucket(b); 
              } catch(e) {
                console.error('Camera switch failed:', e);
                setMsg('Camera switch failed. Please try again.');
              }
            }} style={{background:'rgba(255,255,255,.12)',color:'#fff',border:0,borderRadius:10,padding:'6px'}}>
              {bucketMapRef.current.ultra && <option value="ultra" style={{color:'#000'}}>Back — Ultra‑Wide (0.5×)</option>}
              {bucketMapRef.current.wide && <option value="wide" style={{color:'#000'}}>Back — Wide (1×)</option>}
              {bucketMapRef.current.front && <option value="front" style={{color:'#000'}}>Front</option>}
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
          <button onClick={()=>{
            try {
              savedRef.current=0; 
              setSaved(0);
              repRef.current={phase:'down',lastAbove:0,lastRepAt:0};
              seatedCatsRef.current=[]; 
              firedMilestonesRef.current=new Set();
              effectsRef.current={ 
                fireworks:[], 
                confetti:[], 
                overlay:{active:false,label:'',until:0}, 
                seatedStyle:{mode:'default',until:0} 
              };
              spawnCatCentered(); 
              restartRAF();
            } catch(e) {
              console.error('Reset failed:', e);
              setMsg('Reset failed. Please try again.');
            }
          }} style={btn()}>Reset</button>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto',alignItems:'center',gap:8}}>
          <Labeled label={`Sensitivity (px above rope): ${sensitivity}`}>
            <input type="range" min={8} max={80} step={1} value={sensitivity} onChange={(e)=>{ 
              try {
                const v=parseInt(e.target.value,10); 
                if(!isNaN(v)) {
                  setSensitivity(v); 
                }
              } catch(e) {
                console.error('Sensitivity change failed:', e);
              }
            }} style={{width:'100%'}} />
          </Labeled>
          <label style={{display:'flex',gap:6,alignItems:'center',fontSize:12,opacity:.85}}>
            <input type="checkbox" checked={showPose} onChange={(e)=>{
              try {
                setShowPose(e.target.checked);
              } catch(e) {
                console.error('Show pose toggle failed:', e);
              }
            }} /> Show pose
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
