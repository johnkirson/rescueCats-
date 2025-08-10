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
  
  // Welcome screen state
  const [currentScreen, setCurrentScreen] = useState('welcome'); // 'welcome', 'game', 'results'
  const [playerName, setPlayerName] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
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
  const heartbeatRef=useRef(0);

  // ===== Load sprites =====
  const [imgs,setImgs]=useState({});
  useEffect(()=>{
    const entries=Object.entries(SPRITES); const loaded={}; let left=entries.length;
    entries.forEach(([k,src])=>{ const im=new Image(); im.onload=()=>{ loaded[k]=im; left--; if(left===0) setImgs(loaded); }; im.src=src; });
  },[]);

  // ===== Auto-start camera when sprites are loaded =====
  useEffect(() => {
    // Remove automatic camera start - let user control it
    // if (Object.keys(imgs).length > 0 && !camReady) {
    //   // Auto-start camera after sprites are loaded
    //   setTimeout(() => {
    //     enableCamera();
    //   }, 100);
    // }
  }, [imgs, camReady]);

  // ===== Auto-start RAF for UI rendering =====
  useEffect(() => {
    // Start RAF immediately to show UI
    restartRAF();
  }, []);

  // ===== Screen management functions =====
  const startGame = () => {
    setCurrentScreen('game');
    // Wait for the game screen to render before starting camera
    setTimeout(() => {
      try {
        // Ensure RAF is running first
        if (!rafRef.current) {
          restartRAF();
        }
        // Then start camera
        enableCamera();
      } catch(e) {
        console.error('Failed to start game:', e);
      }
    }, 300);
  };

  const showResults = () => {
    setCurrentScreen('results');
  };

  const backToWelcome = () => {
    setCurrentScreen('welcome');
  };

  const handleLogin = () => {
    if (playerName.trim()) {
      setIsLoggedIn(true);
      setMsg(`Welcome, ${playerName}!`);
    }
  };

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
      // Clean up state
      setMediaRecorder(null);
    };
  }, []);

  // ===== Game screen initialization =====
  useEffect(() => {
    if (currentScreen === 'game') {
      // Ensure proper initialization when game screen becomes active
      setTimeout(() => {
        try {
          // Force resize to update canvas dimensions
          const resizeEvent = new Event('resize');
          window.dispatchEvent(resizeEvent);
          
          // Ensure RAF is running
          if (!rafRef.current) {
            restartRAF();
          }
        } catch(e) {
          console.error('Game screen initialization failed:', e);
        }
      }, 100);
    }
  }, [currentScreen]);

  // ===== Resize handling =====
  useEffect(()=>{
    const resize=()=>{
      const dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1));
      const W=window.innerWidth, H=window.innerHeight;
      
      // Update canvas dimensions
      [baseRef.current,uiRef.current,recRef.current].forEach(cv=>{
        if(!cv) return; 
        cv.style.width=W+'px'; 
        cv.style.height=H+'px'; 
        cv.width=Math.floor(W*dpr); 
        cv.height=Math.floor(H*dpr);
      });
      
      // Set barY if not already set
      if (barYRef.current==null) {
        const mid=Math.floor((uiRef.current?.height || H)*0.5); 
        setBarY(mid);
        barYRef.current = mid;
      }
      
      // Update geometry
      updateGeom(); 
      
      // Spawn cat if it doesn't exist and sprites are loaded
      if(!catRef.current.lastT && Object.keys(imgs).length > 0 && barYRef.current !== null) {
        try {
          spawnCatCentered();
        } catch(e) {
          console.error('Spawn cat in resize failed:', e);
        }
      }
      
      // Ensure RAF is running
      if (!rafRef.current) {
        restartRAF();
      }
    };
    resize(); 
    window.addEventListener('resize',resize);
    return()=>window.removeEventListener('resize',resize);
  },[]);

  function updateGeom(){
    try {
      const video=videoRef.current, base=baseRef.current; 
      if(!video||!base) return;
      
      const W=base.width,H=base.height; 
      
      // Check if video is ready
      if (!video.videoWidth || !video.videoHeight) {
        // Use default dimensions if video is not ready
        const vw=1280, vh=720;
        const scale=Math.max(W/vw,H/vh); 
        const dw=vw*scale, dh=vh*scale; 
        const dx=(W-dw)/2, dy=(H-dh)/2;
        
        geomRef.current={...geomRef.current, W,H,vw,vh,scale,dx,dy};
        return;
      }
      
      const vw=video.videoWidth, vh=video.videoHeight;
      
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
    } catch(e) {
      console.error('Update geometry failed:', e);
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
    // Prevent multiple calls
    if (camReady) return;
    
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
      
      // Ensure barY is set if not already set
      if (barYRef.current === null) {
        const mid = Math.floor((uiRef.current?.height || window.innerHeight) * 0.5);
        setBarY(mid);
        barYRef.current = mid;
        // Wait a bit for the state to update
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setCamReady(true);
      
      // Ensure all elements are properly initialized
      setTimeout(() => {
        try {
          updateGeom();
          // Ensure RAF is running
          if (!rafRef.current) {
          restartRAF();
          }
          // Spawn cat after everything is ready
          if (Object.keys(imgs).length > 0 && barYRef.current !== null) {
            spawnCatCentered();
          }
        } catch(e) {
          console.error('Post-camera initialization failed:', e);
        }
      }, 300);
    }catch(e){ 
      console.error('Camera initialization failed:', e); 
      setMsg('Camera init failed.'); 
      setDebug(`${e.name||'Error'}: ${e.message||e}`); 
    }
  }

  async function switchToBucket(bucket, mapOverride){
    try{
      // Don't switch if camera is not ready
      if (!camReady && !mapOverride) return bucket;
      
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
      
      try {
        setMirrorFromStream(stream); 
        updateGeom();
      } catch(e) {
        console.warn('Failed to set mirror or update geometry:', e);
      }
      
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
  function restartRAF(){ 
    // Allow starting RAF even without camera to show UI
    try {
      cancelAnimationFrame(rafRef.current||0); 
      rafRef.current = requestAnimationFrame(tick); 
    } catch(e) {
      console.error('Restart RAF failed:', e);
    }
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
      b.clearRect(0,0,W,H); 
      
      // Only draw video if it's ready
      if (video.videoWidth && video.videoHeight) {
        const dw=vw*scale, dh=vh*scale;
        if(mirrored){ b.save(); b.translate(W,0); b.scale(-1,1); b.drawImage(video, -dx - dw + W, dy, dw, dh); b.restore(); }
        else { b.drawImage(video, dx, dy, dw, dh); }
      } else {
        // Draw placeholder when video is not ready
        b.fillStyle = 'rgba(0,0,0,0.5)';
        b.fillRect(0, 0, W, H);
        b.fillStyle = 'rgba(255,255,255,0.3)';
        b.font = '24px Arial';
        b.textAlign = 'center';
        b.fillText('Camera initializing...', W/2, H/2);
      }

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
      // Draw game elements when sprites are loaded (even without camera)
      if(Object.keys(imgs).length > 0) {
        // Only draw rope and threshold if barY and sensitivity are set
        if (barYRef.current !== null && sensitivityRef.current !== null) {
          try {
            drawRopeSprite(u,W,H,barYRef.current,imgs.rope);
            drawThreshold(u,W,H,barYRef.current,sensitivityRef.current);
          } catch(e) {
            console.warn('Failed to draw rope/threshold:', e);
          }
        }
        
        try {
          drawSeatedCats(u,imgs);
          drawActiveCat(u,imgs);
        } catch(e) {
          console.warn('Failed to draw cats:', e);
        }
        
        // Ensure cat exists if sprites are loaded and barY is set
        if (!catRef.current.lastT && barYRef.current !== null) {
          try {
            spawnCatCentered();
          } catch(e) {
            console.error('Spawn cat in tick failed:', e);
          }
        }
      }

      // effects layer
      if(camReady) {
        try {
          drawEffectsLayer(u);
        } catch(e) {
          console.warn('Failed to draw effects:', e);
        }
      }

      // HUD
        try {
          drawSavedCounter(u,W,H,savedRef.current);
        } catch(e) {
          console.warn('Failed to draw HUD:', e);
      }
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
    try {
      if(!img||y==null||!ctx||W<=0||H<=0) return;
      if(!img.width||!img.height) return;
      
      const fullW=W*ROPE_SCALE_X; const scaleW=fullW/img.width; const renderW=fullW;
      const renderH0=img.height*scaleW; const renderH=renderH0*ROPE_SCALE_Y;
      const baseline=renderH*(1-ROPE_BASELINE_FROM_BOTTOM);
      const yTop=Math.round(y - baseline); const xLeft=Math.round((W - renderW)/2);
      ctx.drawImage(img, xLeft, yTop, renderW, renderH);
    } catch(e) {
      console.error('Draw rope sprite failed:', e);
    }
  }

  function catWidthPx(state){ const dpr=window.devicePixelRatio||1; const local=(CAT_PER_STATE_SCALE[state] ?? 1)*CAT_GLOBAL_SCALE; return CAT_BASE_WIDTH_PX * local * dpr; }
  function catHeightFor(img, w){ return w * (img.height/img.width); }

  function spawnCatCentered(){
    try {
      // Allow spawning if sprites are loaded and barY is set
      if (Object.keys(imgs).length === 0) return;
      
      const u=uiRef.current; 
      if(!u) return; 
      
      const p=window.devicePixelRatio||1; 
      const W=u.width; 
      const y=barYRef.current ?? Math.floor(u.height*0.5);
      
      // Don't spawn cat if barY is not set
      if (y === null) return;
      
      // Allow spawning if cat doesn't exist OR if cat is in 'seated' mode (ready for next)
      if (catRef.current && catRef.current.lastT && catRef.current.mode !== 'seated') return;
      
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
      
      const c=catRef.current; 
      if(!c || !c.lastT) return; // Don't step if cat doesn't exist
      
      const p=window.devicePixelRatio||1; 
      const H=u.height; 
      const groundY=H-28*p;
      
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
          
          // Reset cat state to allow spawning new cat
          c.lastT = 0;
          c.mode = 'idle';
          c.x = 0;
          c.y = 0;
          c.vx = 0;
          c.vy = 0;
          c.attachDy = 0;
          c.belowStart = 0;
          c.maxDepthBelow = 0;
          c.landUntil = 0;
          
          // Spawn new cat after a short delay
          setTimeout(()=>{ 
            try {
              if (Object.keys(imgs).length > 0) {
                spawnCatCentered(); 
              }
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
      if (!u || !u.width || !u.height) {
        console.warn('Invalid UI element for placing seated cat');
        return {x: 100, y: 100};
      }
      
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
    try {
      const im = imgs.cat_seated; if(!im) return; const w = catWidthPx('seated'); const h = catHeightFor(im,w);
      const fx=effectsRef.current; const mode=fx.seatedStyle.mode; const t=performance.now()/1000;
      for(const s of seatedCatsRef.current){
        let yN=0; if(mode==='dance'){ yN = Math.sin(t*6 + s.x*0.01) * 6 * (window.devicePixelRatio||1); } else if(mode==='chant'){ yN = Math.sin(t*3 + s.x*0.02) * 2 * (window.devicePixelRatio||1); }
        const yNudge=(CAT_Y_NUDGE_PX.seated||0)*(window.devicePixelRatio||1) + yN;
        ctx.drawImage(im, Math.round(s.x - w/2), Math.round(s.y - h + yNudge), w, h);
      }
    } catch(e) {
      console.error('Draw seated cats failed:', e);
    }
  }

  function drawActiveCat(ctx,imgs){
    try {
      const c=catRef.current; 
      if(!c || !c.lastT) return; // Don't draw if cat doesn't exist
      stepActiveCat(); 
      if(c.mode==='seated') return; // не рисуем seated на активном слое (исключает idle‑моргание)
      const state = c.mode==='attached' ? 'attached' : (c.mode==='falling' ? 'falling' : (c.mode==='landing' ? 'landing' : 'idle'));
      const im = state==='attached' ? imgs.cat_attached : (state==='falling' ? imgs.cat_jump : (state==='landing' ? imgs.cat_land : imgs.cat_idle)); if(!im) return;
      const w = catWidthPx(state); const h = catHeightFor(im,w); const yNudge=(CAT_Y_NUDGE_PX[state]||0)*(window.devicePixelRatio||1);
      ctx.drawImage(im, Math.round(c.x - w/2), Math.round(c.y - h + yNudge), w, h);
    } catch(e) {
      console.error('Draw active cat failed:', e);
    }
  }

  function drawSavedCounter(ctx,W,H,val){
    try {
      if(!ctx||W<=0||H<=0) return;
      const p=window.devicePixelRatio||1; const pad=10*p; const boxW=180*p, boxH=56*p; const x=(W-boxW)/2, y=pad;
    ctx.save(); ctx.fillStyle='rgba(0,0,0,.35)'; ctx.beginPath(); const r=12*p; roundRect(ctx,x,y,boxW,boxH,r); ctx.fill();
    ctx.font=`${14*p}px system-ui`; ctx.fillStyle='#fff'; ctx.fillText('Saved', x+14*p, y+20*p);
    ctx.font=`${28*p}px system-ui`; ctx.fillText(`${val}`, x+14*p, y+44*p); ctx.restore();
    } catch(e) {
      console.error('Draw saved counter failed:', e);
    }
  }
  function roundRect(ctx,x,y,w,h,r){ 
    ctx.moveTo(x+r,y); 
    ctx.arcTo(x+w,y,x+w,y+h,r); 
    ctx.arcTo(x+w,y+h,x,y+h,r); 
    ctx.arcTo(x,y+h,x,y,r); 
    ctx.arcTo(x,y,x+w,y,r); 
  }

  function drawThreshold(ctx,W,H,barY,sensitivity){
    try {
      if(barY==null||!ctx||W<=0||H<=0) return; 
      const p=window.devicePixelRatio||1; const thr=barY - sensitivity;
      ctx.save(); ctx.setLineDash([16*p, 10*p]); ctx.lineWidth=4*p; ctx.strokeStyle='#00ff88';
      ctx.beginPath(); ctx.moveTo(0,thr); ctx.lineTo(W,thr); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(W-40*p,thr,10*p,0,Math.PI*2); ctx.fillStyle='#00ff88'; ctx.fill(); ctx.restore();
    } catch(e) {
      console.error('Draw threshold failed:', e);
    }
  }

  // ===== Interactions (drag rope) =====
  const draggingRef=useRef(false);
  function onPointerDown(e){ 
    try {
      // Only allow interaction when camera is ready
      if (!camReady) return;
      
      const y=getY(e); 
      if(y==null) return; 
      setBarY(y); 
      if(catRef.current && catRef.current.mode==='idle') alignCatToBar(); 
      draggingRef.current=true; 
    } catch(e) {
      console.error('Pointer down failed:', e);
    }
  }
  
  function onPointerMove(e){ 
    try {
      if(!draggingRef.current || !camReady) return; 
      const y=getY(e); 
      if(y==null) return; 
      setBarY(y); 
      if(catRef.current && catRef.current.mode==='idle') alignCatToBar(); 
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
      const c=catRef.current;
      if(!c || !c.lastT) return; // Don't align if cat doesn't exist
      
      const p=window.devicePixelRatio||1; 
      c.y = (barYRef.current ?? 0) - CAT_BASELINE_ABOVE_ROPE_PX*p; 
    } catch(e) {
      console.error('Align cat failed:', e);
    }
  }
  
  function onTouchStart(e){ 
    try {
      // Only allow touch interaction when camera is ready
      if (!camReady) return;
      
      e.preventDefault(); 
      onPointerDown(e); 
    } catch(e) {
      console.error('Touch start failed:', e);
    }
  }
  
  function onTouchMove(e){ 
    try {
      // Only allow touch interaction when camera is ready
      if (!camReady) return;
      
      e.preventDefault(); 
      onPointerMove(e); 
    } catch(e) {
      console.error('Touch move failed:', e);
    }
  }
  
  function onTouchEnd(e){ 
    try {
      // Only allow touch interaction when camera is ready
      if (!camReady) return;
      
      e.preventDefault(); 
      onPointerUp(); 
    } catch(e) {
      console.error('Touch end failed:', e);
    }
  }
  
  function getY(e){ 
    try {
      const ui = uiRef.current;
      if (!ui || !ui.width) return null;
      
      const rect = ui.getBoundingClientRect(); 
      if (!rect || rect.width === 0) return null;
      
      const dpr = ui.width/rect.width; 
      if(e.touches && e.touches[0]) return (e.touches[0].clientY-rect.top)*dpr; 
      if(typeof e.clientY==='number') return (e.clientY-rect.top)*dpr; 
      return null; 
    } catch(e) {
      console.error('Get Y failed:', e);
      return null;
    }
  }

  // ===== Recording =====
  const mediaRecorderRef=useRef(null); const recordedChunksRef=useRef([]);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  function isiOSSafari(){ return /iP(hone|ad|od)/.test(navigator.userAgent) && /Safari\//.test(navigator.userAgent) && !/CriOS|FxiOS/.test(navigator.userAgent); }
  function pickMime(){
    const prefer = isiOSSafari() ? ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4'] : [];
    const fall=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
    const opts=[...prefer,...fall];
    for(const o of opts){ try{ if(window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o; }catch{} }
    return '';
  }
  function startRecording(){
    if(!canRecord || !camReady) return; 
    
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
    setMediaRecorder(mr);
    try{ 
      mr.start(500); 
    }catch(e){
      console.warn('500ms timeslice failed, using default:', e);
      mr.start(); 
    }
  }
  function stopRecording(){ 
    if(!camReady) return;
    
    if(mediaRecorderRef.current && mediaRecorderRef.current.state!=='inactive') {
      try {
        mediaRecorderRef.current.stop(); 
      } catch(e) {
        console.error('Error stopping recording:', e);
      }
    }
    setRecording(false); 
    recordingRef.current=false; 
    setMediaRecorder(null);
  }

  // ===== Render =====
  return (
    <div style={{position:'fixed',inset:0,background:'#000',color:'#fff',overflow:'hidden'}}>
      {/* Welcome Screen */}
      {currentScreen === 'welcome' && (
        <div style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          boxSizing: 'border-box'
        }}>
          {/* App Logo */}
          <div style={{
            width: '120px',
            height: '120px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #3b82f6, #1e3a8a)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '24px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
            border: '3px solid rgba(255,255,255,0.2)'
          }}>
            <span style={{ fontSize: '60px' }}>🐱</span>
          </div>

          {/* App Title */}
          <h1 style={{
            fontSize: 'clamp(28px, 6vw, 36px)',
            fontWeight: 'bold',
            color: '#ffffff',
            margin: '0 0 16px 0',
            textAlign: 'center',
            textShadow: '0 2px 8px rgba(0,0,0,0.3)'
          }}>
            Pull-Up Rescue
          </h1>

          {/* App Description */}
          <p style={{
            fontSize: 'clamp(16px, 4vw, 18px)',
            color: 'rgba(255,255,255,0.9)',
            margin: '0 0 40px 0',
            textAlign: 'center',
            lineHeight: '1.5',
            maxWidth: '400px'
          }}>
            Спаси котиков, выполняя<br />
            подтягивания!
          </p>

          {/* Login Section */}
          {!isLoggedIn && (
            <div style={{
              width: '100%',
              maxWidth: '320px',
              marginBottom: '32px'
            }}>
              <input
                type="text"
                placeholder="Введите ваше имя"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                style={{
                  width: '100%',
                  padding: '16px',
                  borderRadius: '12px',
                  border: 'none',
                  fontSize: '16px',
                  marginBottom: '16px',
                  boxSizing: 'border-box',
                  boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
                }}
              />
              <button
                onClick={handleLogin}
                style={{
                  width: '100%',
                  padding: '16px',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #10b981, #059669)',
                  color: '#ffffff',
                  fontSize: '16px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 6px 20px rgba(16, 185, 129, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = '0 4px 16px rgba(16, 185, 129, 0.3)';
                }}
              >
                Войти
              </button>
            </div>
          )}

          {/* Main Navigation Buttons */}
          <div style={{
            width: '100%',
            maxWidth: '320px',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px'
          }}>
            <button
              onClick={startGame}
              style={{
                width: '100%',
                padding: '18px',
                borderRadius: '12px',
                border: 'none',
                background: 'linear-gradient(135deg, #f59e0b, #d97706)',
                color: '#ffffff',
                fontSize: '18px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 16px rgba(245, 158, 11, 0.3)'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 6px 20px rgba(245, 158, 11, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 16px rgba(245, 158, 11, 0.3)';
              }}
            >
              Начать игру
            </button>

            <button
              onClick={showResults}
              style={{
                width: '100%',
                padding: '18px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.3)',
                background: 'rgba(255,255,255,0.1)',
                color: '#ffffff',
                fontSize: '18px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.background = 'rgba(255,255,255,0.2)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.background = 'rgba(255,255,255,0.1)';
              }}
            >
              Посмотреть результаты
            </button>
          </div>
        </div>
      )}

      {/* Results Screen */}
      {currentScreen === 'results' && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          textAlign: 'center'
        }}>
          <h1 style={{
            fontSize: '32px',
            fontWeight: 'bold',
            margin: '0 0 30px 0',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Результаты
          </h1>

          {/* Personal Results */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '16px',
            marginBottom: '20px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)',
            width: '100%',
            maxWidth: '320px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Личные результаты</h3>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#22c55e' }}>
              {saved} котов спасено
            </div>
            <div style={{ fontSize: '14px', opacity: 0.7, marginTop: '8px' }}>
              Лучший результат: 15 котов
            </div>
          </div>

          {/* Overall Results */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '20px',
            borderRadius: '16px',
            marginBottom: '30px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.2)',
            width: '100%',
            maxWidth: '320px'
          }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Общие результаты</h3>
            <div style={{ fontSize: '16px', lineHeight: '1.6' }}>
              <div>Всего игроков: 1,247</div>
              <div>Всего спасено: 8,943 кота</div>
              <div>Рекорд: 42 кота (Игрок123)</div>
            </div>
          </div>

          <button
            onClick={backToWelcome}
            style={{
              padding: '14px 24px',
              borderRadius: '10px',
              background: 'rgba(255,255,255,0.1)',
              color: '#fff',
              fontSize: '16px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              border: '1px solid rgba(255,255,255,0.2)'
            }}
          >
            Назад
          </button>
        </div>
      )}

      {/* Game Screen */}
      {currentScreen === 'game' && (
        <>
          {/* Simple Game Interface - Full Screen Camera */}
          <video
            ref={videoRef}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100vw',
              height: '100vh',
              objectFit: 'cover',
              zIndex: 1
            }}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={() => { 
              try {
                updateGeom(); 
                if(streamRef.current) setMirrorFromStream(streamRef.current); 
              } catch(e) {
                console.error('Video metadata error:', e);
              }
            }}
          />
          
          <canvas
            ref={baseRef}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100vw',
              height: '100vh',
              zIndex: 2,
              touchAction: 'none'
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          />
          
          <canvas
            ref={uiRef}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100vw',
              height: '100vh',
              zIndex: 3,
              touchAction: 'none'
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          />

          <canvas ref={recRef} style={{ display: 'none' }} />

          {/* Simple Controls - Semi-transparent */}
          <div style={{
            position: 'fixed',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            alignItems: 'center'
          }}>
            {/* Main Control Buttons */}
            <div style={{
              display: 'flex',
              gap: '10px',
              justifyContent: 'center'
            }}>
              {!camReady ? (
                <button onClick={enableCamera} style={{
                  ...btn(1, '#22c55e'),
                  padding: '12px 24px',
                  fontSize: '16px'
                }}>
                  Начать игру
                </button>
              ) : !recording ? (
                <button onClick={startRecording} style={{
                  ...btn(1, '#ef4444'),
                  padding: '12px 24px',
                  fontSize: '16px'
                }}>
                  Начать запись
                </button>
              ) : (
                <button onClick={stopRecording} style={{
                  ...btn(1, '#f59e0b'),
                  padding: '12px 24px',
                  fontSize: '16px'
                }}>
                  Остановить запись
                </button>
              )}

              <button onClick={() => {
                try {
                  setSaved(0);
                  setMsg('Счетчик сброшен');
                  if (catRef.current) {
                    catRef.current.lastT = 0;
                    catRef.current.mode = 'idle';
                  }
                  restartRAF();
                } catch(e) {
                  console.error('Reset failed:', e);
                }
              }} style={{
                ...btn(1, '#6b7280'),
                padding: '12px 24px',
                fontSize: '16px'
              }}>
                Сброс
              </button>
            </div>

            {/* Sensitivity and Options */}
            {camReady && (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                alignItems: 'center'
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <span style={{ color: '#ffffff', fontSize: '14px' }}>
                    Чувствительность: {sensitivity}
                  </span>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={sensitivity}
                    onChange={(e) => setSensitivity(parseInt(e.target.value))}
                    style={{
                      width: '100px',
                      height: '4px',
                      borderRadius: '2px',
                      background: 'rgba(255,255,255,0.3)',
                      outline: 'none',
                      cursor: 'pointer'
                    }}
                  />
                </div>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  color: '#ffffff',
                  fontSize: '14px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showPose}
                    onChange={(e) => {
                      try {
                        setShowPose(e.target.checked);
                      } catch(e) {
                        console.error('Show pose toggle failed:', e);
                      }
                    }}
                    style={{
                      width: '16px',
                      height: '16px',
                      cursor: 'pointer'
                    }}
                  />
                  Показать позу
                </label>
              </div>
            )}

            {/* Messages */}
            <div style={{
              fontSize: '14px',
              opacity: 0.9,
              textAlign: 'center',
              color: '#ffffff',
              maxWidth: '300px',
              wordWrap: 'break-word'
            }}>
              {msg}
            </div>
            
            {debug && (
              <div style={{
                fontSize: '12px',
                opacity: 0.7,
                textAlign: 'center',
                color: '#ffffff',
                userSelect: 'all',
                maxWidth: '300px',
                wordWrap: 'break-word'
              }}>
                {debug}
              </div>
            )}

            {/* Back to Menu Button */}
            <button
              onClick={backToWelcome}
              style={{
                ...btn(0.8, 'rgba(0,0,0,0.3)'),
                padding: '8px 16px',
                fontSize: '14px',
                marginTop: '10px'
              }}
            >
              Меню
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function btn(opacity=1,bg){ return {border:0,borderRadius:14,padding:'10px 12px',background:bg||'rgba(255,255,255,.12)',color:'#fff',opacity,backdropFilter:'saturate(120%) blur(6px)'}; }
function Labeled({label,children}){ return (<div><div style={{fontSize:11,opacity:.75,marginBottom:4}}>{label}</div>{children}</div>); }
