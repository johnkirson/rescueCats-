export function drawThreshold(ctx,W,H,barY,sensitivity){
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


