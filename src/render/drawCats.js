import { CAT_PER_STATE_SCALE, CAT_GLOBAL_SCALE, CAT_BASE_WIDTH_PX, CAT_Y_NUDGE_PX } from "../config/calibration";

export function catWidthPx(state){ const dpr=window.devicePixelRatio||1; const local=(CAT_PER_STATE_SCALE[state] ?? 1)*CAT_GLOBAL_SCALE; return CAT_BASE_WIDTH_PX * local * dpr; }
export function catHeightFor(img, w){ return w * (img.height/img.width); }

export function drawSeatedCats(ctx, imgs, seatedCats){
	if(!ctx||!seatedCats||seatedCats.length===0) return;
	try {
		seatedCats.forEach((cat)=>{
			const w=catWidthPx('seated');
			const h=catHeightFor(imgs.cat_seated,w);
			ctx.drawImage(imgs.cat_seated, Math.round(cat.x - w/2), Math.round(cat.y - h), w, h);
		});
	} catch(e) {
		console.error('Draw seated cats failed:', e);
	}
}

export function drawActiveCat(ctx, imgs, catRef, stepActiveCat){
	try {
		const c=catRef.current; 
		if(!c || !c.lastT) return; 
		stepActiveCat(); 
		if(c.mode==='seated') return; 
		const state = c.mode==='attached' ? 'attached' : (c.mode==='falling' ? 'falling' : (c.mode==='landing' ? 'landing' : 'idle'));
		const im = state==='attached' ? imgs.cat_attached : (state==='falling' ? imgs.cat_jump : (state==='landing' ? imgs.cat_land : imgs.cat_idle)); if(!im) return;
		const w = catWidthPx(state); const h = catHeightFor(im,w); const yNudge=(CAT_Y_NUDGE_PX[state]||0)*(window.devicePixelRatio||1);
		ctx.drawImage(im, Math.round(c.x - w/2), Math.round(c.y - h + yNudge), w, h);
	} catch(e) {
		console.error('Draw active cat failed:', e);
	}
}


