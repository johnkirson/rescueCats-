import { ROPE_BASELINE_FROM_BOTTOM, ROPE_SCALE_X, ROPE_SCALE_Y } from "../config/calibration";

export function drawRopeSprite(ctx, W, H, y, img){
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


