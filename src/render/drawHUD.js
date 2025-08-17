export function drawSavedCounter(ctx,W,H,val){
	try {
		if(!ctx||W<=0||H<=0) return;
		const p=window.devicePixelRatio||1; 
		const pad=10*p; 
		const boxW=180*p, boxH=56*p; 
		const x=(W-boxW)/2, y=pad;
		ctx.save(); 
		ctx.fillStyle='rgba(0,0,0,.35)'; 
		ctx.beginPath(); 
		const r=12*p; 
		roundRect(ctx,x,y,boxW,boxH,r); 
		ctx.fill();
		ctx.font=`${14*p}px system-ui`; 
		ctx.fillStyle='#fff'; 
		ctx.textAlign='center';
		ctx.fillText('Saved', x+boxW/2, y+20*p);
		ctx.font=`${28*p}px system-ui`; 
		ctx.fillText(`${val}`, x+boxW/2, y+44*p); 
		ctx.restore();
	} catch(e) {
		console.error('Draw saved counter failed:', e);
	}
}

export function roundRect(ctx,x,y,w,h,r){ 
	ctx.moveTo(x+r,y); 
	ctx.arcTo(x+w,y,x+w,y+h,r); 
	ctx.arcTo(x+w,y+h,x,y+h,r); 
	ctx.arcTo(x,y+h,x,y,r); 
	ctx.arcTo(x,y,x+w,y,r); 
}


