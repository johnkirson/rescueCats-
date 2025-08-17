import { MOVENET_EDGES } from "../config/constants";

export function drawPoseMapped(ctx,kps){
	const edges = MOVENET_EDGES;
	for(const [i,j] of Object.values(edges)){
		const a=kps[i],b=kps[j]; if(a?.score>0.3&&b?.score>0.3){ ctx.beginPath(); ctx.moveTo(a.X,a.Y); ctx.lineTo(b.X,b.Y); ctx.stroke(); }
	}
	for(const k of kps){ if(k.score>0.3){ ctx.beginPath(); ctx.arc(k.X,k.Y,3,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.95)'; ctx.fill(); } }
}


