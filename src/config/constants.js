// Milestones and pose edges constants

export const MILESTONES = [
	{ at: 5,  action: "seatedSwap", style: "dance",  durationMs: 4000 },
	{ at: 10, action: "seatedSwap", style: "chant",  durationMs: 5000 },
	{ at: 15, action: "fireworks",  durationMs: 2500, count: 18 },
	{ at: 20, action: "groupOverlay", label: "METAL BAND", durationMs: 3000 },
	{ at: 30, action: "confetti",   durationMs: 3000, density: "high" },
];

export const MOVENET_EDGES = {
	0:[0,1],1:[1,3],2:[0,2],3:[2,4],4:[5,7],5:[7,9],6:[6,8],7:[8,10],8:[5,6],9:[5,11],10:[6,12],11:[11,12],12:[11,13],13:[13,15],14:[12,14],15:[14,16]
};


