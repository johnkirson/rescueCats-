// Calibration and tuning constants

export const ROPE_BASELINE_FROM_BOTTOM = 0.30; // 0..1 (от низа огня до базовой линии троса)
export const CAT_BASELINE_ABOVE_ROPE_PX = 0;   // px (лапки кота над базовой линией троса)
export const DEFAULT_SENSITIVITY = 27;         // px (порог срабатывания от уровня троса)
export const INFER_EVERY_MS = 70;              // частота инференса позы (~14 Гц)

// Sprite sizing
export const CAT_BASE_WIDTH_PX = 64;    // базовая логическая ширина спрайтов кота
export const CAT_GLOBAL_SCALE = 1.5;    // глобальный масштаб всех спрайтов кота
export const CAT_PER_STATE_SCALE = { idle:1.00, attached:1.00, falling:1.00, landing:1.00, seated:0.80 };
export const CAT_Y_NUDGE_PX     = { idle:0,    attached:0,    falling:0,    landing:0,    seated:0 };

// Rope scaling (X/Y)
export const ROPE_SCALE_X = 1.2;  // ширина fire.png относительно ширины экрана
export const ROPE_SCALE_Y = 1.0;  // вертикальное растяжение fire.png

// Delayed drop
export const DROP_TRAVEL_BELOW_PX = 22; // на сколько «пронести» ниже порога
export const DROP_MIN_TIME_MS     = 500; // сколько времени держать ниже порога

// Landing
export const CAT_LAND_DURATION_MS = 220; // длительность спрайта приземления (7.png)


