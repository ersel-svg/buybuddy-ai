import type { UnitSystem } from "@/types/store-maps";

// ===========================================
// Unit Conversion Functions
// ===========================================

const CM_PER_FOOT = 30.48;
const CM_PER_INCH = 2.54;

/**
 * Convert centimeters to the display unit
 */
export function cmToDisplay(cm: number, unit: UnitSystem): number {
  if (unit === "imperial") {
    return cm / CM_PER_FOOT; // returns feet
  }
  return cm / 100; // returns meters
}

/**
 * Convert display unit to centimeters (internal storage)
 */
export function displayToCm(value: number, unit: UnitSystem): number {
  if (unit === "imperial") {
    return value * CM_PER_FOOT;
  }
  return value * 100; // meters to cm
}

/**
 * Format a measurement value with appropriate unit suffix
 */
export function formatMeasurement(
  cm: number,
  unit: UnitSystem,
  precision: number = 1
): string {
  if (unit === "imperial") {
    const totalInches = cm / CM_PER_INCH;
    const feet = Math.floor(totalInches / 12);
    const inches = totalInches % 12;
    if (feet === 0) return `${inches.toFixed(precision)}in`;
    if (inches < 0.5) return `${feet}ft`;
    return `${feet}ft ${inches.toFixed(0)}in`;
  }

  const meters = cm / 100;
  if (meters < 1) {
    return `${cm.toFixed(precision)}cm`;
  }
  return `${meters.toFixed(precision)}m`;
}

/**
 * Format area (square centimeters to m² or ft²)
 */
export function formatArea(sqCm: number, unit: UnitSystem, precision: number = 1): string {
  if (unit === "imperial") {
    const sqFt = sqCm / (CM_PER_FOOT * CM_PER_FOOT);
    return `${sqFt.toFixed(precision)} ft²`;
  }
  const sqM = sqCm / 10000;
  return `${sqM.toFixed(precision)} m²`;
}

// ===========================================
// Geometry Calculation Functions
// ===========================================

/**
 * Calculate distance between two points (in canvas pixels)
 */
export function distance(
  p1: { x: number; y: number },
  p2: { x: number; y: number }
): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

/**
 * Convert pixel distance to real-world cm using base_ratio
 */
export function pixelsToCm(pixels: number, baseRatio: number): number {
  return pixels * baseRatio;
}

/**
 * Convert real-world cm to pixel distance
 */
export function cmToPixels(cm: number, baseRatio: number): number {
  if (baseRatio === 0) return cm;
  return cm / baseRatio;
}

/**
 * Calculate polygon area (Shoelace formula) - returns in square pixels
 */
export function polygonArea(points: { x: number; y: number }[]): number {
  if (points.length < 3) return 0;
  let area = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += points[i].x * points[j].y;
    area -= points[j].x * points[i].y;
  }
  return Math.abs(area) / 2;
}

/**
 * Calculate polygon perimeter - returns in pixels
 */
export function polygonPerimeter(points: { x: number; y: number }[]): number {
  if (points.length < 2) return 0;
  let perimeter = 0;
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    perimeter += distance(points[i], points[j]);
  }
  return perimeter;
}

/**
 * Calculate circle area from radius in pixels
 */
export function circleArea(radiusPx: number): number {
  return Math.PI * radiusPx * radiusPx;
}

/**
 * Snap a value to the nearest grid point
 */
export function snapToGrid(
  value: number,
  gridSizePx: number,
  precision: "full" | "half" | "quarter" = "full"
): number {
  let snap = gridSizePx;
  if (precision === "half") snap = gridSizePx / 2;
  if (precision === "quarter") snap = gridSizePx / 4;
  return Math.round(value / snap) * snap;
}

// ===========================================
// Grid Size Presets
// ===========================================

export const METRIC_GRID_PRESETS = [
  { label: "25 cm", valueCm: 25 },
  { label: "50 cm", valueCm: 50 },
  { label: "1 m", valueCm: 100 },
  { label: "2 m", valueCm: 200 },
];

export const IMPERIAL_GRID_PRESETS = [
  { label: '6"', valueCm: 6 * CM_PER_INCH },
  { label: "1 ft", valueCm: CM_PER_FOOT },
  { label: "2 ft", valueCm: 2 * CM_PER_FOOT },
  { label: "5 ft", valueCm: 5 * CM_PER_FOOT },
];
