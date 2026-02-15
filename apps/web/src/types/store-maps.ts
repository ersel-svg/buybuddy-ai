// ===========================================
// Store Map Types
// ===========================================

// --- API Response Types (match backend models) ---

export interface StoreMap {
  id: number;
  store_id: number;
  name: string;
  base_ratio: number;
  grid: number;
  // Aggregate counts from list endpoint
  area_count?: number;
  zone_count?: number;
  floor_count?: number;
  scanner_count?: number;
  // Enriched from GET single
  store_name?: string;
  floor_plan_url?: string;
}

export interface MapFloor {
  id: number;
  map_id: number;
  floor: number;
  floor_plan: string; // S3 URL
  // Aggregate counts from list endpoint
  area_count?: number;
  zone_count?: number;
  scanner_count?: number;
}

export interface MapArea {
  id: number;
  name: string;
  floor_id: number;
  updated_at?: string;
}

export interface AreaCoordinate {
  id: number;
  area_id: number;
  x: number;
  y: number;
  z: number;
  r: number;
  circle: boolean;
  updated_at?: string;
}

// --- Nested hierarchy from GET /v1/map/area/coordinate ---

export interface CoordinateHierarchy {
  map: StoreMap;
  floors: FloorWithAreas[];
}

export interface FloorWithAreas extends MapFloor {
  areas: AreaWithCoordinates[];
}

export interface AreaWithCoordinates extends MapArea {
  coordinates: AreaCoordinate[];
}

// --- Request Types ---

export interface CreateMapRequest {
  store_id: number;
  name: string;
  base_ratio?: number;
  grid?: number;
}

export interface UpdateMapRequest {
  name?: string;
  base_ratio?: number;
  grid?: number;
}

export interface CreateFloorRequest {
  floor: number;
  file: File;
}

export interface CreateAreaRequest {
  name: string;
  floor_id: number;
}

export interface UpdateAreaRequest {
  id: number;
  name?: string;
}

export interface CreateCoordinateRequest {
  area_id: number;
  x: number;
  y: number;
  z?: number;
  r?: number;
  circle?: boolean;
}

export interface UpdateCoordinateRequest {
  id: number;
  x?: number;
  y?: number;
  z?: number;
  r?: number;
}

export interface CoordinateListFilters {
  map_id?: number;
  store_id?: number;
  area_id?: number;
  floor_id?: number;
}

// ===========================================
// Frontend-only Types (Canvas, Editor, Measurement)
// ===========================================

// --- Area Types (visual classification) ---

export type AreaType =
  | "shelf"
  | "aisle"
  | "checkout"
  | "storage"
  | "entrance"
  | "promo"
  | "cold_storage"
  | "custom";

export interface AreaTypeConfig {
  type: AreaType;
  label: string;
  color: string;
  bgColor: string;
  icon: string; // lucide icon name
}

export const AREA_TYPE_CONFIGS: Record<AreaType, AreaTypeConfig> = {
  shelf: {
    type: "shelf",
    label: "Reyon",
    color: "#3B82F6",
    bgColor: "rgba(59, 130, 246, 0.15)",
    icon: "ShoppingCart",
  },
  aisle: {
    type: "aisle",
    label: "Koridor",
    color: "#6B7280",
    bgColor: "rgba(107, 114, 128, 0.15)",
    icon: "ArrowRightLeft",
  },
  checkout: {
    type: "checkout",
    label: "Kasa",
    color: "#10B981",
    bgColor: "rgba(16, 185, 129, 0.15)",
    icon: "CreditCard",
  },
  storage: {
    type: "storage",
    label: "Depo",
    color: "#F97316",
    bgColor: "rgba(249, 115, 22, 0.15)",
    icon: "Warehouse",
  },
  entrance: {
    type: "entrance",
    label: "Giris/Cikis",
    color: "#EF4444",
    bgColor: "rgba(239, 68, 68, 0.15)",
    icon: "DoorOpen",
  },
  promo: {
    type: "promo",
    label: "Promosyon",
    color: "#EAB308",
    bgColor: "rgba(234, 179, 8, 0.15)",
    icon: "Tag",
  },
  cold_storage: {
    type: "cold_storage",
    label: "Soguk Depo",
    color: "#0EA5E9",
    bgColor: "rgba(14, 165, 233, 0.15)",
    icon: "Snowflake",
  },
  custom: {
    type: "custom",
    label: "Ozel Alan",
    color: "#8B5CF6",
    bgColor: "rgba(139, 92, 246, 0.15)",
    icon: "Square",
  },
};

// --- Measurement Types ---

export type UnitSystem = "metric" | "imperial";

export type DimensionDisplay = "all" | "selected" | "none";

export type SnapPrecision = "full" | "half" | "quarter";

export interface MeasurementSettings {
  unitSystem: UnitSystem;
  showDimensions: DimensionDisplay;
  snapToGrid: boolean;
  snapPrecision: SnapPrecision;
  showRulers: boolean;
  showGrid: boolean;
  gridOpacity: number;
  majorGridInterval: number; // Every N minor grid lines
}

export const DEFAULT_MEASUREMENT_SETTINGS: MeasurementSettings = {
  unitSystem: "metric",
  showDimensions: "selected",
  snapToGrid: true,
  snapPrecision: "full",
  showRulers: true,
  showGrid: true,
  gridOpacity: 0.3,
  majorGridInterval: 2,
};

// --- Canvas Editor Types ---

export type ToolMode =
  | "select"
  | "rectangle"
  | "polygon"
  | "circle"
  | "measure"
  | "calibrate"
  | "pan";

export interface CanvasAreaObject {
  fabricId: string;
  areaId?: number; // undefined until saved to API
  name: string;
  areaType: AreaType;
  floorId: number;
  coordinates: { x: number; y: number }[];
  isCircle: boolean;
  radius?: number;
  products?: AreaProductMapping[];
  categories?: string[];
}

// --- Product Mapping Types ---

export interface ProductRef {
  product_id: string;
  name: string;
  sku?: string;
  barcode?: string;
  image_url?: string;
}

export interface AreaProductMapping {
  area_id: number;
  products: ProductRef[];
}

// --- Editor State ---

export interface EditorHistoryEntry {
  areas: CanvasAreaObject[];
  description: string;
}
