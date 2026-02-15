import { create } from "zustand";
import type {
  ToolMode,
  CanvasAreaObject,
  MeasurementSettings,
  AreaType,
} from "@/types/store-maps";
import { DEFAULT_MEASUREMENT_SETTINGS } from "@/types/store-maps";

// ===========================================
// Canvas Editor Store
// ===========================================

interface CanvasState {
  // Map info
  mapId: number | null;
  activeFloorId: number | null;

  // Tool
  activeTool: ToolMode;
  setActiveTool: (tool: ToolMode) => void;

  // Selection
  selectedAreaId: string | null; // fabricId
  setSelectedAreaId: (id: string | null) => void;

  // Areas
  areas: CanvasAreaObject[];
  addArea: (area: CanvasAreaObject) => void;
  updateArea: (fabricId: string, updates: Partial<CanvasAreaObject>) => void;
  removeArea: (fabricId: string) => void;
  setAreas: (areas: CanvasAreaObject[]) => void;

  // New area being created
  pendingAreaType: AreaType;
  setPendingAreaType: (type: AreaType) => void;

  // Floor management
  setActiveFloorId: (id: number | null) => void;
  setMapId: (id: number | null) => void;

  // Measurement
  measurement: MeasurementSettings;
  updateMeasurement: (updates: Partial<MeasurementSettings>) => void;

  // Canvas state
  zoom: number;
  setZoom: (zoom: number) => void;
  isCalibrated: boolean;
  setIsCalibrated: (v: boolean) => void;
  baseRatio: number; // cm per pixel
  setBaseRatio: (ratio: number) => void;
  gridSizeCm: number; // grid cell size in cm
  setGridSizeCm: (size: number) => void;

  // Undo/Redo
  history: CanvasAreaObject[][];
  historyIndex: number;
  pushHistory: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // Save state
  isDirty: boolean;
  setIsDirty: (v: boolean) => void;
  isSaving: boolean;
  setIsSaving: (v: boolean) => void;

  // Panels
  showPropertiesPanel: boolean;
  setShowPropertiesPanel: (v: boolean) => void;
  showProductPanel: boolean;
  setShowProductPanel: (v: boolean) => void;
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  // Map info
  mapId: null,
  activeFloorId: null,

  // Tool
  activeTool: "select",
  setActiveTool: (tool) => set({ activeTool: tool }),

  // Selection
  selectedAreaId: null,
  setSelectedAreaId: (id) => set({ selectedAreaId: id }),

  // Areas
  areas: [],
  addArea: (area) =>
    set((state) => {
      const newAreas = [...state.areas, area];
      return { areas: newAreas, isDirty: true };
    }),
  updateArea: (fabricId, updates) =>
    set((state) => ({
      areas: state.areas.map((a) =>
        a.fabricId === fabricId ? { ...a, ...updates } : a
      ),
      isDirty: true,
    })),
  removeArea: (fabricId) =>
    set((state) => ({
      areas: state.areas.filter((a) => a.fabricId !== fabricId),
      isDirty: true,
      selectedAreaId:
        state.selectedAreaId === fabricId ? null : state.selectedAreaId,
    })),
  setAreas: (areas) => set({ areas }),

  // New area type
  pendingAreaType: "shelf",
  setPendingAreaType: (type) => set({ pendingAreaType: type }),

  // Floor
  setActiveFloorId: (id) => set({ activeFloorId: id }),
  setMapId: (id) => set({ mapId: id }),

  // Measurement
  measurement: DEFAULT_MEASUREMENT_SETTINGS,
  updateMeasurement: (updates) =>
    set((state) => ({
      measurement: { ...state.measurement, ...updates },
    })),

  // Canvas
  zoom: 1,
  setZoom: (zoom) => set({ zoom }),
  isCalibrated: false,
  setIsCalibrated: (v) => set({ isCalibrated: v }),
  baseRatio: 1, // default: 1 pixel = 1 cm
  setBaseRatio: (ratio) => set({ baseRatio: ratio, isCalibrated: true }),
  gridSizeCm: 50, // default: 50cm grid
  setGridSizeCm: (size) => set({ gridSizeCm: size }),

  // Undo/Redo
  history: [[]],
  historyIndex: 0,
  pushHistory: () =>
    set((state) => {
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push(JSON.parse(JSON.stringify(state.areas)));
      return {
        history: newHistory,
        historyIndex: newHistory.length - 1,
      };
    }),
  undo: () =>
    set((state) => {
      if (state.historyIndex <= 0) return state;
      const newIndex = state.historyIndex - 1;
      return {
        historyIndex: newIndex,
        areas: JSON.parse(JSON.stringify(state.history[newIndex])),
        isDirty: true,
      };
    }),
  redo: () =>
    set((state) => {
      if (state.historyIndex >= state.history.length - 1) return state;
      const newIndex = state.historyIndex + 1;
      return {
        historyIndex: newIndex,
        areas: JSON.parse(JSON.stringify(state.history[newIndex])),
        isDirty: true,
      };
    }),
  canUndo: () => get().historyIndex > 0,
  canRedo: () => get().historyIndex < get().history.length - 1,

  // Save state
  isDirty: false,
  setIsDirty: (v) => set({ isDirty: v }),
  isSaving: false,
  setIsSaving: (v) => set({ isSaving: v }),

  // Panels
  showPropertiesPanel: true,
  setShowPropertiesPanel: (v) => set({ showPropertiesPanel: v }),
  showProductPanel: false,
  setShowProductPanel: (v) => set({ showProductPanel: v }),
}));
