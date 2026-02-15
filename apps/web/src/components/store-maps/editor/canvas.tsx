"use client";

import { useRef, useEffect, useCallback } from "react";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import { AREA_TYPE_CONFIGS } from "@/types/store-maps";
import type { CanvasAreaObject } from "@/types/store-maps";
import {
  snapToGrid,
  pixelsToCm,
  formatMeasurement,
  formatArea,
  polygonArea,
  cmToPixels,
} from "@/lib/store-maps/measurement";

// Fabric.js is only available client-side
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let fabric: any = null;

function generateId(): string {
  return `area_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

interface StoreMapCanvasProps {
  floorPlanUrl?: string;
  canvasRef: React.MutableRefObject<import("fabric").Canvas | null>;
}

export function StoreMapCanvas({ floorPlanUrl, canvasRef }: StoreMapCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const htmlCanvasRef = useRef<HTMLCanvasElement>(null);
  const isInitialized = useRef(false);
  const isPanning = useRef(false);
  const lastPanPoint = useRef<{ x: number; y: number } | null>(null);
  const drawingPoints = useRef<{ x: number; y: number }[]>([]);
  const tempObjects = useRef<import("fabric").FabricObject[]>([]);

  const {
    activeTool,
    setActiveTool,
    measurement,
    zoom,
    setZoom,
    baseRatio,
    gridSizeCm,
    addArea,
    areas,
    selectedAreaId,
    setSelectedAreaId,
    pendingAreaType,
    pushHistory,
    setBaseRatio,
    setIsCalibrated,
  } = useCanvasStore();

  // ============================================
  // Initialize Fabric Canvas
  // ============================================
  useEffect(() => {
    if (!htmlCanvasRef.current || !containerRef.current) return;
    
    // Check if canvas is already initialized on this HTML element
    const htmlCanvas = htmlCanvasRef.current;
    if ((htmlCanvas as any).__fabric) {
      // Canvas already initialized, skip
      return;
    }

    if (isInitialized.current) return;

    const initCanvas = async () => {
      fabric = await import("fabric");

      const container = containerRef.current!;
      const canvas = new fabric.Canvas(htmlCanvasRef.current!, {
        width: container.clientWidth,
        height: container.clientHeight,
        backgroundColor: "#1a1a2e",
        selection: true,
        preserveObjectStacking: true,
      });

      canvasRef.current = canvas;
      isInitialized.current = true;

      // Load floor plan if available
      if (floorPlanUrl) {
        loadFloorPlan(canvas, floorPlanUrl);
      }

      // Draw grid
      drawGrid(canvas);
    };

    initCanvas();

    return () => {
      if (canvasRef.current) {
        canvasRef.current.dispose();
        canvasRef.current = null;
        isInitialized.current = false;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ============================================
  // Load Floor Plan
  // ============================================
  const loadFloorPlan = useCallback(
    async (canvas: import("fabric").Canvas, url: string) => {
      if (!fabric) return;
      try {
        const img = await fabric.FabricImage.fromURL(url, { crossOrigin: "anonymous" });
        img.set({
          selectable: false,
          evented: false,
          hoverCursor: "default",
          objectCaching: false,
        });
        (img as Record<string, unknown>).isFloorPlan = true;
        canvas.insertAt(0, img);
        canvas.renderAll();
      } catch {
        console.warn("Failed to load floor plan image");
      }
    },
    []
  );

  // ============================================
  // Draw Grid
  // ============================================
  const drawGrid = useCallback(
    (canvas: import("fabric").Canvas) => {
      if (!fabric || !measurement.showGrid) return;

      // Remove old grid lines
      const objects = canvas.getObjects();
      objects.forEach((obj) => {
        if ((obj as unknown as Record<string, unknown>).isGrid) canvas.remove(obj);
      });

      const gridPx = cmToPixels(gridSizeCm, baseRatio);
      if (gridPx < 5) return; // Too small to render

      const majorInterval = measurement.majorGridInterval;
      const width = canvas.width! * 2;
      const height = canvas.height! * 2;

      const lines: import("fabric").FabricObject[] = [];

      // Vertical lines
      for (let x = 0; x < width; x += gridPx) {
        const isMajor = Math.round(x / gridPx) % majorInterval === 0;
        const line = new fabric.Line([x, -height, x, height], {
          stroke: isMajor ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)",
          strokeWidth: isMajor ? 1 : 0.5,
          selectable: false,
          evented: false,
        });
        (line as Record<string, unknown>).isGrid = true;
        lines.push(line);
      }

      // Horizontal lines
      for (let y = 0; y < height; y += gridPx) {
        const isMajor = Math.round(y / gridPx) % majorInterval === 0;
        const line = new fabric.Line([-width, y, width, y], {
          stroke: isMajor ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)",
          strokeWidth: isMajor ? 1 : 0.5,
          selectable: false,
          evented: false,
        });
        (line as Record<string, unknown>).isGrid = true;
        lines.push(line);
      }

      lines.forEach((l) => canvas.insertAt(1, l));
      canvas.renderAll();
    },
    [measurement.showGrid, measurement.majorGridInterval, gridSizeCm, baseRatio]
  );

  // ============================================
  // Redraw grid on settings change
  // ============================================
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawGrid(canvas);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measurement.showGrid, gridSizeCm, baseRatio, measurement.majorGridInterval]);

  // ============================================
  // Resize observer
  // ============================================
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.setDimensions({ width, height });
        canvas.renderAll();
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [canvasRef]);

  // ============================================
  // Mouse Events for Drawing, Pan, Zoom
  // ============================================
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !fabric) return;

    const getSnappedPoint = (x: number, y: number) => {
      if (!measurement.snapToGrid) return { x, y };
      const gridPx = cmToPixels(gridSizeCm, baseRatio);
      return {
        x: snapToGrid(x, gridPx, measurement.snapPrecision),
        y: snapToGrid(y, gridPx, measurement.snapPrecision),
      };
    };

    // --- Mouse Down ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseDown = (opt: any) => {
      const evt = opt.e as MouseEvent;
      const pointer = canvas.getViewportPoint(evt);

      // Pan with space or pan tool
      if (activeTool === "pan" || evt.altKey) {
        isPanning.current = true;
        lastPanPoint.current = { x: evt.clientX, y: evt.clientY };
        canvas.selection = false;
        canvas.setCursor("grabbing");
        return;
      }

      // Rectangle tool
      if (activeTool === "rectangle") {
        const pt = getSnappedPoint(pointer.x, pointer.y);
        drawingPoints.current = [pt];
        return;
      }

      // Polygon tool - add point
      if (activeTool === "polygon") {
        const pt = getSnappedPoint(pointer.x, pointer.y);

        // Check if closing the polygon (clicking near first point)
        if (drawingPoints.current.length >= 3) {
          const first = drawingPoints.current[0];
          const dist = Math.hypot(pt.x - first.x, pt.y - first.y);
          if (dist < 15) {
            // Close polygon
            finishPolygon();
            return;
          }
        }

        drawingPoints.current.push(pt);
        drawTempPolygon();
        return;
      }

      // Circle tool
      if (activeTool === "circle") {
        const pt = getSnappedPoint(pointer.x, pointer.y);
        drawingPoints.current = [pt];
        return;
      }

      // Calibrate tool
      if (activeTool === "calibrate") {
        const pt = { x: pointer.x, y: pointer.y };
        if (drawingPoints.current.length === 0) {
          drawingPoints.current = [pt];
        } else {
          drawingPoints.current.push(pt);
          finishCalibration();
        }
        return;
      }

      // Measure tool
      if (activeTool === "measure") {
        const pt = { x: pointer.x, y: pointer.y };
        if (drawingPoints.current.length === 0) {
          drawingPoints.current = [pt];
        } else {
          drawingPoints.current.push(pt);
          showMeasurement();
          drawingPoints.current = [];
        }
        return;
      }

      // Select tool - handled by Fabric default
    };

    // --- Mouse Move ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseMove = (opt: any) => {
      const evt = opt.e as MouseEvent;

      // Pan
      if (isPanning.current && lastPanPoint.current) {
        const vpt = canvas.viewportTransform!;
        vpt[4] += evt.clientX - lastPanPoint.current.x;
        vpt[5] += evt.clientY - lastPanPoint.current.y;
        lastPanPoint.current = { x: evt.clientX, y: evt.clientY };
        canvas.requestRenderAll();
        return;
      }

      const pointer = canvas.getViewportPoint(evt);

      // Rectangle drawing preview
      if (activeTool === "rectangle" && drawingPoints.current.length === 1) {
        const pt = getSnappedPoint(pointer.x, pointer.y);
        clearTemp();
        const start = drawingPoints.current[0];
        const config = AREA_TYPE_CONFIGS[pendingAreaType];
        const rect = new fabric.Rect({
          left: Math.min(start.x, pt.x),
          top: Math.min(start.y, pt.y),
          width: Math.abs(pt.x - start.x),
          height: Math.abs(pt.y - start.y),
          fill: config.bgColor,
          stroke: config.color,
          strokeWidth: 2,
          selectable: false,
          evented: false,
        });
        tempObjects.current = [rect];
        canvas.add(rect);

        // Show live dimensions
        const widthCm = pixelsToCm(Math.abs(pt.x - start.x), baseRatio);
        const heightCm = pixelsToCm(Math.abs(pt.y - start.y), baseRatio);
        const dimText = new fabric.FabricText(
          `${formatMeasurement(widthCm, measurement.unitSystem)} × ${formatMeasurement(heightCm, measurement.unitSystem)}`,
          {
            left: Math.min(start.x, pt.x) + Math.abs(pt.x - start.x) / 2,
            top: Math.min(start.y, pt.y) + Math.abs(pt.y - start.y) / 2,
            fontSize: 13,
            fill: "#ffffff",
            fontFamily: "monospace",
            originX: "center",
            originY: "center",
            selectable: false,
            evented: false,
            backgroundColor: "rgba(0,0,0,0.6)",
            padding: 4,
          }
        );
        tempObjects.current.push(dimText);
        canvas.add(dimText);
        canvas.renderAll();
      }

      // Circle drawing preview
      if (activeTool === "circle" && drawingPoints.current.length === 1) {
        const pt = { x: pointer.x, y: pointer.y };
        clearTemp();
        const center = drawingPoints.current[0];
        const radius = Math.hypot(pt.x - center.x, pt.y - center.y);
        const config = AREA_TYPE_CONFIGS[pendingAreaType];
        const circle = new fabric.Circle({
          left: center.x - radius,
          top: center.y - radius,
          radius,
          fill: config.bgColor,
          stroke: config.color,
          strokeWidth: 2,
          selectable: false,
          evented: false,
        });
        tempObjects.current = [circle];
        canvas.add(circle);

        const radiusCm = pixelsToCm(radius, baseRatio);
        const dimText = new fabric.FabricText(
          `r: ${formatMeasurement(radiusCm, measurement.unitSystem)}`,
          {
            left: center.x,
            top: center.y,
            fontSize: 13,
            fill: "#ffffff",
            fontFamily: "monospace",
            originX: "center",
            originY: "center",
            selectable: false,
            evented: false,
            backgroundColor: "rgba(0,0,0,0.6)",
            padding: 4,
          }
        );
        tempObjects.current.push(dimText);
        canvas.add(dimText);
        canvas.renderAll();
      }
    };

    // --- Mouse Up ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseUp = (opt: any) => {
      // Pan end
      if (isPanning.current) {
        isPanning.current = false;
        lastPanPoint.current = null;
        canvas.selection = activeTool === "select";
        canvas.setCursor("default");
        return;
      }

      const pointer = canvas.getViewportPoint(opt.e as MouseEvent);

      // Finish rectangle
      if (activeTool === "rectangle" && drawingPoints.current.length === 1) {
        const start = drawingPoints.current[0];
        const end = getSnappedPoint(pointer.x, pointer.y);
        const width = Math.abs(end.x - start.x);
        const height = Math.abs(end.y - start.y);

        if (width > 5 && height > 5) {
          clearTemp();
          const left = Math.min(start.x, end.x);
          const top = Math.min(start.y, end.y);
          createRectangleArea(left, top, width, height);
        } else {
          clearTemp();
        }
        drawingPoints.current = [];
      }

      // Finish circle
      if (activeTool === "circle" && drawingPoints.current.length === 1) {
        const center = drawingPoints.current[0];
        const radius = Math.hypot(pointer.x - center.x, pointer.y - center.y);
        if (radius > 5) {
          clearTemp();
          createCircleArea(center.x, center.y, radius);
        } else {
          clearTemp();
        }
        drawingPoints.current = [];
      }
    };

    // --- Mouse Wheel (Zoom) ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleWheel = (opt: any) => {
      const evt = opt.e as WheelEvent;
      evt.preventDefault();
      evt.stopPropagation();

      const delta = evt.deltaY;
      let newZoom = canvas.getZoom();
      newZoom *= 0.999 ** delta;
      newZoom = Math.min(Math.max(0.1, newZoom), 10);

      canvas.zoomToPoint(new fabric.Point(evt.offsetX, evt.offsetY), newZoom);
      setZoom(newZoom);
    };

    // --- Selection ---
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleSelection = (opt: any) => {
      const selected = opt.selected;
      if (selected && selected.length === 1) {
        const fabricId = (selected[0] as Record<string, unknown>).areaFabricId as string | undefined;
        if (fabricId) {
          setSelectedAreaId(fabricId);
        }
      }
    };

    const handleSelectionCleared = () => {
      setSelectedAreaId(null);
    };

    canvas.on("mouse:down", handleMouseDown);
    canvas.on("mouse:move", handleMouseMove);
    canvas.on("mouse:up", handleMouseUp);
    canvas.on("mouse:wheel", handleWheel);
    canvas.on("selection:created", handleSelection);
    canvas.on("selection:updated", handleSelection);
    canvas.on("selection:cleared", handleSelectionCleared);

    // Set cursor based on tool
    if (activeTool === "pan") {
      canvas.defaultCursor = "grab";
      canvas.selection = false;
    } else if (activeTool === "select") {
      canvas.defaultCursor = "default";
      canvas.selection = true;
    } else {
      canvas.defaultCursor = "crosshair";
      canvas.selection = false;
    }

    return () => {
      canvas.off("mouse:down", handleMouseDown);
      canvas.off("mouse:move", handleMouseMove);
      canvas.off("mouse:up", handleMouseUp);
      canvas.off("mouse:wheel", handleWheel);
      canvas.off("selection:created", handleSelection);
      canvas.off("selection:updated", handleSelection);
      canvas.off("selection:cleared", handleSelectionCleared);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTool, measurement, baseRatio, gridSizeCm, pendingAreaType]);

  // ============================================
  // Helper functions
  // ============================================
  const clearTemp = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    tempObjects.current.forEach((obj) => canvas.remove(obj));
    tempObjects.current = [];
  };

  const createRectangleArea = (left: number, top: number, width: number, height: number) => {
    if (!fabric || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const config = AREA_TYPE_CONFIGS[pendingAreaType];
    const fabricId = generateId();

    const rect = new fabric.Rect({
      left,
      top,
      width,
      height,
      fill: config.bgColor,
      stroke: config.color,
      strokeWidth: 2,
      cornerColor: config.color,
      cornerStyle: "circle",
      cornerSize: 8,
      transparentCorners: false,
    });
    (rect as Record<string, unknown>).areaFabricId = fabricId;

    // Name label
    const label = new fabric.FabricText(config.label, {
      left: left + width / 2,
      top: top + height / 2,
      fontSize: 12,
      fill: config.color,
      fontFamily: "sans-serif",
      fontWeight: "600",
      originX: "center",
      originY: "center",
      selectable: false,
      evented: false,
    });
    (label as Record<string, unknown>).isLabel = true;
    (label as Record<string, unknown>).parentFabricId = fabricId;

    canvas.add(rect);
    canvas.add(label);
    canvas.renderAll();

    const coords = [
      { x: left, y: top },
      { x: left + width, y: top },
      { x: left + width, y: top + height },
      { x: left, y: top + height },
    ];

    const areaObj: CanvasAreaObject = {
      fabricId,
      name: config.label,
      areaType: pendingAreaType,
      floorId: useCanvasStore.getState().activeFloorId ?? 0,
      coordinates: coords,
      isCircle: false,
    };

    addArea(areaObj);
    pushHistory();
    setSelectedAreaId(fabricId);
  };

  const createCircleArea = (cx: number, cy: number, radius: number) => {
    if (!fabric || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const config = AREA_TYPE_CONFIGS[pendingAreaType];
    const fabricId = generateId();

    const circle = new fabric.Circle({
      left: cx - radius,
      top: cy - radius,
      radius,
      fill: config.bgColor,
      stroke: config.color,
      strokeWidth: 2,
      cornerColor: config.color,
      cornerStyle: "circle",
      cornerSize: 8,
      transparentCorners: false,
    });
    (circle as Record<string, unknown>).areaFabricId = fabricId;

    const label = new fabric.FabricText(config.label, {
      left: cx,
      top: cy,
      fontSize: 12,
      fill: config.color,
      fontFamily: "sans-serif",
      fontWeight: "600",
      originX: "center",
      originY: "center",
      selectable: false,
      evented: false,
    });
    (label as Record<string, unknown>).isLabel = true;
    (label as Record<string, unknown>).parentFabricId = fabricId;

    canvas.add(circle);
    canvas.add(label);
    canvas.renderAll();

    const areaObj: CanvasAreaObject = {
      fabricId,
      name: config.label,
      areaType: pendingAreaType,
      floorId: useCanvasStore.getState().activeFloorId ?? 0,
      coordinates: [{ x: cx, y: cy }],
      isCircle: true,
      radius,
    };

    addArea(areaObj);
    pushHistory();
    setSelectedAreaId(fabricId);
  };

  const finishPolygon = () => {
    if (!fabric || !canvasRef.current || drawingPoints.current.length < 3) return;
    const canvas = canvasRef.current;
    clearTemp();

    const config = AREA_TYPE_CONFIGS[pendingAreaType];
    const fabricId = generateId();
    const points = [...drawingPoints.current];

    const polygon = new fabric.Polygon(
      points.map((p) => new fabric.Point(p.x, p.y)),
      {
        fill: config.bgColor,
        stroke: config.color,
        strokeWidth: 2,
        cornerColor: config.color,
        cornerStyle: "circle",
        cornerSize: 8,
        transparentCorners: false,
      }
    );
    (polygon as Record<string, unknown>).areaFabricId = fabricId;

    // Calculate center for label
    const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const cy = points.reduce((s, p) => s + p.y, 0) / points.length;

    // Area measurement label
    const areaPx = polygonArea(points);
    const areaCmSq = areaPx * baseRatio * baseRatio;
    const areaText = formatArea(areaCmSq, measurement.unitSystem);

    const label = new fabric.FabricText(`${config.label}\n${areaText}`, {
      left: cx,
      top: cy,
      fontSize: 11,
      fill: config.color,
      fontFamily: "sans-serif",
      fontWeight: "600",
      originX: "center",
      originY: "center",
      textAlign: "center",
      selectable: false,
      evented: false,
    });
    (label as Record<string, unknown>).isLabel = true;
    (label as Record<string, unknown>).parentFabricId = fabricId;

    canvas.add(polygon);
    canvas.add(label);
    canvas.renderAll();

    const areaObj: CanvasAreaObject = {
      fabricId,
      name: config.label,
      areaType: pendingAreaType,
      floorId: useCanvasStore.getState().activeFloorId ?? 0,
      coordinates: points,
      isCircle: false,
    };

    addArea(areaObj);
    pushHistory();
    setSelectedAreaId(fabricId);
    drawingPoints.current = [];
  };

  const drawTempPolygon = () => {
    if (!fabric || !canvasRef.current) return;
    const canvas = canvasRef.current;
    clearTemp();

    const config = AREA_TYPE_CONFIGS[pendingAreaType];
    const points = drawingPoints.current;

    // Draw lines between points
    for (let i = 0; i < points.length - 1; i++) {
      const line = new fabric.Line(
        [points[i].x, points[i].y, points[i + 1].x, points[i + 1].y],
        {
          stroke: config.color,
          strokeWidth: 2,
          selectable: false,
          evented: false,
        }
      );
      tempObjects.current.push(line);
      canvas.add(line);
    }

    // Draw dots at each point
    points.forEach((pt, idx) => {
      const dot = new fabric.Circle({
        left: pt.x - 4,
        top: pt.y - 4,
        radius: 4,
        fill: idx === 0 ? "#22C55E" : config.color,
        stroke: "#fff",
        strokeWidth: 1,
        selectable: false,
        evented: false,
      });
      tempObjects.current.push(dot);
      canvas.add(dot);
    });

    canvas.renderAll();
  };

  const finishCalibration = () => {
    if (drawingPoints.current.length < 2) return;
    const p1 = drawingPoints.current[0];
    const p2 = drawingPoints.current[1];
    const pixelDist = Math.hypot(p2.x - p1.x, p2.y - p1.y);

    // Prompt user for real-world distance
    const input = window.prompt(
      `Enter the real-world distance for this line (in ${measurement.unitSystem === "metric" ? "meters" : "feet"}):`,
      "10"
    );

    if (input && !isNaN(parseFloat(input))) {
      const realDist = parseFloat(input);
      const realCm =
        measurement.unitSystem === "metric" ? realDist * 100 : realDist * 30.48;
      const ratio = realCm / pixelDist;
      setBaseRatio(ratio);
      setIsCalibrated(true);
    }

    drawingPoints.current = [];
    setActiveTool("select");
  };

  const showMeasurement = () => {
    if (!fabric || !canvasRef.current || drawingPoints.current.length < 2) return;
    const canvas = canvasRef.current;
    const p1 = drawingPoints.current[0];
    const p2 = drawingPoints.current[1];
    const pixelDist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const realCm = pixelsToCm(pixelDist, baseRatio);

    // Draw measurement line
    const line = new fabric.Line([p1.x, p1.y, p2.x, p2.y], {
      stroke: "#FACC15",
      strokeWidth: 2,
      strokeDashArray: [6, 4],
      selectable: false,
      evented: false,
    });
    (line as Record<string, unknown>).isMeasurement = true;

    const mx = (p1.x + p2.x) / 2;
    const my = (p1.y + p2.y) / 2;
    const text = new fabric.FabricText(
      formatMeasurement(realCm, measurement.unitSystem),
      {
        left: mx,
        top: my - 15,
        fontSize: 13,
        fill: "#FACC15",
        fontFamily: "monospace",
        originX: "center",
        originY: "center",
        selectable: false,
        evented: false,
        backgroundColor: "rgba(0,0,0,0.7)",
        padding: 4,
      }
    );
    (text as Record<string, unknown>).isMeasurement = true;

    canvas.add(line);
    canvas.add(text);
    canvas.renderAll();

    // Auto-remove after 8 seconds
    setTimeout(() => {
      canvas.remove(line);
      canvas.remove(text);
      canvas.renderAll();
    }, 8000);
  };

  // ============================================
  // Keyboard shortcuts
  // ============================================
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;

      switch (e.key.toLowerCase()) {
        case "v":
          setActiveTool("select");
          break;
        case "r":
          setActiveTool("rectangle");
          break;
        case "p":
          setActiveTool("polygon");
          break;
        case "c":
          setActiveTool("circle");
          break;
        case "m":
          setActiveTool("measure");
          break;
        case " ":
          e.preventDefault();
          setActiveTool("pan");
          break;
        case "escape":
          // Cancel current drawing
          clearTemp();
          drawingPoints.current = [];
          setActiveTool("select");
          break;
        case "delete":
        case "backspace":
          if (selectedAreaId && canvasRef.current) {
            // Remove from canvas
            const canvas = canvasRef.current;
            const objects = canvas.getObjects();
            objects.forEach((obj) => {
              const custom = obj as unknown as Record<string, unknown>;
              if (custom.areaFabricId === selectedAreaId || custom.parentFabricId === selectedAreaId) {
                canvas.remove(obj);
              }
            });
            canvas.renderAll();
            useCanvasStore.getState().removeArea(selectedAreaId);
            pushHistory();
          }
          break;
        case "z":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            if (e.shiftKey) {
              useCanvasStore.getState().redo();
            } else {
              useCanvasStore.getState().undo();
            }
          }
          break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === " ") {
        setActiveTool("select");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAreaId]);

  return (
    <div ref={containerRef} className="flex-1 relative overflow-hidden bg-neutral-900">
      <canvas ref={htmlCanvasRef} />
      {/* Status bar */}
      <div className="absolute bottom-0 left-0 right-0 h-6 bg-card/80 backdrop-blur-sm border-t flex items-center px-3 text-xs text-muted-foreground gap-4">
        <span>
          {areas.filter((a) => a.floorId === useCanvasStore.getState().activeFloorId).length} areas
        </span>
        <span>Zoom: {Math.round(zoom * 100)}%</span>
        <span>
          Grid: {formatMeasurement(gridSizeCm, measurement.unitSystem)}
        </span>
        {useCanvasStore.getState().isCalibrated && (
          <span className="text-green-500">Scale calibrated</span>
        )}
      </div>
    </div>
  );
}
