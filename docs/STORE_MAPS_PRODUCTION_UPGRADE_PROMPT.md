# STORE MAP SYSTEM - PRODUCTION UPGRADE & SOTA IMPLEMENTATION

## CONTEXT & CURRENT STATE

You are Claude Sonnet 5.3 Codex, the most advanced AI coding agent. You have been assigned to upgrade a retail store layout mapping system from MVP to production-ready, state-of-the-art implementation.

### Project Overview
- **Platform**: BuyBuddy AI - Retail analytics and AI platform
- **Tech Stack**: 
  - Frontend: React 19 + Next.js 16 (App Router), TypeScript, Tailwind CSS v4, shadcn/ui, TanStack Query v5, Zustand, Fabric.js 6.x
  - Backend: FastAPI (Python), acting as proxy to BuyBuddy Legacy API
  - Canvas Library: Fabric.js 6.x for interactive 2D editing
- **Codebase Location**: `/Users/erselgokmen/Ai-pipeline/buybuddy-ai/`
- **PRD Document**: `/Users/erselgokmen/.cursor/plans/store_layout_builder_prd_348adfb7.plan.md`

### What Has Been Built (MVP - Phase 1)

**✅ Implemented Features:**
1. **Map CRUD**: List, create, update, delete store maps
2. **Floor Management**: Floor creation, floor plan image upload (S3), floor switching
3. **Canvas Editor**: Fabric.js-based full-screen editor with zoom, pan, selection
4. **Drawing Tools**: Rectangle, polygon, circle drawing tools
5. **Measurement System**: 
   - Unit system (metric/imperial) with internal cm storage
   - Grid system (real-world units: 25cm, 50cm, 1m, 2m or 6in, 1ft, 2ft, 5ft)
   - Snap-to-grid functionality
   - Live dimension display during drawing
   - Measurement utility functions (unit conversion, area calculation, perimeter)
6. **Area Management**: Area creation, naming, type selection (Reyon, Koridor, Kasa, Depo, etc.), properties panel
7. **Basic Toolbar**: Tool selection, undo/redo, zoom controls, unit toggle, grid toggle, save button
8. **API Integration**: All map/floor/area/coordinate endpoints connected to BuyBuddy Legacy API
9. **Authentication**: User authentication with token management, store assignment

**📂 Key Files:**
- Frontend Canvas: `apps/web/src/components/store-maps/editor/canvas.tsx` (934 lines)
- Toolbar: `apps/web/src/components/store-maps/editor/toolbar.tsx` (352 lines)
- Types: `apps/web/src/types/store-maps.ts` (279 lines)
- Measurement Utils: `apps/web/src/lib/store-maps/measurement.ts` (161 lines)
- Canvas Store: `apps/web/src/hooks/store-maps/use-canvas-store.ts` (186 lines)
- Backend Router: `apps/api/src/api/v1/map.py` (428 lines)
- BuyBuddy Service: `apps/api/src/services/buybuddy.py` (map methods at lines 647-900+)

**🚨 Current Issues & Limitations (Identified During MVP Development):**

1. **Canvas Performance**:
   - No object virtualization or culling - all objects rendered at once
   - Grid redraws on every state change (inefficient)
   - No debouncing on mouse events
   - Large maps (100+ areas) will likely cause frame drops

2. **User Experience Gaps**:
   - No visual feedback for drawing operations (rubber band, ghost shapes)
   - No tooltips on canvas objects (hover shows nothing)
   - No keyboard shortcuts beyond tool selection
   - No minimap for navigation
   - No rulers (mentioned in PRD but not implemented)
   - Calibration tool defined but not functional
   - Measurement tool exists but no visual annotations

3. **Data Management**:
   - No auto-save implementation (only manual save button)
   - Undo/redo exists but history stack is naive (no memory limits)
   - No optimistic updates - every change waits for API
   - No conflict resolution for concurrent edits
   - No data validation before API calls

4. **Missing Core Features from PRD**:
   - **Rulers**: Top and left rulers showing real-world measurements
   - **Calibration Tool**: UI for clicking floor plan and entering known distance
   - **Measurement Annotations**: Temporary measurement lines with labels
   - **Live Dimension Labels**: Floating dimension labels on shapes during drawing/editing
   - **Area Dimension Display**: Edge lengths, area (m²/ft²), perimeter on completed areas
   - **Product Panel**: Left/right panel for product catalog (search, filter, drag source)
   - **Product Mapping**: Drag & drop products to areas, assigned products list
   - **Category Mapping**: Category tree picker, category assignment to areas
   - **Planogram Editor**: Entire Phase 2 - shelf structure, product placement, facing management

5. **Code Quality Issues**:
   - Massive `canvas.tsx` file (934 lines) - needs refactoring into smaller modules
   - TypeScript `any` types used liberally for Fabric.js (type safety compromised)
   - No error boundaries in React components
   - No loading states for async operations
   - Inline styles mixed with Tailwind classes
   - No comprehensive error handling in API calls
   - Console.log statements left in production code

6. **Accessibility**:
   - Canvas is not keyboard-accessible (no focus management)
   - No ARIA labels on toolbar buttons
   - No screen reader support for canvas content
   - No high contrast mode

7. **Testing**:
   - **Zero tests** - no unit tests, integration tests, or E2E tests
   - No test infrastructure set up
   - No mocking for Fabric.js or API calls

8. **Documentation**:
   - No JSDoc comments on complex functions
   - No inline documentation for canvas event handlers
   - No architecture decision records (ADRs)

### What Needs to Be Built (Phase 2 + Production Hardening)

**Priority 1 - Complete Phase 1 (Missing Features):**
1. **Rulers Component**: Top/left rulers with real-world unit labels, zoom-adaptive scaling
2. **Calibration Tool**: 
   - Click-to-measure mode on floor plan
   - Dialog for entering known distance
   - Auto-calculate and set `base_ratio`
3. **Measurement Tool Enhancements**:
   - Visual line with distance label
   - Multiple measurements on canvas
   - "Clear All Measurements" button
4. **Enhanced Dimension Display**:
   - Floating dimension labels during drawing (width × height)
   - Edge length labels on completed polygons
   - Area (m²/ft²) label inside shapes
   - Perimeter in properties panel
5. **Product Integration**:
   - Product catalog panel (collapsible, left or right side)
   - Product search API integration
   - Category filter tree
   - Drag & drop from panel to canvas areas
   - Assigned products list in properties panel
   - Product count badge on areas
6. **Category Mapping**:
   - Category tree picker (hierarchical combobox)
   - Category assignment to areas
   - "Category View" mode - areas colored by category
   - Legend panel for category colors

**Priority 2 - Planogram Editor (Phase 2):**
1. **Planogram Modal/View**:
   - Triggered by double-click on area or "Edit Planogram" button
   - Front-facing view (not top-down)
   - Breadcrumb navigation back to map
2. **Shelf Structure**:
   - Visual shelf unit (rectangle container)
   - Horizontal shelf level lines (draggable height)
   - Shelf type indicator (standard, cooler, peg hook, basket)
   - Dimension inputs (width, height in cm/ft)
3. **Product Placement on Shelves**:
   - Drag products from catalog to shelf
   - Snap to adjacent products
   - Facing controls (+/- buttons, drag to expand)
   - Product thumbnail display on shelf
   - Empty space indicators (dashed lines)
4. **Planogram Data Model**:
   - New Zustand store: `usePlanogramStore`
   - Types: `Planogram`, `Shelf`, `ShelfProduct`
   - Serialization to/from API (or localStorage as fallback)
5. **Planogram Features**:
   - Copy/paste planogram between areas
   - Fill percentage indicator
   - Product reordering (drag within shelf, drag between shelves)
   - Auto-sync with area product list

**Priority 3 - Performance & Production Hardening:**
1. **Canvas Optimization**:
   - Object pooling for grid lines
   - Viewport culling (only render visible objects)
   - Debounced mouse events (throttle to 60fps)
   - Lazy rendering for complex shapes
   - Virtual scrolling for product list
2. **Auto-Save**:
   - Debounced auto-save (2-3 seconds after last change)
   - Visual indicator (saving/saved/error)
   - Retry logic with exponential backoff
   - Offline queue (save when connection returns)
3. **Error Handling**:
   - React Error Boundaries around editor
   - Toast notifications for API errors
   - Graceful degradation (show cached data if API fails)
   - User-friendly error messages (not raw API errors)
4. **Code Refactoring**:
   - Split `canvas.tsx` into modules:
     - `canvas-core.tsx` (init, event setup)
     - `canvas-drawing.tsx` (drawing tool logic)
     - `canvas-selection.tsx` (selection, transform logic)
     - `canvas-measurement.tsx` (live dimensions, annotations)
     - `canvas-serialization.tsx` (API sync, save/load)
   - Extract Fabric.js type wrappers to `@/lib/store-maps/fabric-types.ts`
   - Create custom hooks:
     - `useCanvasTools()` - drawing logic
     - `useCanvasSelection()` - selection state
     - `useCanvasMeasurement()` - dimension calculation
     - `useCanvasSync()` - API sync, auto-save
5. **Type Safety**:
   - Replace all `any` with proper Fabric.js types or custom wrappers
   - Strict null checks enabled
   - Discriminated unions for tool state
6. **Accessibility**:
   - Keyboard navigation for toolbar (Tab, Arrow keys)
   - Focus management (trap focus in modals)
   - ARIA labels on all interactive elements
   - Canvas selection via keyboard (Tab through areas, Enter to select)
   - Screen reader announcements for state changes

**Priority 4 - Testing & Documentation:**
1. **Unit Tests** (Vitest):
   - Measurement utils: `measurement.ts` (100% coverage)
   - Canvas store: `use-canvas-store.ts` (state transitions)
   - Area type configs: `store-maps.ts`
   - Serialization logic: `canvas-serialization.tsx`
2. **Integration Tests** (React Testing Library):
   - Toolbar interactions
   - Drawing workflows (rectangle, polygon)
   - Properties panel updates
   - Product assignment flow
3. **E2E Tests** (Playwright):
   - Full user journey: create map → draw area → assign products → save
   - Calibration workflow
   - Planogram creation and editing
4. **Documentation**:
   - JSDoc comments on all public functions
   - Architecture overview (mermaid diagrams)
   - Canvas event flow diagram
   - API integration guide

**Priority 5 - Advanced Features (Nice-to-Have):**
1. **Minimap**: Small overview panel showing full canvas, viewport indicator
2. **Layers Panel**: Z-order management, visibility toggle per area
3. **Copy/Paste**: Copy area with all properties, paste at mouse position
4. **Multi-select**: Shift+click to select multiple areas, bulk operations
5. **Snapping Enhancements**: Snap to edges, snap to center, snap guides (Figma-style)
6. **History Panel**: Visual undo/redo timeline with thumbnails
7. **Export/Import**: Export map as JSON, import from external format
8. **Templates**: Predefined store layouts (supermarket, convenience store, warehouse)
9. **Collaboration**: Real-time cursor indicators, presence awareness (optional, advanced)

---

## YOUR MISSION

**Transform this MVP into a production-grade, state-of-the-art retail store mapping system.**

### Success Criteria

1. **Performance**:
   - 60fps on maps with 200+ areas
   - Load time < 2s for typical map
   - Canvas interaction latency < 16ms

2. **Reliability**:
   - Zero data loss (auto-save, offline queue)
   - Graceful error handling (no white screens)
   - API retry with exponential backoff

3. **User Experience**:
   - Smooth drawing experience (visual feedback, ghost shapes, live dimensions)
   - Intuitive product assignment (drag & drop, search, filter)
   - Professional UI (rulers, minimap, keyboard shortcuts)
   - Accessible (keyboard navigation, screen readers)

4. **Code Quality**:
   - Modular architecture (no files > 300 lines)
   - Type-safe (no `any` types)
   - Test coverage > 80%
   - Comprehensive documentation

5. **Feature Completeness**:
   - All Phase 1 features from PRD implemented
   - Planogram editor (Phase 2) fully functional
   - Product/category mapping working end-to-end

---

## EXECUTION STRATEGY

### Phase A: Performance & Architecture Foundation (Day 1-2)

**Goal**: Establish solid foundation before adding complexity.

1. **Refactor Canvas Component**:
   - Split `canvas.tsx` into 5 modules (core, drawing, selection, measurement, serialization)
   - Extract drawing logic to `useCanvasTools` hook
   - Create Fabric.js type wrappers in `fabric-types.ts`
   - Remove all `any` types

2. **Optimize Rendering**:
   - Implement object pooling for grid lines
   - Add viewport culling (render only visible objects)
   - Debounce mouse events (use lodash.throttle or custom implementation)
   - Memoize expensive calculations (area, perimeter)

3. **Set Up Testing Infrastructure**:
   - Install Vitest + React Testing Library
   - Create test setup file (`setup-tests.ts`)
   - Write first unit test for `measurement.ts`
   - Add test scripts to `package.json`

### Phase B: Complete Phase 1 Features (Day 3-5)

**Goal**: Implement missing PRD features to match original specification.

1. **Rulers Component**:
   - Create `apps/web/src/components/store-maps/editor/rulers.tsx`
   - Horizontal ruler (top): render tick marks, labels based on zoom and baseRatio
   - Vertical ruler (left): same logic
   - Auto-scale labels (cm → m when zoomed out, in → ft)
   - Update `store-map-editor.tsx` to include rulers

2. **Calibration Tool**:
   - Add "calibrate" tool mode to `canvas-drawing.tsx`
   - Two-click line drawing on floor plan
   - Dialog: `calibration-dialog.tsx` with distance input
   - Calculate: `base_ratio = pixel_distance / (real_distance * 100)` (convert m to cm)
   - Update map via API: `PATCH /api/v1/map/{id}` with new `base_ratio`

3. **Measurement Tool Enhancements**:
   - Create `canvas-measurement-annotations.tsx` for temporary measurement lines
   - Store measurements in `useCanvasStore` as `measurements: {id, p1, p2, distanceCm}[]`
   - Render as dashed lines with distance label
   - "Clear Measurements" button in toolbar

4. **Enhanced Dimension Display**:
   - During drawing: floating label showing "W × H" or "r: X"
   - After creation: edge length labels (use Fabric.js Text objects, positioned on edge midpoints)
   - Area label inside shape (center of bounding box)
   - Update properties panel to show area (m²/ft²), perimeter, bounding box

5. **Product Panel**:
   - Create `apps/web/src/components/store-maps/editor/product-panel.tsx`
   - Collapsible side panel (use shadcn Sheet or custom drawer)
   - Search input with debounced API call
   - Category filter (tree structure, use shadcn Popover + Tree component or custom)
   - Product cards (small thumbnail, name, SKU)
   - Drag source: use HTML5 drag-drop API or react-dnd
   - Update `store-map-editor.tsx` layout to include product panel

6. **Product Assignment**:
   - Drop handler on canvas: detect area under mouse, assign product
   - Store mapping in Zustand: `areaProducts: {[areaId: string]: ProductRef[]}`
   - Update properties panel: "Products" tab with assigned list
   - Remove product button (X icon)
   - Product count badge on areas (Fabric.js Text object at top-right corner of area)

7. **Category Mapping**:
   - Create `category-picker.tsx` (hierarchical tree, use shadcn Combobox or custom)
   - "Categories" tab in properties panel
   - Store in area object: `categories: string[]`
   - "Category View" toggle in toolbar (changes area colors to category colors)
   - Legend panel showing active categories (small floating panel)

### Phase C: Planogram Editor (Day 6-8)

**Goal**: Build Phase 2 - shelf-level product placement.

1. **Planogram Data Model**:
   - Create `apps/web/src/hooks/store-maps/use-planogram-store.ts` (Zustand)
   - Types in `types/store-maps.ts`: `Planogram`, `Shelf`, `ShelfProduct`
   - Initial state: `activePlanogramId`, `planograms`, `shelves`, `shelfProducts`

2. **Planogram View**:
   - Create `apps/web/src/components/store-maps/planogram/planogram-editor.tsx`
   - Modal or slide-over panel (use shadcn Dialog at full width, or custom slide-in)
   - Breadcrumb: "Map > [Area Name] > Planogram"
   - Toolbar: Add Shelf, Save, Close

3. **Shelf Canvas**:
   - Create `planogram/shelf-canvas.tsx` (new Fabric.js canvas, front-facing view)
   - Render gondola outline (tall rectangle)
   - Horizontal shelf lines (Fabric.js Line objects, draggable)
   - Shelf level labels (1, 2, 3... from top or bottom, configurable)

4. **Product Placement on Shelves**:
   - Drag from product panel to shelf canvas
   - Snap to shelf line (y-axis) and adjacent products (x-axis)
   - Render as small rectangle with product image (use Fabric.js Image + Group)
   - Facing controls: overlay buttons (+/-) or drag handles to expand width
   - Store: `ShelfProduct {shelf_id, product_id, x_position, facings, width, height}`

5. **Planogram Features**:
   - "Fill %" indicator: sum of product widths / shelf width * 100
   - Empty space visualization (dashed rectangle)
   - Drag to reorder within shelf
   - Drag between shelves
   - Delete product (right-click context menu or X button)
   - Save: serialize to localStorage or API (if endpoint exists)

6. **Sync with Area Products**:
   - On planogram save: extract unique products, update area's product list
   - Warning: "Some products in area not placed on planogram"

### Phase D: Production Hardening (Day 9-10)

**Goal**: Ensure reliability, error handling, auto-save.

1. **Auto-Save**:
   - `useCanvasSync` hook: watch `isDirty`, debounce 3s, trigger save
   - Save function: serialize areas → API calls (batched if possible)
   - Retry logic: exponential backoff (1s, 2s, 4s, 8s)
   - Offline queue: store pending changes in IndexedDB, sync on reconnect

2. **Error Boundaries**:
   - Wrap `<StoreMapEditor>` in React Error Boundary
   - Fallback UI: "Something went wrong" + "Reload" button
   - Log errors to console (or external service like Sentry)

3. **Loading States**:
   - Skeleton loaders for product panel, properties panel
   - Spinner on canvas while loading map data
   - Disable toolbar during save operation

4. **Input Validation**:
   - Area name: max length 100 chars, no empty strings
   - Coordinate values: clamp to canvas bounds
   - Product assignment: check product exists before adding

5. **User Feedback**:
   - Toast on save success/failure (use sonner)
   - Toast on product assignment ("Product added to area")
   - Hover tooltips on canvas objects (area name, type, product count)

### Phase E: Testing & Documentation (Day 11-12)

**Goal**: Achieve 80%+ coverage, document architecture.

1. **Unit Tests**:
   - `measurement.ts`: all functions (15+ tests)
   - `use-canvas-store.ts`: state mutations (10+ tests)
   - `canvas-serialization.tsx`: toAPI/fromAPI conversions (8+ tests)
   - `use-planogram-store.ts`: shelf operations (10+ tests)

2. **Integration Tests**:
   - Toolbar: tool switching, undo/redo (5 tests)
   - Drawing: simulate mouse events, verify area creation (8 tests)
   - Product panel: search, filter, drag-drop (6 tests)

3. **E2E Tests** (Playwright):
   - Happy path: create map → upload floor plan → calibrate → draw area → save (1 test)
   - Product assignment: open product panel → search → drag to area → verify (1 test)
   - Planogram: double-click area → add shelf → place product → save (1 test)

4. **Documentation**:
   - JSDoc on all exported functions (use `/** ... */` format)
   - `STORE_MAPS_ARCHITECTURE.md`: component tree, data flow diagrams (Mermaid)
   - `CANVAS_EVENTS.md`: event handler flow, coordinate systems
   - `API_INTEGRATION.md`: endpoint mapping, error codes

### Phase F: Advanced Features (Day 13-14, Optional)

**Goal**: Polish, minimap, copy/paste, templates.

1. **Minimap**:
   - `minimap.tsx`: small canvas (200×150px) in bottom-right corner
   - Render simplified shapes (no textures, just fills)
   - Viewport indicator (draggable rectangle)
   - Click to pan

2. **Keyboard Shortcuts**:
   - V: Select, R: Rectangle, P: Polygon, C: Circle, M: Measure
   - Ctrl+Z: Undo, Ctrl+Shift+Z: Redo
   - Ctrl+S: Save
   - Ctrl+C/V: Copy/Paste selected area
   - Delete: Remove selected area

3. **Copy/Paste**:
   - Store clipboard in `useCanvasStore`: `clipboard: CanvasAreaObject | null`
   - On paste: clone area, offset by 20px, assign new fabricId

4. **Templates**:
   - `templates/`: JSON files with predefined layouts
   - "New from Template" button in map list
   - Templates: "Small Convenience Store", "Supermarket", "Warehouse"

---

## DETAILED IMPLEMENTATION GUIDELINES

### Code Style & Patterns

1. **Component Structure**:
   ```tsx
   // Good: Small, focused component
   export function RulerHorizontal({ baseRatio, zoom, width }: RulerProps) {
     const ticks = useMemo(() => calculateTicks(width, baseRatio, zoom), [width, baseRatio, zoom]);
     return <div className="ruler-h">{ticks.map(...)}</div>;
   }
   ```

2. **Hook Extraction**:
   ```tsx
   // Extract complex logic to custom hooks
   function useCanvasDrawing(canvas: Canvas | null, tool: ToolMode) {
     useEffect(() => {
       if (!canvas || tool !== 'rectangle') return;
       const handler = (e: FabricEvent) => { /* ... */ };
       canvas.on('mouse:down', handler);
       return () => canvas.off('mouse:down', handler);
     }, [canvas, tool]);
   }
   ```

3. **Type Safety**:
   ```tsx
   // Create type wrappers for Fabric.js
   export interface FabricRect extends fabric.Rect {
     customData?: { areaId: string; areaType: AreaType };
   }
   
   // Use discriminated unions for tool state
   type DrawingState =
     | { mode: 'idle' }
     | { mode: 'drawing'; startPoint: Point }
     | { mode: 'complete'; shape: FabricObject };
   ```

4. **Error Handling**:
   ```tsx
   async function saveArea(area: CanvasAreaObject) {
     try {
       const result = await apiClient.createMapArea({ name: area.name, floor_id: area.floorId });
       toast.success('Area saved');
       return result;
     } catch (error) {
       if (error instanceof ApiError) {
         toast.error(`Failed to save: ${error.message}`);
       } else {
         toast.error('Unknown error occurred');
       }
       throw error;
     }
   }
   ```

5. **Performance**:
   ```tsx
   // Debounce expensive operations
   const debouncedSave = useMemo(
     () => debounce(async () => { await saveToAPI(); }, 3000),
     []
   );
   
   useEffect(() => {
     if (isDirty) debouncedSave();
   }, [isDirty, debouncedSave]);
   ```

### Testing Patterns

1. **Unit Test Example** (measurement.ts):
   ```typescript
   describe('formatMeasurement', () => {
     it('formats metric < 1m as cm', () => {
       expect(formatMeasurement(75, 'metric', 1)).toBe('75.0cm');
     });
     
     it('formats metric >= 1m as m', () => {
       expect(formatMeasurement(250, 'metric', 2)).toBe('2.50m');
     });
     
     it('formats imperial as ft and in', () => {
       expect(formatMeasurement(152.4, 'imperial', 0)).toBe('5ft');
     });
   });
   ```

2. **Integration Test Example** (toolbar):
   ```tsx
   describe('EditorToolbar', () => {
     it('switches tool on button click', () => {
       render(<EditorToolbar {...mockProps} />);
       const rectButton = screen.getByLabelText(/rectangle/i);
       fireEvent.click(rectButton);
       expect(mockStore.setActiveTool).toHaveBeenCalledWith('rectangle');
     });
   });
   ```

3. **E2E Test Example** (Playwright):
   ```typescript
   test('create map and draw area', async ({ page }) => {
     await page.goto('/store-maps');
     await page.click('text=New Map');
     await page.fill('[name="name"]', 'Test Store');
     await page.selectOption('[name="store_id"]', '1');
     await page.click('text=Create');
     await page.waitForURL(/\/editor\?id=/);
     
     await page.click('[aria-label="Rectangle"]');
     await page.mouse.click(200, 200);
     await page.mouse.click(400, 350);
     
     await expect(page.locator('text=Area created')).toBeVisible();
   });
   ```

### API Integration Best Practices

1. **Optimistic Updates**:
   ```tsx
   const mutation = useMutation({
     mutationFn: apiClient.createMapArea,
     onMutate: async (newArea) => {
       // Cancel in-flight queries
       await queryClient.cancelQueries({ queryKey: ['areas'] });
       // Snapshot previous value
       const prev = queryClient.getQueryData(['areas']);
       // Optimistically update
       queryClient.setQueryData(['areas'], (old) => [...old, newArea]);
       return { prev };
     },
     onError: (err, newArea, context) => {
       // Rollback on error
       queryClient.setQueryData(['areas'], context.prev);
     },
     onSettled: () => {
       // Refetch after mutation
       queryClient.invalidateQueries({ queryKey: ['areas'] });
     },
   });
   ```

2. **Retry Logic**:
   ```tsx
   const queryClient = new QueryClient({
     defaultOptions: {
       queries: {
         retry: 3,
         retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
       },
     },
   });
   ```

### Accessibility Checklist

- [ ] All toolbar buttons have `aria-label`
- [ ] Keyboard navigation works (Tab, Shift+Tab)
- [ ] Focus is visible (focus-visible:ring)
- [ ] Canvas selection announced ("Area selected: Reyon 1")
- [ ] Dialogs trap focus (no tabbing out)
- [ ] Error messages have role="alert"
- [ ] Color contrast >= 4.5:1 (WCAG AA)

---

## OUTPUT EXPECTATIONS

After completing this task, you should deliver:

1. **Working System**:
   - All Phase 1 features functional
   - Planogram editor operational
   - Auto-save working
   - No critical bugs

2. **Refactored Codebase**:
   - No files > 300 lines
   - Zero `any` types
   - Consistent code style
   - Modular architecture

3. **Test Suite**:
   - 80%+ coverage on critical modules
   - All tests passing
   - E2E tests for main workflows

4. **Documentation**:
   - JSDoc on all public APIs
   - Architecture document (with diagrams)
   - API integration guide

5. **Performance Metrics**:
   - Canvas renders at 60fps with 200 areas
   - Auto-save latency < 200ms (debounced)
   - Load time < 2s for typical map

---

## CRITICAL SUCCESS FACTORS

1. **Do NOT over-engineer**: Focus on PRD requirements, not speculative features
2. **Test as you go**: Write tests alongside implementation, not after
3. **Prioritize UX**: Smooth drawing experience > fancy algorithms
4. **Performance matters**: Profile before optimizing, but don't ship laggy UI
5. **Accessibility is not optional**: Keyboard nav and screen readers from day 1
6. **Document decisions**: If you make a trade-off, explain why in a comment
7. **Communicate progress**: Update the user on milestones (e.g., "Rulers complete", "Planogram working")

---

## START NOW

Begin with **Phase A: Performance & Architecture Foundation**.

First step: Read the current `canvas.tsx` file, analyze its structure, and propose a refactoring plan (which functions go into which new modules). Wait for my approval before proceeding with the split.

Then move systematically through Phases B → C → D → E → F.

Good luck, Sonnet 5.3. Show me what state-of-the-art means. 🚀
