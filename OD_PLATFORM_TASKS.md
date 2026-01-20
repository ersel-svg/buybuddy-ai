# Object Detection Platform - Implementation Tasks

> Bu dosya, OD Platform implementasyonunu adım adım takip etmek için kullanılacak.
> Her task tamamlandığında `[ ]` → `[x]` olarak işaretlenecek.

---

## 📋 Overview

| Phase | Açıklama | Tahmini Task Sayısı |
|-------|----------|---------------------|
| Phase 0 | Navigation & Layout Setup | 5 |
| Phase 1 | Database & Backend Foundation | 15 |
| Phase 2 | Image Management (Roboflow-Level) | **45+** |
| Phase 3 | Dataset Management | **35+** |
| Phase 4 | Class Management | 8 |
| Phase 5 | Annotation Editor | 20 |
| Phase 6 | AI Auto-Annotation Worker | 12 |
| Phase 7 | Export/Import | 10 |
| Phase 8 | Training Pipeline | 14 |
| Phase 9 | Model Registry | 8 |
| **Total** | | **~170+ tasks** |

### 🎯 Key Features (Roboflow-Level)
- **Multi-source import:** Upload, URL, BuyBuddy API, Cloud Storage, ZIP
- **Annotated dataset import:** COCO, YOLO, Pascal VOC, LabelMe, CVAT
- **Smart class mapping:** Auto-match similar names, create new, skip
- **Duplicate detection:** Perceptual hash (pHash) based
- **Bulk operations:** Multi-select, batch processing
- **Advanced filtering:** By source, status, tags, datasets, dimensions

---

## 🎨 Tech Stack & UI/UX

### Frontend Stack
- **Framework:** Next.js 16 (App Router)
- **UI Library:** shadcn/ui (Radix UI + Tailwind CSS)
- **State:** Zustand
- **Data Fetching:** TanStack Query
- **Tables:** TanStack Table
- **Charts:** Recharts
- **Icons:** Lucide React

### UI/UX Prensipleri
- **Consistency:** Mevcut buybuddy-ai patterns'ını takip et
- **Keyboard First:** Annotation editor'da full keyboard support
- **Progressive Disclosure:** Tabs ile organize et
- **Visual Feedback:** Loading states, save indicators, progress bars
- **Error Prevention:** Confirmation dialogs, undo support
- **Efficiency:** Bulk actions, shortcuts, quick navigation

### Kullanılacak shadcn/ui Components
- `Card`, `Dialog`, `Sheet`, `Tabs` - Layout
- `Table`, `DataTable` - List views
- `Button`, `Input`, `Select`, `Checkbox` - Forms
- `DropdownMenu`, `ContextMenu` - Actions
- `Toast` (sonner), `AlertDialog` - Feedback
- `Progress`, `Badge` - Status indicators
- `Tooltip`, `Popover` - Help

---

## 🗺️ Sistem Entegrasyonu

### Navigation Yapısı (Grouped Sidebar)

```
┌─────────────────────────────────────────┐
│  🏠 Dashboard                           │
│                                         │
│  ─── PRODUCT RECOGNITION ───            │
│  📦 Products                            │
│  📊 Datasets                            │
│  🏋️ Training                            │
│  🧬 Embeddings                          │
│  🖼️ Cutouts                             │
│  🔗 Matching                            │
│                                         │
│  ─── OBJECT DETECTION ───    🆕         │
│  📷 Images                              │
│  📦 Datasets                            │
│  ✏️ Annotate                            │
│  🏋️ Training                            │
│  🤖 Models                              │
│                                         │
└─────────────────────────────────────────┘
```

### Route Yapısı

```
# Mevcut (değişmez)
/products
/datasets
/training
/embeddings
/cutouts
/matching

# Yeni OD Routes
/od                              # OD Dashboard
/od/images                       # Image library
/od/datasets                     # Dataset list
/od/datasets/[id]                # Dataset detail (tabs: overview, images, classes, versions, export)
/od/annotate                     # Annotation queue (select dataset/image)
/od/annotate/[datasetId]/[imageId]  # Annotation editor
/od/classes                      # Global class management
/od/training                     # Training runs list
/od/training/[id]                # Training detail
/od/training/new                 # New training wizard
/od/models                       # Model registry
/od/models/[id]                  # Model detail
```

### Database Tabloları (Ayrı, çakışma yok)

```
# Mevcut (değişmez)
datasets              → Embedding datasets
dataset_products      → Product-dataset ilişkisi
training_runs         → Embedding training
trained_models        → Embedding models

# Yeni OD Tables (od_ prefix)
od_images             → Shelf/reyon görselleri
od_classes            → Detection class tanımları
od_datasets           → Detection datasets
od_dataset_images     → Dataset-image ilişkisi
od_annotations        → Bounding box annotations
od_dataset_versions   → Dataset snapshots
od_training_runs      → Detection training
od_trained_models     → Detection models
```

---

## Phase 0: Navigation & Layout Setup

> Bu phase, OD modülünün mevcut sisteme entegrasyonunu sağlar.

### 0.1 Sidebar Navigation Update

- [x] **0.1.1** `apps/web/src/components/layout/sidebar.tsx` güncelle
  - [x] "Object Detection" section header eklendi
  - [x] OD menü öğeleri eklendi: Images, Datasets, Annotate, Training, Models
  - [x] İkonlar: `ImageIconOD`, `FolderOpen`, `PenTool`, `Brain`, `Cpu`

### 0.2 OD Route Structure

- [x] **0.2.1** `apps/web/src/app/od/` klasör yapısı oluşturuldu
  - [x] `page.tsx` - OD Dashboard
  - [x] `images/page.tsx` - placeholder
  - [x] `datasets/page.tsx` - placeholder
  - [x] `datasets/[id]/page.tsx` - placeholder
  - [x] `annotate/page.tsx` - placeholder
  - [x] `annotate/[datasetId]/[imageId]/page.tsx` - placeholder
  - [x] `classes/page.tsx` - placeholder
  - [x] `training/page.tsx` - placeholder
  - [x] `training/new/page.tsx` - placeholder
  - [x] `training/[id]/page.tsx` - placeholder
  - [x] `models/page.tsx` - placeholder
  - [x] `models/[id]/page.tsx` - placeholder

- [x] **0.2.2** `/od/page.tsx` - OD Dashboard içeriği
  - [x] Stats cards: Total Images, Datasets, Annotations, Models
  - [x] Quick Actions: Upload Images, Create Dataset, Start Annotating
  - [x] Getting Started guide

### 0.3 Backend OD Router Setup

- [x] **0.3.1** `apps/api/src/api/v1/od/__init__.py` oluşturuldu
- [x] **0.3.2** `apps/api/src/main.py` - OD router eklendi (`/api/v1/od`)
- [x] **0.3.3** `GET /api/v1/od/health` - Health check endpoint
- [x] **0.3.4** `GET /api/v1/od/stats` - Stats endpoint

### 0.4 Frontend API Client

- [x] **0.4.1** `apps/web/src/lib/api-client.ts` - OD API metodları eklendi
- [x] **0.4.2** `apps/web/src/types/od.ts` - Base TypeScript types oluşturuldu

---

## Phase 1: Database & Backend Foundation

### 1.1 Database Migration

- [ ] **1.1.1** `infra/supabase/migrations/030_object_detection.sql` oluştur
  - [ ] `od_images` tablosu
  - [ ] `od_classes` tablosu
  - [ ] `od_datasets` tablosu
  - [ ] `od_dataset_images` tablosu
  - [ ] `od_annotations` tablosu
  - [ ] `od_dataset_versions` tablosu
  - [ ] `od_training_runs` tablosu
  - [ ] `od_trained_models` tablosu
  - [ ] `od_class_changes` tablosu
  - [ ] Indexes oluştur
  - [ ] Triggers oluştur (updated_at)

- [ ] **1.1.2** Migration'ı Supabase'de çalıştır ve test et

### 1.2 Backend API Structure

- [ ] **1.2.1** `apps/api/src/routers/od/` klasörü oluştur
  - [ ] `__init__.py`
  - [ ] `images.py` - Image endpoints
  - [ ] `classes.py` - Class endpoints
  - [ ] `datasets.py` - Dataset endpoints
  - [ ] `annotations.py` - Annotation endpoints
  - [ ] `training.py` - Training endpoints
  - [ ] `models.py` - Model registry endpoints

- [ ] **1.2.2** `apps/api/src/services/od/` klasörü oluştur
  - [ ] `__init__.py`
  - [ ] `image_service.py`
  - [ ] `class_service.py`
  - [ ] `dataset_service.py`
  - [ ] `annotation_service.py`
  - [ ] `export_service.py`
  - [ ] `import_service.py`
  - [ ] `training_service.py`
  - [ ] `buybuddy_sync_service.py`

- [ ] **1.2.3** Main router'a OD router'ı ekle (`apps/api/src/main.py`)

### 1.3 Types & Schemas

- [ ] **1.3.1** `apps/api/src/schemas/od.py` - Pydantic schemas
  - [ ] Image schemas (Create, Update, Response)
  - [ ] Class schemas
  - [ ] Dataset schemas
  - [ ] Annotation schemas
  - [ ] Training schemas
  - [ ] Model schemas

- [ ] **1.3.2** `apps/web/src/types/od.ts` - TypeScript types
  - [ ] ODImage, ODClass, ODDataset, etc.

---

## Phase 2: Image Management (Roboflow-Level)

> Roboflow benzeri gelişmiş image management sistemi

### 2.1 Backend - Core Image API

- [ ] **2.1.1** `GET /api/v1/od/images` - List images
  - [ ] Pagination (limit, offset, cursor-based)
  - [ ] Filters (source, folder, tags, has_annotations, dataset_id)
  - [ ] Search (filename, fuzzy search)
  - [ ] Sort (created_at, filename, annotation_count)
  - [ ] Bulk selection support (return ids for bulk ops)

- [ ] **2.1.2** `POST /api/v1/od/images/upload` - Upload image(s)
  - [ ] Single file upload
  - [ ] Bulk upload support (multipart)
  - [ ] Thumbnail generation (multiple sizes: 128, 256, 512)
  - [ ] Metadata extraction (width, height, format, EXIF)
  - [ ] Auto-orient based on EXIF
  - [ ] Perceptual hash (pHash) calculation for duplicate detection
  - [ ] Supabase Storage'a kaydet
  - [ ] Optional auto-resize (max dimension config)

- [ ] **2.1.3** `GET /api/v1/od/images/{id}` - Get image detail
  - [ ] Include annotation count
  - [ ] Include datasets it belongs to
  - [ ] Include similar images (by pHash)

- [ ] **2.1.4** `PATCH /api/v1/od/images/{id}` - Update image metadata
  - [ ] Folder assignment
  - [ ] Tags (add/remove)
  - [ ] Custom metadata fields

- [ ] **2.1.5** `DELETE /api/v1/od/images/{id}` - Delete image
  - [ ] Storage'dan da sil
  - [ ] İlişkili annotation'ları kontrol et
  - [ ] Cascade delete option (force=true)
  - [ ] Soft delete support (is_deleted flag)

- [ ] **2.1.6** `POST /api/v1/od/images/bulk` - Bulk operations
  - [ ] Bulk delete
  - [ ] Bulk tag assignment
  - [ ] Bulk folder move
  - [ ] Bulk add to dataset

### 2.2 Backend - Import APIs

- [ ] **2.2.1** `POST /api/v1/od/images/import/url` - Import from URL(s)
  - [ ] Single URL
  - [ ] URL list (newline separated)
  - [ ] Download with retry
  - [ ] Validate image format
  - [ ] Return import job ID for tracking

- [ ] **2.2.2** `POST /api/v1/od/images/import/buybuddy` - BuyBuddy API sync
  - [ ] Date range filter
  - [ ] Store filter (multi-select)
  - [ ] Merchant filter
  - [ ] Status filter (approved, pending, rejected)
  - [ ] Preview mode (dry-run, count only)
  - [ ] Duplicate check (buybuddy_image_id + pHash)
  - [ ] Auto-tag by store name
  - [ ] Save source metadata (store, merchant, capture_date)
  - [ ] Pagination & batch processing
  - [ ] Background job with progress tracking

- [ ] **2.2.3** `POST /api/v1/od/images/import/annotated` - Import annotated dataset
  - [ ] Support COCO JSON format
  - [ ] Support YOLO format (data.yaml + labels/)
  - [ ] Support Pascal VOC (XML per image)
  - [ ] Support LabelMe JSON
  - [ ] Support CVAT XML export
  - [ ] Class mapping interface (source → target)
  - [ ] Auto-create missing classes option
  - [ ] Validation (bbox bounds, class existence)
  - [ ] Conflict resolution (skip/replace/merge annotations)
  - [ ] Preview mode with stats
  - [ ] Background job for large imports

- [ ] **2.2.4** `POST /api/v1/od/images/import/zip` - Import from ZIP
  - [ ] Extract and process images
  - [ ] Detect annotation files (COCO, YOLO, VOC)
  - [ ] Auto-detect format
  - [ ] Nested folder support

- [ ] **2.2.5** `GET /api/v1/od/images/import/{jobId}` - Import job status
  - [ ] Progress percentage
  - [ ] Processed / total count
  - [ ] Errors list
  - [ ] Result summary

### 2.3 Backend - Duplicate Detection

- [ ] **2.3.1** `POST /api/v1/od/images/check-duplicates` - Check for duplicates
  - [ ] Input: file hash or pHash
  - [ ] Return matching images with similarity score
  - [ ] Threshold configuration

- [ ] **2.3.2** `GET /api/v1/od/images/duplicates` - List duplicate groups
  - [ ] Group by pHash similarity
  - [ ] Include annotation status per duplicate
  - [ ] Suggest which to keep (most annotated)

- [ ] **2.3.3** `POST /api/v1/od/images/duplicates/resolve` - Resolve duplicates
  - [ ] Keep one, delete others
  - [ ] Merge annotations option
  - [ ] Bulk resolve

### 2.4 Frontend - Image Library Page

- [ ] **2.4.1** `/od/images/page.tsx` - Main image library
  - [ ] Grid view (thumbnail cards)
  - [ ] Table view toggle
  - [ ] Infinite scroll / pagination toggle
  - [ ] Multi-select with Shift+Click, Ctrl+Click
  - [ ] Select all / deselect all
  - [ ] Bulk action bar (when items selected)

- [ ] **2.4.2** Filter sidebar
  - [ ] Source filter (upload, url, buybuddy, import)
  - [ ] Folder tree navigation
  - [ ] Tag filter (multi-select)
  - [ ] Annotation status (annotated, unannotated, partial)
  - [ ] Dataset membership filter
  - [ ] Date range filter
  - [ ] Dimension filter (min/max width/height)

- [ ] **2.4.3** Search & sort
  - [ ] Filename search (fuzzy)
  - [ ] Sort by: date, name, annotation count, size
  - [ ] Ascending / descending toggle

- [ ] **2.4.4** Image card component
  - [ ] Thumbnail with lazy loading
  - [ ] Selection checkbox
  - [ ] Quick annotation count badge
  - [ ] Source icon indicator
  - [ ] Hover: show preview + quick actions

### 2.5 Frontend - Advanced Import Modal

- [ ] **2.5.1** Import modal - Tab structure
  ```
  [📤 Upload] [🔗 URL] [📦 Annotated] [🛒 BuyBuddy] [☁️ Cloud]
  ```

- [ ] **2.5.2** Upload Tab
  - [ ] Drag & drop zone (full modal size)
  - [ ] File picker button
  - [ ] Folder upload support
  - [ ] Upload queue list with status per file
  - [ ] Duplicate indicator (⚠️ already exists)
  - [ ] Remove from queue button
  - [ ] Progress bar per file + total
  - [ ] Options: skip duplicates, auto-resize, auto-tag

- [ ] **2.5.3** URL Import Tab
  - [ ] Text area for URL list
  - [ ] Paste from clipboard button
  - [ ] URL validation (format check)
  - [ ] Preview thumbnails before import
  - [ ] Detected count display

- [ ] **2.5.4** Annotated Import Tab (Key Feature!)
  - [ ] Format selector: COCO, YOLO, VOC, LabelMe, CVAT
  - [ ] File drop zone for annotation file + images
  - [ ] ZIP upload option
  - [ ] **Preview panel:**
    - [ ] Image count
    - [ ] Class list found
    - [ ] Annotation count
    - [ ] Sample image with annotations overlay
  - [ ] **Class mapping table:**
    - [ ] Source class name
    - [ ] Target class dropdown (existing classes + "Create New")
    - [ ] Auto-map similar names (fuzzy match)
    - [ ] Skip class option
    - [ ] Color preview
  - [ ] **Options:**
    - [ ] Skip duplicate images
    - [ ] Replace existing annotations
    - [ ] Merge annotations (add to existing)
    - [ ] Auto-create missing classes
    - [ ] Assign to dataset (optional)
  - [ ] **Validation errors display**
  - [ ] Import progress with detailed log

- [ ] **2.5.5** BuyBuddy Sync Tab
  - [ ] Date range picker
  - [ ] Store multi-select dropdown (fetch from API)
  - [ ] Merchant filter
  - [ ] Status checkboxes (approved, pending, rejected)
  - [ ] **Preview button** → shows count before import
  - [ ] Preview panel:
    - [ ] Total found
    - [ ] New images (not in system)
    - [ ] Duplicates (will skip)
    - [ ] Sample thumbnails
  - [ ] Options:
    - [ ] Import only new
    - [ ] Save metadata (store, merchant, date)
    - [ ] Auto-tag by store
  - [ ] Progress tracking for large syncs

- [ ] **2.5.6** Cloud Storage Tab (Phase 2 - Optional)
  - [ ] S3 bucket configuration
  - [ ] Path/prefix input
  - [ ] List files button
  - [ ] Select files to import
  - [ ] GCS support
  - [ ] Azure Blob support

### 2.6 Frontend - Image Detail

- [ ] **2.6.1** Image detail modal/drawer
  - [ ] Large image preview with zoom
  - [ ] Annotation overlay toggle
  - [ ] Metadata panel (dimensions, format, size, dates)
  - [ ] Source info (origin, buybuddy metadata)
  - [ ] Tags editor (add/remove)
  - [ ] Folder selector
  - [ ] Datasets list (which datasets contain this image)
  - [ ] Similar images section (by pHash)
  - [ ] Quick actions: Edit in annotator, Delete, Add to dataset

### 2.7 Frontend - Duplicate Management

- [ ] **2.7.1** Duplicates page/modal
  - [ ] List duplicate groups
  - [ ] Side-by-side comparison
  - [ ] Annotation status per duplicate
  - [ ] "Keep this one" button
  - [ ] Bulk resolve all duplicates
  - [ ] Merge annotations option

---

## Phase 3: Dataset Management

### 3.1 Backend - Dataset API

- [ ] **3.1.1** `GET /api/v1/od/datasets` - List datasets
  - [ ] Include stats (image_count, annotated_count, class_count)
  - [ ] Filter by status (active, archived)
  - [ ] Sort options

- [ ] **3.1.2** `POST /api/v1/od/datasets` - Create dataset
  - [ ] Name, description
  - [ ] Annotation type (bbox, polygon, both)
  - [ ] Initial class assignment (from templates or custom)
  - [ ] Auto-create from import option

- [ ] **3.1.3** `GET /api/v1/od/datasets/{id}` - Get dataset with stats
  - [ ] Full stats (images, annotations, classes)
  - [ ] Progress metrics
  - [ ] Recent activity

- [ ] **3.1.4** `PATCH /api/v1/od/datasets/{id}` - Update dataset
- [ ] **3.1.5** `DELETE /api/v1/od/datasets/{id}` - Delete dataset
  - [ ] Soft delete (archive) option
  - [ ] Hard delete with confirmation

### 3.2 Backend - Dataset Images API

- [ ] **3.2.1** `GET /api/v1/od/datasets/{id}/images` - List images in dataset
  - [ ] Pagination
  - [ ] Filter by status (pending, annotated, reviewed, skipped)
  - [ ] Filter by annotation count
  - [ ] Sort by various fields
  - [ ] Include annotation preview

- [ ] **3.2.2** `POST /api/v1/od/datasets/{id}/images` - Add images to dataset
  - [ ] Add from image library (image IDs list)
  - [ ] Skip duplicates option
  - [ ] Copy annotations option (if image has annotations from other dataset)
  - [ ] Batch processing

- [ ] **3.2.3** `POST /api/v1/od/datasets/{id}/images/upload` - Upload directly to dataset
  - [ ] Same as main upload but auto-assigns to dataset
  - [ ] Inherits dataset classes

- [ ] **3.2.4** `POST /api/v1/od/datasets/{id}/images/import` - Import annotated to dataset
  - [ ] Import with annotations directly into dataset
  - [ ] Class mapping scoped to dataset classes
  - [ ] Auto-create classes in dataset

- [ ] **3.2.5** `DELETE /api/v1/od/datasets/{id}/images/{imageId}` - Remove from dataset
  - [ ] Remove link only (keep image in library)
  - [ ] Delete annotations in this dataset option

- [ ] **3.2.6** `POST /api/v1/od/datasets/{id}/images/bulk` - Bulk operations
  - [ ] Bulk remove from dataset
  - [ ] Bulk status change
  - [ ] Bulk annotation operations

### 3.3 Backend - Dataset Stats

- [ ] **3.3.1** `GET /api/v1/od/datasets/{id}/stats` - Dataset statistics
  - [ ] Image count by status (pending/annotated/reviewed/skipped)
  - [ ] Annotation count by class
  - [ ] Class distribution data (for charts)
  - [ ] Annotation density stats (avg per image)
  - [ ] Progress over time (daily/weekly)

- [ ] **3.3.2** `GET /api/v1/od/datasets/{id}/health` - Dataset health check
  - [ ] Images without annotations
  - [ ] Classes without usage
  - [ ] Annotation quality metrics
  - [ ] Potential issues list

### 3.4 Frontend - Dataset List Page

- [ ] **3.4.1** `/od/datasets/page.tsx` - Dataset list
  - [ ] Card grid view with preview
  - [ ] Table view toggle
  - [ ] Stats preview (image count, progress bar)
  - [ ] Quick actions (annotate, export, delete)
  - [ ] Create dataset button
  - [ ] Import dataset button (creates new dataset from import)

### 3.5 Frontend - Create Dataset

- [ ] **3.5.1** Create dataset modal/wizard
  - [ ] Step 1: Basic info (name, description)
  - [ ] Step 2: Annotation type (bbox, polygon)
  - [ ] Step 3: Initial classes
    - [ ] Use template classes
    - [ ] Import from another dataset
    - [ ] Create custom classes
    - [ ] Skip (add later)
  - [ ] Step 4: Add initial images (optional)
    - [ ] From library
    - [ ] Upload new
    - [ ] Import annotated
    - [ ] Skip (add later)

### 3.6 Frontend - Dataset Detail Page

- [ ] **3.6.1** `/od/datasets/[id]/page.tsx` - Dataset detail with tabs

#### Overview Tab
- [ ] **3.6.2** Stats cards (total images, annotated, annotations, classes)
- [ ] **3.6.3** Progress bar (annotation completion)
- [ ] **3.6.4** Class distribution chart (pie/bar)
- [ ] **3.6.5** Annotation activity chart (over time)
- [ ] **3.6.6** Quick actions (Start annotating, Export, Add images)
- [ ] **3.6.7** Health alerts (if issues detected)

#### Images Tab
- [ ] **3.6.8** Image grid with status indicators
- [ ] **3.6.9** Status filter tabs (All, Pending, Annotated, Reviewed, Skipped)
- [ ] **3.6.10** Multi-select support
- [ ] **3.6.11** Bulk actions (remove, change status, re-annotate)
- [ ] **3.6.12** **"Add Images" button → Advanced Add Modal**

#### Classes Tab
- [ ] **3.6.13** Class list with annotation counts
- [ ] **3.6.14** Add/edit/delete classes
- [ ] **3.6.15** Class color management
- [ ] **3.6.16** Merge classes (within dataset)
- [ ] **3.6.17** Import classes from template/other dataset

#### Versions Tab (Phase 7)
#### Export Tab (Phase 7)

### 3.7 Frontend - Add Images to Dataset Modal (Key Feature!)

> Dataset'e image eklemek için gelişmiş modal - Roboflow seviyesinde

- [ ] **3.7.1** Add Images Modal - Tab structure
  ```
  [📚 Library] [📤 Upload] [📦 Import Annotated] [🛒 BuyBuddy]
  ```

- [ ] **3.7.2** Library Tab (Add from existing images)
  - [ ] Image grid from main library
  - [ ] Filter: not in this dataset
  - [ ] Multi-select with checkboxes
  - [ ] Search and filter
  - [ ] Preview selection count
  - [ ] "Add X images" button
  - [ ] Option: Copy existing annotations (if any)

- [ ] **3.7.3** Upload Tab (Upload new images directly to dataset)
  - [ ] Drag & drop zone
  - [ ] File picker
  - [ ] Upload queue with status
  - [ ] Duplicate check (within dataset + library)
  - [ ] Progress tracking
  - [ ] Options:
    - [ ] Also add to main library
    - [ ] Skip duplicates
    - [ ] Auto-resize

- [ ] **3.7.4** Import Annotated Tab (Import with annotations to dataset)
  - [ ] Format selector (COCO, YOLO, VOC, etc.)
  - [ ] File upload (annotations + images)
  - [ ] **Class mapping UI:**
    - [ ] Map to existing dataset classes
    - [ ] Create new class in dataset
    - [ ] Skip class
    - [ ] Auto-map by name similarity
  - [ ] Preview:
    - [ ] Image count
    - [ ] Annotation count
    - [ ] Sample with overlay
  - [ ] Options:
    - [ ] Add images to library also
    - [ ] Merge/replace existing annotations
  - [ ] Validation errors display
  - [ ] Import progress

- [ ] **3.7.5** BuyBuddy Tab (Sync from BuyBuddy to dataset)
  - [ ] Same filters as main BuyBuddy import
  - [ ] Preview count
  - [ ] Direct add to this dataset
  - [ ] Option: Also add to library

- [ ] **3.7.6** Common footer
  - [ ] Selected count summary
  - [ ] Cancel button
  - [ ] Add/Import button with count

---

## Phase 4: Class Management

### 4.1 Backend - Class API

- [ ] **4.1.1** `GET /api/v1/od/classes` - List classes
  - [ ] Filter by category
  - [ ] Search by name
  - [ ] Include annotation count

- [ ] **4.1.2** `POST /api/v1/od/classes` - Create class
- [ ] **4.1.3** `PATCH /api/v1/od/classes/{id}` - Update/rename class
  - [ ] Alias'lara eski ismi ekle
  - [ ] Audit log kaydı

- [ ] **4.1.4** `DELETE /api/v1/od/classes/{id}` - Delete class
  - [ ] İlişkili annotation kontrolü
  - [ ] Audit log

- [ ] **4.1.5** `POST /api/v1/od/classes/merge` - Merge classes
  - [ ] Source class annotation'larını target'a taşı
  - [ ] Source class'ı sil
  - [ ] Audit log

### 4.2 Frontend - Class Management

- [ ] **4.2.1** `/od/classes/page.tsx` - Class list
  - [ ] Table view
  - [ ] Category filter
  - [ ] Search
  - [ ] Color indicator

- [ ] **4.2.2** Create/Edit class modal
  - [ ] Name input
  - [ ] Display name
  - [ ] Color picker
  - [ ] Category select

- [ ] **4.2.3** Merge classes modal
  - [ ] Source class(es) selection
  - [ ] Target class selection
  - [ ] Affected annotation count preview

---

## Phase 5: Annotation Editor

### 5.1 Backend - Annotation API

- [ ] **5.1.1** `GET /api/v1/od/datasets/{id}/images/{imgId}/annotations`
- [ ] **5.1.2** `POST /api/v1/od/datasets/{id}/images/{imgId}/annotations`
- [ ] **5.1.3** `PATCH /api/v1/od/annotations/{annId}`
- [ ] **5.1.4** `DELETE /api/v1/od/annotations/{annId}`
- [ ] **5.1.5** `POST /api/v1/od/annotations/bulk` - Bulk create/update
- [ ] **5.1.6** `DELETE /api/v1/od/annotations/bulk` - Bulk delete

- [ ] **5.1.7** Image status update endpoint
  - [ ] Mark as completed
  - [ ] Mark as skipped
  - [ ] Update annotation count

### 5.2 Frontend - Annotation Editor

- [ ] **5.2.1** `/od/annotate/[datasetId]/[imageId]/page.tsx` - Main editor page
  - [ ] Layout: Canvas + Sidebar

- [ ] **5.2.2** Canvas Component (`AnnotationCanvas.tsx`)
  - [ ] Image rendering
  - [ ] Zoom (scroll wheel)
  - [ ] Pan (middle mouse / space+drag)
  - [ ] Fit to screen

- [ ] **5.2.3** BBox Drawing
  - [ ] Draw mode (click+drag)
  - [ ] BBox rendering with class color
  - [ ] Class label display on bbox

- [ ] **5.2.4** BBox Editing
  - [ ] Select bbox (click)
  - [ ] Move bbox (drag)
  - [ ] Resize bbox (corner handles)
  - [ ] Delete bbox (Delete key)

- [ ] **5.2.5** Sidebar - Annotation List
  - [ ] List all annotations
  - [ ] Click to select/highlight
  - [ ] Delete button
  - [ ] Class dropdown per annotation

- [ ] **5.2.6** Sidebar - Class Selector
  - [ ] Class list with colors
  - [ ] Quick select (keyboard 1-9)
  - [ ] Search classes

- [ ] **5.2.7** Toolbar
  - [ ] Tool selection (Select, Draw)
  - [ ] Zoom controls
  - [ ] Undo/Redo buttons
  - [ ] Save status indicator

- [ ] **5.2.8** Keyboard Shortcuts
  - [ ] `1-9` - Select class
  - [ ] `D` - Delete selected
  - [ ] `Ctrl+Z` - Undo
  - [ ] `Ctrl+Y` - Redo
  - [ ] `Ctrl+S` - Save
  - [ ] `Space` - Pan mode
  - [ ] `Escape` - Deselect

- [ ] **5.2.9** Navigation
  - [ ] Previous/Next image buttons
  - [ ] Image counter (5/120)
  - [ ] Keyboard: Arrow Left/Right

- [ ] **5.2.10** Undo/Redo System
  - [ ] Action history stack
  - [ ] Undo: create, delete, move, resize
  - [ ] Redo support

- [ ] **5.2.11** Auto-save
  - [ ] Debounced save (2 saniye inactivity sonrası)
  - [ ] Save on navigate away
  - [ ] Unsaved changes warning

---

## Phase 6: AI Auto-Annotation Worker

### 6.1 Worker Setup

- [ ] **6.1.1** `workers/od-annotation/` klasör yapısı
  ```
  workers/od-annotation/
  ├── Dockerfile
  ├── handler.py
  ├── requirements.txt
  └── models/
      ├── __init__.py
      ├── grounding_dino.py
      ├── owlv2.py
      ├── florence2.py
      ├── sam2.py
      └── custom_model.py
  ```

- [ ] **6.1.2** Dockerfile oluştur
  - [ ] CUDA base image
  - [ ] Python dependencies
  - [ ] Model weights download

- [ ] **6.1.3** `handler.py` - RunPod handler
  - [ ] Job input parsing
  - [ ] Model selection
  - [ ] Batch processing
  - [ ] Result formatting

### 6.2 Model Integrations

- [ ] **6.2.1** Grounding DINO 1.5
  - [ ] Model loading
  - [ ] Text prompt → bbox inference
  - [ ] Confidence threshold

- [ ] **6.2.2** OWLv2
  - [ ] Model loading
  - [ ] Text queries → bbox inference

- [ ] **6.2.3** Florence-2
  - [ ] Model loading
  - [ ] Detection task
  - [ ] Phrase grounding task

- [ ] **6.2.4** SAM 2.1
  - [ ] Model loading
  - [ ] Point prompt → mask
  - [ ] Box prompt → mask
  - [ ] Auto segmentation mode
  - [ ] Mask → bbox conversion

- [ ] **6.2.5** Custom Model (trained models)
  - [ ] Model download from Supabase
  - [ ] Dynamic loading
  - [ ] Inference

### 6.3 Backend - AI Annotation API

- [ ] **6.3.1** `POST /api/v1/od/ai/annotate` - Start auto-annotation job
  - [ ] RunPod job trigger
  - [ ] Job status tracking

- [ ] **6.3.2** `GET /api/v1/od/ai/annotate/{jobId}` - Get job status

- [ ] **6.3.3** `POST /api/v1/od/ai/segment` - SAM segmentation (real-time)
  - [ ] Point/box input
  - [ ] Return mask

- [ ] **6.3.4** Webhook endpoint for RunPod callback
  - [ ] Save predictions to database
  - [ ] Update image status

### 6.4 Frontend - AI Annotation UI

- [ ] **6.4.1** AI Annotation panel in editor
  - [ ] Model selection dropdown
  - [ ] Text prompt input (for Grounding DINO, OWLv2)
  - [ ] Confidence threshold slider
  - [ ] "Auto-Annotate" button

- [ ] **6.4.2** AI predictions display
  - [ ] Different style (dashed border)
  - [ ] Confidence badge
  - [ ] Accept/Reject buttons
  - [ ] Accept All / Reject All

- [ ] **6.4.3** SAM integration
  - [ ] Click on image → point prompt
  - [ ] Draw box → box prompt
  - [ ] Show mask overlay
  - [ ] Convert to bbox

- [ ] **6.4.4** Bulk auto-annotation modal
  - [ ] Select images (all, unannotated, selected)
  - [ ] Model selection
  - [ ] Configuration
  - [ ] Progress tracking

---

## Phase 7: Export/Import

### 7.1 Import Service

- [ ] **7.1.1** COCO Import (`import_service.py`)
  - [ ] Parse annotations.json
  - [ ] Download/copy images
  - [ ] Create od_images records
  - [ ] Map categories → od_classes
  - [ ] Create od_annotations

- [ ] **7.1.2** YOLO Import
  - [ ] Parse data.yaml
  - [ ] Read label .txt files
  - [ ] Map class indices → od_classes
  - [ ] Create annotations

- [ ] **7.1.3** Pascal VOC Import
  - [ ] Parse XML files
  - [ ] Extract bboxes
  - [ ] Create annotations

- [ ] **7.1.4** Import API endpoints
  - [ ] `POST /api/v1/od/images/import/coco`
  - [ ] `POST /api/v1/od/images/import/yolo`
  - [ ] `POST /api/v1/od/images/import/voc`

### 7.2 Export Service

- [ ] **7.2.1** YOLO Export (`export_service.py`)
  - [ ] Generate data.yaml
  - [ ] Generate label .txt files
  - [ ] Copy images to train/val folders
  - [ ] Create ZIP

- [ ] **7.2.2** COCO Export
  - [ ] Generate annotations JSON
  - [ ] Copy images
  - [ ] Create ZIP

- [ ] **7.2.3** Export API endpoints
  - [ ] `POST /api/v1/od/datasets/{id}/export/yolo`
  - [ ] `POST /api/v1/od/datasets/{id}/export/coco`
  - [ ] `GET /api/v1/od/export/{jobId}/download`

### 7.3 Dataset Versioning

- [ ] **7.3.1** Version API
  - [ ] `POST /api/v1/od/datasets/{id}/versions` - Create version
  - [ ] `GET /api/v1/od/datasets/{id}/versions` - List versions
  - [ ] `GET /api/v1/od/datasets/{id}/versions/{v}` - Get version

- [ ] **7.3.2** Version creation logic
  - [ ] Snapshot current state
  - [ ] Freeze class mapping
  - [ ] Generate train/val/test split
  - [ ] Store split config

### 7.4 Frontend - Export/Import UI

- [ ] **7.4.1** Import tab in modal (complete)
  - [ ] COCO format option
  - [ ] YOLO format option
  - [ ] File upload
  - [ ] Progress tracking

- [ ] **7.4.2** Export tab in dataset detail
  - [ ] Format selection
  - [ ] Preprocessing options (resize, normalize)
  - [ ] Augmentation options
  - [ ] Split configuration
  - [ ] Download button

- [ ] **7.4.3** Versions tab in dataset detail
  - [ ] Version list
  - [ ] Create version button
  - [ ] Version detail (stats, split info)
  - [ ] Quick export from version

---

## Phase 8: Training Pipeline

### 8.1 Training Worker

- [ ] **8.1.1** `workers/od-training/` klasör yapısı
  ```
  workers/od-training/
  ├── Dockerfile
  ├── handler.py
  ├── requirements.txt
  └── trainers/
      ├── __init__.py
      ├── rf_detr_trainer.py
      ├── rt_detr_trainer.py
      └── yolo_nas_trainer.py
  ```

- [ ] **8.1.2** Dockerfile oluştur
  - [ ] CUDA base image
  - [ ] Training dependencies
  - [ ] Framework installations

- [ ] **8.1.3** RF-DETR Trainer
  - [ ] Dataset preparation (COCO format)
  - [ ] Model initialization
  - [ ] Training loop
  - [ ] Checkpoint saving
  - [ ] Metrics logging

- [ ] **8.1.4** RT-DETRv2 Trainer
  - [ ] Similar to RF-DETR

- [ ] **8.1.5** YOLO-NAS Trainer
  - [ ] Dataset preparation (YOLO format)
  - [ ] SuperGradients integration
  - [ ] Training loop

- [ ] **8.1.6** Common training utilities
  - [ ] Progress callback → API webhook
  - [ ] Metrics calculation (mAP)
  - [ ] Best checkpoint tracking
  - [ ] Model upload to Supabase

### 8.2 Backend - Training API

- [ ] **8.2.1** `POST /api/v1/od/training` - Start training
  - [ ] Validate dataset version
  - [ ] Prepare config
  - [ ] Trigger RunPod job
  - [ ] Create training_run record

- [ ] **8.2.2** `GET /api/v1/od/training` - List training runs
- [ ] **8.2.3** `GET /api/v1/od/training/{id}` - Get training run detail
- [ ] **8.2.4** `DELETE /api/v1/od/training/{id}` - Cancel training
- [ ] **8.2.5** `GET /api/v1/od/training/{id}/metrics` - Get metrics history
- [ ] **8.2.6** `GET /api/v1/od/training/{id}/logs` - Get training logs

- [ ] **8.2.7** Webhook endpoint
  - [ ] Update progress
  - [ ] Update metrics
  - [ ] Handle completion
  - [ ] Handle failure

### 8.3 Frontend - Training UI

- [ ] **8.3.1** `/od/training/page.tsx` - Training runs list
  - [ ] Table with status, progress
  - [ ] Filter by status
  - [ ] New training button

- [ ] **8.3.2** New training wizard
  - [ ] Step 1: Select dataset/version
  - [ ] Step 2: Select model (RF-DETR, RT-DETR, YOLO-NAS)
  - [ ] Step 3: Configure hyperparameters
  - [ ] Step 4: Review & start

- [ ] **8.3.3** `/od/training/[id]/page.tsx` - Training detail
  - [ ] Progress bar
  - [ ] Loss curve chart (recharts)
  - [ ] mAP chart
  - [ ] Current epoch / total
  - [ ] Logs viewer
  - [ ] Cancel button
  - [ ] Download model (when complete)

---

## Phase 9: Model Registry

### 9.1 Backend - Model API

- [ ] **9.1.1** `GET /api/v1/od/models` - List trained models
- [ ] **9.1.2** `GET /api/v1/od/models/{id}` - Get model detail
- [ ] **9.1.3** `PATCH /api/v1/od/models/{id}` - Update model (name, active)
- [ ] **9.1.4** `DELETE /api/v1/od/models/{id}` - Delete model
- [ ] **9.1.5** `POST /api/v1/od/models/{id}/set-default` - Set as annotation default

- [ ] **9.1.6** `POST /api/v1/od/models/{id}/inference` - Test inference
  - [ ] Upload image
  - [ ] Run inference
  - [ ] Return predictions

- [ ] **9.1.7** `POST /api/v1/od/models/{id}/export` - Export model
  - [ ] ONNX export
  - [ ] TensorRT export (optional)

### 9.2 Frontend - Model Registry UI

- [ ] **9.2.1** `/od/models/page.tsx` - Model list
  - [ ] Table view
  - [ ] Metrics display (mAP)
  - [ ] Active/default badges
  - [ ] Filter by base model

- [ ] **9.2.2** `/od/models/[id]/page.tsx` - Model detail
  - [ ] Metrics summary
  - [ ] Class mapping
  - [ ] Training run link
  - [ ] Test inference panel
  - [ ] Export buttons
  - [ ] Set as default button

- [ ] **9.2.3** Test inference panel
  - [ ] Image upload
  - [ ] Run inference button
  - [ ] Display predictions on image
  - [ ] Confidence threshold slider

---

## 🎯 Implementation Order

Önerilen sıralama:

1. **Phase 0** → Navigation & Layout Setup (sisteme entegrasyon)
2. **Phase 1** → Database & backend foundation (her şeyin temeli)
3. **Phase 2** → Image management (veri yükleyebilmek için)
4. **Phase 4** → Class management (annotation'dan önce class lazım)
5. **Phase 3** → Dataset management (image + class hazır olunca)
6. **Phase 5** → Annotation editor (core feature)
7. **Phase 6** → AI auto-annotation (productivity boost)
8. **Phase 7** → Export/Import (training için veri hazırlığı)
9. **Phase 8** → Training pipeline
10. **Phase 9** → Model registry

---

## 📝 Notes

- Her phase tamamlandığında bu dosyayı güncelleyeceğiz
- Blocking issue'lar için `⚠️` işareti kullanacağız
- Completed tasks: `[x]`
- In progress: `[~]`
- Blocked: `[!]`

---

*Son güncelleme: 2026-01-20*

---

## 📊 Feature Comparison: Before vs After

| Feature | Eski (Basit) | Yeni (Roboflow-Level) |
|---------|--------------|----------------------|
| Image Upload | Tek dosya/bulk | Drag&drop, folder, queue, progress |
| URL Import | ❌ | ✅ Tek/çoklu URL, validation |
| Annotated Import | Phase 7'de | ✅ COCO, YOLO, VOC, LabelMe, CVAT |
| Class Mapping | ❌ | ✅ Auto-map, create, skip UI |
| Duplicate Detection | ❌ | ✅ pHash based, similarity groups |
| BuyBuddy Sync | Basit | ✅ Preview, filters, metadata, tags |
| Dataset Image Add | Basit modal | ✅ Multi-tab advanced modal |
| Bulk Operations | ❌ | ✅ Multi-select, batch actions |
| Import Progress | ❌ | ✅ Real-time progress, job tracking |
