# Roboflow Trained Model Integration Plan

## Özet

Roboflow'da train edilmiş modelleri BuyBuddy AI sistemine entegre ederek AI Bulk Annotate özelliğinde kullanılabilir hale getirme.

## Gereksinimler

1. **UI'dan model seçimi** - Kullanıcı Roboflow'daki modellerini görebilmeli ve seçebilmeli (resim seçer gibi)
2. **Local inference** - Roboflow API değil, kendi worker'ımızda inference yapılacak
3. **Hem batch hem real-time** - Bulk annotation ve preview için kullanılabilmeli

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                     │
├─────────────────────────────────────────────────────────────────────┤
│  AI Annotate Modal                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Model Selection:                                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │ Grounding    │  │ Florence-2   │  │ Roboflow     │      │   │
│  │  │ DINO         │  │              │  │ Models ▼     │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │
│  │                                       ↓                      │   │
│  │                         ┌────────────────────────┐          │   │
│  │                         │ slot-detection v15     │          │   │
│  │                         │ product-detection v3   │          │   │
│  │                         │ price-tag-detector v2  │          │   │
│  │                         └────────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           API                                        │
├─────────────────────────────────────────────────────────────────────┤
│  GET  /api/v1/od/ai/roboflow/models     → List available models     │
│  POST /api/v1/od/ai/roboflow/download   → Download model weights    │
│  GET  /api/v1/od/ai/roboflow/models/{id}/status → Download status   │
│                                                                      │
│  POST /api/v1/od/ai/predict             → Single image (+ roboflow) │
│  POST /api/v1/od/ai/batch               → Batch job (+ roboflow)    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OD ANNOTATION WORKER                            │
├─────────────────────────────────────────────────────────────────────┤
│  models/                                                             │
│  ├── grounding_dino.py                                              │
│  ├── florence2.py                                                   │
│  ├── sam2.py                                                        │
│  ├── sam3.py                                                        │
│  └── roboflow_yolo.py  ← NEW                                        │
│                                                                      │
│  weights/                                                            │
│  ├── grounding_dino/                                                │
│  ├── florence2/                                                     │
│  └── roboflow/         ← NEW (downloaded YOLO weights)              │
│      ├── slot-detection-v15.pt                                      │
│      ├── product-detection-v3.pt                                    │
│      └── ...                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Faz 1: Backend - Model Yönetimi

### 1.1 Roboflow Model Listesi Endpoint'i

**Endpoint:** `GET /api/v1/od/ai/roboflow/models`

```python
# Response
{
  "models": [
    {
      "id": "slot-detection/15",
      "name": "Slot Detection",
      "project": "slot-detection-7d5qx",
      "version": 15,
      "type": "yolov8",
      "classes": ["slot"],
      "map": 0.95,
      "status": "available",  # available | downloading | downloaded
      "size_mb": 45,
      "downloaded_at": null
    },
    {
      "id": "product-detection/3",
      "name": "Product Detection",
      "project": "objectdetectionbuybuddy",
      "version": 3,
      "type": "yolov8",
      "classes": ["product", "price_tag"],
      "map": 0.89,
      "status": "downloaded",
      "size_mb": 52,
      "downloaded_at": "2026-01-21T10:00:00Z"
    }
  ]
}
```

### 1.2 Model Download Endpoint'i

**Endpoint:** `POST /api/v1/od/ai/roboflow/download`

```python
# Request
{
  "project": "slot-detection-7d5qx",
  "version": 15,
  "format": "yolov8"  # yolov8 | yolov5 | onnx
}

# Response
{
  "job_id": "uuid",
  "status": "downloading",
  "message": "Downloading model weights..."
}
```

### 1.3 Database Schema

```sql
-- roboflow_models table
CREATE TABLE roboflow_models (
  id UUID PRIMARY KEY,
  project VARCHAR NOT NULL,
  version INT NOT NULL,
  name VARCHAR,
  type VARCHAR,  -- yolov8, yolov5, etc.
  classes JSONB,
  metrics JSONB,  -- mAP, precision, recall
  weights_path VARCHAR,  -- /app/weights/roboflow/...
  status VARCHAR,  -- pending, downloading, ready, failed
  size_bytes BIGINT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  UNIQUE(project, version)
);
```

---

## Faz 2: Worker - Roboflow YOLO Model

### 2.1 RoboflowYOLOModel Class

**File:** `workers/od-annotation/models/roboflow_yolo.py`

```python
from ultralytics import YOLO
from .base import BaseModel

class RoboflowYOLOModel(BaseModel):
    """Roboflow'dan export edilmiş YOLO modeli için wrapper."""

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self.class_names = []

    def _load_model(self):
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)
        self.class_names = self.model.names
        return self.model

    def predict(
        self,
        image_url: str,
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> list[dict]:
        """Run inference on image."""
        image = self.download_image(image_url)

        results = self.model.predict(
            image,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )[0]

        predictions = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Normalized
            predictions.append({
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "label": self.class_names[int(box.cls)],
                "confidence": float(box.conf),
            })

        return predictions
```

### 2.2 Model Factory Güncellemesi

**File:** `workers/od-annotation/models/__init__.py`

```python
def get_model(model_name: str, cache: dict, **kwargs) -> Any:
    if model_name in cache:
        return cache[model_name]

    if model_name == "grounding_dino":
        model = GroundingDINOModel(...)
    elif model_name == "florence2":
        model = Florence2Model(...)
    elif model_name == "sam2":
        model = SAM2Model(...)
    elif model_name == "sam3":
        model = SAM3Model(...)
    elif model_name.startswith("roboflow:"):
        # Format: "roboflow:slot-detection/15"
        _, model_id = model_name.split(":", 1)
        weights_path = f"/app/weights/roboflow/{model_id.replace('/', '-')}.pt"
        model = RoboflowYOLOModel(weights_path=weights_path, device=config.device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cache[model_name] = model
    return model
```

---

## Faz 3: Frontend - Model Seçimi UI

### 3.1 Roboflow Models Hook

**File:** `hooks/use-roboflow-models.ts`

```typescript
export function useRoboflowModels() {
  return useQuery({
    queryKey: ["roboflow-models"],
    queryFn: () => apiClient.getRoboflowAIModels(),
  });
}

export function useDownloadRoboflowModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (params: { project: string; version: number }) =>
      apiClient.downloadRoboflowModel(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["roboflow-models"] });
    },
  });
}
```

### 3.2 Model Selector Component

**File:** `components/od/roboflow-model-selector.tsx`

```tsx
export function RoboflowModelSelector({
  value,
  onChange
}: {
  value: string | null;
  onChange: (modelId: string) => void;
}) {
  const { data: models, isLoading } = useRoboflowModels();
  const downloadMutation = useDownloadRoboflowModel();

  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger>
        <SelectValue placeholder="Select Roboflow model..." />
      </SelectTrigger>
      <SelectContent>
        {models?.map((model) => (
          <SelectItem
            key={model.id}
            value={`roboflow:${model.id}`}
            disabled={model.status === "downloading"}
          >
            <div className="flex items-center gap-2">
              <span>{model.name} v{model.version}</span>
              {model.status === "downloaded" && (
                <Badge variant="success">Ready</Badge>
              )}
              {model.status === "downloading" && (
                <Badge variant="secondary">
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  Downloading
                </Badge>
              )}
              {model.status === "available" && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    downloadMutation.mutate({
                      project: model.project,
                      version: model.version,
                    });
                  }}
                >
                  <Download className="h-3 w-3" />
                </Button>
              )}
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
```

---

## Faz 4: AI Annotate Modal Güncellemesi

### 4.1 Model Kategorileri

```tsx
// AI Annotate Modal'da model seçimi
<Tabs defaultValue="general">
  <TabsList>
    <TabsTrigger value="general">General Models</TabsTrigger>
    <TabsTrigger value="roboflow">Roboflow Models</TabsTrigger>
  </TabsList>

  <TabsContent value="general">
    {/* Grounding DINO, Florence-2, SAM3 */}
    <Select>
      <SelectItem value="grounding_dino">Grounding DINO</SelectItem>
      <SelectItem value="florence2">Florence-2</SelectItem>
      <SelectItem value="sam3">SAM3</SelectItem>
    </Select>
  </TabsContent>

  <TabsContent value="roboflow">
    <RoboflowModelSelector
      value={selectedModel}
      onChange={setSelectedModel}
    />
  </TabsContent>
</Tabs>
```

---

## Implementasyon Sırası

| Adım | Görev | Dosyalar | Zorluk |
|------|-------|----------|--------|
| 1 | Database schema oluştur | Supabase migration | Kolay |
| 2 | Roboflow model listesi endpoint'i | `api/v1/od/ai.py` | Kolay |
| 3 | Model download endpoint'i | `api/v1/od/ai.py` | Orta |
| 4 | RoboflowYOLOModel class | `workers/od-annotation/models/roboflow_yolo.py` | Orta |
| 5 | Model factory güncellemesi | `workers/od-annotation/models/__init__.py` | Kolay |
| 6 | Frontend hook'ları | `hooks/use-roboflow-models.ts` | Kolay |
| 7 | Model selector component | `components/od/roboflow-model-selector.tsx` | Orta |
| 8 | AI Annotate modal güncellemesi | `components/od/ai-annotate-modal.tsx` | Orta |
| 9 | Docker image güncellemesi | `workers/od-annotation/Dockerfile` | Kolay |
| 10 | Test ve dokümantasyon | - | Orta |

---

## Notlar

### Roboflow Model Export Formatları

| Format | Avantaj | Dezavantaj |
|--------|---------|------------|
| YOLOv8 | Hızlı, Ultralytics desteği | Sadece detection |
| YOLOv5 | Yaygın kullanım | Eski versiyon |
| ONNX | Platform bağımsız | Ekstra setup |
| TensorRT | En hızlı inference | NVIDIA only |

**Öneri:** YOLOv8 formatı kullan (Ultralytics ile kolay entegrasyon)

### Model Weights Storage

```
/app/weights/roboflow/
├── slot-detection-15.pt        # YOLOv8 weights
├── slot-detection-15.yaml      # Class names, metadata
├── product-detection-3.pt
└── product-detection-3.yaml
```

### API Key Yönetimi

- Roboflow API key `.env`'de saklanacak
- Model download işlemi için gerekli
- Inference sırasında API key'e gerek yok (local weights)

---

## Zaman Tahmini

- Faz 1 (Backend): 2-3 saat
- Faz 2 (Worker): 2-3 saat
- Faz 3 (Frontend): 2-3 saat
- Faz 4 (Integration): 1-2 saat
- Test: 1-2 saat

**Toplam:** ~10-13 saat (1.5-2 gün)
