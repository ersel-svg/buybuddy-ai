# Roboflow Streaming Import - Implementation Plan

## Problem
Mevcut sistem 10GB+ ZIP indiriyor, bu:
- Çok yavaş (önce indir, sonra işle)
- Disk alanı gerektirir
- ZIP bozulursa baştan başlamak gerekir

## Çözüm
Roboflow API ile direkt stream: Her görseli tek tek al, direkt Supabase'e yükle.

---

## Değişiklik Yapılacak Dosyalar

### 1. `services/roboflow.py` (Yeni metodlar)

```python
async def list_all_images(self, api_key, workspace, project) -> AsyncGenerator[dict, None]:
    """Generator that yields all images from a project."""

async def get_image_with_annotations(self, api_key, workspace, project, image_id) -> dict:
    """Get single image with full details including URLs and annotations."""
```

### 2. `services/roboflow_streaming.py` (YENİ DOSYA)

Streaming import servisi:
```python
class RoboflowStreamingImporter:
    async def import_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        dataset_id: str,
        class_mapping: list,
        concurrency: int = 20,
        progress_callback: Callable = None,
    ) -> ImportResult
```

### 3. `api/v1/od/images.py` (Import endpoint güncelleme)

Yeni `_run_roboflow_import_streaming()` fonksiyonu:
- Eski ZIP-based yerine streaming kullanacak
- Aynı endpoint, sadece backend değişiyor

---

## Yeni Akış

```
1. project.search_all() → Tüm image ID listesi (hızlı, sadece ID'ler)

2. Her image için paralel (20 concurrent):
   ├─ project.image(id) → URL + annotation al
   ├─ URL'den stream → Supabase Storage'a upload
   ├─ od_images tablosuna insert
   └─ od_annotations tablosuna insert

3. Her 50 görsel → Checkpoint kaydet
```

---

## Checkpoint Yapısı (Streaming için)

```json
{
  "stage": "streaming",
  "total_images": 12647,
  "processed_images": ["id1", "id2", ...],
  "processed_count": 5000,
  "failed_images": {"id3": "error msg"},
  "uploaded_storage_map": {"id1": "uuid1.jpg", ...}
}
```

---

## Implementasyon Adımları

| # | Görev | Dosya |
|---|-------|-------|
| 1 | Roboflow SDK entegrasyonu | `services/roboflow.py` |
| 2 | Streaming importer servisi | `services/roboflow_streaming.py` |
| 3 | Import fonksiyonunu güncelle | `api/v1/od/images.py` |
| 4 | Checkpoint'i streaming'e adapte et | `services/import_checkpoint.py` |
| 5 | Test | Manual test |

---

## Roboflow API Kullanımı

```python
from roboflow import Roboflow

rf = Roboflow(api_key="xxx")
project = rf.workspace().project("slot-detection-7d5qx")

# Tüm image'ları listele (generator)
for img_data in project.search_all():
    image_id = img_data['id']

    # Detaylı bilgi al
    full_image = project.image(image_id)

    # full_image içeriği:
    # {
    #   "urls": {"original": "https://...", "thumb": "https://..."},
    #   "annotation": {"boxes": [...], "width": 1080, "height": 1920},
    #   "split": "train|valid|test"
    # }
```

---

## Concurrency ve Rate Limiting

- Roboflow API rate limit: ~100 req/sec
- Safe concurrency: 20 parallel requests
- Her request: ~100ms latency
- Throughput: ~200 images/sec (teorik)
- 12,647 görsel: ~1 dakika (API calls only)
- Supabase upload dahil: ~5-10 dakika

---

## Backward Compatibility

- Mevcut endpoint aynı kalacak
- Request/response formatı değişmeyecek
- Config'de `use_streaming: true` flag eklenebilir (opsiyonel)
