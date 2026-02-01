-- =====================================================
-- DUPLICATE ANNOTATION CLEANUP SCRIPT (OPTIMIZED)
-- =====================================================
-- Bu script aynı koordinatlarda birden fazla oluşturulmuş
-- duplicate annotation'ları tespit eder ve siler.
--
-- TIMEOUT SORUNU İÇİN OPTİMİZE EDİLMİŞ VERSİYON:
-- - İndeks kullanımı eklendi
-- - Batch bazlı silme
-- - Daha küçük sorgular
--
-- KULLANIM:
-- 1. Önce STEP 0'ı çalıştırarak index oluştur (bir kez)
-- 2. STEP 1 ile duplicate sayısını kontrol et
-- 3. STEP 2 ile batch batch sil
-- 4. STEP 3 ile sayıları düzelt
-- 5. STEP 4 ile doğrula
-- =====================================================

-- =====================================================
-- STEP 0: INDEX OLUŞTUR (sadece bir kez çalıştır)
-- =====================================================

-- Bu index duplicate tespitini çok hızlandırır
CREATE INDEX IF NOT EXISTS idx_od_annotations_duplicate_check
ON od_annotations (dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height);

-- =====================================================
-- STEP 1: HIZLI DUPLICATE SAYISI KONTROLÜ
-- =====================================================

-- Toplam annotation sayısı
SELECT COUNT(*) as total_annotations FROM od_annotations;

-- Unique annotation sayısı (duplicate olmadan)
SELECT COUNT(*) as unique_annotations FROM (
    SELECT DISTINCT dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height
    FROM od_annotations
) sub;

-- Fark = silinecek duplicate sayısı
-- (total - unique = duplicates to delete)

-- =====================================================
-- STEP 1B: DATASET BAZINDA DUPLICATE SAYILARI (opsiyonel)
-- =====================================================
-- Bu sorgu büyük tablolarda yavaş olabilir, gerekirse atla

SELECT
    a.dataset_id,
    COUNT(*) as total,
    COUNT(DISTINCT (a.image_id, a.class_id, a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height)) as unique_count
FROM od_annotations a
GROUP BY a.dataset_id
HAVING COUNT(*) > COUNT(DISTINCT (a.image_id, a.class_id, a.bbox_x, a.bbox_y, a.bbox_width, a.bbox_height));

-- =====================================================
-- STEP 2: BATCH BAZLI SİLME (10,000'lik gruplar halinde)
-- =====================================================

-- Bu sorguyu duplicate kalmayıncaya kadar tekrar tekrar çalıştır
-- Her çalıştırmada 10,000 duplicate siler

DELETE FROM od_annotations
WHERE id IN (
    SELECT id FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height
                ORDER BY created_at ASC
            ) as rn
        FROM od_annotations
    ) ranked
    WHERE rn > 1
    LIMIT 10000
);

-- Kaç satır silindi kontrol et
-- Eğer 0 dönerse tüm duplicate'lar temizlenmiş demektir
-- Eğer 10000 dönerse tekrar çalıştır

-- =====================================================
-- STEP 2B: TEK SEFERDE HIZLI SİLME (alternatif yöntem)
-- =====================================================
-- Eğer STEP 2 hala yavaşsa bu yöntemi dene

-- Önce silinmeyecek (tutulacak) ID'leri temp tabloya al
CREATE TEMP TABLE keep_ids AS
SELECT DISTINCT ON (dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height)
    id
FROM od_annotations
ORDER BY dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at ASC;

-- Index ekle
CREATE INDEX ON keep_ids(id);

-- keep_ids'de OLMAYAN her şeyi sil
DELETE FROM od_annotations
WHERE id NOT IN (SELECT id FROM keep_ids);

-- Temp tabloyu temizle
DROP TABLE keep_ids;

-- =====================================================
-- STEP 2C: EN HIZLI YÖNTEM (çok büyük tablolar için)
-- =====================================================
-- Bu yöntem tabloyu yeniden oluşturur, en hızlısı ama dikkatli ol!

/*
-- 1. Yeni tablo oluştur (sadece unique kayıtlarla)
CREATE TABLE od_annotations_clean AS
SELECT DISTINCT ON (dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height)
    *
FROM od_annotations
ORDER BY dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at ASC;

-- 2. Index'leri yeni tabloya ekle
CREATE INDEX idx_clean_dataset ON od_annotations_clean(dataset_id);
CREATE INDEX idx_clean_image ON od_annotations_clean(image_id);
CREATE INDEX idx_clean_class ON od_annotations_clean(class_id);
-- ... diğer index'ler

-- 3. Tabloları swap et (TRANSACTION içinde)
BEGIN;
ALTER TABLE od_annotations RENAME TO od_annotations_old;
ALTER TABLE od_annotations_clean RENAME TO od_annotations;
COMMIT;

-- 4. Eski tabloyu sil (emin olduktan sonra)
DROP TABLE od_annotations_old;
*/

-- =====================================================
-- STEP 3: RECOUNT - Sayıları yeniden hesapla
-- =====================================================
-- Bu sorguları STEP 2 tamamlandıktan sonra çalıştır

-- 3A: Image annotation count güncelle
UPDATE od_dataset_images di
SET annotation_count = sub.cnt
FROM (
    SELECT dataset_id, image_id, COUNT(*) as cnt
    FROM od_annotations
    GROUP BY dataset_id, image_id
) sub
WHERE di.dataset_id = sub.dataset_id
  AND di.image_id = sub.image_id
  AND di.annotation_count != sub.cnt;

-- 3B: Annotation'ı olmayan image'ları 0'la
UPDATE od_dataset_images di
SET annotation_count = 0
WHERE annotation_count > 0
  AND NOT EXISTS (
    SELECT 1 FROM od_annotations a
    WHERE a.dataset_id = di.dataset_id AND a.image_id = di.image_id
);

-- 3C: Image status güncelle
UPDATE od_dataset_images di
SET status = CASE
    WHEN di.status IN ('completed', 'skipped') THEN di.status
    WHEN di.annotation_count > 0 THEN 'annotated'
    ELSE 'pending'
END
WHERE di.status NOT IN ('completed', 'skipped')
  AND ((di.annotation_count > 0 AND di.status != 'annotated')
       OR (di.annotation_count = 0 AND di.status != 'pending'));

-- 3D: Class annotation count güncelle
UPDATE od_classes c
SET annotation_count = COALESCE(sub.cnt, 0)
FROM (
    SELECT class_id, COUNT(*) as cnt
    FROM od_annotations
    GROUP BY class_id
) sub
WHERE c.id = sub.class_id
  AND c.annotation_count != sub.cnt;

-- 3E: Dataset annotation count güncelle
UPDATE od_datasets d
SET
    annotation_count = COALESCE(ann.cnt, 0),
    annotated_image_count = COALESCE(img.cnt, 0)
FROM (
    SELECT dataset_id, COUNT(*) as cnt
    FROM od_annotations
    GROUP BY dataset_id
) ann,
(
    SELECT dataset_id, COUNT(*) as cnt
    FROM od_dataset_images
    WHERE status IN ('annotated', 'completed')
    GROUP BY dataset_id
) img
WHERE d.id = ann.dataset_id
  AND d.id = img.dataset_id
  AND (d.annotation_count != ann.cnt OR d.annotated_image_count != img.cnt);

-- =====================================================
-- STEP 4: VERIFY - Doğrulama
-- =====================================================

-- Duplicate kalmadığını doğrula (0 olmalı)
SELECT
    (SELECT COUNT(*) FROM od_annotations) -
    (SELECT COUNT(*) FROM (
        SELECT DISTINCT dataset_id, image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height
        FROM od_annotations
    ) sub) as remaining_duplicates;

-- Dataset sayıları doğru mu kontrol et
SELECT
    d.id,
    d.name,
    d.annotation_count as stored,
    (SELECT COUNT(*) FROM od_annotations a WHERE a.dataset_id = d.id) as actual
FROM od_datasets d
WHERE d.annotation_count != (SELECT COUNT(*) FROM od_annotations a WHERE a.dataset_id = d.id);

-- =====================================================
-- OPSIYONEL: INDEX'İ KALDIR (cleanup bittikten sonra)
-- =====================================================
-- Eğer duplicate önleme backend'de yapılıyorsa bu index gereksiz olabilir

-- DROP INDEX IF EXISTS idx_od_annotations_duplicate_check;
