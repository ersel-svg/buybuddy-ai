-- Remove unsupported models from pretrained models
-- Migration 065:
-- - YOLO-NAS disabled due to super-gradients dependency conflict (onnxruntime==1.13.1 incompatible with Python 3.11)
-- - OWL-v2 disabled due to Owlv2Processor API incompatibility

-- Remove YOLO-NAS models
DELETE FROM wf_pretrained_models
WHERE id IN ('yolo-nas-s', 'yolo-nas-m', 'yolo-nas-l');

-- Remove OWL-v2 models (keep OWL-ViT v1 which works)
DELETE FROM wf_pretrained_models
WHERE id IN ('owlv2-base', 'owlv2-large');

COMMENT ON TABLE wf_pretrained_models IS 'Registry of pretrained models - YOLO-NAS and OWL-v2 removed due to dependency/API conflicts';
