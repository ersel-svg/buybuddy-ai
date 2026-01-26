-- Additional Detection Models for RunPod Worker
-- Migration 064: Add OWL-ViT, YOLO-NAS, and D-FINE models

-- ============================================
-- OWL-ViT / OWL-v2 (Open-Vocabulary Detection)
-- ============================================

INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('owlvit-base-patch32', 'OWL-ViT Base', 'Open-vocabulary object detection with text queries', 'detection', 'huggingface', 'google/owlvit-base-patch32',
     '{"confidence": 0.1, "text_queries": ["object"]}'::jsonb, 768),

    ('owlvit-large-patch14', 'OWL-ViT Large', 'Open-vocabulary detection - Higher accuracy', 'detection', 'huggingface', 'google/owlvit-large-patch14',
     '{"confidence": 0.1, "text_queries": ["object"]}'::jsonb, 840),

    ('owlv2-base', 'OWL-v2 Base', 'Improved open-vocabulary detection with ensemble', 'detection', 'huggingface', 'google/owlv2-base-patch16-ensemble',
     '{"confidence": 0.1, "text_queries": ["object"]}'::jsonb, 960),

    ('owlv2-large', 'OWL-v2 Large', 'OWL-v2 large - Best open-vocab accuracy', 'detection', 'huggingface', 'google/owlv2-large-patch14-ensemble',
     '{"confidence": 0.1, "text_queries": ["object"]}'::jsonb, 1008)

ON CONFLICT (id) DO NOTHING;

-- ============================================
-- YOLO-NAS Models (Neural Architecture Search)
-- ============================================

INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, class_count, default_config, input_size)
VALUES
    ('yolo-nas-s', 'YOLO-NAS Small', 'YOLO-NAS from DECI - Small and fast (COCO)', 'detection', 'super-gradients', 'yolo_nas_s',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "iou_threshold": 0.45}'::jsonb, 640),

    ('yolo-nas-m', 'YOLO-NAS Medium', 'YOLO-NAS from DECI - Medium balanced (COCO)', 'detection', 'super-gradients', 'yolo_nas_m',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "iou_threshold": 0.45}'::jsonb, 640),

    ('yolo-nas-l', 'YOLO-NAS Large', 'YOLO-NAS from DECI - Large highest accuracy (COCO)', 'detection', 'super-gradients', 'yolo_nas_l',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "iou_threshold": 0.45}'::jsonb, 640)

ON CONFLICT (id) DO NOTHING;

-- ============================================
-- D-FINE Models (DETR with Fine-grained)
-- ============================================

INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, class_count, default_config, input_size)
VALUES
    ('dfine-s', 'D-FINE Small', 'D-FINE DETR - Small (COCO)', 'detection', 'huggingface', 'ustc-community/dfine-small-coco',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640),

    ('dfine-m', 'D-FINE Medium', 'D-FINE DETR - Medium (COCO)', 'detection', 'huggingface', 'ustc-community/dfine-medium-coco',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640),

    ('dfine-l', 'D-FINE Large', 'D-FINE DETR - Large (COCO)', 'detection', 'huggingface', 'ustc-community/dfine-large-coco',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640),

    ('dfine-x', 'D-FINE XLarge', 'D-FINE DETR - Extra Large (COCO)', 'detection', 'huggingface', 'ustc-community/dfine-xlarge-coco',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640)

ON CONFLICT (id) DO NOTHING;

-- ============================================
-- Additional Embedding Models
-- ============================================

INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, embedding_dim, input_size)
VALUES
    ('dinov2-giant', 'DINOv2 Giant', 'DINOv2 - Giant model (1536-dim)', 'embedding', 'huggingface', 'facebook/dinov2-giant',
     '{"normalize": true, "pooling": "cls"}'::jsonb, 1536, 518),

    ('siglip-large', 'SigLIP Large', 'SigLIP with sigmoid loss - Large (1024-dim)', 'embedding', 'huggingface', 'google/siglip-large-patch16-384',
     '{"normalize": true}'::jsonb, 1024, 384),

    ('clip-vit-l-14', 'CLIP ViT-L/14', 'OpenAI CLIP Large (768-dim)', 'embedding', 'huggingface', 'openai/clip-vit-large-patch14',
     '{"normalize": true}'::jsonb, 768, 224)

ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE wf_pretrained_models IS 'Registry of pretrained models - includes OWL-ViT, YOLO-NAS, D-FINE, and additional embedding models';
