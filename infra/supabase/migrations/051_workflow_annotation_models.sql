-- Buybuddy AI Platform - Workflow Annotation Models
-- Migration 051: Add annotation and zero-shot detection models
--
-- Adds:
-- - Grounding DINO (open-vocabulary detection)
-- - Florence-2 (vision foundation model)
-- - SAM2 (improved segmentation)
-- - RT-DETR (real-time DETR variants)

-- ============================================
-- ANNOTATION / ZERO-SHOT DETECTION MODELS
-- ============================================

-- Grounding DINO Models (Open-vocabulary object detection)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('grounding-dino-tiny', 'Grounding DINO Tiny', 'Open-vocabulary detection with text prompts - Fast', 'detection', 'huggingface', 'IDEA-Research/grounding-dino-tiny',
     '{"confidence": 0.3, "box_threshold": 0.3, "text_threshold": 0.25}'::jsonb, 800),

    ('grounding-dino-base', 'Grounding DINO Base', 'Open-vocabulary detection with text prompts - Balanced', 'detection', 'huggingface', 'IDEA-Research/grounding-dino-base',
     '{"confidence": 0.3, "box_threshold": 0.3, "text_threshold": 0.25}'::jsonb, 800)

ON CONFLICT (id) DO NOTHING;

-- SAM3 Detection Models (Text-prompted detection like Grounding DINO)
-- SAM3 can detect objects using natural language and return bounding boxes
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('sam3-detect', 'SAM3 (Text Detection)', 'Segment Anything 3 - Text-prompted object detection', 'detection', 'huggingface', 'facebook/sam3',
     '{"confidence": 0.3, "box_threshold": 0.3}'::jsonb, 1024)

ON CONFLICT (id) DO NOTHING;

-- Florence-2 Models (Microsoft Vision Foundation)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('florence2-base', 'Florence-2 Base', 'Microsoft vision foundation model - Detection, Captioning, OCR', 'detection', 'huggingface', 'microsoft/Florence-2-base',
     '{"task": "object_detection", "confidence": 0.3}'::jsonb, 768),

    ('florence2-large', 'Florence-2 Large', 'Microsoft vision foundation model - Higher accuracy', 'detection', 'huggingface', 'microsoft/Florence-2-large',
     '{"task": "object_detection", "confidence": 0.3}'::jsonb, 768)

ON CONFLICT (id) DO NOTHING;

-- SAM2 Models (Segment Anything 2)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('sam2-tiny', 'SAM2 Tiny', 'Segment Anything 2 - Fast', 'segmentation', 'huggingface', 'facebook/sam2-hiera-tiny',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2-small', 'SAM2 Small', 'Segment Anything 2 - Balanced', 'segmentation', 'huggingface', 'facebook/sam2-hiera-small',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2-base', 'SAM2 Base+', 'Segment Anything 2 - Higher quality', 'segmentation', 'huggingface', 'facebook/sam2-hiera-base-plus',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2-large', 'SAM2 Large', 'Segment Anything 2 - Best quality', 'segmentation', 'huggingface', 'facebook/sam2-hiera-large',
     '{"points_per_side": 32}'::jsonb, 1024)

ON CONFLICT (id) DO NOTHING;

-- SAM 2.1 Models (Point/Box Segmentation)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('sam2.1-tiny', 'SAM 2.1 Tiny', 'Segment Anything 2.1 - Fastest', 'segmentation', 'huggingface', 'facebook/sam2.1-hiera-tiny',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2.1-small', 'SAM 2.1 Small', 'Segment Anything 2.1 - Fast', 'segmentation', 'huggingface', 'facebook/sam2.1-hiera-small',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2.1-base', 'SAM 2.1 Base+', 'Segment Anything 2.1 - Balanced', 'segmentation', 'huggingface', 'facebook/sam2.1-hiera-base-plus',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam2.1-large', 'SAM 2.1 Large', 'Segment Anything 2.1 - Best quality', 'segmentation', 'huggingface', 'facebook/sam2.1-hiera-large',
     '{"points_per_side": 32}'::jsonb, 1024)

ON CONFLICT (id) DO NOTHING;

-- SAM3 Segmentation Model (Text-prompted segmentation)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('sam3-segment', 'SAM3 (Text Segmentation)', 'Segment Anything 3 - Text-prompted segmentation with masks', 'segmentation', 'huggingface', 'facebook/sam3',
     '{"points_per_side": 32}'::jsonb, 1024)

ON CONFLICT (id) DO NOTHING;

-- RT-DETR Models (Real-time Detection Transformer)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, class_count, default_config, input_size)
VALUES
    ('rtdetr-l', 'RT-DETR Large', 'Real-time DETR - Large (COCO)', 'detection', 'ultralytics', 'rtdetr-l.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640),

    ('rtdetr-x', 'RT-DETR XLarge', 'Real-time DETR - Extra Large (COCO)', 'detection', 'ultralytics', 'rtdetr-x.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5}'::jsonb, 640)

ON CONFLICT (id) DO NOTHING;

-- Additional YOLO Models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, class_count, default_config, input_size)
VALUES
    ('yolo11m', 'YOLO11 Medium', 'Latest YOLO - medium detection (COCO)', 'detection', 'ultralytics', 'yolo11m.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640),

    ('yolo11l', 'YOLO11 Large', 'Latest YOLO - large detection (COCO)', 'detection', 'ultralytics', 'yolo11l.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640),

    ('yolo11x', 'YOLO11 XLarge', 'Latest YOLO - extra large detection (COCO)', 'detection', 'ultralytics', 'yolo11x.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640)

ON CONFLICT (id) DO NOTHING;

-- Additional Classification Models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, class_count, input_size)
VALUES
    ('vit-base', 'ViT Base', 'Vision Transformer - ImageNet-1k', 'classification', 'huggingface', 'google/vit-base-patch16-224',
     '{"top_k": 5}'::jsonb, 1000, 224),

    ('convnext-base', 'ConvNeXt Base', 'ConvNeXt - ImageNet-1k', 'classification', 'huggingface', 'facebook/convnext-base-224',
     '{"top_k": 5}'::jsonb, 1000, 224),

    ('swin-base', 'Swin Base', 'Swin Transformer - ImageNet-1k', 'classification', 'huggingface', 'microsoft/swin-base-patch4-window7-224',
     '{"top_k": 5}'::jsonb, 1000, 224),

    ('efficientnet-b0', 'EfficientNet B0', 'EfficientNet - ImageNet-1k', 'classification', 'huggingface', 'google/efficientnet-b0',
     '{"top_k": 5}'::jsonb, 1000, 224)

ON CONFLICT (id) DO NOTHING;

-- SigLIP Embedding Model
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, embedding_dim, input_size)
VALUES
    ('siglip-base', 'SigLIP Base', 'Improved CLIP with sigmoid loss (768-dim)', 'embedding', 'huggingface', 'google/siglip-base-patch16-224',
     '{"normalize": true}'::jsonb, 768, 224)

ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE wf_pretrained_models IS 'Registry of pretrained models available for workflows (updated with annotation models)';
