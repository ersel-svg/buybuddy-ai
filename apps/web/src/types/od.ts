/**
 * Object Detection Platform Types
 *
 * Base types for the OD module. Will be extended as features are implemented.
 */

// ===========================================
// Image Types
// ===========================================

export interface ODImage {
  id: string;
  filename: string;
  image_url: string;
  thumbnail_url?: string;
  width: number;
  height: number;
  file_size_bytes?: number;
  source: "upload" | "buybuddy_sync" | "import";
  buybuddy_image_id?: string;
  folder?: string;
  tags?: string[];
  status: "pending" | "annotating" | "completed" | "skipped";
  annotation_count: number;
  created_at: string;
  updated_at: string;
}

export interface ODImagesResponse {
  images: ODImage[];
  total: number;
  page: number;
  limit: number;
}

// ===========================================
// Class Types
// ===========================================

export interface ODClass {
  id: string;
  name: string;
  display_name?: string;
  color: string;
  category?: string;
  aliases?: string[];
  annotation_count: number;
  created_at: string;
  updated_at: string;
}

// ===========================================
// Dataset Types
// ===========================================

export interface ODDataset {
  id: string;
  name: string;
  description?: string;
  annotation_type: "bbox" | "polygon" | "segmentation";
  image_count: number;
  annotated_image_count: number;
  annotation_count: number;
  class_count: number;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ODDatasetWithImages extends ODDataset {
  images: ODDatasetImage[];
}

export interface ODDatasetImage {
  id: string;
  image_id: string;
  dataset_id: string;
  image: ODImage;
  status: "pending" | "annotating" | "completed" | "skipped";
  annotation_count: number;
  added_at: string;
}

// ===========================================
// Annotation Types
// ===========================================

export interface ODAnnotation {
  id: string;
  dataset_id: string;
  image_id: string;
  class_id: string;
  class_name?: string;
  class_color?: string;

  // Bounding box (normalized 0-1)
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };

  // Optional polygon points (for polygon annotations)
  polygon?: { x: number; y: number }[];

  // Optional segmentation mask URL
  mask_url?: string;

  // AI prediction metadata
  is_ai_generated: boolean;
  confidence?: number;
  ai_model?: string;

  created_at: string;
  updated_at: string;
}

// ===========================================
// Training Types
// ===========================================

export interface ODTrainingRun {
  id: string;
  name: string;
  description?: string;
  dataset_id: string;
  dataset_version_id?: string;
  model_type: "rf-detr" | "rt-detr" | "yolo-nas";
  model_size: "small" | "medium" | "large";
  status: "pending" | "preparing" | "running" | "completed" | "failed" | "cancelled";
  config: ODTrainingConfig;
  current_epoch: number;
  total_epochs: number;
  best_map?: number;
  best_epoch?: number;
  runpod_job_id?: string;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at: string;
}

export interface ODTrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size: number;
  augmentation?: {
    horizontal_flip: boolean;
    vertical_flip: boolean;
    rotation: number;
    brightness: number;
    contrast: number;
  };
  split?: {
    train_ratio: number;
    val_ratio: number;
    test_ratio: number;
  };
}

// ===========================================
// Model Types
// ===========================================

export interface ODTrainedModel {
  id: string;
  training_run_id: string;
  name: string;
  description?: string;
  model_type: "rf-detr" | "rt-detr" | "yolo-nas";
  checkpoint_url: string;
  onnx_url?: string;
  map: number;
  map_50?: number;
  map_75?: number;
  class_count: number;
  is_active: boolean;
  is_default: boolean;
  created_at: string;
}

// ===========================================
// Stats Types
// ===========================================

export interface ODStats {
  total_images: number;
  total_datasets: number;
  total_annotations: number;
  total_models: number;
  images_by_status: {
    pending: number;
    annotating: number;
    completed: number;
    skipped: number;
  };
}

// ===========================================
// API Request/Response Types
// ===========================================

export interface ODCreateDatasetRequest {
  name: string;
  description?: string;
  annotation_type?: "bbox" | "polygon" | "segmentation";
}

export interface ODCreateClassRequest {
  name: string;
  display_name?: string;
  color?: string;
  category?: string;
}

export interface ODCreateAnnotationRequest {
  class_id: string;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  polygon?: { x: number; y: number }[];
  is_ai_generated?: boolean;
  confidence?: number;
  ai_model?: string;
}

export interface ODBulkAnnotationRequest {
  annotations: ODCreateAnnotationRequest[];
}

export interface ODAutoAnnotateRequest {
  image_ids: string[];
  model: "grounding-dino" | "owlv2" | "florence-2" | "custom";
  custom_model_id?: string;
  text_prompt?: string;
  class_mapping?: Record<string, string>;
  confidence_threshold?: number;
}

export interface ODStartTrainingRequest {
  name: string;
  description?: string;
  dataset_id: string;
  dataset_version_id?: string;
  model_type: "rf-detr" | "rt-detr" | "yolo-nas";
  model_size?: "small" | "medium" | "large";
  config?: Partial<ODTrainingConfig>;
}

// ===========================================
// AI Annotation Types (Phase 6)
// ===========================================

export type AIModelType = "grounding_dino" | "sam3" | "sam2" | "florence2";

export interface AIPredictRequest {
  image_id: string;
  model: AIModelType;
  text_prompt: string;
  box_threshold?: number;
  text_threshold?: number;
}

export interface AISegmentRequest {
  image_id: string;
  model: "sam2" | "sam3";
  prompt_type: "point" | "box";
  point?: [number, number];
  box?: [number, number, number, number];
  label?: number;
  text_prompt?: string;
}

export interface AIBatchAnnotateRequest {
  dataset_id: string;
  image_ids?: string[];
  model: AIModelType;
  text_prompt: string;
  box_threshold?: number;
  text_threshold?: number;
  auto_accept?: boolean;
  class_mapping?: Record<string, string>;
}

export interface AIPrediction {
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  label: string;
  confidence: number;
  mask?: string;
}

export interface AIPredictResponse {
  predictions: AIPrediction[];
  model: string;
  processing_time_ms?: number;
}

export interface AISegmentResponse {
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  mask?: string;
  processing_time_ms?: number;
}

export interface AIBatchJobResponse {
  job_id: string;
  status: string;
  total_images: number;
  message: string;
}

export interface AIJobStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  total_images: number;
  predictions_generated: number;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
}

export interface AIModelInfo {
  id: string;
  name: string;
  description: string;
  tasks: string[];
  requires_prompt: boolean;
}
