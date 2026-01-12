// ===========================================
// Core Domain Types
// ===========================================

export interface Product {
  id: string;
  barcode: string;
  video_id?: number;
  video_url?: string;
  brand_name?: string;
  sub_brand?: string;
  product_name?: string;
  variant_flavor?: string;
  category?: string;
  container_type?: string;
  net_quantity?: string;
  nutrition_facts?: NutritionFacts;
  claims?: string[];
  grounding_prompt?: string;
  visibility_score?: number;
  frame_count: number;
  frames_path?: string;
  primary_image_url?: string;
  status: ProductStatus;
  version?: number;
  created_at: string;
  updated_at: string;
}

export type ProductStatus =
  | "pending"
  | "processing"
  | "needs_matching"
  | "ready"
  | "rejected";

export interface NutritionFacts {
  calories?: number;
  protein?: string;
  carbohydrates?: string;
  fat?: string;
  fiber?: string;
  sodium?: string;
  sugar?: string;
  [key: string]: string | number | undefined;
}

export interface ProductsResponse {
  items: Product[];
  total: number;
  page: number;
  limit: number;
}

// ===========================================
// Dataset Types
// ===========================================

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  product_count: number;
  version?: number;
  created_at: string;
  updated_at: string;
}

export interface DatasetWithProducts extends Dataset {
  products: Product[];
}

export interface CreateDatasetRequest {
  name: string;
  description?: string;
  product_ids?: string[];
  filters?: ProductFilters;
}

export interface ProductFilters {
  status?: ProductStatus;
  category?: string;
  search?: string;
}

// ===========================================
// Job Types
// ===========================================

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  progress: number;
  config?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
  runpod_job_id?: string;
  created_at: string;
  updated_at: string;
}

export type JobType =
  | "video_processing"
  | "augmentation"
  | "training"
  | "embedding_extraction";

export type JobStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

// ===========================================
// Training Types
// ===========================================

export interface TrainingJob extends Job {
  dataset_id: string;
  dataset_name?: string;
  epochs: number;
  epochs_completed?: number;
  batch_size: number;
  learning_rate: number;
  final_loss?: number;
  checkpoint_url?: string;
  metrics?: TrainingMetrics;
}

export interface TrainingMetrics {
  train_loss: number[];
  val_loss?: number[];
  learning_rate: number[];
  epoch_times: number[];
}

export interface TrainingConfig {
  dataset_id: string;
  model_name: "facebook/dinov2-large" | "facebook/dinov2-base";
  proj_dim: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  label_smoothing: number;
  warmup_epochs: number;
  grad_clip: number;
  llrd_decay: number;
  domain_aware_ratio: number;
  hard_negative_pool_size: number;
  use_hardest_negatives: boolean;
  use_mixed_precision: boolean;
  train_ratio: number;
  valid_ratio: number;
  test_ratio: number;
  save_every: number;
  seed: number;
}

export interface ModelArtifact {
  id: string;
  name: string;
  version: string;
  training_job_id: string;
  checkpoint_url: string;
  embedding_dim: number;
  num_classes: number;
  final_loss: number;
  is_active: boolean;
  created_at: string;
}

// ===========================================
// Matching Types
// ===========================================

export interface ProductSummary {
  id: string;
  barcode?: string;
  brand_name?: string;
  product_name?: string;
  primary_image_url?: string;
  frame_count: number;
  real_image_count: number;
  status?: string;
}

export interface RealImage {
  id: string;
  product_id: string;
  image_url: string;
  image_path?: string;
  source?: string;
  similarity?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface MatchCandidate {
  id: string;
  image_path: string;
  image_url: string;
  similarity: number;
  metadata?: Record<string, unknown>;
}

export interface ProductMatch {
  id: string;
  reference_upc: string;
  candidate_path: string;
  similarity: number;
  is_approved?: boolean;
  created_at: string;
}

// ===========================================
// Video Types
// ===========================================

export interface Video {
  id: number;
  barcode: string;
  video_url: string;
  status: "pending" | "processing" | "completed" | "failed";
  product_id?: string;
  created_at: string;
}

export interface VideoSyncResponse {
  synced_count: number;
  new_videos: Video[];
}

// ===========================================
// Embedding Types
// ===========================================

export interface EmbeddingIndex {
  id: string;
  name: string;
  model_artifact_id: string;
  model_name?: string;
  vector_count: number;
  index_path: string;
  created_at: string;
}

// ===========================================
// Dashboard Types
// ===========================================

export interface DashboardStats {
  total_products: number;
  products_by_status: Record<ProductStatus, number>;
  total_datasets: number;
  active_jobs: number;
  completed_jobs_today: number;
  recent_products: Product[];
  recent_jobs: Job[];
}

// ===========================================
// Resource Lock Types (Multi-user)
// ===========================================

export interface ResourceLock {
  id: string;
  resource_type: "product" | "dataset";
  resource_id: string;
  user_id: string;
  user_email?: string;
  locked_at: string;
  expires_at: string;
}
