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
  manufacturer_country?: string;
  product_name?: string;
  variant_flavor?: string;
  category?: string;
  container_type?: string;
  net_quantity?: string;
  pack_configuration?: PackConfiguration;
  identifiers?: ProductIdentifiers;
  nutrition_facts?: NutritionFacts;
  claims?: string[];
  marketing_description?: string;
  grounding_prompt?: string;
  visibility_score?: number;
  issues_detected?: string[];
  frame_count: number;
  frames_path?: string;
  primary_image_url?: string;
  status: ProductStatus;
  version?: number;
  created_at: string;
  updated_at: string;
  // New fields for multiple identifiers and custom fields
  identifiers_list?: ProductIdentifier[];
  custom_fields?: Record<string, string>;
}

export interface PackConfiguration {
  type: "single_unit" | "multipack";
  item_count: number;
}

export interface ProductIdentifiers {
  barcode?: string;
  sku_model_code?: string;
}

// ===========================================
// Product Identifier Types (New System)
// ===========================================

export type IdentifierType =
  | "barcode"
  | "short_code"
  | "sku"
  | "upc"
  | "ean"
  | "custom";

export interface ProductIdentifier {
  id: string;
  identifier_type: IdentifierType;
  identifier_value: string;
  custom_label?: string;
  is_primary: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface ProductIdentifierCreate {
  identifier_type: IdentifierType;
  identifier_value: string;
  custom_label?: string;
  is_primary?: boolean;
}

export const IDENTIFIER_TYPE_LABELS: Record<IdentifierType, string> = {
  barcode: "Barcode",
  short_code: "Short Code",
  sku: "SKU",
  upc: "UPC",
  ean: "EAN",
  custom: "Custom",
};

export type ProductStatus =
  | "pending"
  | "processing"
  | "needs_matching"
  | "ready"
  | "rejected";

export interface NutritionFacts {
  serving_size?: string;
  calories?: number;
  total_fat?: string;
  protein?: string;
  carbohydrates?: string;
  sugar?: string;
  fiber?: string;
  sodium?: string;
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

export interface FrameCounts {
  synthetic: number;
  real: number;
  augmented: number;
}

export interface ProductWithFrameCounts extends Product {
  frame_counts?: FrameCounts;
  total_frames?: number;
}

export interface DatasetWithProducts extends Dataset {
  products: ProductWithFrameCounts[];
  total_synthetic?: number;
  total_real?: number;
  total_augmented?: number;
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
// Augmentation Types
// ===========================================

export type AugmentationPreset = "clean" | "normal" | "realistic" | "extreme" | "custom";

export interface AugmentationConfig {
  // Preset selection
  preset: AugmentationPreset;

  // Transform probabilities (0.0 - 1.0)
  PROB_HEAVY_AUGMENTATION?: number;
  PROB_NEIGHBORING_PRODUCTS?: number;
  PROB_TIPPED_OVER_NEIGHBOR?: number;

  // Shelf elements
  PROB_PRICE_TAG?: number;
  PROB_SHELF_RAIL?: number;
  PROB_CAMPAIGN_STICKER?: number;

  // Lighting effects
  PROB_FLUORESCENT_BANDING?: number;
  PROB_COLOR_TRANSFER?: number;
  PROB_SHELF_REFLECTION?: number;
  PROB_SHADOW?: number;

  // Camera effects
  PROB_PERSPECTIVE_CHANGE?: number;
  PROB_LENS_DISTORTION?: number;
  PROB_CHROMATIC_ABERRATION?: number;
  PROB_CAMERA_NOISE?: number;

  // Refrigerator effects
  PROB_CONDENSATION?: number;
  PROB_FROST_CRYSTALS?: number;
  PROB_COLD_COLOR_FILTER?: number;
  PROB_WIRE_RACK?: number;

  // Color adjustments
  PROB_HSV_SHIFT?: number;
  PROB_RGB_SHIFT?: number;
  PROB_MEDIAN_BLUR?: number;
  PROB_ISO_NOISE?: number;
  PROB_CLAHE?: number;
  PROB_SHARPEN?: number;
  PROB_HORIZONTAL_FLIP?: number;

  // Neighbor settings
  MIN_NEIGHBORS?: number;
  MAX_NEIGHBORS?: number;

  // Color shift limits
  HSV_HUE_LIMIT?: number;
  HSV_SAT_LIMIT?: number;
  HSV_VAL_LIMIT?: number;
  RGB_SHIFT_LIMIT?: number;

  // Geometric transform limits (degrees)
  ROTATION_LIMIT?: number;
  SHEAR_LIMIT?: number;
}

export interface AugmentationRequest {
  syn_target: number;
  real_target: number;
  use_diversity_pyramid: boolean;
  include_neighbors: boolean;
  // Frame interval for angle diversity (1 = all frames, 20 = every 20th frame)
  frame_interval: number;
  augmentation_config?: AugmentationConfig;
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
