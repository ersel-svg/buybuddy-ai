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
  // Frame counts by type (when include_frame_counts=true)
  frame_counts?: {
    synthetic: number;
    real: number;
    augmented: number;
  };
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

export interface FilterOption {
  value: string;
  label: string;
  count: number;
}

export interface BooleanFilterCounts {
  trueCount: number;
  falseCount: number;
}

export interface RangeFilterValues {
  min: number;
  max: number;
}

export interface FilterOptionsResponse {
  status: FilterOption[];
  category: FilterOption[];
  brand: FilterOption[];
  subBrand: FilterOption[];
  productName: FilterOption[];
  flavor: FilterOption[];
  container: FilterOption[];
  netQuantity: FilterOption[];
  packType: FilterOption[];
  country: FilterOption[];
  claims: FilterOption[];
  issueTypes: FilterOption[];
  hasVideo: BooleanFilterCounts;
  hasImage: BooleanFilterCounts;
  hasNutrition: BooleanFilterCounts;
  hasDescription: BooleanFilterCounts;
  hasPrompt: BooleanFilterCounts;
  hasIssues: BooleanFilterCounts;
  frameCount: RangeFilterValues;
  visibilityScore: RangeFilterValues;
  totalProducts: number;
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
  products_total?: number;
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

// ===========================================
// Cutout Image Types (Matching System)
// ===========================================

export interface CutoutImage {
  id: string;
  external_id: number;
  image_url: string;
  predicted_upc?: string;
  qdrant_point_id?: string;
  embedding_model_id?: string;
  has_embedding: boolean;
  matched_product_id?: string;
  match_similarity?: number;
  matched_by?: string;
  matched_at?: string;
  synced_at: string;
  created_at: string;
}

export interface CutoutStats {
  total: number;
  with_embedding: number;
  without_embedding: number;
  matched: number;
  unmatched: number;
}

export interface CutoutSyncResponse {
  synced_count: number;
  skipped_count: number;
  total_fetched: number;
  highest_external_id?: number;
  lowest_external_id?: number;
  stopped_early?: boolean;
  last_page?: number;
}

export interface CutoutSyncState {
  min_synced_external_id?: number;
  max_synced_external_id?: number;
  total_synced: number;
  backfill_completed: boolean;
  last_sync_new_at?: string;
  last_backfill_at?: string;
  last_backfill_page?: number;
  buybuddy_max_id?: number;
  estimated_remaining?: number;
}

export interface CutoutsResponse {
  items: CutoutImage[];
  total: number;
  page: number;
  limit: number;
}

// ===========================================
// Product Matching Types
// ===========================================

export type CandidateMatchType = "barcode" | "similarity" | "both";

export interface CutoutCandidate {
  id: string;
  external_id: number;
  image_url: string;
  predicted_upc?: string;
  similarity?: number;
  match_type: CandidateMatchType;
  has_embedding: boolean;
  is_matched: boolean;
}

export interface ProductCandidatesResponse {
  product: Product;
  candidates: CutoutCandidate[];
  barcode_match_count: number;
  similarity_match_count: number;
  total_count: number;
  has_product_embedding: boolean;
}

// ===========================================
// Embedding Model Types (Matching System)
// ===========================================

export type EmbeddingModelType =
  // DINOv2 family
  | "dinov2-small"
  | "dinov2-base"
  | "dinov2-large"
  // DINOv3 family
  | "dinov3-small"
  | "dinov3-base"
  | "dinov3-large"
  // CLIP family
  | "clip-vit-l-14"
  // Custom
  | "custom";

export type EmbeddingModelFamily = "dinov2" | "dinov3" | "clip" | "custom";

export interface EmbeddingModel {
  id: string;
  name: string;
  model_type: EmbeddingModelType;
  model_family?: EmbeddingModelFamily;
  hf_model_id?: string;
  model_path?: string;
  checkpoint_url?: string;
  embedding_dim: number;
  config?: Record<string, unknown>;
  qdrant_collection?: string;
  product_collection?: string;
  cutout_collection?: string;
  qdrant_vector_count: number;
  is_matching_active: boolean;
  is_pretrained?: boolean;
  is_default?: boolean;
  base_model_id?: string;
  training_run_id?: string;
  created_at: string;
  updated_at: string;
}

export type EmbeddingJobStatus = "pending" | "running" | "completed" | "failed";
export type EmbeddingJobType = "full" | "incremental";
export type EmbeddingSource = "cutouts" | "products" | "both";

export interface EmbeddingJob {
  id: string;
  embedding_model_id: string;
  job_type: EmbeddingJobType;
  source: EmbeddingSource;
  status: EmbeddingJobStatus;
  total_images: number;
  processed_images: number;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
}

export type ExportFormat = "json" | "numpy" | "faiss" | "qdrant_snapshot";

export interface EmbeddingExport {
  id: string;
  embedding_model_id: string;
  format: ExportFormat;
  file_url?: string;
  vector_count: number;
  file_size_bytes?: number;
  created_at: string;
}

// ===========================================
// Qdrant Collection Types
// ===========================================

export interface CollectionInfo {
  name: string;
  vectors_count: number;
  points_count: number;
  vector_size: number;
  status: string;
  size_bytes?: number;
  model_name?: string;
}

// ===========================================
// Advanced Embedding Extraction Types
// ===========================================

export type FrameSelection = "first" | "key_frames" | "interval" | "all";

export interface ProductExtractionConfig {
  frame_selection: FrameSelection;
  frame_interval: number;
  max_frames: number;
  include_augmented?: boolean;
}

export interface CutoutExtractionConfig {
  filter_has_upc?: boolean;
  synced_after?: string;
  batch_size?: number;
}

export interface CollectionStrategy {
  separate_collections: boolean;
  collection_prefix?: string;
}

export interface EmbeddingJobCreateAdvanced {
  model_id: string;
  job_type: EmbeddingJobType;
  source: EmbeddingSource;
  product_config?: ProductExtractionConfig;
  cutout_config?: CutoutExtractionConfig;
  collection_strategy?: CollectionStrategy;
}

// ===========================================
// 3-Tab Embedding Extraction Types
// ===========================================

export type ProductSource = "all" | "selected" | "dataset" | "filter" | "matched" | "new";
export type ImageType = "synthetic" | "real" | "augmented";
export type CollectionMode = "create" | "replace" | "append";
export type OutputTarget = "qdrant" | "file";
export type JobPurpose = "matching" | "training" | "evaluation";

// Tab 1: Matching Extraction
export interface MatchingExtractionRequest {
  model_id?: string;
  product_source: ProductSource;
  product_ids?: string[];
  product_dataset_id?: string;
  product_filter?: Record<string, unknown>;
  include_cutouts: boolean;
  cutout_filter_has_upc?: boolean;
  collection_mode: CollectionMode;
  product_collection_name?: string;
  cutout_collection_name?: string;
  frame_selection: FrameSelection;
  frame_interval?: number;
  max_frames?: number;
}

export interface MatchingExtractionResponse {
  job_id: string;
  status: string;
  product_collection: string;
  cutout_collection?: string;
  product_count: number;
  cutout_count: number;
  total_embeddings: number;
  failed_count: number;
}

// Tab 2: Training Extraction
export interface TrainingExtractionRequest {
  model_id?: string;
  image_types: ImageType[];
  frame_selection: FrameSelection | "all";
  frame_interval?: number;
  max_frames?: number;
  include_matched_cutouts: boolean;
  output_target: OutputTarget;
  collection_mode: CollectionMode;
  collection_name?: string;
  file_format?: "npz" | "json";
}

export interface TrainingExtractionResponse {
  job_id: string;
  status: string;
  collection_name?: string;
  matched_product_count: number;
  total_embeddings: number;
  failed_count: number;
  output_target: OutputTarget;
  output_file_url?: string;
}

// Tab 3: Evaluation Extraction
export interface EvaluationExtractionRequest {
  model_id: string;
  dataset_id: string;
  image_types: ImageType[];
  frame_selection: FrameSelection | "all";
  frame_interval?: number;
  max_frames?: number;
  collection_mode: CollectionMode;
  collection_name?: string;
}

export interface EvaluationExtractionResponse {
  job_id: string;
  status: string;
  collection_name: string;
  dataset_id: string;
  dataset_name: string;
  product_count: number;
  total_embeddings: number;
  failed_count: number;
}

// Matched Products Stats
export interface MatchedProductsStats {
  total_matched_products: number;
  sample: Product[];
}

// Embedding Collection Metadata
export type CollectionType = "products" | "cutouts" | "training" | "evaluation";

export interface EmbeddingCollection {
  id: string;
  name: string;
  collection_type: CollectionType;
  source_type?: ProductSource;
  source_dataset_id?: string;
  source_product_ids?: string[];
  source_filter?: Record<string, unknown>;
  embedding_model_id?: string;
  vector_count: number;
  image_types?: ImageType[];
  frame_selection?: string;
  created_at: string;
  updated_at: string;
  last_sync_at?: string;
}

// Enhanced Embedding Job (with new fields)
export interface EmbeddingJobExtended extends EmbeddingJob {
  purpose?: JobPurpose;
  image_types?: ImageType[];
  output_target?: OutputTarget;
  output_file_url?: string;
  source_config?: Record<string, unknown>;
}

// ===========================================
// Reverse Matching Types (Cutout → Products)
// ===========================================

export interface ProductMatchCandidate {
  id: string;
  barcode?: string;
  brand_name?: string;
  product_name?: string;
  primary_image_url?: string;
  similarity: number;
  match_type: CandidateMatchType;
}

export interface CutoutProductsResponse {
  cutout: CutoutImage;
  candidates: ProductMatchCandidate[];
  barcode_match_count: number;
  similarity_match_count: number;
  total_count: number;
  has_cutout_embedding: boolean;
}

// ===========================================
// Collection Products Types
// ===========================================

export interface CollectionProductsResponse {
  products: Product[];
  total_count: number;
  page: number;
  limit: number;
  collection_name: string;
}

// ===========================================
// Training System Types (Flexible Label Field)
// ===========================================

export type TrainingRunStatus = "pending" | "preparing" | "running" | "completed" | "failed" | "cancelled";
export type TrainingDataSource = "all_products" | "matched_products" | "dataset" | "selected";

// Label field options for training (dynamic - any product field can be used)
export type LabelFieldType = string;

// Known label fields for display purposes
export const KNOWN_LABEL_FIELDS: Record<string, { label: string; description: string }> = {
  product_id: { label: "Product ID", description: "Each product becomes its own class" },
  category: { label: "Category", description: "Train a category classifier" },
  brand_name: { label: "Brand", description: "Train a brand classifier" },
  container_type: { label: "Container Type", description: "Group by container (bottle, can, box, etc.)" },
  sub_brand: { label: "Sub-Brand", description: "Group by sub-brand variants" },
  manufacturer_country: { label: "Country", description: "Group by manufacturer country" },
  variant_flavor: { label: "Flavor/Variant", description: "Group by flavor or variant" },
  net_quantity: { label: "Size/Quantity", description: "Group by product size" },
};

export interface LabelConfig {
  label_field: LabelFieldType;
  custom_mapping?: Record<string, string>;
  min_samples_per_class: number;
}

// Identifier info for inference (product_id -> product details)
export interface ProductIdentifierInfo {
  barcode?: string;
  product_name?: string;
  brand_name?: string;
  category?: string;
  identifiers?: Record<string, string>;  // Additional identifiers (UPC, EAN, etc.)
}

export interface IdentifierMappingResponse {
  run_id: string;
  run_name: string;
  label_field: LabelFieldType;
  num_products: number;
  identifier_mapping: Record<string, ProductIdentifierInfo>;
}

export interface LabelFieldStats {
  label: string;
  description: string;
  total_products: number;
  total_classes: number;
  min_samples_per_class: number;
  max_samples_per_class: number;
  avg_samples_per_class: number;
  top_classes?: Array<[string, number]>;
  unknown_count?: number;
  coverage_percent?: number;
  is_custom?: boolean;
}

export interface LabelStatsResponse {
  data_source: TrainingDataSource;
  total_products: number;
  label_fields: Record<string, LabelFieldStats>;
}

export interface TrainingSplitConfig {
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
  seed: number;
  stratify_by?: "brand_name" | "category";
}

// ===========================================
// SOTA Training Configuration
// ===========================================

export interface SOTALossConfig {
  arcface_weight: number;
  triplet_weight: number;
  domain_weight: number;
  arcface_margin: number;
  arcface_scale: number;
  triplet_margin: number;
}

export interface SOTASamplingConfig {
  products_per_batch: number;
  samples_per_product: number;
  synthetic_ratio: number;
}

export interface SOTACurriculumConfig {
  warmup_epochs: number;
  easy_epochs: number;
  hard_epochs: number;
  finetune_epochs: number;
}

export interface SOTAConfig {
  enabled: boolean;
  use_combined_loss: boolean;
  use_pk_sampling: boolean;
  use_curriculum: boolean;
  use_domain_adaptation: boolean;
  use_early_stopping: boolean;
  early_stopping_patience: number;
  loss: SOTALossConfig;
  sampling: SOTASamplingConfig;
  curriculum: SOTACurriculumConfig;
  triplet_mining_run_id?: string;
}

export const DEFAULT_SOTA_CONFIG: SOTAConfig = {
  enabled: true,
  use_combined_loss: true,
  use_pk_sampling: true,
  use_curriculum: false,
  use_domain_adaptation: true,
  use_early_stopping: true,
  early_stopping_patience: 5,
  loss: {
    arcface_weight: 1.0,
    triplet_weight: 0.5,
    domain_weight: 0.1,
    arcface_margin: 0.5,
    arcface_scale: 64.0,
    triplet_margin: 0.3,
  },
  sampling: {
    products_per_batch: 8,
    samples_per_product: 4,
    synthetic_ratio: 0.5,
  },
  curriculum: {
    warmup_epochs: 2,
    easy_epochs: 5,
    hard_epochs: 10,
    finetune_epochs: 3,
  },
};

// Image configuration for training
export interface ImageConfig {
  image_types: ImageType[];
  frame_selection: FrameSelection;
  frame_interval: number;
  max_frames_per_type: number;
  include_matched_cutouts: boolean;
}

// Default image config
export const DEFAULT_IMAGE_CONFIG: ImageConfig = {
  image_types: ["synthetic", "real", "augmented"],
  frame_selection: "key_frames",
  frame_interval: 5,
  max_frames_per_type: 10,
  include_matched_cutouts: true,
};

export interface TrainingRunCreate {
  name: string;
  description?: string;
  base_model_type: string;
  data_source: TrainingDataSource;
  dataset_id?: string;
  product_ids?: string[];
  image_config?: ImageConfig;  // Which image types and frames to include
  label_config?: LabelConfig;  // What to train the model to classify
  split_config: TrainingSplitConfig;
  training_config: TrainingRunConfig;
  sota_config?: SOTAConfig;  // SOTA training features (multi-loss, P-K sampling, etc.)
  hard_negative_pairs?: Array<[string, string]>;  // Pairs of product IDs that look similar but are different
}

export interface TrainingRunConfig {
  // Training hyperparameters
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  warmup_epochs: number;
  early_stopping_patience: number;

  // Model config
  embedding_dim: number;
  use_arcface: boolean;
  arcface_margin: number;
  arcface_scale: number;

  // Optimization
  use_llrd: boolean;
  llrd_factor: number;
  gradient_accumulation_steps: number;
  mixed_precision: boolean;
  label_smoothing: number;

  // Checkpointing
  save_every_n_epochs: number;
}

export interface TrainingRun {
  id: string;
  name: string;
  description?: string;
  base_model_type: string;
  data_source: TrainingDataSource;
  dataset_id?: string;

  // Label configuration (what the model is trained to classify)
  label_config?: LabelConfig;
  label_mapping?: Record<string, number>;  // label -> class_index
  identifier_mapping?: Record<string, ProductIdentifierInfo>;  // product_id -> identifiers for inference

  // Split info (based on label_field, not just product_id)
  split_config: TrainingSplitConfig;
  train_product_ids: string[];
  val_product_ids: string[];
  test_product_ids: string[];
  train_product_count: number;
  val_product_count: number;
  test_product_count: number;
  train_image_count: number;
  val_image_count: number;
  test_image_count: number;
  num_classes: number;  // Number of unique labels (classes)

  // Config
  training_config: TrainingRunConfig;
  sota_config?: SOTAConfig;  // SOTA training configuration
  sota_enabled?: boolean;  // Whether SOTA features were used
  config_id?: string;

  // RunPod
  runpod_job_id?: string;
  runpod_endpoint_id?: string;

  // Status
  status: TrainingRunStatus;
  current_epoch: number;
  total_epochs: number;

  // Metrics
  best_val_loss?: number;
  best_val_recall_at_1?: number;
  best_val_recall_at_5?: number;
  best_epoch?: number;

  // Error
  error_message?: string;
  error_traceback?: string;

  // Timestamps
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at: string;
}

export interface TrainingCheckpoint {
  id: string;
  training_run_id: string;
  epoch: number;
  step?: number;
  checkpoint_url: string;
  file_size_bytes?: number;
  train_loss?: number;
  val_loss?: number;
  val_recall_at_1?: number;
  val_recall_at_5?: number;
  val_recall_at_10?: number;
  val_map?: number;
  is_best: boolean;
  is_final: boolean;
  created_at: string;
}

export interface TrainedModel {
  id: string;
  training_run_id: string;
  checkpoint_id: string;
  name: string;
  description?: string;
  embedding_model_id?: string;
  test_evaluated: boolean;
  test_metrics?: TrainingMetricsResult;
  test_evaluated_at?: string;
  cross_domain_metrics?: CrossDomainMetrics;
  identifier_mapping_url?: string;
  is_default: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;

  // Joined data
  training_run?: Partial<TrainingRun>;
  checkpoint?: Partial<TrainingCheckpoint>;
}

export interface TrainingMetricsResult {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10?: number;
  map: number;
  accuracy?: number;
}

export interface CrossDomainMetrics {
  real_to_synth?: TrainingMetricsResult;
  synth_to_real?: TrainingMetricsResult;
}

export interface ModelEvaluation {
  id: string;
  trained_model_id: string;
  eval_config?: Record<string, unknown>;
  overall_metrics: TrainingMetricsResult;
  real_to_synthetic?: TrainingMetricsResult;
  synthetic_to_real?: TrainingMetricsResult;
  per_category_metrics?: Record<string, TrainingMetricsResult>;
  worst_product_ids?: Array<{ product_id: string; recall_at_1: number }>;
  most_confused_pairs?: Array<{ product_id_1: string; product_id_2: string; similarity: number }>;
  created_at: string;
}

// Model comparison result (for side-by-side comparison)
export interface ModelComparisonResult {
  id: string;
  name: string;
  test_metrics?: TrainingMetricsResult;
  cross_domain_metrics?: CrossDomainMetrics;
}

export interface TrainingConfigPreset {
  id: string;
  name: string;
  description?: string;
  base_model_type: string;
  config: TrainingRunConfig;
  is_default: boolean;
  created_by?: string;
  created_at: string;
  updated_at: string;
}

export interface ModelPreset {
  model_type: string;
  model_family: string;
  name: string;
  hf_model_id: string;
  embedding_dim: number;
  image_size: number;
  description: string;
  recommended_for: string[];
}

export interface ModelPresetsResponse {
  presets: ModelPreset[];
  families: Array<{
    id: string;
    name: string;
    description: string;
  }>;
}

export interface TrainingRunsResponse {
  items: TrainingRun[];
  total: number;
}

export interface TrainingProductsResponse {
  products: Array<{
    id: string;
    barcode?: string;
    short_code?: string;
    upc?: string;
    brand_name?: string;
    frames_path?: string;
    frame_count: number;
  }>;
  total: number;
}

// ===========================================
// Triplet Mining Types
// ===========================================

export type TripletMiningStatus = "pending" | "running" | "completed" | "failed" | "cancelled";
export type TripletDifficulty = "hard" | "semi_hard" | "easy";
export type TripletDomain = "synthetic" | "real" | "unknown";

export interface TripletMiningRun {
  id: string;
  name: string;
  description?: string;
  dataset_id?: string;
  embedding_model_id: string;
  collection_name: string;
  hard_negative_threshold: number;
  positive_threshold: number;
  max_triplets_per_anchor: number;
  include_cross_domain: boolean;
  total_anchors?: number;
  total_triplets?: number;
  hard_triplets?: number;
  semi_hard_triplets?: number;
  easy_triplets?: number;
  cross_domain_triplets?: number;
  status: TripletMiningStatus;
  error_message?: string;
  output_url?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at: string;
}

export interface MinedTriplet {
  id: string;
  mining_run_id: string;
  anchor_product_id: string;
  positive_product_id: string;
  negative_product_id: string;
  anchor_frame_idx: number;
  positive_frame_idx: number;
  negative_frame_idx: number;
  anchor_positive_sim: number;
  anchor_negative_sim: number;
  margin: number;
  difficulty: TripletDifficulty;
  is_cross_domain: boolean;
  anchor_domain?: TripletDomain;
  positive_domain?: TripletDomain;
  negative_domain?: TripletDomain;
  created_at: string;
}

export interface TripletMiningRequest {
  name: string;
  description?: string;
  dataset_id?: string;
  embedding_model_id?: string;
  collection_name?: string;
  hard_negative_threshold?: number;
  positive_threshold?: number;
  max_triplets_per_anchor?: number;
  include_cross_domain?: boolean;
}

export interface TripletMiningStats {
  total_triplets: number;
  hard_count: number;
  semi_hard_count: number;
  easy_count: number;
  cross_domain_count: number;
  avg_margin: number;
  min_margin: number;
  max_margin: number;
}

export interface TripletExportResult {
  export_id: string;
  format: "json" | "csv";
  total_triplets: number;
  file_url: string;
  file_size_bytes?: number;
}

export interface MatchingFeedback {
  id: string;
  cutout_id?: string;
  cutout_image_url?: string;
  predicted_product_id?: string;
  predicted_similarity?: number;
  model_id?: string;
  collection_name?: string;
  feedback_type: "correct" | "wrong" | "uncertain";
  correct_product_id?: string;
  user_id?: string;
  feedback_source: "web" | "api" | "review" | "auto";
  notes?: string;
  created_at: string;
}

export interface MatchingFeedbackCreate {
  cutout_id?: string;
  cutout_image_url?: string;
  predicted_product_id?: string;
  predicted_similarity?: number;
  model_id?: string;
  collection_name?: string;
  feedback_type: "correct" | "wrong" | "uncertain";
  correct_product_id?: string;
  notes?: string;
}

export interface FeedbackStats {
  total: number;
  correct: number;
  wrong: number;
  uncertain: number;
  accuracy_rate: number;
}

export interface HardExample {
  cutout_image_url: string;
  correct_product_id: string;
  wrong_product_id: string;
}
