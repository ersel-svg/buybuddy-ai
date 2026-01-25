import type {
  Product,
  ProductsResponse,
  FilterOptionsResponse,
  ProductSummary,
  RealImage,
  Dataset,
  DatasetWithProducts,
  CreateDatasetRequest,
  Job,
  TrainingJob,
  TrainingConfig,
  ModelArtifact,
  MatchCandidate,
  Video,
  VideoSyncResponse,
  EmbeddingIndex,
  DashboardStats,
  ResourceLock,
  AugmentationRequest,
  ProductIdentifier,
  ProductIdentifierCreate,
  CutoutImage,
  CutoutsResponse,
  CutoutStats,
  CutoutSyncResponse,
  CutoutSyncState,
  EmbeddingModel,
  EmbeddingJob,
  EmbeddingExport,
  CutoutCandidate,
  ProductCandidatesResponse,
  CollectionInfo,
  EmbeddingJobCreateAdvanced,
  // New 4-tab extraction types
  MatchingExtractionRequest,
  MatchingExtractionResponse,
  TrainingExtractionRequest,
  TrainingExtractionResponse,
  EvaluationExtractionRequest,
  EvaluationExtractionResponse,
  ProductionExtractionRequest,
  ProductionExtractionResponse,
  MatchedProductsStats,
  EmbeddingCollection,
  CutoutProductsResponse,
  CollectionProductsResponse,
  // Training system types (product_id based)
  TrainingRun,
  TrainingRunCreate,
  TrainingRunConfig,
  TrainingRunsResponse,
  TrainingCheckpoint,
  TrainedModel,
  ModelEvaluation,
  TrainingConfigPreset,
  TrainingProductsResponse,
  ModelPresetsResponse,
  LabelStatsResponse,
  IdentifierMappingResponse,
  ModelComparisonResult,
  // Triplet Mining types
  TripletMiningRun,
  TripletMiningRequest,
  TripletMiningStats,
  TripletExportResult,
  MinedTriplet,
  MatchingFeedbackCreate,
  MatchingFeedback,
  FeedbackStats,
  HardExample,
  TrainingMetricsHistory,
  TrainingProgress,
  // Scan Requests
  ScanRequest,
  ScanRequestCreate,
  ScanRequestsResponse,
  DuplicateCheckResponse,
} from "@/types";
import { getAuthHeader, clearAuth } from "./auth";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Export filter options for download/export endpoints
export interface ExportFilters {
  product_ids?: string[];
  search?: string;
  status?: string[];
  category?: string[];
  brand?: string[];
  sub_brand?: string[];
  product_name?: string[];
  variant_flavor?: string[];
  container_type?: string[];
  net_quantity?: string[];
  pack_type?: string[];
  manufacturer_country?: string[];
  claims?: string[];
  has_video?: boolean;
  has_image?: boolean;
  has_nutrition?: boolean;
  has_description?: boolean;
  has_prompt?: boolean;
  has_issues?: boolean;
  frame_count_min?: number;
  frame_count_max?: number;
  visibility_score_min?: number;
  visibility_score_max?: number;
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  params?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const { params, body, ...fetchOptions } = options;

    let url = `${this.baseUrl}${endpoint}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      const queryString = searchParams.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader(),
        ...fetchOptions.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));

      // Handle 401 Unauthorized - clear auth and redirect to login
      // But only if we're not already on the login page (to prevent infinite loop)
      if (response.status === 401) {
        clearAuth();
        if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
          window.location.href = "/login";
        }
      }

      // Handle case where error.detail might be an object
      const errorMessage = typeof error.detail === 'string'
        ? error.detail
        : (error.detail?.message || error.message || JSON.stringify(error.detail) || `HTTP ${response.status}`);

      throw new ApiError(
        errorMessage,
        response.status,
        error
      );
    }

    // Handle empty responses
    const text = await response.text();
    if (!text) return {} as T;

    return JSON.parse(text);
  }

  private async requestBlob(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<Blob> {
    const { params, body, ...fetchOptions } = options;

    let url = `${this.baseUrl}${endpoint}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      const queryString = searchParams.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader(),
        ...fetchOptions.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));

      // Handle 401 Unauthorized - clear auth and redirect to login
      // But only if we're not already on the login page (to prevent infinite loop)
      if (response.status === 401) {
        clearAuth();
        if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
          window.location.href = "/login";
        }
      }

      // Handle case where error.detail might be an object
      const errorMessage = typeof error.detail === 'string'
        ? error.detail
        : (error.detail?.message || error.message || JSON.stringify(error.detail) || `HTTP ${response.status}`);

      throw new ApiError(
        errorMessage,
        response.status,
        error
      );
    }

    return response.blob();
  }

  // ===========================================
  // Dashboard
  // ===========================================

  async getDashboardStats(): Promise<DashboardStats> {
    return this.request<DashboardStats>("/api/v1/dashboard/stats");
  }

  // ===========================================
  // Products
  // ===========================================

  async getProducts(params?: {
    page?: number;
    limit?: number;
    search?: string;
    // Sorting parameters
    sort_by?: string;
    sort_order?: "asc" | "desc";
    // All filter parameters (comma-separated for multiple values)
    status?: string;
    category?: string;
    brand?: string;
    sub_brand?: string;
    product_name?: string;
    variant_flavor?: string;
    container_type?: string;
    net_quantity?: string;
    pack_type?: string;
    manufacturer_country?: string;
    claims?: string;
    // Boolean filters
    has_video?: boolean;
    has_image?: boolean;
    has_nutrition?: boolean;
    has_description?: boolean;
    has_prompt?: boolean;
    has_issues?: boolean;
    // Range filters
    frame_count_min?: number;
    frame_count_max?: number;
    visibility_score_min?: number;
    visibility_score_max?: number;
    // Exclusion filters
    exclude_dataset_id?: string;
    include_frame_counts?: boolean;
  }): Promise<ProductsResponse> {
    return this.request<ProductsResponse>("/api/v1/products", { params });
  }

  async getFilterOptions(params?: {
    // Current filter selections (for cascading filters)
    status?: string;
    category?: string;
    brand?: string;
    sub_brand?: string;
    product_name?: string;
    variant_flavor?: string;
    container_type?: string;
    net_quantity?: string;
    pack_type?: string;
    manufacturer_country?: string;
    claims?: string;
    has_video?: boolean;
    has_image?: boolean;
    has_nutrition?: boolean;
    has_description?: boolean;
    has_prompt?: boolean;
    has_issues?: boolean;
    frame_count_min?: number;
    frame_count_max?: number;
    visibility_score_min?: number;
    visibility_score_max?: number;
    exclude_dataset_id?: string;
  }): Promise<FilterOptionsResponse> {
    return this.request<FilterOptionsResponse>("/api/v1/products/filter-options", { params });
  }

  async getProduct(id: string): Promise<Product> {
    return this.request<Product>(`/api/v1/products/${id}`);
  }

  async updateProduct(
    id: string,
    data: Partial<Product>,
    version?: number
  ): Promise<Product> {
    return this.request<Product>(`/api/v1/products/${id}`, {
      method: "PATCH",
      body: { ...data, version },
    });
  }

  async deleteProduct(id: string): Promise<void> {
    await this.request<void>(`/api/v1/products/${id}`, { method: "DELETE" });
  }

  async deleteProducts(ids: string[]): Promise<{ deleted_count: number }> {
    return this.request<{ deleted_count: number }>(
      "/api/v1/products/bulk-delete",
      {
        method: "POST",
        body: { product_ids: ids },
      }
    );
  }

  async setPrimaryImage(
    productId: string,
    imageUrl: string,
    version?: number
  ): Promise<Product> {
    return this.request<Product>(`/api/v1/products/${productId}`, {
      method: "PATCH",
      body: { primary_image_url: imageUrl, version },
    });
  }

  async getProductFrames(
    id: string,
    imageType?: "synthetic" | "real" | "augmented"
  ): Promise<{
    frames: {
      id: string | null;
      url: string;
      index: number;
      image_type: "synthetic" | "real" | "augmented";
      source: string;
    }[];
    counts: { synthetic: number; real: number; augmented: number };
    total: number;
  }> {
    const params = imageType ? `?image_type=${imageType}` : "";
    return this.request(`/api/v1/products/${id}/frames${params}`);
  }

  // ===========================================
  // Product Images Browse (for CLS Visual Picker)
  // ===========================================

  async browseProductImages(params: {
    page?: number;
    limit?: number;
    categories?: string[];
    brands?: string[];
    statuses?: string[];
    image_types?: string[];
    search?: string;
  }): Promise<{
    images: {
      id: string;
      image_url: string;
      product_id: string;
      product_name: string | null;
      brand_name: string | null;
      category: string | null;
      image_type: string;
      status: string | null;
    }[];
    total: number;
    page: number;
    limit: number;
    has_more: boolean;
    filters: {
      categories: string[];
      brands: string[];
      statuses: string[];
      image_types: string[];
    };
  }> {
    return this.request("/api/v1/products/images/browse", {
      method: "POST",
      body: params,
    });
  }

  async getProductImageFilterOptions(): Promise<{
    categories: string[];
    brands: string[];
    statuses: string[];
    image_types: string[];
  }> {
    // Filter options are included in browse response
    const result = await this.browseProductImages({ page: 1, limit: 1 });
    return result.filters;
  }

  async deleteProductFrames(
    id: string,
    frameIds: string[]
  ): Promise<{ deleted: number; storage_deleted: number }> {
    const params = frameIds.map((fid) => `frame_ids=${fid}`).join("&");
    return this.request(`/api/v1/products/${id}/frames?${params}`, {
      method: "DELETE",
    });
  }

  async getProductCategories(): Promise<string[]> {
    return this.request<string[]>("/api/v1/products/categories");
  }

  // Product Downloads - with full filter support
  async downloadProducts(filters: ExportFilters): Promise<Blob> {
    return this.requestBlob("/api/v1/products/download", {
      method: "POST",
      body: filters,
    });
  }

  async downloadAllProducts(filters?: ExportFilters): Promise<Blob> {
    return this.requestBlob("/api/v1/products/download", {
      method: "POST",
      body: filters || {},
    });
  }

  async downloadProduct(id: string): Promise<Blob> {
    return this.requestBlob(`/api/v1/products/${id}/download`);
  }

  async exportProductsCSV(filters?: ExportFilters): Promise<Blob> {
    return this.requestBlob("/api/v1/products/export/csv", {
      method: "POST",
      body: filters || {},
    });
  }

  async exportProductsJSON(filters?: ExportFilters): Promise<Blob> {
    return this.requestBlob("/api/v1/products/export/json", {
      method: "POST",
      body: filters || {},
    });
  }

  // ===========================================
  // Product Identifiers
  // ===========================================

  async getProductIdentifiers(productId: string): Promise<ProductIdentifier[]> {
    return this.request<ProductIdentifier[]>(
      `/api/v1/products/${productId}/identifiers`
    );
  }

  async addProductIdentifier(
    productId: string,
    data: ProductIdentifierCreate
  ): Promise<ProductIdentifier> {
    return this.request<ProductIdentifier>(
      `/api/v1/products/${productId}/identifiers`,
      {
        method: "POST",
        body: data,
      }
    );
  }

  async updateProductIdentifiers(
    productId: string,
    identifiers: ProductIdentifierCreate[]
  ): Promise<ProductIdentifier[]> {
    return this.request<ProductIdentifier[]>(
      `/api/v1/products/${productId}/identifiers`,
      {
        method: "PUT",
        body: { identifiers },
      }
    );
  }

  async updateProductIdentifier(
    productId: string,
    identifierId: string,
    data: Partial<ProductIdentifierCreate>
  ): Promise<ProductIdentifier> {
    return this.request<ProductIdentifier>(
      `/api/v1/products/${productId}/identifiers/${identifierId}`,
      {
        method: "PATCH",
        body: data,
      }
    );
  }

  async deleteProductIdentifier(
    productId: string,
    identifierId: string
  ): Promise<void> {
    await this.request<void>(
      `/api/v1/products/${productId}/identifiers/${identifierId}`,
      { method: "DELETE" }
    );
  }

  async setPrimaryIdentifier(
    productId: string,
    identifierId: string
  ): Promise<ProductIdentifier> {
    return this.request<ProductIdentifier>(
      `/api/v1/products/${productId}/identifiers/${identifierId}/set-primary`,
      { method: "POST" }
    );
  }

  // ===========================================
  // Custom Fields
  // ===========================================

  async getCustomFields(productId: string): Promise<Record<string, string>> {
    return this.request<Record<string, string>>(
      `/api/v1/products/${productId}/custom-fields`
    );
  }

  async updateCustomFields(
    productId: string,
    fields: Record<string, string>
  ): Promise<Record<string, string>> {
    return this.request<Record<string, string>>(
      `/api/v1/products/${productId}/custom-fields`,
      {
        method: "PUT",
        body: { custom_fields: fields },
      }
    );
  }

  async patchCustomFields(
    productId: string,
    updates: Record<string, string | null>
  ): Promise<Record<string, string>> {
    return this.request<Record<string, string>>(
      `/api/v1/products/${productId}/custom-fields`,
      {
        method: "PATCH",
        body: updates,
      }
    );
  }

  // ===========================================
  // Videos
  // ===========================================

  async getVideos(): Promise<Video[]> {
    return this.request<Video[]>("/api/v1/videos");
  }

  async syncVideos(limit?: number): Promise<VideoSyncResponse> {
    return this.request<VideoSyncResponse>("/api/v1/videos/sync", {
      method: "POST",
      params: limit ? { limit } : undefined,
    });
  }

  async processVideo(
    videoId: number,
    options?: { sampleRate?: number; maxFrames?: number }
  ): Promise<Job> {
    return this.request<Job>("/api/v1/videos/process", {
      method: "POST",
      body: {
        video_id: videoId,
        sample_rate: options?.sampleRate,
        max_frames: options?.maxFrames,
      },
    });
  }

  async processVideos(
    videoIds: number[],
    options?: { sampleRate?: number; maxFrames?: number; geminiModel?: string }
  ): Promise<Job[]> {
    return this.request<Job[]>("/api/v1/videos/process/batch", {
      method: "POST",
      body: {
        video_ids: videoIds,
        sample_rate: options?.sampleRate,
        max_frames: options?.maxFrames,
        gemini_model: options?.geminiModel,
      },
    });
  }

  async reprocessVideo(videoId: number): Promise<{ job: Job; cleanup: { frames_deleted: number; files_deleted: number }; message: string }> {
    return this.request(`/api/v1/videos/${videoId}/reprocess`, {
      method: "POST",
    });
  }

  async reprocessProduct(
    productId: string,
    options?: {
      custom_prompts?: string[];
      points?: Array<{ x: number; y: number; label: number }>;
    }
  ): Promise<{ job: Job; cleanup: { frames_deleted: number; files_deleted: number }; message: string }> {
    return this.request(`/api/v1/products/${productId}/reprocess`, {
      method: "POST",
      body: options || undefined,
    });
  }

  async previewSegmentation(
    productId: string,
    options: {
      text_prompts?: string[];
      points?: Array<{ x: number; y: number; label: number }>;
    }
  ): Promise<{
    mask_image: string;
    first_frame: string;
    mask_stats: {
      pixel_count: number;
      coverage_percent: number;
      width: number;
      height: number;
    };
  }> {
    return this.request(`/api/v1/products/${productId}/preview-segmentation`, {
      method: "POST",
      body: options,
    });
  }

  async syncRunpodStatus(): Promise<{
    checked: number;
    updated: number;
    completed: number;
    failed: number;
    still_running: number;
    errors: string[];
  }> {
    return this.request("/api/v1/videos/sync-runpod-status", {
      method: "POST",
    });
  }

  async clearStuckVideos(): Promise<{
    checked: number;
    cleared: number;
    no_job: number;
    job_finished: number;
  }> {
    return this.request("/api/v1/videos/clear-stuck", {
      method: "POST",
    });
  }

  // ===========================================
  // Datasets
  // ===========================================

  async getDatasets(): Promise<Dataset[]> {
    return this.request<Dataset[]>("/api/v1/datasets");
  }

  async getDataset(
    id: string,
    params?: {
      page?: number;
      limit?: number;
      search?: string;
      sort_by?: string;
      sort_order?: "asc" | "desc";
      status?: string;
      category?: string;
      brand?: string;
      sub_brand?: string;
      product_name?: string;
      variant_flavor?: string;
      container_type?: string;
      net_quantity?: string;
      pack_type?: string;
      manufacturer_country?: string;
      claims?: string;
      has_video?: boolean;
      has_image?: boolean;
      has_nutrition?: boolean;
      has_description?: boolean;
      has_prompt?: boolean;
      has_issues?: boolean;
      frame_count_min?: number;
      frame_count_max?: number;
      visibility_score_min?: number;
      visibility_score_max?: number;
      include_frame_counts?: boolean;
    }
  ): Promise<DatasetWithProducts> {
    return this.request<DatasetWithProducts>(`/api/v1/datasets/${id}`, { params });
  }

  async getDatasetFilterOptions(datasetId: string): Promise<FilterOptionsResponse> {
    return this.request<FilterOptionsResponse>(`/api/v1/datasets/${datasetId}/filter-options`);
  }

  async createDataset(data: CreateDatasetRequest): Promise<Dataset> {
    return this.request<Dataset>("/api/v1/datasets", {
      method: "POST",
      body: data,
    });
  }

  async updateDataset(
    id: string,
    data: Partial<Dataset>,
    version?: number
  ): Promise<Dataset> {
    return this.request<Dataset>(`/api/v1/datasets/${id}`, {
      method: "PATCH",
      body: { ...data, version },
    });
  }

  async deleteDataset(id: string): Promise<void> {
    await this.request<void>(`/api/v1/datasets/${id}`, { method: "DELETE" });
  }

  async addProductsToDataset(
    datasetId: string,
    productIds: string[]
  ): Promise<{ added_count: number }> {
    return this.request<{ added_count: number }>(
      `/api/v1/datasets/${datasetId}/products`,
      {
        method: "POST",
        body: { product_ids: productIds },
      }
    );
  }

  async addFilteredProductsToDataset(
    datasetId: string,
    filters: ExportFilters
  ): Promise<{ 
    added_count: number; 
    skipped_count?: number; 
    total_matching?: number;
    duration_ms?: number;
  }> {
    return this.request<{ 
      added_count: number; 
      skipped_count?: number; 
      total_matching?: number;
      duration_ms?: number;
    }>(
      `/api/v1/datasets/${datasetId}/products`,
      {
        method: "POST",
        body: { filters },
      }
    );
  }

  async removeProductFromDataset(
    datasetId: string,
    productId: string
  ): Promise<void> {
    await this.request<void>(
      `/api/v1/datasets/${datasetId}/products/${productId}`,
      { method: "DELETE" }
    );
  }

  // Dataset Actions
  async startAugmentation(
    datasetId: string,
    config: AugmentationRequest
  ): Promise<Job> {
    return this.request<Job>(`/api/v1/datasets/${datasetId}/augment`, {
      method: "POST",
      body: config,
    });
  }

  async startTrainingFromDataset(
    datasetId: string,
    config: TrainingConfig
  ): Promise<Job> {
    return this.request<Job>(`/api/v1/datasets/${datasetId}/train`, {
      method: "POST",
      body: config,
    });
  }

  async startEmbeddingExtraction(
    datasetId: string,
    modelId: string
  ): Promise<Job> {
    return this.request<Job>(`/api/v1/datasets/${datasetId}/extract`, {
      method: "POST",
      body: { model_id: modelId },
    });
  }

  // ===========================================
  // Jobs
  // ===========================================

  async getJobs(type?: string): Promise<Job[]> {
    return this.request<Job[]>("/api/v1/jobs", {
      params: type ? { type } : undefined,
    });
  }

  async getJob(id: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${id}`);
  }

  async cancelJob(id: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${id}/cancel`, { method: "POST" });
  }

  async cancelJobsBatch(params?: {
    job_ids?: string[];
    job_type?: string;
  }): Promise<{ cancelled_count: number; failed_count: number; errors: string[] }> {
    return this.request("/api/v1/jobs/batch/cancel", {
      method: "POST",
      body: params || {},
    });
  }

  async getActiveJobsCount(type?: string): Promise<{ count: number }> {
    return this.request<{ count: number }>("/api/v1/jobs/active/count", {
      params: type ? { job_type: type } : undefined,
    });
  }

  // ===========================================
  // Training
  // ===========================================

  async getTrainingJobs(): Promise<TrainingJob[]> {
    return this.request<TrainingJob[]>("/api/v1/training/jobs");
  }

  async getTrainingJob(id: string): Promise<TrainingJob> {
    return this.request<TrainingJob>(`/api/v1/training/jobs/${id}`);
  }

  async startTrainingJob(config: TrainingConfig): Promise<Job> {
    return this.request<Job>("/api/v1/training/start", {
      method: "POST",
      body: config,
    });
  }

  async getTrainingModels(): Promise<ModelArtifact[]> {
    return this.request<ModelArtifact[]>("/api/v1/training/models");
  }

  async activateModel(modelId: string): Promise<ModelArtifact> {
    return this.request<ModelArtifact>(
      `/api/v1/training/models/${modelId}/activate`,
      { method: "POST" }
    );
  }

  async downloadModel(modelId: string): Promise<Blob> {
    return this.requestBlob(`/api/v1/training/models/${modelId}/download`);
  }

  // ===========================================
  // Matching
  // ===========================================

  async getMatchingProducts(params?: {
    page?: number;
    limit?: number;
    search?: string;
  }): Promise<ProductSummary[]> {
    return this.request<ProductSummary[]>("/api/v1/matching/products", {
      params,
    });
  }

  async getMatchingProduct(productId: string): Promise<{
    product: Product;
    synthetic_frames: { url: string; index: number; type: string }[];
    real_images: RealImage[];
  }> {
    return this.request(`/api/v1/matching/products/${productId}`);
  }

  async getProductRealImages(productId: string): Promise<{ images: RealImage[] }> {
    return this.request(`/api/v1/matching/products/${productId}/real-images`);
  }

  async addRealImages(
    productId: string,
    imageUrls: string[]
  ): Promise<{ added: number }> {
    return this.request(`/api/v1/matching/products/${productId}/real-images`, {
      method: "POST",
      body: imageUrls,
    });
  }

  async removeRealImages(
    productId: string,
    imageIds: string[]
  ): Promise<{ status: string }> {
    return this.request(`/api/v1/matching/products/${productId}/real-images`, {
      method: "DELETE",
      body: imageIds,
    });
  }

  // Legacy methods (for compatibility)
  async getMatchingUPCs(): Promise<string[]> {
    return this.request<string[]>("/api/v1/matching/upcs");
  }

  async searchMatches(
    upc: string,
    topK?: number
  ): Promise<{ candidates: MatchCandidate[] }> {
    return this.request<{ candidates: MatchCandidate[] }>(
      `/api/v1/matching/upcs/${upc}/search`,
      {
        method: "POST",
        body: { top_k: topK || 10 },
      }
    );
  }

  async approveMatch(matchId: string, approved: boolean): Promise<void> {
    await this.request<void>("/api/v1/matching/approve", {
      method: "POST",
      body: { match_id: matchId, is_approved: approved },
    });
  }

  // ===========================================
  // Embeddings
  // ===========================================

  async getEmbeddingIndexes(): Promise<EmbeddingIndex[]> {
    return this.request<EmbeddingIndex[]>("/api/v1/embeddings/indexes");
  }

  async createEmbeddingIndex(
    name: string,
    modelId: string
  ): Promise<EmbeddingIndex> {
    return this.request<EmbeddingIndex>("/api/v1/embeddings/indexes", {
      method: "POST",
      body: { name, model_id: modelId },
    });
  }

  // ===========================================
  // Resource Locks (Multi-user)
  // ===========================================

  async acquireLock(
    resourceType: "product" | "dataset",
    resourceId: string
  ): Promise<ResourceLock> {
    return this.request<ResourceLock>("/api/v1/locks/acquire", {
      method: "POST",
      body: { resource_type: resourceType, resource_id: resourceId },
    });
  }

  async releaseLock(lockId: string): Promise<void> {
    await this.request<void>(`/api/v1/locks/${lockId}`, { method: "DELETE" });
  }

  async getLockInfo(
    resourceType: "product" | "dataset",
    resourceId: string
  ): Promise<ResourceLock | null> {
    try {
      return await this.request<ResourceLock>(
        `/api/v1/locks/${resourceType}/${resourceId}`
      );
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async refreshLock(lockId: string): Promise<ResourceLock> {
    return this.request<ResourceLock>(`/api/v1/locks/${lockId}/refresh`, {
      method: "POST",
    });
  }

  // ===========================================
  // Cutout Images (Matching System)
  // ===========================================

  async getCutouts(params?: {
    page?: number;
    limit?: number;
    has_embedding?: boolean;
    is_matched?: boolean;
    predicted_upc?: string;
    // NEW: Visual picker filters
    matched_category?: string;
    matched_brand?: string;
    date_from?: string;
    date_to?: string;
    search?: string;
  }): Promise<CutoutsResponse> {
    return this.request<CutoutsResponse>("/api/v1/cutouts", { params });
  }

  async getCutoutFilterOptions(): Promise<{
    categories: string[];
    brands: string[];
  }> {
    return this.request("/api/v1/cutouts/filter-options");
  }

  async getCutout(id: string): Promise<CutoutImage> {
    return this.request<CutoutImage>(`/api/v1/cutouts/${id}`);
  }

  async getCutoutStats(): Promise<CutoutStats> {
    return this.request<CutoutStats>("/api/v1/cutouts/stats");
  }

  async syncCutouts(params?: {
    max_pages?: number;
    page_size?: number;
    sort_order?: "asc" | "desc";  // "desc" for newest first, "asc" for oldest first
  }): Promise<CutoutSyncResponse> {
    return this.request<CutoutSyncResponse>("/api/v1/cutouts/sync", {
      method: "POST",
      body: params || {},
    });
  }

  async getSyncState(): Promise<CutoutSyncState> {
    return this.request<CutoutSyncState>("/api/v1/cutouts/sync/state");
  }

  async syncNewCutouts(params?: {
    max_items?: number;
    page_size?: number;
  }): Promise<CutoutSyncResponse> {
    return this.request<CutoutSyncResponse>("/api/v1/cutouts/sync/new", {
      method: "POST",
      body: params || {},
    });
  }

  async backfillCutouts(params?: {
    max_items?: number;
    page_size?: number;
    start_page?: number;
  }): Promise<CutoutSyncResponse> {
    return this.request<CutoutSyncResponse>("/api/v1/cutouts/sync/backfill", {
      method: "POST",
      body: params || {},
    });
  }

  async matchCutout(
    cutoutId: string,
    productId: string,
    similarity?: number
  ): Promise<CutoutImage> {
    return this.request<CutoutImage>(`/api/v1/cutouts/${cutoutId}/match`, {
      method: "POST",
      body: { product_id: productId, similarity },
    });
  }

  async unmatchCutout(cutoutId: string): Promise<CutoutImage> {
    return this.request<CutoutImage>(`/api/v1/cutouts/${cutoutId}/unmatch`, {
      method: "POST",
    });
  }

  async deleteCutout(id: string): Promise<void> {
    await this.request<void>(`/api/v1/cutouts/${id}`, { method: "DELETE" });
  }

  // ===========================================
  // Product Matching (New Product-Centric)
  // ===========================================

  async getProductCandidates(
    productId: string,
    params?: {
      min_similarity?: number;
      include_matched?: boolean;
      limit?: number;
      match_type_filter?: string;
      product_collection?: string;
      cutout_collection?: string;
    }
  ): Promise<ProductCandidatesResponse> {
    return this.request<ProductCandidatesResponse>(
      `/api/v1/matching/products/${productId}/candidates`,
      { params }
    );
  }

  async bulkMatchCutouts(
    productId: string,
    cutoutIds: string[],
    similarityScores?: number[]
  ): Promise<{ matched_count: number; images_added: number }> {
    return this.request(`/api/v1/matching/products/${productId}/match`, {
      method: "POST",
      body: {
        cutout_ids: cutoutIds,
        similarity_scores: similarityScores,
      },
    });
  }

  async bulkUnmatchCutouts(
    productId: string,
    cutoutIds: string[]
  ): Promise<{ unmatched_count: number }> {
    return this.request(`/api/v1/matching/products/${productId}/unmatch`, {
      method: "POST",
      body: cutoutIds,
    });
  }

  // ===========================================
  // Embedding Models (Matching System)
  // ===========================================

  async getEmbeddingModels(): Promise<EmbeddingModel[]> {
    return this.request<EmbeddingModel[]>("/api/v1/embeddings/models");
  }

  async getEmbeddingModel(id: string): Promise<EmbeddingModel> {
    return this.request<EmbeddingModel>(`/api/v1/embeddings/models/${id}`);
  }

  async getActiveEmbeddingModel(): Promise<EmbeddingModel | null> {
    try {
      return await this.request<EmbeddingModel>("/api/v1/embeddings/models/active");
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async activateEmbeddingModel(modelId: string): Promise<EmbeddingModel> {
    return this.request<EmbeddingModel>(
      `/api/v1/embeddings/models/${modelId}/activate`,
      { method: "POST" }
    );
  }

  async deleteEmbeddingModel(modelId: string): Promise<void> {
    await this.request<void>(`/api/v1/embeddings/models/${modelId}`, {
      method: "DELETE",
    });
  }

  // ===========================================
  // Embedding Jobs (Matching System)
  // ===========================================

  async getEmbeddingJobs(status?: string): Promise<EmbeddingJob[]> {
    return this.request<EmbeddingJob[]>("/api/v1/embeddings/jobs", {
      params: status ? { status } : undefined,
    });
  }

  async getEmbeddingJob(id: string): Promise<EmbeddingJob> {
    return this.request<EmbeddingJob>(`/api/v1/embeddings/jobs/${id}`);
  }

  async startEmbeddingJob(params: {
    model_id: string;
    job_type: "full" | "incremental";
    source: "cutouts" | "products" | "both";
  }): Promise<EmbeddingJob> {
    return this.request<EmbeddingJob>("/api/v1/embeddings/jobs", {
      method: "POST",
      body: params,
    });
  }

  async startEmbeddingJobAdvanced(params: EmbeddingJobCreateAdvanced): Promise<EmbeddingJob> {
    return this.request<EmbeddingJob>("/api/v1/embeddings/jobs/advanced", {
      method: "POST",
      body: params,
    });
  }

  async cancelEmbeddingJob(jobId: string): Promise<{ status: string; message: string; job_id: string }> {
    return this.request(`/api/v1/embeddings/jobs/${jobId}/cancel`, {
      method: "POST",
    });
  }

  // ===========================================
  // Qdrant Collections
  // ===========================================

  async getQdrantCollections(): Promise<CollectionInfo[]> {
    return this.request<CollectionInfo[]>("/api/v1/embeddings/collections");
  }

  async getQdrantCollectionStats(collectionName: string): Promise<CollectionInfo> {
    return this.request<CollectionInfo>(`/api/v1/embeddings/collections/${collectionName}/stats`);
  }

  async deleteQdrantCollection(collectionName: string): Promise<void> {
    await this.request<void>(`/api/v1/embeddings/collections/${collectionName}`, {
      method: "DELETE",
    });
  }

  async exportCollection(
    collectionName: string,
    format: "json" | "numpy" | "faiss" = "json"
  ): Promise<{
    export_id: string;
    collection_name: string;
    format: string;
    vector_count: number;
    file_url: string;
    file_size_bytes: number;
  }> {
    return this.request(`/api/v1/embeddings/collections/${collectionName}/export`, {
      method: "POST",
      body: { format },
    });
  }

  async getCollectionProducts(
    collectionName: string,
    params?: {
      page?: number;
      limit?: number;
      search?: string;
    }
  ): Promise<CollectionProductsResponse> {
    return this.request<CollectionProductsResponse>(
      `/api/v1/embeddings/collections/${collectionName}/products`,
      { params }
    );
  }

  // ===========================================
  // Embedding Exports (Matching System)
  // ===========================================

  async getEmbeddingExports(): Promise<EmbeddingExport[]> {
    return this.request<EmbeddingExport[]>("/api/v1/embeddings/exports");
  }

  async createEmbeddingExport(params: {
    model_id: string;
    format: "json" | "numpy" | "faiss" | "qdrant_snapshot";
  }): Promise<EmbeddingExport> {
    return this.request<EmbeddingExport>("/api/v1/embeddings/exports", {
      method: "POST",
      body: params,
    });
  }

  async downloadEmbeddingExport(exportId: string): Promise<Blob> {
    return this.requestBlob(`/api/v1/embeddings/exports/${exportId}/download`);
  }

  // ===========================================
  // 3-Tab Embedding Extraction
  // ===========================================

  // Tab 1: Matching Extraction
  async startMatchingExtraction(
    params: MatchingExtractionRequest
  ): Promise<MatchingExtractionResponse> {
    return this.request<MatchingExtractionResponse>("/api/v1/embeddings/jobs/matching", {
      method: "POST",
      body: params,
    });
  }

  // Tab 2: Training Extraction
  async startTrainingExtraction(
    params: TrainingExtractionRequest
  ): Promise<TrainingExtractionResponse> {
    return this.request<TrainingExtractionResponse>("/api/v1/embeddings/jobs/training", {
      method: "POST",
      body: params,
    });
  }

  // Tab 3: Evaluation Extraction
  async startEvaluationExtraction(
    params: EvaluationExtractionRequest
  ): Promise<EvaluationExtractionResponse> {
    return this.request<EvaluationExtractionResponse>("/api/v1/embeddings/jobs/evaluation", {
      method: "POST",
      body: params,
    });
  }

  // Tab 4: Production Extraction (SOTA: multi-view + augmentation for inference)
  async startProductionExtraction(
    params: ProductionExtractionRequest
  ): Promise<ProductionExtractionResponse> {
    return this.request<ProductionExtractionResponse>("/api/v1/embeddings/jobs/production", {
      method: "POST",
      body: params,
    });
  }

  // Matched Products Stats (for Training tab)
  async getMatchedProductsStats(): Promise<MatchedProductsStats> {
    return this.request<MatchedProductsStats>("/api/v1/embeddings/matched-products/stats");
  }

  // ===========================================
  // Embedding Collection Metadata
  // ===========================================

  async getEmbeddingCollections(params?: {
    collection_type?: string;
  }): Promise<EmbeddingCollection[]> {
    return this.request<EmbeddingCollection[]>("/api/v1/embeddings/collection-metadata", {
      params,
    });
  }

  async getEmbeddingCollection(id: string): Promise<EmbeddingCollection> {
    return this.request<EmbeddingCollection>(`/api/v1/embeddings/collection-metadata/${id}`);
  }

  // ===========================================
  // Reverse Matching (Cutout → Products)
  // ===========================================

  async getCutoutProductCandidates(
    cutoutId: string,
    params?: {
      min_similarity?: number;
      limit?: number;
      product_collection?: string;
      cutout_collection?: string;
    }
  ): Promise<CutoutProductsResponse> {
    return this.request<CutoutProductsResponse>(
      `/api/v1/matching/cutouts/${cutoutId}/products`,
      { params }
    );
  }

  // ===========================================
  // Training System (product_id based)
  // ===========================================

  // Training Runs
  async getTrainingRuns(params?: {
    status?: string;
    base_model_type?: string;
    limit?: number;
    offset?: number;
  }): Promise<TrainingRunsResponse> {
    return this.request<TrainingRunsResponse>("/api/v1/training/runs", { params });
  }

  async getTrainingRun(id: string): Promise<TrainingRun> {
    return this.request<TrainingRun>(`/api/v1/training/runs/${id}`);
  }

  async createTrainingRun(data: TrainingRunCreate): Promise<TrainingRun> {
    return this.request<TrainingRun>("/api/v1/training/runs", {
      method: "POST",
      body: data,
    });
  }

  async cancelTrainingRun(id: string): Promise<TrainingRun> {
    return this.request<TrainingRun>(`/api/v1/training/runs/${id}/cancel`, {
      method: "POST",
    });
  }

  async resumeTrainingRun(id: string): Promise<{
    new_run_id: string;
    resumed_from_epoch: number;
    remaining_epochs: number;
    checkpoint_url: string;
  }> {
    return this.request(`/api/v1/training/runs/${id}/resume`, {
      method: "POST",
    });
  }

  async deleteTrainingRun(id: string, force: boolean = false): Promise<void> {
    return this.request<void>(`/api/v1/training/runs/${id}`, {
      method: "DELETE",
      params: force ? { force: "true" } : undefined,
    });
  }

  async getIdentifierMapping(runId: string): Promise<IdentifierMappingResponse> {
    return this.request<IdentifierMappingResponse>(`/api/v1/training/runs/${runId}/identifier-mapping`);
  }

  // Training Checkpoints
  async getTrainingCheckpoints(runId: string, params?: {
    is_best?: boolean;
  }): Promise<TrainingCheckpoint[]> {
    return this.request<TrainingCheckpoint[]>(`/api/v1/training/runs/${runId}/checkpoints`, { params });
  }

  async getTrainingCheckpoint(checkpointId: string): Promise<TrainingCheckpoint> {
    return this.request<TrainingCheckpoint>(`/api/v1/training/checkpoints/${checkpointId}`);
  }

  async deleteTrainingCheckpoint(checkpointId: string): Promise<void> {
    return this.request<void>(`/api/v1/training/checkpoints/${checkpointId}`, {
      method: "DELETE",
    });
  }

  // Training Metrics History
  async getTrainingMetricsHistory(runId: string): Promise<TrainingMetricsHistory[]> {
    return this.request<TrainingMetricsHistory[]>(`/api/v1/training/runs/${runId}/metrics-history`);
  }

  async getTrainingProgress(runId: string): Promise<TrainingProgress> {
    return this.request<TrainingProgress>(`/api/v1/training/runs/${runId}/progress`);
  }

  // Trained Models
  async getTrainedModels(params?: {
    is_active?: boolean;
    limit?: number;
  }): Promise<TrainedModel[]> {
    return this.request<TrainedModel[]>("/api/v1/training/models", { params });
  }

  async getTrainedModel(id: string): Promise<TrainedModel> {
    return this.request<TrainedModel>(`/api/v1/training/models/${id}`);
  }

  async registerTrainedModel(data: {
    checkpoint_id: string;
    name: string;
    description?: string;
  }): Promise<TrainedModel> {
    return this.request<TrainedModel>("/api/v1/training/models", {
      method: "POST",
      body: data,
    });
  }

  async activateTrainedModel(id: string): Promise<TrainedModel> {
    return this.request<TrainedModel>(`/api/v1/training/models/${id}/activate`, {
      method: "POST",
    });
  }

  async evaluateTrainedModel(id: string): Promise<TrainedModel> {
    return this.request<TrainedModel>(`/api/v1/training/models/${id}/evaluate`, {
      method: "POST",
    });
  }

  async deleteTrainedModel(id: string): Promise<void> {
    return this.request<void>(`/api/v1/training/models/${id}`, {
      method: "DELETE",
    });
  }

  async compareModels(modelIds: string[]): Promise<ModelComparisonResult[]> {
    return this.request<ModelComparisonResult[]>("/api/v1/training/models/compare", {
      method: "POST",
      body: modelIds,
    });
  }

  // Model Evaluations
  async getModelEvaluations(modelId: string): Promise<ModelEvaluation[]> {
    return this.request<ModelEvaluation[]>(`/api/v1/training/models/${modelId}/evaluations`);
  }

  // Training Config Presets
  async getTrainingConfigPresets(): Promise<TrainingConfigPreset[]> {
    return this.request<TrainingConfigPreset[]>("/api/v1/training/configs/presets");
  }

  async getTrainingConfigPreset(baseModelType: string): Promise<TrainingRunConfig> {
    return this.request<TrainingRunConfig>(`/api/v1/training/configs/presets/${baseModelType}`);
  }

  async getTrainingSavedConfigs(params?: {
    base_model_type?: string;
  }): Promise<TrainingConfigPreset[]> {
    return this.request<TrainingConfigPreset[]>("/api/v1/training/configs", { params });
  }

  async saveTrainingConfig(data: {
    name: string;
    description?: string;
    base_model_type: string;
    config: TrainingRunConfig;
  }): Promise<TrainingConfigPreset> {
    return this.request<TrainingConfigPreset>("/api/v1/training/configs", {
      method: "POST",
      body: data,
    });
  }

  async deleteTrainingConfig(id: string): Promise<void> {
    return this.request<void>(`/api/v1/training/configs/${id}`, {
      method: "DELETE",
    });
  }

  // Products for Training
  async getProductsForTraining(params?: {
    data_source?: string;
    dataset_id?: string;
    min_frames?: number;
    limit?: number;
  }): Promise<TrainingProductsResponse> {
    return this.request<TrainingProductsResponse>("/api/v1/training/products", { params });
  }

  // Label Field Statistics (for flexible label configuration)
  async getLabelFieldStats(params?: {
    data_source?: string;
    dataset_id?: string;
  }): Promise<LabelStatsResponse> {
    return this.request<LabelStatsResponse>("/api/v1/training/label-stats", { params });
  }

  // Model Presets (for selecting base models)
  async getModelPresets(): Promise<ModelPresetsResponse> {
    return this.request<ModelPresetsResponse>("/api/v1/embeddings/models/presets");
  }

  // ===========================================
  // Triplet Mining
  // ===========================================

  async startTripletMining(params: TripletMiningRequest): Promise<TripletMiningRun> {
    return this.request<TripletMiningRun>("/api/v1/triplets/mine", {
      method: "POST",
      body: params,
    });
  }

  async getTripletMiningRuns(params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<TripletMiningRun[]> {
    return this.request<TripletMiningRun[]>("/api/v1/triplets/runs", { params });
  }

  async getTripletMiningRun(runId: string): Promise<TripletMiningRun> {
    return this.request<TripletMiningRun>(`/api/v1/triplets/runs/${runId}`);
  }

  async getTripletMiningStats(runId: string): Promise<TripletMiningStats> {
    return this.request<TripletMiningStats>(`/api/v1/triplets/runs/${runId}/stats`);
  }

  async getMinedTriplets(
    runId: string,
    params?: {
      difficulty?: string;
      is_cross_domain?: boolean;
      limit?: number;
      offset?: number;
    }
  ): Promise<{ triplets: MinedTriplet[]; total: number }> {
    return this.request(`/api/v1/triplets/runs/${runId}/triplets`, { params });
  }

  async deleteTripletMiningRun(runId: string): Promise<void> {
    return this.request<void>(`/api/v1/triplets/runs/${runId}`, {
      method: "DELETE",
    });
  }

  async exportTriplets(
    runId: string,
    format: "json" | "csv" = "json"
  ): Promise<TripletExportResult> {
    return this.request<TripletExportResult>(`/api/v1/triplets/runs/${runId}/export`, {
      method: "POST",
      body: { format },
    });
  }

  // ===========================================
  // Matching Feedback (Active Learning)
  // ===========================================

  async submitMatchingFeedback(data: MatchingFeedbackCreate): Promise<MatchingFeedback> {
    return this.request<MatchingFeedback>("/api/v1/triplets/feedback", {
      method: "POST",
      body: data,
    });
  }

  async getFeedbackStats(modelId?: string): Promise<FeedbackStats> {
    return this.request<FeedbackStats>("/api/v1/triplets/feedback/stats", {
      params: modelId ? { model_id: modelId } : undefined,
    });
  }

  async getHardExamplesFromFeedback(params?: {
    model_id?: string;
    limit?: number;
  }): Promise<HardExample[]> {
    return this.request<HardExample[]>("/api/v1/triplets/feedback/hard-examples", { params });
  }

  // ===========================================
  // Scan Requests
  // ===========================================

  async getScanRequests(params?: {
    page?: number;
    limit?: number;
    status?: string;
    search?: string;
  }): Promise<ScanRequestsResponse> {
    return this.request<ScanRequestsResponse>("/api/v1/scan-requests", { params });
  }

  async getScanRequest(id: string): Promise<ScanRequest> {
    return this.request<ScanRequest>(`/api/v1/scan-requests/${id}`);
  }

  async createScanRequest(data: ScanRequestCreate): Promise<ScanRequest> {
    return this.request<ScanRequest>("/api/v1/scan-requests", {
      method: "POST",
      body: data,
    });
  }

  async updateScanRequest(id: string, data: { status?: string; notes?: string }): Promise<ScanRequest> {
    return this.request<ScanRequest>(`/api/v1/scan-requests/${id}`, {
      method: "PATCH",
      body: data,
    });
  }

  async deleteScanRequest(id: string): Promise<void> {
    await this.request(`/api/v1/scan-requests/${id}`, { method: "DELETE" });
  }

  async checkDuplicateScanRequest(barcode: string): Promise<DuplicateCheckResponse> {
    return this.request<DuplicateCheckResponse>("/api/v1/scan-requests/check-duplicate", {
      params: { barcode },
    });
  }

  // ===========================================
  // Object Detection Module
  // ===========================================

  async getODHealth(): Promise<{ status: string; module: string; version: string }> {
    return this.request("/api/v1/od/health");
  }

  async getODStats(): Promise<{
    total_images: number;
    total_datasets: number;
    total_annotations: number;
    total_classes: number;
    total_models: number;
    images_by_status: Record<string, number>;
    recent_datasets: Array<{ id: string; name: string; image_count: number; created_at: string }>;
  }> {
    return this.request("/api/v1/od/stats");
  }

  // OD Images
  async getODImages(params?: {
    page?: number;
    limit?: number;
    // Single value filters (backwards compatible)
    status?: string;
    source?: string;
    folder?: string;
    search?: string;
    merchant_id?: number;
    store_id?: number;
    // Multi-select filters (comma-separated)
    statuses?: string;
    sources?: string;
    folders?: string;
    merchant_ids?: string;
    store_ids?: string;
  }): Promise<{
    images: Array<{
      id: string;
      filename: string;
      image_url: string;
      thumbnail_url?: string;
      width: number;
      height: number;
      status: string;
      source: string;
      folder?: string;
      merchant_id?: number;
      merchant_name?: string;
      store_id?: number;
      store_name?: string;
      created_at: string;
    }>;
    total: number;
    page: number;
    limit: number;
  }> {
    return this.request("/api/v1/od/images", { params });
  }

  async getODImageFilterOptions(): Promise<{
    // New format with counts for FilterDrawer
    status: Array<{ value: string; label: string; count: number }>;
    source: Array<{ value: string; label: string; count: number }>;
    folder: Array<{ value: string; label: string; count: number }>;
    merchant: Array<{ value: string; label: string; count: number }>;
    store: Array<{ value: string; label: string; count: number }>;
    // Backwards compatible
    merchants: Array<{ id: number; name: string }>;
    stores: Array<{ id: number; name: string }>;
  }> {
    return this.request("/api/v1/od/images/filters/options");
  }

  async getODImage(id: string): Promise<{
    id: string;
    filename: string;
    image_url: string;
    width: number;
    height: number;
    status: string;
    source: string;
    folder?: string;
    tags?: string[];
    created_at: string;
  }> {
    return this.request(`/api/v1/od/images/${id}`);
  }

  async uploadODImage(file: File, folder?: string): Promise<{
    id: string;
    filename: string;
    image_url: string;
  }> {
    const formData = new FormData();
    formData.append("file", file);
    if (folder) formData.append("folder", folder);

    const response = await fetch(`${this.baseUrl}/api/v1/od/images`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async uploadODImagesBulk(files: File[]): Promise<Array<{ id: string; filename: string; image_url: string }>> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const response = await fetch(`${this.baseUrl}/api/v1/od/images/bulk`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async updateODImage(id: string, data: { folder?: string; tags?: string[]; status?: string }): Promise<{ id: string }> {
    return this.request(`/api/v1/od/images/${id}`, { method: "PATCH", body: data });
  }

  async deleteODImage(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/od/images/${id}`, { method: "DELETE" });
  }

  async deleteODImagesBulk(imageIds: string[]): Promise<{ deleted: number; errors: string[] }> {
    return this.request("/api/v1/od/images/bulk", { method: "DELETE", body: imageIds });
  }

  // Bulk operations by filters (for "select all filtered" mode)
  async deleteODImagesByFilters(filters: {
    search?: string;
    statuses?: string;
    sources?: string;
    folders?: string;
    merchant_ids?: string;
    store_ids?: string;
  }): Promise<{ deleted: number; total_matched: number; errors: string[] }> {
    return this.request("/api/v1/od/images/bulk/delete-by-filters", {
      method: "POST",
      body: filters,
    });
  }

  async deleteODImagesByFiltersAsync(
    filters: {
      search?: string;
      statuses?: string;
      sources?: string;
      folders?: string;
      merchant_ids?: string;
      store_ids?: string;
    },
    skipInDatasets: boolean = true
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("/api/v1/od/images/bulk/delete-by-filters/async", {
      method: "POST",
      body: filters,
      params: { skip_in_datasets: skipInDatasets },
    });
  }

  async addFilteredImagesToODDataset(
    datasetId: string,
    filters: {
      search?: string;
      statuses?: string;
      sources?: string;
      folders?: string;
      merchant_ids?: string;
      store_ids?: string;
    }
  ): Promise<{ added: number; skipped: number; total_matched: number; errors: string[] }> {
    return this.request(`/api/v1/od/images/bulk/add-to-dataset-by-filters?dataset_id=${datasetId}`, {
      method: "POST",
      body: filters,
    });
  }

  async addFilteredImagesToODDatasetAsync(
    datasetId: string,
    filters: {
      search?: string;
      statuses?: string;
      sources?: string;
      folders?: string;
      merchant_ids?: string;
      store_ids?: string;
    }
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request(`/api/v1/od/images/bulk/add-to-dataset-by-filters/async?dataset_id=${datasetId}`, {
      method: "POST",
      body: filters,
    });
  }

  // OD Classes
  async getODClasses(params?: {
    dataset_id?: string;
    category?: string;
    is_active?: boolean;
    include_templates?: boolean;
  }): Promise<Array<{
    id: string;
    name: string;
    display_name?: string;
    color: string;
    category?: string;
    annotation_count: number;
    is_system: boolean;
    dataset_id?: string;
  }>> {
    return this.request("/api/v1/od/classes", { params });
  }

  async getODDatasetClasses(datasetId: string): Promise<Array<{
    id: string;
    name: string;
    display_name?: string;
    color: string;
    category?: string;
    annotation_count: number;
    is_system: boolean;
  }>> {
    return this.request(`/api/v1/od/datasets/${datasetId}/classes`);
  }

  async createODClass(data: {
    name: string;
    display_name?: string;
    color?: string;
    category?: string;
    dataset_id?: string;
  }): Promise<{ id: string; name: string; color: string }> {
    return this.request("/api/v1/od/classes", { method: "POST", body: data });
  }

  async createODDatasetClass(datasetId: string, data: {
    name: string;
    display_name?: string;
    color?: string;
    category?: string;
  }): Promise<{ id: string; name: string; color: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/classes`, { method: "POST", body: data });
  }

  async updateODClass(id: string, data: {
    name?: string;
    display_name?: string;
    color?: string;
    category?: string;
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/od/classes/${id}`, { method: "PATCH", body: data });
  }

  async deleteODClass(id: string, force?: boolean): Promise<{ status: string }> {
    return this.request(`/api/v1/od/classes/${id}`, { method: "DELETE", params: force ? { force: "true" } : undefined });
  }

  async mergeODClasses(sourceClassIds: string[], targetClassId: string): Promise<{ merged_count: number; annotations_moved: number }> {
    return this.request("/api/v1/od/classes/merge", { method: "POST", body: { source_class_ids: sourceClassIds, target_class_id: targetClassId } });
  }

  async getODClassDuplicates(datasetId: string, threshold?: number): Promise<{
    groups: Array<{
      classes: Array<{
        id: string;
        name: string;
        display_name?: string;
        annotation_count: number;
        is_system: boolean;
        color?: string;
        similarity?: number;
      }>;
      max_similarity: number;
      suggested_target: string;
      suggested_sources: string[];
      total_annotations: number;
    }>;
    total_groups: number;
  }> {
    return this.request("/api/v1/od/classes/duplicates", { params: { dataset_id: datasetId, ...(threshold ? { threshold: threshold.toString() } : {}) } });
  }

  // OD Datasets
  async getODDatasets(): Promise<Array<{
    id: string;
    name: string;
    description?: string;
    annotation_type: string;
    image_count: number;
    annotated_image_count: number;
    annotation_count: number;
    created_at: string;
  }>> {
    return this.request("/api/v1/od/datasets");
  }

  async getODDataset(id: string): Promise<{
    id: string;
    name: string;
    description?: string;
    image_count: number;
    annotated_image_count: number;
    annotation_count: number;
    created_at: string;
  }> {
    return this.request(`/api/v1/od/datasets/${id}`);
  }

  async getODDatasetImages(datasetId: string, params?: {
    page?: number;
    limit?: number;
    status?: string;
  }): Promise<{
    images: Array<{
      id: string;
      image_id: string;
      status: string;
      annotation_count: number;
      image: { id: string; filename: string; image_url: string; width: number; height: number };
    }>;
    total: number;
    page: number;
    limit: number;
  }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images`, { params });
  }

  async createODDataset(data: { name: string; description?: string; annotation_type?: string }): Promise<{ id: string; name: string }> {
    return this.request("/api/v1/od/datasets", { method: "POST", body: data });
  }

  async updateODDataset(id: string, data: { name?: string; description?: string }): Promise<{ id: string }> {
    return this.request(`/api/v1/od/datasets/${id}`, { method: "PATCH", body: data });
  }

  async deleteODDataset(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/od/datasets/${id}`, { method: "DELETE" });
  }

  async addImagesToODDataset(datasetId: string, imageIds: string[]): Promise<{ added: number; skipped: number }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images`, { method: "POST", body: { image_ids: imageIds } });
  }

  async removeImageFromODDataset(datasetId: string, imageId: string): Promise<{ status: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images/${imageId}`, { method: "DELETE" });
  }

  async removeImagesFromODDatasetBulk(datasetId: string, imageIds: string[]): Promise<{ removed: number }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images/bulk-remove`, { method: "POST", body: imageIds });
  }

  async removeImagesFromODDatasetBulkAsync(
    datasetId: string,
    options: {
      imageIds?: string[];
      statuses?: string;
      hasAnnotations?: boolean;
      deleteCompletely?: boolean;
    }
  ): Promise<{ job_id: string; status: string; message: string }> {
    const params: Record<string, string | boolean> = {};
    if (options.statuses) params.statuses = options.statuses;
    if (options.hasAnnotations !== undefined) params.has_annotations = options.hasAnnotations;
    if (options.deleteCompletely !== undefined) params.delete_completely = options.deleteCompletely;

    return this.request(`/api/v1/od/datasets/${datasetId}/images/bulk-remove/async`, {
      method: "POST",
      body: options.imageIds || null,
      params,
    });
  }

  async updateODDatasetImageStatus(datasetId: string, imageId: string, status: string): Promise<{ id: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images/${imageId}/status`, { method: "PATCH", params: { status } });
  }

  async updateODDatasetImageStatusBulk(datasetId: string, imageIds: string[], status: string): Promise<{ updated: number; status: string; message: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/images/bulk-status`, { method: "POST", body: imageIds, params: { status } });
  }

  async updateODDatasetImageStatusByFilter(
    datasetId: string,
    newStatus: string,
    options?: { currentStatus?: string; hasAnnotations?: boolean }
  ): Promise<{ updated: number; status: string; message: string }> {
    const params: Record<string, string | boolean> = { new_status: newStatus };
    if (options?.currentStatus) params.current_status = options.currentStatus;
    if (options?.hasAnnotations !== undefined) params.has_annotations = options.hasAnnotations;
    return this.request(`/api/v1/od/datasets/${datasetId}/images/bulk-status-by-filter`, { method: "POST", params });
  }

  async updateODDatasetImageStatusBulkAsync(
    datasetId: string,
    newStatus: string,
    options: {
      imageIds?: string[];
      currentStatus?: string;
      hasAnnotations?: boolean;
    }
  ): Promise<{ job_id: string; status: string; message: string }> {
    const params: Record<string, string | boolean> = { new_status: newStatus };
    if (options.currentStatus) params.current_status = options.currentStatus;
    if (options.hasAnnotations !== undefined) params.has_annotations = options.hasAnnotations;

    return this.request(`/api/v1/od/datasets/${datasetId}/images/bulk-status/async`, {
      method: "POST",
      body: options.imageIds || null,
      params,
    });
  }

  async getODDatasetStats(datasetId: string): Promise<{
    dataset: { id: string; name: string };
    images_by_status: Record<string, number>;
    total_annotations: number;
  }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/stats`);
  }

  /**
   * Get detailed dataset stats for training wizard
   * Returns class info, image sizes, annotation stats etc.
   */
  async getODDatasetTrainingStats(datasetId: string, versionId?: string): Promise<{
    name: string;
    image_count: number;
    annotated_image_count: number;
    annotation_count: number;
    class_names: string[];
    class_distribution: Record<string, number>;
    avg_annotations_per_image: number;
    min_image_size: { width: number; height: number };
    max_image_size: { width: number; height: number };
    avg_image_size: { width: number; height: number };
  }> {
    const endpoint = versionId
      ? `/api/v1/od/datasets/${datasetId}/versions/${versionId}/stats`
      : `/api/v1/od/datasets/${datasetId}/stats`;
    return this.request(endpoint);
  }

  // OD Annotations
  async getODAnnotations(datasetId: string, imageId: string): Promise<Array<{
    id: string;
    class_id: string;
    class_name: string;
    class_color: string;
    bbox: { x: number; y: number; width: number; height: number };
    is_ai_generated: boolean;
    confidence?: number;
  }>> {
    return this.request(`/api/v1/od/annotations/datasets/${datasetId}/images/${imageId}`);
  }

  async createODAnnotation(datasetId: string, imageId: string, data: {
    class_id: string;
    bbox: { x: number; y: number; width: number; height: number };
    is_ai_generated?: boolean;
    confidence?: number;
  }): Promise<{
    id: string;
    class_id: string;
    class_name: string;
    class_color: string;
    bbox: { x: number; y: number; width: number; height: number };
    is_ai_generated: boolean;
    confidence?: number;
  }> {
    return this.request(`/api/v1/od/annotations/datasets/${datasetId}/images/${imageId}`, { method: "POST", body: data });
  }

  async createODAnnotationsBulk(datasetId: string, imageId: string, annotations: Array<{
    class_id: string;
    bbox: { x: number; y: number; width: number; height: number };
    is_ai_generated?: boolean;
    confidence?: number;
  }>): Promise<{ created: number; annotation_ids: string[] }> {
    return this.request(`/api/v1/od/annotations/datasets/${datasetId}/images/${imageId}/bulk`, { method: "POST", body: { annotations } });
  }

  async updateODAnnotation(annotationId: string, data: {
    class_id?: string;
    bbox?: { x: number; y: number; width: number; height: number };
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/od/annotations/${annotationId}`, { method: "PATCH", body: data });
  }

  async deleteODAnnotation(annotationId: string): Promise<{ status: string }> {
    return this.request(`/api/v1/od/annotations/${annotationId}`, { method: "DELETE" });
  }

  async deleteODAnnotationsBulk(datasetId: string, imageId: string, annotationIds?: string[]): Promise<{ deleted: number }> {
    return this.request(`/api/v1/od/annotations/datasets/${datasetId}/images/${imageId}/bulk`, { method: "DELETE", body: annotationIds });
  }

  // ===========================================
  // OD Image Import (Advanced)
  // ===========================================

  async importODImagesFromUrls(data: {
    urls: string[];
    folder?: string;
    skip_duplicates?: boolean;
    dataset_id?: string;
  }): Promise<{
    success: boolean;
    images_imported: number;
    images_skipped: number;
    duplicates_found: number;
    errors: string[];
  }> {
    return this.request("/api/v1/od/images/import/url", { method: "POST", body: data });
  }

  async previewODImport(file: File): Promise<{
    format_detected: string;
    total_images: number;
    total_annotations: number;
    classes_found: string[];
    sample_images: Array<{
      filename: string;
      annotation_count: number;
      classes: string[];
    }>;
    errors: string[];
  }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/api/v1/od/images/import/preview`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Preview failed");
    }

    return response.json();
  }

  async importODAnnotatedDataset(
    file: File,
    datasetId: string,
    classMapping: Array<{
      source_name: string;
      target_class_id?: string;
      create_new?: boolean;
      skip?: boolean;
      color?: string;
    }>,
    options?: {
      skip_duplicates?: boolean;
      merge_annotations?: boolean;
    }
  ): Promise<{
    success: boolean;
    images_imported: number;
    annotations_imported: number;
    images_skipped: number;
    duplicates_found: number;
    errors: string[];
  }> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("dataset_id", datasetId);
    formData.append("class_mapping_json", JSON.stringify(classMapping));
    if (options?.skip_duplicates !== undefined) {
      formData.append("skip_duplicates", String(options.skip_duplicates));
    }
    if (options?.merge_annotations !== undefined) {
      formData.append("merge_annotations", String(options.merge_annotations));
    }

    const response = await fetch(`${this.baseUrl}/api/v1/od/images/import/annotated`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Import failed");
    }

    return response.json();
  }

  // ===========================================
  // OD Duplicate Detection
  // ===========================================

  async checkODImageDuplicate(file: File): Promise<{
    is_duplicate: boolean;
    similar_images: Array<{
      id: string;
      filename: string;
      image_url: string;
      similarity: number;
    }>;
  }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/api/v1/od/images/check-duplicate`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Duplicate check failed");
    }

    return response.json();
  }

  async getODImageDuplicateGroups(threshold?: number): Promise<{
    groups: Array<{
      images: Array<{
        id: string;
        filename: string;
        image_url: string;
        similarity?: number;
      }>;
      max_similarity: number;
    }>;
    total_groups: number;
  }> {
    return this.request("/api/v1/od/images/duplicates", {
      params: threshold ? { threshold } : undefined,
    });
  }

  async resolveODImageDuplicates(
    keepImageId: string,
    deleteImageIds: string[],
    mergeToDatasets?: boolean
  ): Promise<{ deleted: number; errors: string[] }> {
    return this.request("/api/v1/od/images/duplicates/resolve", {
      method: "POST",
      params: {
        keep_image_id: keepImageId,
        merge_to_datasets: mergeToDatasets,
      },
      body: deleteImageIds,
    });
  }

  // ===========================================
  // OD Bulk Operations
  // ===========================================

  async bulkTagODImages(
    imageIds: string[],
    tags: string[],
    operation: "add" | "remove" | "replace" = "add"
  ): Promise<{
    success: boolean;
    affected_count: number;
    errors: string[];
  }> {
    return this.request("/api/v1/od/images/bulk/tags", {
      method: "POST",
      body: { image_ids: imageIds, tags, operation },
    });
  }

  async bulkMoveODImages(
    imageIds: string[],
    folder?: string
  ): Promise<{
    success: boolean;
    affected_count: number;
    errors: string[];
  }> {
    return this.request("/api/v1/od/images/bulk/move", {
      method: "POST",
      body: { image_ids: imageIds, folder },
    });
  }

  async bulkAddODImagesToDataset(
    imageIds: string[],
    datasetId: string
  ): Promise<{
    success: boolean;
    affected_count: number;
    errors: string[];
  }> {
    return this.request("/api/v1/od/images/bulk/add-to-dataset", {
      method: "POST",
      params: { dataset_id: datasetId },
      body: imageIds,
    });
  }

  // ===========================================
  // BuyBuddy Sync for OD
  // ===========================================

  async checkBuyBuddySyncStatus(): Promise<{
    configured: boolean;
    accessible: boolean;
    message: string;
  }> {
    return this.request("/api/v1/od/images/buybuddy/status");
  }

  async previewBuyBuddySync(options?: {
    start_date?: string;
    end_date?: string;
    store_id?: number;
    is_annotated?: boolean;
    is_approved?: boolean;
    limit?: number;
  }): Promise<{
    total_available: number;
    sample_count: number;
    sample_images: Array<{
      buybuddy_image_id: string;
      image_url: string;
      image_type?: string;
      inserted_at?: string;
      basket_id?: number;
      basket_identifier?: string;
      merchant_id?: number;
      merchant_name?: string;
      store_id?: number;
      store_name?: string;
      store_code?: string;
      is_annotated?: boolean;
      is_approved?: boolean;
      annotation_id?: number;
      in_datasets?: number[];
    }>;
    filters_applied: {
      start_date?: string;
      end_date?: string;
      store_id?: number;
      is_annotated?: boolean;
      is_approved?: boolean;
    };
  }> {
    return this.request("/api/v1/od/images/buybuddy/preview", {
      params: options,
    });
  }

  async syncFromBuyBuddy(options: {
    start_date?: string;
    end_date?: string;
    store_id?: number;
    is_annotated?: boolean;
    is_approved?: boolean;
    max_images?: number;
    dataset_id?: string;
    tags?: string[];
  }): Promise<{
    job_id: string;
    status: string;
    message: string;
  }> {
    return this.request("/api/v1/od/images/buybuddy/sync", {
      method: "POST",
      body: options,
    });
  }

  async getBuyBuddySyncStatus(jobId: string): Promise<{
    job_id: string;
    status: string;
    progress: number;
    result?: {
      stage?: string;
      message?: string;
      can_resume?: boolean;
      synced?: number;
      skipped?: number;
      total_found?: number;
      errors?: string[];
      images_per_second?: number;
      eta_seconds?: number;
    };
    error?: string;
    created_at: string;
    can_resume: boolean;
  }> {
    return this.request(`/api/v1/od/images/buybuddy/sync/${jobId}`);
  }

  async retryBuyBuddySync(jobId: string): Promise<{
    job_id: string;
    status: string;
    message: string;
    resumed_from_checkpoint: boolean;
  }> {
    return this.request(`/api/v1/od/images/buybuddy/sync/${jobId}/retry`, {
      method: "POST",
    });
  }

  async uploadScanRequestImage(file: File): Promise<{ success: boolean; path: string; url: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/api/v1/scan-requests/upload-image`, {
      method: "POST",
      headers: await getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Upload failed" }));
      // Handle case where detail might be an object
      const errorMessage = typeof error.detail === 'string'
        ? error.detail
        : (error.detail?.message || error.message || JSON.stringify(error.detail) || "Upload failed");
      throw new Error(errorMessage);
    }

    return response.json();
  }

  // ===========================================
  // Roboflow Integration
  // ===========================================

  async validateRoboflowKey(apiKey: string): Promise<{
    valid: boolean;
    error?: string;
    workspaces: Array<{ name: string; url: string; projects: number }>;
    projects?: Array<{ id: string; name: string; type: string; images: number }>;
  }> {
    return this.request("/api/v1/od/images/roboflow/validate-key", {
      method: "POST",
      body: { api_key: apiKey },
    });
  }

  async getRoboflowProjects(apiKey: string, workspace: string): Promise<Array<{
    id: string;
    name: string;
    type: string;
    created: string;
    updated: string;
    images: number;
    classes: string[];
    versions: number;
  }>> {
    return this.request("/api/v1/od/images/roboflow/projects", {
      params: { api_key: apiKey, workspace },
    });
  }

  async getRoboflowVersions(
    apiKey: string,
    workspace: string,
    project: string
  ): Promise<Array<{
    id: string;
    name: string;
    version: number;
    images: Record<string, number>;
    splits: Record<string, number>;
    classes: string[];
    preprocessing: Record<string, unknown>;
    augmentation: Record<string, unknown>;
    created: string;
    exports: string[];
  }>> {
    return this.request("/api/v1/od/images/roboflow/versions", {
      params: { api_key: apiKey, workspace, project },
    });
  }

  async previewRoboflowImport(
    apiKey: string,
    workspace: string,
    project: string,
    version: number
  ): Promise<{
    workspace: string;
    project: string;
    version: number;
    version_name: string;
    total_images: number;
    splits: Record<string, number>;
    classes: Array<{ name: string; count: number }>;
    class_count: number;
    preprocessing: Record<string, unknown>;
    augmentation: Record<string, unknown>;
  }> {
    return this.request("/api/v1/od/images/roboflow/preview", {
      params: { api_key: apiKey, workspace, project, version },
    });
  }

  async importFromRoboflow(params: {
    api_key: string;
    workspace: string;
    project: string;
    version: number;
    dataset_id: string;
    format?: "coco" | "yolov8" | "yolov5pytorch" | "voc";
    class_mapping?: Array<{
      source_name: string;
      target_class_id?: string;
      create_new?: boolean;
      skip?: boolean;
      color?: string;
    }>;
  }): Promise<{
    job_id: string;
    status: string;
    message: string;
  }> {
    return this.request("/api/v1/od/images/roboflow/import", {
      method: "POST",
      body: params,
    });
  }

  async getRoboflowImportStatus(jobId: string): Promise<{
    job_id: string;
    status: "pending" | "running" | "completed" | "failed";
    progress: number;
    result?: {
      stage: string;
      message?: string;
      success?: boolean;
      images_imported?: number;
      annotations_imported?: number;
      images_skipped?: number;
      duplicates_found?: number;
      errors?: string[];
    };
    error?: string;
    created_at: string;
    updated_at?: string;
  }> {
    return this.request(`/api/v1/od/images/roboflow/import/${jobId}`);
  }

  async retryRoboflowImport(jobId: string): Promise<{
    job_id: string;
    status: string;
    message: string;
  }> {
    return this.request(`/api/v1/od/images/roboflow/import/${jobId}/retry`, {
      method: "POST",
    });
  }

  // ===========================================
  // OD AI Annotation (Phase 6)
  // ===========================================

  async getODAIModels(): Promise<{
    detection_models: Array<{
      id: string;
      name: string;
      description: string;
      tasks: string[];
      requires_prompt: boolean;
      model_type?: "open_vocab" | "closed_vocab";
      classes?: string[];  // For closed-vocab Roboflow models
      architecture?: string;  // e.g., "yolov8m", "rf-detr"
    }>;
    segmentation_models: Array<{
      id: string;
      name: string;
      description: string;
      tasks: string[];
      requires_prompt: boolean;
    }>;
  }> {
    return this.request("/api/v1/od/ai/models");
  }

  async predictODAI(params: {
    image_id: string;
    model: string;
    text_prompt?: string;  // Optional - not needed for Roboflow closed-vocab models
    box_threshold?: number;
    text_threshold?: number;
    use_nms?: boolean;
    nms_threshold?: number;
    filter_classes?: string[];  // Filter to only include these classes (useful for Roboflow models)
  }): Promise<{
    predictions: Array<{
      bbox: { x: number; y: number; width: number; height: number };
      label: string;
      confidence: number;
      mask?: string;
    }>;
    model: string;
    processing_time_ms?: number;
    nms_applied?: boolean;
  }> {
    return this.request("/api/v1/od/ai/predict", {
      method: "POST",
      body: params,
    });
  }

  async segmentODAI(params: {
    image_id: string;
    model: string;
    prompt_type: "point" | "box";
    point?: [number, number];
    box?: [number, number, number, number];
    label?: number;
    text_prompt?: string;
  }): Promise<{
    bbox: { x: number; y: number; width: number; height: number };
    confidence: number;
    mask?: string;
    processing_time_ms?: number;
  }> {
    return this.request("/api/v1/od/ai/segment", {
      method: "POST",
      body: params,
    });
  }

  async batchAnnotateODAI(params: {
    dataset_id: string;
    image_ids?: string[];
    model: string;
    text_prompt?: string;  // Required for open-vocab models, optional for Roboflow models
    box_threshold?: number;
    text_threshold?: number;
    auto_accept?: boolean;
    class_mapping?: Record<string, string>;
    filter_classes?: string[];  // Filter to only include these classes (for Roboflow models)
  }): Promise<{
    job_id: string;
    status: string;
    total_images: number;
    message: string;
  }> {
    return this.request("/api/v1/od/ai/batch", {
      method: "POST",
      body: params,
    });
  }

  async getODAIJobStatus(jobId: string): Promise<{
    job_id: string;
    status: string;
    progress: number;
    total_images: number;
    predictions_generated: number;
    error_message?: string;
    started_at?: string;
    completed_at?: string;
  }> {
    return this.request(`/api/v1/od/ai/jobs/${jobId}`);
  }

  // ===========================================
  // OD Dataset Export
  // ===========================================

  async exportODDataset(
    datasetId: string,
    params: {
      format: "yolo" | "coco";
      include_images?: boolean;
      version_id?: string;
      split?: string;
      config?: {
        train_split?: number;
        val_split?: number;
        test_split?: number;
        image_size?: number;
        include_unannotated?: boolean;
      };
    }
  ): Promise<{
    job_id: string;
    status: string;
    download_url?: string;
    progress: number;
    result?: {
      total_images: number;
      total_annotations: number;
      total_classes: number;
    };
    error?: string;
    created_at: string;
    completed_at?: string;
  }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/export`, {
      method: "POST",
      body: params,
    });
  }

  async exportODDatasetAsync(
    datasetId: string,
    params: {
      format: "yolo" | "coco";
      include_images?: boolean;
      version_id?: string;
      split?: string;
      config?: {
        train_split?: number;
        val_split?: number;
        test_split?: number;
      };
    }
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/export/async`, {
      method: "POST",
      body: params,
    });
  }

  // ===========================================
  // OD Dataset Versioning
  // ===========================================

  async getODDatasetVersions(datasetId: string): Promise<
    Array<{
      id: string;
      dataset_id: string;
      version_number: number;
      name?: string;
      description?: string;
      image_count: number;
      annotation_count: number;
      class_count: number;
      train_count: number;
      val_count: number;
      test_count: number;
      class_mapping?: Record<string, number>;
      created_at: string;
    }>
  > {
    return this.request(`/api/v1/od/datasets/${datasetId}/versions`);
  }

  async createODDatasetVersion(
    datasetId: string,
    params: {
      name?: string;
      description?: string;
      train_split?: number;
      val_split?: number;
      test_split?: number;
    }
  ): Promise<{
    id: string;
    dataset_id: string;
    version_number: number;
    name?: string;
    description?: string;
    image_count: number;
    annotation_count: number;
    class_count: number;
    train_count: number;
    val_count: number;
    test_count: number;
    class_mapping?: Record<string, number>;
    created_at: string;
  }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/versions`, {
      method: "POST",
      body: params,
    });
  }

  async getODDatasetVersion(
    datasetId: string,
    versionId: string
  ): Promise<{
    id: string;
    dataset_id: string;
    version_number: number;
    name?: string;
    description?: string;
    image_count: number;
    annotation_count: number;
    class_count: number;
    train_count: number;
    val_count: number;
    test_count: number;
    class_mapping?: Record<string, number>;
    created_at: string;
  }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/versions/${versionId}`);
  }

  async deleteODDatasetVersion(datasetId: string, versionId: string): Promise<{ status: string; id: string }> {
    return this.request(`/api/v1/od/datasets/${datasetId}/versions/${versionId}`, {
      method: "DELETE",
    });
  }

  // ===========================================
  // OD Training (Phase 8)
  // ===========================================

  async getODTrainingRuns(params?: {
    dataset_id?: string;
    status?: string;
    limit?: number;
  }): Promise<
    Array<{
      id: string;
      name: string;
      description?: string;
      dataset_id: string;
      dataset_version_id?: string;
      model_type: string;
      model_size: string;
      status: string;
      current_epoch: number;
      total_epochs: number;
      best_map?: number;
      best_epoch?: number;
      error_message?: string;
      started_at?: string;
      completed_at?: string;
      created_at: string;
      updated_at: string;
    }>
  > {
    return this.request("/api/v1/od/training", { params });
  }

  async getODTrainingRun(trainingId: string): Promise<{
    id: string;
    name: string;
    description?: string;
    dataset_id: string;
    dataset_version_id?: string;
    model_type: string;
    model_size: string;
    config: Record<string, unknown>;
    status: string;
    current_epoch: number;
    total_epochs: number;
    best_map?: number;
    best_epoch?: number;
    metrics_history?: Array<{
      epoch: number;
      loss?: number;
      map?: number;
      map_50?: number;
      timestamp: string;
    }>;
    logs?: string[];
    error_message?: string;
    runpod_job_id?: string;
    started_at?: string;
    completed_at?: string;
    created_at: string;
    updated_at: string;
  }> {
    return this.request(`/api/v1/od/training/${trainingId}`);
  }

  async createODTrainingRun(params: {
    name: string;
    description?: string;
    dataset_id: string;
    dataset_version_id?: string;
    model_type: string;
    model_size: string;
    config?: {
      epochs?: number;
      batch_size?: number;
      learning_rate?: number;
      image_size?: number;
      // SOTA features
      augmentation_preset?: string;
      use_ema?: boolean;
      ema_decay?: number;
      llrd_decay?: number;
      head_lr_factor?: number;
      warmup_epochs?: number;
      mixed_precision?: boolean;
      weight_decay?: number;
      gradient_clip?: number;
      multi_scale?: boolean;
      patience?: number;
      save_freq?: number;
    };
  }): Promise<{
    id: string;
    name: string;
    dataset_id: string;
    model_type: string;
    model_size: string;
    status: string;
    created_at: string;
  }> {
    return this.request("/api/v1/od/training", {
      method: "POST",
      body: params,
    });
  }

  async cancelODTrainingRun(trainingId: string): Promise<{ status: string; id: string }> {
    return this.request(`/api/v1/od/training/${trainingId}/cancel`, {
      method: "POST",
    });
  }

  async deleteODTrainingRun(trainingId: string): Promise<{ status: string; id: string }> {
    return this.request(`/api/v1/od/training/${trainingId}`, {
      method: "DELETE",
    });
  }

  async getODTrainingMetrics(trainingId: string): Promise<
    Array<{
      epoch: number;
      loss?: number;
      map?: number;
      map_50?: number;
      timestamp: string;
    }>
  > {
    return this.request(`/api/v1/od/training/${trainingId}/metrics-history`);
  }

  async getODTrainingLogs(trainingId: string, limit?: number): Promise<string[]> {
    return this.request(`/api/v1/od/training/${trainingId}/logs`, {
      params: limit ? { limit } : undefined,
    });
  }

  // ===========================================
  // OD Trained Models (Phase 9)
  // ===========================================

  async getODTrainedModels(params?: {
    is_active?: boolean;
    model_type?: string;
    limit?: number;
  }): Promise<
    Array<{
      id: string;
      training_run_id?: string;
      name: string;
      description?: string;
      model_type: string;
      checkpoint_url: string;
      map?: number;
      map_50?: number;
      class_count: number;
      is_active: boolean;
      is_default: boolean;
      created_at: string;
    }>
  > {
    return this.request("/api/v1/od/models", { params });
  }

  async getODTrainedModel(modelId: string): Promise<{
    id: string;
    training_run_id?: string;
    name: string;
    description?: string;
    model_type: string;
    checkpoint_url: string;
    map?: number;
    map_50?: number;
    class_count: number;
    class_names?: string[];
    is_active: boolean;
    is_default: boolean;
    created_at: string;
  }> {
    return this.request(`/api/v1/od/models/${modelId}`);
  }

  async updateODTrainedModel(
    modelId: string,
    data: {
      name?: string;
      description?: string;
      is_active?: boolean;
    }
  ): Promise<{ id: string }> {
    return this.request(`/api/v1/od/models/${modelId}`, {
      method: "PATCH",
      body: data,
    });
  }

  async setDefaultODModel(modelId: string): Promise<{ id: string; is_default: boolean }> {
    return this.request(`/api/v1/od/models/${modelId}/set-default`, {
      method: "POST",
    });
  }

  async deleteODTrainedModel(modelId: string): Promise<{ status: string }> {
    return this.request(`/api/v1/od/models/${modelId}`, {
      method: "DELETE",
    });
  }

  async downloadODModel(modelId: string): Promise<Blob> {
    return this.requestBlob(`/api/v1/od/models/${modelId}/download`);
  }

  // ===========================================
  // Product Matcher
  // ===========================================

  async uploadProductMatcherFile(file: File): Promise<{
    file_name: string;
    columns: string[];
    total_rows: number;
    preview: Record<string, unknown>[];
  }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/api/v1/product-matcher/upload`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async matchProducts(
    rows: Record<string, unknown>[],
    matchRules: Array<{ source_column: string; target_field: string; priority: number }>
  ): Promise<{
    matched: Array<{
      source_row: Record<string, unknown>;
      product: {
        id: string;
        barcode: string;
        product_name?: string;
        brand_name?: string;
        category?: string;
        status: string;
      };
      matched_by: string;
    }>;
    unmatched: Array<{ source_row: Record<string, unknown> }>;
    summary: {
      total: number;
      matched_count: number;
      unmatched_count: number;
      match_rate: number;
    };
  }> {
    return this.request("/api/v1/product-matcher/match", {
      method: "POST",
      body: {
        rows,
        mapping_config: { match_rules: matchRules },
      },
    });
  }

  async createBulkScanRequests(params: {
    items: Array<{ barcode: string; product_name?: string; brand_name?: string }>;
    requester_name: string;
    requester_email: string;
    source_file?: string;
    notes?: string;
  }): Promise<{
    created_count: number;
    skipped_count: number;
    skipped_barcodes: string[];
  }> {
    return this.request("/api/v1/product-matcher/create-scan-requests", {
      method: "POST",
      body: params,
    });
  }

  async exportMatchedProducts(
    items: Array<{
      source_row: Record<string, unknown>;
      product: Record<string, unknown>;
      matched_by: string;
    }>,
    columns: string[]
  ): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/product-matcher/export/matched`, {
      method: "POST",
      headers: {
        ...getAuthHeader(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ items, columns }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Export failed");
    }

    return response.blob();
  }

  async exportUnmatchedProducts(
    items: Array<{ source_row: Record<string, unknown> }>,
    columns: string[]
  ): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/product-matcher/export/unmatched`, {
      method: "POST",
      headers: {
        ...getAuthHeader(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ items, columns }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Export failed");
    }

    return response.blob();
  }

  async getProductMatcherSystemFields(): Promise<
    Array<{ value: string; label: string; description: string }>
  > {
    return this.request("/api/v1/product-matcher/system-fields");
  }

  // ===========================================
  // Bulk Update API
  // ===========================================

  async getBulkUpdateSystemFields(): Promise<
    Array<{ id: string; label: string; group: string; editable: boolean }>
  > {
    return this.request("/api/v1/products/bulk-update/system-fields");
  }

  async previewBulkUpdate(params: {
    rows: Record<string, unknown>[];
    identifier_column: string;
    field_mappings: Array<{ source_column: string; target_field: string }>;
  }): Promise<{
    matches: Array<{
      row_index: number;
      product_id: string;
      barcode: string;
      current_values: Record<string, unknown>;
      new_values: Record<string, unknown>;
      product_field_changes: string[];
      identifier_field_changes: string[];
    }>;
    not_found: Array<{
      row_index: number;
      identifier_value: string;
      source_row: Record<string, unknown>;
    }>;
    validation_errors: Array<{
      row_index: number;
      field: string;
      value: unknown;
      error: string;
    }>;
    summary: {
      total_rows: number;
      matched: number;
      not_found: number;
      validation_errors: number;
      will_update: number;
    };
  }> {
    return this.request("/api/v1/products/bulk-update/preview", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async executeBulkUpdate(params: {
    updates: Array<{ product_id: string; fields: Record<string, unknown> }>;
    mode: "strict" | "lenient";
  }): Promise<{
    success: boolean;
    updated_count: number;
    failed: Array<{ product_id: string; error: string }>;
    execution_time_ms: number;
  }> {
    return this.request("/api/v1/products/bulk-update/execute", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async executeBulkUpdateAsync(params: {
    updates: Array<{ product_id: string; fields: Record<string, unknown> }>;
    mode: "strict" | "lenient";
  }): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("/api/v1/products/bulk-update/execute/async", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  // ===========================================
  // Classification Module
  // ===========================================

  async getCLSHealth(): Promise<{ status: string; module: string; version: string }> {
    return this.request("/api/v1/classification/health");
  }

  async getCLSStats(): Promise<{
    total_images: number;
    total_datasets: number;
    total_labels: number;
    total_classes: number;
    total_models: number;
    active_training_runs: number;
    images_by_status: Record<string, number>;
    recent_datasets: Array<{ id: string; name: string; image_count: number; created_at: string }>;
  }> {
    return this.request("/api/v1/classification/stats");
  }

  // CLS Images
  async getCLSImages(params?: {
    page?: number;
    limit?: number;
    status?: string;
    source?: string;
    folder?: string;
    search?: string;
    dataset_id?: string;
  }): Promise<{
    images: Array<{
      id: string;
      filename: string;
      image_url: string;
      thumbnail_url?: string;
      width?: number;
      height?: number;
      status: string;
      source: string;
      folder?: string;
      tags?: string[];
      created_at: string;
    }>;
    total: number;
    page: number;
    limit: number;
  }> {
    return this.request("/api/v1/classification/images", { params });
  }

  async getCLSImageFilterOptions(): Promise<{
    status: Array<{ value: string; label: string; count: number }>;
    source: Array<{ value: string; label: string; count: number }>;
    folder: Array<{ value: string; label: string; count: number }>;
    dataset: Array<{ value: string; label: string; count: number }>;
  }> {
    return this.request("/api/v1/classification/images/filters/options");
  }

  async getCLSImage(id: string): Promise<{
    id: string;
    filename: string;
    image_url: string;
    thumbnail_url?: string;
    width?: number;
    height?: number;
    status: string;
    source: string;
    folder?: string;
    tags?: string[];
    created_at: string;
  }> {
    return this.request(`/api/v1/classification/images/${id}`);
  }

  async uploadCLSImage(file: File, datasetId?: string, label?: string | null): Promise<{
    id: string;
    filename: string;
    image_url: string;
  }> {
    const formData = new FormData();
    formData.append("file", file);
    if (datasetId) formData.append("dataset_id", datasetId);
    if (label) formData.append("label", label);

    const response = await fetch(`${this.baseUrl}/api/v1/classification/images`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async uploadCLSImagesBulk(files: File[]): Promise<Array<{ id: string; filename: string; image_url: string }>> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const response = await fetch(`${this.baseUrl}/api/v1/classification/images/bulk`, {
      method: "POST",
      headers: getAuthHeader(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async deleteCLSImage(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/classification/images/${id}`, { method: "DELETE" });
  }

  async deleteCLSImagesBulk(imageIds: string[]): Promise<{ deleted: number; errors: string[] }> {
    return this.request("/api/v1/classification/images/bulk", { method: "DELETE", body: { image_ids: imageIds } });
  }

  // CLS Classes
  async getCLSClasses(params?: {
    page?: number;
    limit?: number;
    search?: string;
    is_active?: boolean;
  }): Promise<Array<{
    id: string;
    name: string;
    display_name?: string;
    description?: string;
    color: string;
    image_count: number;
    is_active: boolean;
    created_at: string;
  }>> {
    return this.request("/api/v1/classification/classes", { params });
  }

  async getCLSClass(id: string): Promise<{
    id: string;
    name: string;
    display_name?: string;
    description?: string;
    color: string;
    image_count: number;
    is_active: boolean;
    created_at: string;
  }> {
    return this.request(`/api/v1/classification/classes/${id}`);
  }

  async createCLSClass(data: {
    dataset_id: string;
    name: string;
    display_name?: string;
    description?: string;
    color?: string;
  }): Promise<{ id: string; name: string; dataset_id: string }> {
    return this.request("/api/v1/classification/classes", { method: "POST", body: data });
  }

  async updateCLSClass(id: string, data: {
    name?: string;
    display_name?: string;
    description?: string;
    color?: string;
    is_active?: boolean;
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/classification/classes/${id}`, { method: "PATCH", body: data });
  }

  async deleteCLSClass(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/classification/classes/${id}`, { method: "DELETE" });
  }

  async createCLSClassesBulk(classes: Array<{
    name: string;
    display_name?: string;
    description?: string;
    color?: string;
  }>): Promise<{ created: number; classes: Array<{ id: string; name: string }> }> {
    return this.request("/api/v1/classification/classes/bulk", { method: "POST", body: { classes } });
  }

  // CLS Datasets
  async getCLSDatasets(): Promise<Array<{
    id: string;
    name: string;
    description?: string;
    task_type: string;
    image_count: number;
    labeled_image_count: number;
    class_count: number;
    version: number;
    created_at: string;
  }>> {
    return this.request("/api/v1/classification/datasets");
  }

  async getCLSDataset(id: string): Promise<{
    id: string;
    name: string;
    description?: string;
    task_type: string;
    image_count: number;
    labeled_image_count: number;
    class_count: number;
    split_ratios: { train: number; val: number; test: number };
    preprocessing: Record<string, unknown>;
    version: number;
    created_at: string;
  }> {
    return this.request(`/api/v1/classification/datasets/${id}`);
  }

  async createCLSDataset(data: {
    name: string;
    description?: string;
    task_type?: "single_label" | "multi_label";
    split_ratios?: { train: number; val: number; test: number };
  }): Promise<{ id: string; name: string }> {
    return this.request("/api/v1/classification/datasets", { method: "POST", body: data });
  }

  async updateCLSDataset(id: string, data: {
    name?: string;
    description?: string;
    split_ratios?: { train: number; val: number; test: number };
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/classification/datasets/${id}`, { method: "PATCH", body: data });
  }

  async deleteCLSDataset(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/classification/datasets/${id}`, { method: "DELETE" });
  }

  async getCLSDatasetImages(datasetId: string, params?: {
    page?: number;
    limit?: number;
    status?: string;
    split?: string;
    class_id?: string;
    has_label?: boolean;
  }): Promise<{
    images: Array<{
      id: string;
      dataset_id: string;
      image_id: string;
      status: string;
      split?: string;
      added_at: string;
      image: {
        id: string;
        filename: string;
        image_url: string;
        thumbnail_url?: string;
      };
      labels?: Array<{
        id: string;
        class_id: string;
        class_name?: string;
        class_color?: string;
        confidence?: number;
      }>;
    }>;
    total: number;
    page: number;
    limit: number;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/images`, { params });
  }

  async addImagesToCLSDataset(datasetId: string, imageIds: string[]): Promise<{
    added: number;
    skipped: number;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/images`, {
      method: "POST",
      body: { image_ids: imageIds },
    });
  }

  async removeImagesFromCLSDataset(datasetId: string, imageIds: string[]): Promise<{
    removed: number;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/images/remove`, {
      method: "POST",
      body: { image_ids: imageIds },
    });
  }

  async autoSplitCLSDataset(datasetId: string, params: {
    train_ratio?: number;
    val_ratio?: number;
    test_ratio?: number;
    stratified?: boolean;
    seed?: number;
  }): Promise<{
    train_count: number;
    val_count: number;
    test_count: number;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/auto-split`, {
      method: "POST",
      body: params,
    });
  }

  async getCLSDatasetSplitStats(datasetId: string): Promise<{
    train_count: number;
    val_count: number;
    test_count: number;
    unassigned_count: number;
    class_distribution: Record<string, { train: number; val: number; test: number }>;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/split-stats`);
  }

  async getCLSDatasetClasses(datasetId: string): Promise<Array<{
    id: string;
    name: string;
    display_name?: string;
    color: string;
    image_count: number;
    is_active: boolean;
  }>> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/classes`);
  }

  async getCLSDatasetVersions(datasetId: string): Promise<Array<{
    id: string;
    version_number: number;
    name?: string;
    image_count: number;
    labeled_image_count: number;
    class_count: number;
    created_at: string;
  }>> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/versions`);
  }

  async createCLSDatasetVersion(datasetId: string, data?: {
    name?: string;
    description?: string;
  }): Promise<{
    id: string;
    version_number: number;
  }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/versions`, {
      method: "POST",
      body: data || {},
    });
  }

  // CLS Labels
  async setCLSLabel(datasetId: string, imageId: string, data: {
    class_id: string;
    confidence?: number;
    is_ai_generated?: boolean;
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/classification/labels/${datasetId}/${imageId}`, {
      method: "POST",
      body: data,
    });
  }

  async clearCLSLabels(datasetId: string, imageId: string): Promise<{ cleared: number }> {
    return this.request(`/api/v1/classification/labels/${datasetId}/${imageId}`, {
      method: "DELETE",
    });
  }

  async bulkSetCLSLabels(datasetId: string, imageIds: string[], classId: string): Promise<{
    labeled: number;
  }> {
    return this.request(`/api/v1/classification/labels/${datasetId}/bulk`, {
      method: "POST",
      body: { image_ids: imageIds, class_id: classId },
    });
  }

  async bulkClearCLSLabels(datasetId: string, imageIds: string[]): Promise<{
    cleared: number;
  }> {
    return this.request(`/api/v1/classification/labels/${datasetId}/bulk`, {
      method: "DELETE",
      body: { image_ids: imageIds },
    });
  }

  // CLS Async Bulk Operations
  async deleteCLSImagesAsync(
    imageIds: string[]
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("/api/v1/classification/images/bulk/delete/async", {
      method: "POST",
      body: { image_ids: imageIds },
    });
  }

  async addCLSImagesToDatasetAsync(
    datasetId: string,
    imageIds: string[]
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("/api/v1/classification/images/bulk/add-to-dataset/async", {
      method: "POST",
      body: { dataset_id: datasetId, image_ids: imageIds },
    });
  }

  async removeCLSImagesFromDatasetAsync(
    datasetId: string,
    imageIds: string[],
    deleteCompletely: boolean = false
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request(`/api/v1/classification/datasets/${datasetId}/images/remove/async`, {
      method: "POST",
      body: { image_ids: imageIds },
      params: { delete_completely: deleteCompletely },
    });
  }

  async updateCLSTagsAsync(
    imageIds: string[],
    action: "add" | "remove" | "replace",
    tags: string[]
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("/api/v1/classification/images/bulk/tags/async", {
      method: "POST",
      body: { image_ids: imageIds, action, tags },
    });
  }

  async bulkSetCLSLabelsAsync(
    datasetId: string,
    imageIds: string[],
    classId: string
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request(`/api/v1/classification/labels/datasets/${datasetId}/bulk/async`, {
      method: "POST",
      body: { image_ids: imageIds, class_id: classId },
    });
  }

  async bulkClearCLSLabelsAsync(
    datasetId: string,
    imageIds: string[]
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request(`/api/v1/classification/labels/datasets/${datasetId}/bulk-clear/async`, {
      method: "POST",
      body: { image_ids: imageIds },
    });
  }

  // CLS Labeling Workflow
  async getCLSLabelingQueue(datasetId: string, params?: {
    mode?: "all" | "unlabeled" | "review" | "random" | "low_confidence";
    split?: string;
    class_id?: string;
    limit?: number;
  }): Promise<{
    image_ids: string[];
    total: number;
    mode: string;
  }> {
    return this.request(`/api/v1/classification/labeling/${datasetId}/queue`, { params });
  }

  async getCLSLabelingImage(datasetId: string, imageId: string): Promise<{
    image: {
      id: string;
      filename: string;
      image_url: string;
      thumbnail_url?: string;
      width?: number;
      height?: number;
    };
    current_labels: Array<{
      id: string;
      class_id: string;
      class_name?: string;
      class_color?: string;
      confidence?: number;
    }>;
    dataset_image_status: string;
    position: number;
    total: number;
    prev_image_id?: string;
    next_image_id?: string;
  }> {
    return this.request(`/api/v1/classification/labeling/${datasetId}/image/${imageId}`);
  }

  async submitCLSLabeling(datasetId: string, imageId: string, data: {
    class_id?: string;
    class_ids?: string[];
    action: "label" | "skip" | "review";
    confidence?: number;
  }): Promise<{
    success: boolean;
    next_image_id?: string;
  }> {
    return this.request(`/api/v1/classification/labeling/${datasetId}/image/${imageId}`, {
      method: "POST",
      body: data,
    });
  }

  async getCLSLabelingProgress(datasetId: string): Promise<{
    total: number;
    labeled: number;
    pending: number;
    review: number;
    completed: number;
    skipped: number;
    progress_pct: number;
  }> {
    return this.request(`/api/v1/classification/labeling/${datasetId}/progress`);
  }

  // CLS Training
  async getCLSTrainingRuns(params?: {
    page?: number;
    limit?: number;
    status?: string;
    dataset_id?: string;
  }): Promise<Array<{
    id: string;
    name: string;
    status: string;
    dataset_id: string;
    model_type: string;
    model_size: string;
    current_epoch: number;
    total_epochs: number;
    best_accuracy?: number;
    best_f1?: number;
    created_at: string;
    started_at?: string;
    completed_at?: string;
  }>> {
    return this.request("/api/v1/classification/training", { params });
  }

  async getCLSTrainingRun(id: string): Promise<{
    id: string;
    name: string;
    description?: string;
    status: string;
    dataset_id: string;
    dataset_version_id?: string;
    task_type: string;
    num_classes: number;
    model_type: string;
    model_size: string;
    config: Record<string, unknown>;
    current_epoch: number;
    total_epochs: number;
    best_accuracy?: number;
    best_f1?: number;
    best_top5_accuracy?: number;
    best_epoch?: number;
    metrics_history?: Array<Record<string, unknown>>;
    runpod_job_id?: string;
    error_message?: string;
    started_at?: string;
    completed_at?: string;
    created_at: string;
  }> {
    return this.request(`/api/v1/classification/training/${id}`);
  }

  async getCLSTrainingMetricsHistory(trainingRunId: string): Promise<Array<{
    epoch: number;
    train_loss?: number;
    val_loss?: number;
    val_accuracy?: number;
    val_f1?: number;
    learning_rate?: number;
    created_at: string;
  }>> {
    return this.request(`/api/v1/classification/training/${trainingRunId}/metrics-history`);
  }

  async createCLSTrainingRun(data: {
    name: string;
    description?: string;
    dataset_id: string;
    dataset_version_id?: string;
    config: {
      model_type: "vit" | "convnext" | "efficientnet" | "swin" | "dinov2" | "clip";
      model_size: string;
      epochs?: number;
      batch_size?: number;
      learning_rate?: number;
      use_ema?: boolean;
      mixed_precision?: boolean;
      augmentation_preset?: "sota" | "heavy" | "medium" | "light" | "none";
      image_size?: number;
      early_stopping?: boolean;
      [key: string]: unknown;
    };
  }): Promise<{ id: string; name: string; status: string }> {
    return this.request("/api/v1/classification/training", { method: "POST", body: data });
  }

  async cancelCLSTrainingRun(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/classification/training/${id}/cancel`, { method: "POST" });
  }

  async deleteCLSTrainingRun(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/classification/training/${id}`, { method: "DELETE" });
  }

  async getCLSAugmentationPresets(): Promise<Record<string, {
    name: string;
    description: string;
    transforms: Record<string, unknown>;
  }>> {
    return this.request("/api/v1/classification/training/augmentation-presets");
  }

  async getCLSModelConfigs(): Promise<Record<string, {
    name: string;
    description: string;
    sizes: Record<string, { name: string; params: string; default_image_size: number }>;
  }>> {
    return this.request("/api/v1/classification/training/model-configs");
  }

  // CLS Models
  async getCLSModels(params?: {
    page?: number;
    limit?: number;
    model_type?: string;
    is_active?: boolean;
    training_run_id?: string;
  }): Promise<Array<{
    id: string;
    name: string;
    model_type: string;
    model_size?: string;
    task_type: string;
    accuracy?: number;
    f1_score?: number;
    num_classes: number;
    is_active: boolean;
    is_default: boolean;
    created_at: string;
  }>> {
    return this.request("/api/v1/classification/models", { params });
  }

  async getCLSModel(id: string): Promise<{
    id: string;
    training_run_id?: string;
    name: string;
    description?: string;
    model_type: string;
    model_size?: string;
    task_type: string;
    checkpoint_url: string;
    onnx_url?: string;
    num_classes: number;
    class_names: string[];
    class_mapping: Record<string, number>;
    accuracy?: number;
    f1_score?: number;
    top5_accuracy?: number;
    precision_macro?: number;
    recall_macro?: number;
    confusion_matrix?: number[][];
    per_class_metrics?: Record<string, { precision: number; recall: number; f1: number }>;
    is_active: boolean;
    is_default: boolean;
    created_at: string;
  }> {
    return this.request(`/api/v1/classification/models/${id}`);
  }

  async updateCLSModel(id: string, data: {
    name?: string;
    description?: string;
    is_active?: boolean;
  }): Promise<{ id: string }> {
    return this.request(`/api/v1/classification/models/${id}`, { method: "PATCH", body: data });
  }

  async deleteCLSModel(id: string): Promise<{ success: boolean }> {
    return this.request(`/api/v1/classification/models/${id}`, { method: "DELETE" });
  }

  async activateCLSModel(id: string): Promise<{ success: boolean }> {
    return this.request(`/api/v1/classification/models/${id}/activate`, { method: "POST" });
  }

  async deactivateCLSModel(id: string): Promise<{ success: boolean }> {
    return this.request(`/api/v1/classification/models/${id}/deactivate`, { method: "POST" });
  }

  async setDefaultCLSModel(id: string): Promise<{ success: boolean }> {
    return this.request(`/api/v1/classification/models/${id}/set-default`, { method: "POST" });
  }

  async getCLSModelMetrics(id: string): Promise<{
    overall: {
      accuracy?: number;
      f1_score?: number;
      top5_accuracy?: number;
      precision_macro?: number;
      recall_macro?: number;
    };
    confusion_matrix?: number[][];
    per_class_metrics?: Record<string, { precision: number; recall: number; f1: number }>;
    class_names?: string[];
  }> {
    return this.request(`/api/v1/classification/models/${id}/metrics`);
  }

  async getCLSModelDownloadUrls(id: string): Promise<{
    name: string;
    checkpoint_url?: string;
    onnx_url?: string;
    torchscript_url?: string;
  }> {
    return this.request(`/api/v1/classification/models/${id}/download`);
  }

  async getDefaultCLSModel(modelType: string): Promise<{
    id: string;
    name: string;
    model_type: string;
    accuracy?: number;
    is_default: boolean;
  }> {
    return this.request(`/api/v1/classification/models/default/${modelType}`);
  }

  // CLS Import
  async importCLSImagesFromURLs(params: {
    urls: string[];
    folder?: string;
    skip_duplicates?: boolean;
    dataset_id?: string;
    label?: string | null; // Custom label for all imported images
  }): Promise<{
    success: boolean;
    images_imported: number;
    images_skipped: number;
    duplicates_found: number;
    errors: string[];
  }> {
    return this.request("/api/v1/classification/images/import/url", {
      method: "POST",
      body: params,
    });
  }

  async importCLSImagesFromProducts(params: {
    product_ids?: string[];
    label_source?: string; // "product_id" | "category" | "brand" | "product_name" | "custom" | "none"
    custom_label?: string | null; // Custom label when label_source is "custom"
    image_types?: string[];
    max_frames_per_product?: number;
    skip_duplicates?: boolean;
    dataset_id?: string;
    status?: string;
    category?: string;
    brand?: string;
  }): Promise<{
    success: boolean;
    images_imported: number;
    labels_created: number;
    classes_created: number;
    errors: string[];
  }> {
    return this.request("/api/v1/classification/images/import/products", {
      method: "POST",
      body: params,
    });
  }

  async importCLSImagesFromCutouts(params: {
    cutout_ids?: string[];
    label_source?: string; // "matched_product_id" | "custom" | "none"
    custom_label?: string | null; // Custom label when label_source is "custom"
    only_matched?: boolean;
    skip_duplicates?: boolean;
    dataset_id?: string;
  }): Promise<{
    success: boolean;
    images_imported: number;
    labels_created: number;
    classes_created: number;
    errors: string[];
  }> {
    return this.request("/api/v1/classification/images/import/cutouts", {
      method: "POST",
      body: params,
    });
  }

  async importCLSImagesFromOD(params: {
    od_image_ids?: string[];
    skip_duplicates?: boolean;
    dataset_id?: string;
    label?: string | null; // Custom label for all imported images
  }): Promise<{
    success: boolean;
    images_imported: number;
    images_skipped: number;
    errors: string[];
  }> {
    return this.request("/api/v1/classification/images/import/od-images", {
      method: "POST",
      body: params,
    });
  }

  // ===========================================
  // Bulk IDs (for Select All Filtered)
  // ===========================================

  async getProductBulkIds(params: {
    search?: string;
    category?: string;
    brand?: string;
    has_image?: boolean;
    limit?: number;
  }): Promise<{ ids: string[]; total: number }> {
    return this.request("/api/v1/classification/images/bulk-ids/products", {
      params,
    });
  }

  async getCutoutBulkIds(params: {
    search?: string;
    is_matched?: boolean;
    limit?: number;
  }): Promise<{ ids: string[]; total: number }> {
    return this.request("/api/v1/classification/images/bulk-ids/cutouts", {
      params,
    });
  }

  async getODImageBulkIds(params: {
    search?: string;
    folder?: string;
    limit?: number;
  }): Promise<{ ids: string[]; total: number }> {
    return this.request("/api/v1/classification/images/bulk-ids/od-images", {
      params,
    });
  }

  // ===========================================
  // Classification AI Labeling
  // ===========================================

  async getCLSAIModels(): Promise<{
    zero_shot_models: Array<{
      id: string;
      name: string;
      description: string;
      model_type: string;
      requires_classes: boolean;
    }>;
    trained_models: Array<{
      id: string;
      name: string;
      description: string;
      model_type: string;
      requires_classes: boolean;
      num_classes?: number;
    }>;
  }> {
    return this.request("/api/v1/classification/ai/models");
  }

  async predictCLSAI(params: {
    image_id: string;
    dataset_id: string;
    model?: string;
    top_k?: number;
    threshold?: number;
  }): Promise<{
    predictions: Array<{
      class_id: string;
      class_name: string;
      confidence: number;
    }>;
    model: string;
    processing_time_ms: number;
  }> {
    return this.request("/api/v1/classification/ai/predict", {
      method: "POST",
      body: params,
    });
  }

  async batchClassifyCLSAI(params: {
    dataset_id: string;
    image_ids?: string[];
    model?: string;
    top_k?: number;
    threshold?: number;
    auto_accept?: boolean;
    overwrite_existing?: boolean;
  }): Promise<{
    job_id: string;
    status: string;
    total_images: number;
    message: string;
  }> {
    return this.request("/api/v1/classification/ai/batch", {
      method: "POST",
      body: params,
    });
  }

  async getCLSAIJobStatus(jobId: string): Promise<{
    job_id: string;
    status: string;
    progress: number;
    total_images: number;
    predictions_generated: number;
    labels_created: number;
    error_message?: string;
    started_at?: string;
    completed_at?: string;
  }> {
    return this.request(`/api/v1/classification/ai/jobs/${jobId}`);
  }

  // ===========================================
  // Workflow API Methods
  // ===========================================

  // Workflows CRUD
  async getWorkflows(params?: {
    status?: string;
    search?: string;
    page?: number;
    limit?: number;
  }): Promise<{
    workflows: Array<{
      id: string;
      name: string;
      description?: string;
      status: string;
      definition: { nodes: unknown[]; edges: unknown[] };
      run_count: number;
      last_run_at?: string;
      avg_duration_ms?: number;
      created_at: string;
      updated_at: string;
    }>;
    total: number;
  }> {
    return this.request("/api/v1/workflows", { params });
  }

  async getWorkflow(id: string): Promise<{
    id: string;
    name: string;
    description?: string;
    status: string;
    definition: { nodes: unknown[]; edges: unknown[] };
    created_at: string;
    updated_at: string;
  }> {
    return this.request(`/api/v1/workflows/${id}`);
  }

  async createWorkflow(data: {
    name: string;
    description?: string;
    definition?: { nodes: unknown[]; edges: unknown[] };
  }): Promise<{
    id: string;
    name: string;
    description?: string;
    status: string;
    definition: { nodes: unknown[]; edges: unknown[] };
    created_at: string;
    updated_at: string;
  }> {
    return this.request("/api/v1/workflows", {
      method: "POST",
      body: data,
    });
  }

  async updateWorkflow(
    id: string,
    data: {
      name?: string;
      description?: string;
      status?: string;
      definition?: { nodes: unknown[]; edges: unknown[] };
    }
  ): Promise<{
    id: string;
    name: string;
    description?: string;
    status: string;
    definition: { nodes: unknown[]; edges: unknown[] };
    created_at: string;
    updated_at: string;
  }> {
    return this.request(`/api/v1/workflows/${id}`, {
      method: "PATCH",
      body: data,
    });
  }

  async deleteWorkflow(id: string): Promise<{ status: string }> {
    return this.request(`/api/v1/workflows/${id}`, { method: "DELETE" });
  }

  // Workflow Executions
  async getWorkflowExecutions(params?: {
    workflow_id?: string;
    status?: string;
    page?: number;
    limit?: number;
  }): Promise<{
    items: Array<{
      id: string;
      workflow_id: string;
      workflow_name?: string;
      status: string;
      inputs: Record<string, unknown>;
      outputs?: Record<string, unknown>;
      error_message?: string;
      started_at?: string;
      completed_at?: string;
      created_at: string;
      total_duration_ms?: number;
    }>;
    total: number;
  }> {
    return this.request("/api/v1/workflows/executions", { params });
  }

  async getWorkflowExecution(executionId: string): Promise<{
    id: string;
    workflow_id: string;
    workflow_name?: string;
    status: string;
    inputs: Record<string, unknown>;
    outputs?: Record<string, unknown>;
    error_message?: string;
    node_outputs?: Record<string, unknown>;
    node_errors?: Record<string, string>;
    started_at?: string;
    completed_at?: string;
    created_at: string;
    total_duration_ms?: number;
  }> {
    return this.request(`/api/v1/workflows/executions/${executionId}`);
  }

  async executeWorkflow(
    workflowId: string,
    inputs: Record<string, unknown>
  ): Promise<{
    id: string;
    workflow_id: string;
    status: string;
    inputs: Record<string, unknown>;
    created_at: string;
  }> {
    return this.request(`/api/v1/workflows/${workflowId}/execute`, {
      method: "POST",
      body: { inputs },
    });
  }

  async cancelExecution(executionId: string): Promise<{ status: string }> {
    return this.request(`/api/v1/workflows/executions/${executionId}/cancel`, {
      method: "POST",
    });
  }

  // Workflow Models
  async getWorkflowModels(params?: {
    model_type?: string;
    include_inactive?: boolean;
  }): Promise<{
    detection: {
      pretrained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean }>;
      trained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean; checkpoint_url?: string }>;
    };
    classification: {
      pretrained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean }>;
      trained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean; checkpoint_url?: string }>;
    };
    embedding: {
      pretrained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean }>;
      trained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean; checkpoint_url?: string }>;
    };
    segmentation: {
      pretrained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean }>;
      trained: Array<{ id: string; name: string; description?: string; model_type: string; is_active: boolean; checkpoint_url?: string }>;
    };
  }> {
    return this.request("/api/v1/workflows/models", { params });
  }

  // Get flattened list of all models for workflow blocks
  // Uses the new /models/list endpoint for server-side flattening
  async getWorkflowModelsList(params?: {
    model_type?: string;
    source?: "pretrained" | "trained";
    include_inactive?: boolean;
  }): Promise<{
    items: Array<{
      id: string;
      name: string;
      model_type: string;
      category: string;
      source: "pretrained" | "trained";
      is_active: boolean;
      is_default?: boolean;
      checkpoint_url?: string;
      class_mapping?: Record<string, string>;
      created_at?: string;
    }>;
    total: number;
  }> {
    const queryParams = new URLSearchParams();
    if (params?.model_type) queryParams.append("model_type", params.model_type);
    if (params?.source) queryParams.append("source", params.source);
    if (params?.include_inactive) queryParams.append("include_inactive", "true");

    const queryString = queryParams.toString();
    const url = `/api/v1/workflows/models/list${queryString ? `?${queryString}` : ""}`;
    return this.request<{
      items: Array<{
        id: string;
        name: string;
        model_type: string;
        category: string;
        source: "pretrained" | "trained";
        is_active: boolean;
        is_default?: boolean;
        checkpoint_url?: string;
        class_mapping?: Record<string, string>;
        created_at?: string;
      }>;
      total: number;
    }>(url, { method: "GET" });
  }

  // Block Registry
  async getWorkflowBlocks(): Promise<
    Array<{
      type: string;
      display_name: string;
      description: string;
      category: string;
      input_ports: Array<{
        name: string;
        type: string;
        required: boolean;
        description?: string;
      }>;
      output_ports: Array<{
        name: string;
        type: string;
        description?: string;
      }>;
      config_schema: Record<string, unknown>;
    }>
  > {
    const response = await this.request<{
      blocks: Record<string, {
        type: string;
        name: string;
        description: string;
        inputs: Array<{ name: string; type: string; description?: string }>;
        outputs: Array<{ name: string; type: string; description?: string }>;
        config_schema: Record<string, unknown>;
      }>;
      categories: Record<string, { name: string; color: string; blocks: string[] }>;
    }>("/api/v1/workflows/blocks");

    // Build category lookup from categories
    const categoryMap: Record<string, string> = {};
    for (const [category, data] of Object.entries(response.categories)) {
      for (const blockType of data.blocks) {
        categoryMap[blockType] = category;
      }
    }

    // Convert blocks object to array with category
    return Object.values(response.blocks).map((block) => ({
      type: block.type,
      display_name: block.name,
      description: block.description,
      category: categoryMap[block.type] || "logic",
      input_ports: block.inputs.map((i) => ({ ...i, required: true })),
      output_ports: block.outputs,
      config_schema: block.config_schema,
    }));
  }
}

// Custom error class for API errors
export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }

  isConflict(): boolean {
    return this.status === 409;
  }

  isNotFound(): boolean {
    return this.status === 404;
  }

  isUnauthorized(): boolean {
    return this.status === 401;
  }
}

// Export singleton instance
export const apiClient = new ApiClient(API_BASE_URL);

// Export class for testing
export { ApiClient };
