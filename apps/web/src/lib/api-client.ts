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
  // New 3-tab extraction types
  MatchingExtractionRequest,
  MatchingExtractionResponse,
  TrainingExtractionRequest,
  TrainingExtractionResponse,
  EvaluationExtractionRequest,
  EvaluationExtractionResponse,
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

  async reprocessProduct(productId: string): Promise<{ job: Job; cleanup: { frames_deleted: number; files_deleted: number }; message: string }> {
    return this.request(`/api/v1/products/${productId}/reprocess`, {
      method: "POST",
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
  ): Promise<{ added_count: number }> {
    return this.request<{ added_count: number }>(
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
  }): Promise<CutoutsResponse> {
    return this.request<CutoutsResponse>("/api/v1/cutouts", { params });
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

  async deleteTrainingRun(id: string): Promise<void> {
    return this.request<void>(`/api/v1/training/runs/${id}`, {
      method: "DELETE",
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
      body: JSON.stringify(data),
    });
  }

  async updateScanRequest(id: string, data: { status?: string; notes?: string }): Promise<ScanRequest> {
    return this.request<ScanRequest>(`/api/v1/scan-requests/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
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
