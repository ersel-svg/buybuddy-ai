import type {
  Product,
  ProductsResponse,
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
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
        ...fetchOptions.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(
        error.detail || `HTTP ${response.status}`,
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
        ...fetchOptions.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(
        error.detail || `HTTP ${response.status}`,
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
    status?: string;
    category?: string;
  }): Promise<ProductsResponse> {
    return this.request<ProductsResponse>("/api/v1/products", { params });
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

  async getProductFrames(
    id: string
  ): Promise<{ frames: { url: string; index: number }[] }> {
    return this.request<{ frames: { url: string; index: number }[] }>(
      `/api/v1/products/${id}/frames`
    );
  }

  async getProductCategories(): Promise<string[]> {
    return this.request<string[]>("/api/v1/products/categories");
  }

  // Product Downloads
  async downloadProducts(ids: string[]): Promise<Blob> {
    return this.requestBlob("/api/v1/products/download", {
      method: "POST",
      body: { product_ids: ids },
    });
  }

  async downloadAllProducts(filters?: {
    status?: string;
    category?: string;
  }): Promise<Blob> {
    return this.requestBlob("/api/v1/products/download/all", {
      method: "GET",
      params: filters,
    });
  }

  async downloadProduct(id: string): Promise<Blob> {
    return this.requestBlob(`/api/v1/products/${id}/download`);
  }

  async exportProductsCSV(ids?: string[]): Promise<Blob> {
    return this.requestBlob("/api/v1/products/export/csv", {
      method: "POST",
      body: { product_ids: ids },
    });
  }

  async exportProductsJSON(ids?: string[]): Promise<Blob> {
    return this.requestBlob("/api/v1/products/export/json", {
      method: "POST",
      body: { product_ids: ids },
    });
  }

  // ===========================================
  // Videos
  // ===========================================

  async getVideos(): Promise<Video[]> {
    return this.request<Video[]>("/api/v1/videos");
  }

  async syncVideos(): Promise<VideoSyncResponse> {
    return this.request<VideoSyncResponse>("/api/v1/videos/sync", {
      method: "POST",
    });
  }

  async processVideo(videoId: number): Promise<Job> {
    return this.request<Job>("/api/v1/videos/process", {
      method: "POST",
      body: { video_id: videoId },
    });
  }

  async processVideos(videoIds: number[]): Promise<Job[]> {
    return this.request<Job[]>("/api/v1/videos/process/batch", {
      method: "POST",
      body: { video_ids: videoIds },
    });
  }

  // ===========================================
  // Datasets
  // ===========================================

  async getDatasets(): Promise<Dataset[]> {
    return this.request<Dataset[]>("/api/v1/datasets");
  }

  async getDataset(id: string): Promise<DatasetWithProducts> {
    return this.request<DatasetWithProducts>(`/api/v1/datasets/${id}`);
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
    config: Record<string, unknown>
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
