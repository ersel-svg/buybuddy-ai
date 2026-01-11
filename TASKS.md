# Buybuddy AI Platform - Implementation Tasks

> **LLM-Friendly Implementation Guide**
>
> Bu dosya projeyi sıfırdan kurmak için detaylı task listesi içerir.
> Her task bağımsız olarak uygulanabilir ve acceptance criteria ile doğrulanabilir.

---

## Quick Reference

**Tech Stack:**
- Frontend: Next.js 14 + shadcn/ui + TypeScript
- Backend: FastAPI + Python 3.11+
- Database: Supabase (PostgreSQL + Storage)
- GPU Workers: Runpod Serverless
- Monorepo: Turborepo + pnpm

**Key Documents:**
- `TASKS.md` (this file) - Implementation tasks
- `CONTEXT.md` - API credentials & schemas
- `Eski kodlar/` - Reference implementations

---

## Phase 0: Project Initialization

### TASK-001: Create Monorepo Structure
**Prerequisites:** None
**Estimated Complexity:** Low

**Description:**
Turborepo ile monorepo yapısını oluştur.

**Steps:**
```bash
# 1. Root directory'de başla
cd /Users/erselgokmen/Ai-pipeline/buybuddy-ai

# 2. Mevcut dosyaları yedekle
mkdir -p _backup
mv worker _backup/
mv notebooks _backup/

# 3. Monorepo oluştur
pnpm dlx create-turbo@latest . --example basic

# 4. Klasör yapısını düzenle
mkdir -p apps/web apps/api
mkdir -p workers/video-segmentation workers/augmentation workers/training workers/embedding-extraction
mkdir -p libs/core libs/ml-utils libs/db-client
mkdir -p infra/supabase/migrations infra/docker
mkdir -p scripts docs
```

**Final Structure:**
```
buybuddy-ai/
├── apps/
│   ├── web/           # Next.js (will be created in TASK-002)
│   └── api/           # FastAPI (will be created in TASK-010)
├── workers/
│   ├── video-segmentation/
│   ├── augmentation/
│   ├── training/
│   └── embedding-extraction/
├── libs/
│   ├── core/
│   ├── ml-utils/
│   └── db-client/
├── infra/
├── scripts/
├── docs/
├── turbo.json
├── pnpm-workspace.yaml
└── package.json
```

**Acceptance Criteria:**
- [ ] `pnpm install` çalışıyor
- [ ] `pnpm dev` komutu tanımlı (henüz app yok)
- [ ] Klasör yapısı yukarıdaki gibi

---

### TASK-002: Setup Next.js Frontend
**Prerequisites:** TASK-001
**Estimated Complexity:** Medium

**Description:**
Next.js 14 + shadcn/ui frontend uygulamasını oluştur.

**Steps:**
```bash
# 1. apps/web dizinine Next.js kur
cd apps/web
pnpm create next-app@latest . --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

# 2. shadcn/ui kur
pnpm dlx shadcn@latest init

# shadcn config seçenekleri:
# - Style: Default
# - Base color: Slate
# - CSS variables: Yes

# 3. Temel shadcn componentlerini ekle
pnpm dlx shadcn@latest add button card input label table tabs dialog sheet dropdown-menu separator badge avatar tooltip

# 4. Ek dependencies
pnpm add @tanstack/react-query @tanstack/react-table lucide-react recharts date-fns zod react-hook-form @hookform/resolvers
```

**Create Layout Structure:**

`apps/web/src/app/layout.tsx`:
```typescript
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { QueryProvider } from "@/lib/query-provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Buybuddy AI Platform",
  description: "AI-powered product video processing pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <QueryProvider>
          <div className="flex h-screen">
            <Sidebar />
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-auto p-6 bg-gray-50">
                {children}
              </main>
            </div>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
```

**Acceptance Criteria:**
- [ ] `pnpm dev` ile http://localhost:3000 açılıyor
- [ ] Sidebar ve Header görünüyor
- [ ] shadcn components import edilebiliyor
- [ ] Tailwind CSS çalışıyor

---

### TASK-003: Create Sidebar Navigation
**Prerequisites:** TASK-002
**Estimated Complexity:** Low

**Description:**
Sol sidebar navigation component'ini oluştur.

**File:** `apps/web/src/components/layout/sidebar.tsx`

```typescript
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Video,
  Package,
  FolderOpen,
  GitCompare,
  Sparkles,
  GraduationCap,
  Database,
  Settings,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Videos", href: "/videos", icon: Video },
  { name: "Products", href: "/products", icon: Package },
  { name: "Datasets", href: "/datasets", icon: FolderOpen },
  { name: "Matching", href: "/matching", icon: GitCompare },
  { name: "Training", href: "/training", icon: GraduationCap },
  { name: "Embeddings", href: "/embeddings", icon: Database },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 bg-slate-900 text-white flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-slate-700">
        <span className="text-xl font-bold">Buybuddy AI</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navigation.map((item) => {
          const isActive = pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                isActive
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Settings */}
      <div className="p-3 border-t border-slate-700">
        <Link
          href="/settings"
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800"
        >
          <Settings className="h-5 w-5" />
          Settings
        </Link>
      </div>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Sidebar tüm sayfalarda görünüyor
- [ ] Aktif sayfa highlight ediliyor
- [ ] Navigation linkleri çalışıyor (henüz sayfalar yok olabilir)

---

### TASK-004: Create Header Component
**Prerequisites:** TASK-002
**Estimated Complexity:** Low

**Description:**
Üst header component'ini oluştur.

**File:** `apps/web/src/components/layout/header.tsx`

```typescript
"use client";

import { Bell, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export function Header() {
  return (
    <header className="h-16 border-b bg-white flex items-center justify-between px-6">
      {/* Search */}
      <div className="flex items-center gap-4 flex-1 max-w-md">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search products, datasets..."
            className="pl-10"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-4">
        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full" />
        </Button>

        {/* User */}
        <Avatar>
          <AvatarFallback>BB</AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
}
```

**Acceptance Criteria:**
- [ ] Header tüm sayfalarda görünüyor
- [ ] Search input var
- [ ] Notification icon var

---

### TASK-005: Create Dashboard Page
**Prerequisites:** TASK-003, TASK-004
**Estimated Complexity:** Medium

**Description:**
Dashboard ana sayfasını oluştur.

**File:** `apps/web/src/app/page.tsx`

```typescript
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Package, Video, FolderOpen, Database } from "lucide-react";

// Stats data (will come from API later)
const stats = [
  { name: "Products", value: "1,234", icon: Package, change: "+12%" },
  { name: "Videos Processed", value: "856", icon: Video, change: "+8%" },
  { name: "Datasets", value: "12", icon: FolderOpen, change: "+2" },
  { name: "Embeddings", value: "50K", icon: Database, change: "+5K" },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-gray-500">Overview of your AI pipeline</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.name}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-500">
                {stat.name}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-gray-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-green-600">{stat.change} from last week</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recent Activity & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-500">No recent activity</p>
              {/* Will be populated from API */}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
              🔄 Sync Videos from Buybuddy
            </button>
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
              📁 Create New Dataset
            </button>
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
              🎯 Start Training
            </button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Dashboard `/` adresinde görünüyor
- [ ] 4 stat card görünüyor
- [ ] Recent Activity section var
- [ ] Quick Actions section var

---

### TASK-006: Setup React Query Provider
**Prerequisites:** TASK-002
**Estimated Complexity:** Low

**Description:**
React Query provider'ını oluştur.

**File:** `apps/web/src/lib/query-provider.tsx`

```typescript
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { useState } from "react";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
```

**Acceptance Criteria:**
- [ ] React Query devtools görünüyor (development mode)
- [ ] useQuery hook kullanılabilir

---

### TASK-007: Create API Client
**Prerequisites:** TASK-006
**Estimated Complexity:** Low

**Description:**
FastAPI backend ile iletişim için API client oluştur.

**File:** `apps/web/src/lib/api-client.ts`

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface RequestOptions extends RequestInit {
  params?: Record<string, string>;
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
    const { params, ...fetchOptions } = options;

    let url = `${this.baseUrl}${endpoint}`;
    if (params) {
      const searchParams = new URLSearchParams(params);
      url += `?${searchParams.toString()}`;
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        "Content-Type": "application/json",
        ...fetchOptions.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Products
  async getProducts(params?: { page?: number; limit?: number; search?: string }) {
    return this.request<ProductsResponse>("/api/v1/products", { params: params as any });
  }

  async getProduct(id: string) {
    return this.request<Product>(`/api/v1/products/${id}`);
  }

  async updateProduct(id: string, data: Partial<Product>) {
    return this.request<Product>(`/api/v1/products/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  // Videos
  async syncVideos() {
    return this.request<{ count: number }>("/api/v1/videos/sync", { method: "POST" });
  }

  async processVideo(videoId: string) {
    return this.request<Job>("/api/v1/videos/process", {
      method: "POST",
      body: JSON.stringify({ video_id: videoId }),
    });
  }

  // Datasets
  async getDatasets() {
    return this.request<Dataset[]>("/api/v1/datasets");
  }

  async createDataset(data: CreateDatasetRequest) {
    return this.request<Dataset>("/api/v1/datasets", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Jobs
  async getJobs(type?: string) {
    return this.request<Job[]>("/api/v1/jobs", { params: type ? { type } : undefined });
  }

  async getJob(id: string) {
    return this.request<Job>(`/api/v1/jobs/${id}`);
  }

  // Training
  async getTrainingJobs() {
    return this.request<TrainingJob[]>("/api/v1/training/jobs");
  }

  async getTrainingModels() {
    return this.request<ModelArtifact[]>("/api/v1/training/models");
  }

  async startTrainingJob(config: TrainingConfig) {
    return this.request<Job>("/api/v1/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    });
  }

  async activateModel(modelId: string) {
    return this.request<ModelArtifact>(`/api/v1/training/models/${modelId}/activate`, {
      method: "POST",
    });
  }

  // Product Download & Export
  async downloadProducts(ids: string[]): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/products/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_ids: ids }),
    });
    if (!response.ok) throw new Error("Download failed");
    return response.blob();
  }

  async downloadAllProducts(filters?: { status?: string; category?: string }): Promise<Blob> {
    const params = new URLSearchParams();
    if (filters?.status) params.append("status", filters.status);
    if (filters?.category) params.append("category", filters.category);

    const response = await fetch(
      `${this.baseUrl}/api/v1/products/download/all?${params}`,
      { method: "GET" }
    );
    if (!response.ok) throw new Error("Download failed");
    return response.blob();
  }

  async downloadProduct(id: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/products/${id}/download`);
    if (!response.ok) throw new Error("Download failed");
    return response.blob();
  }

  async exportProductsCSV(ids?: string[]): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/products/export/csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_ids: ids }),
    });
    if (!response.ok) throw new Error("Export failed");
    return response.blob();
  }

  async exportProductsJSON(ids?: string[]): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/products/export/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_ids: ids }),
    });
    if (!response.ok) throw new Error("Export failed");
    return response.blob();
  }

  async getProductCategories(): Promise<string[]> {
    return this.request<string[]>("/api/v1/products/categories");
  }

  async deleteProducts(ids: string[]): Promise<void> {
    await this.request<void>("/api/v1/products/bulk-delete", {
      method: "POST",
      body: JSON.stringify({ product_ids: ids }),
    });
  }

  async addProductsToDataset(datasetId: string, productIds: string[]): Promise<void> {
    await this.request<void>(`/api/v1/datasets/${datasetId}/products`, {
      method: "POST",
      body: JSON.stringify({ product_ids: productIds }),
    });
  }
}

export const apiClient = new ApiClient(API_BASE_URL);

// Types (will move to separate file)
export interface Product {
  id: string;
  barcode: string;
  brand_name: string;
  product_name: string;
  status: string;
  frame_count: number;
  created_at: string;
}

export interface ProductsResponse {
  items: Product[];
  total: number;
  page: number;
  limit: number;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  product_count: number;
  created_at: string;
}

export interface CreateDatasetRequest {
  name: string;
  description?: string;
  product_ids?: string[];
  filters?: Record<string, any>;
}

export interface Job {
  id: string;
  type: string;
  status: string;
  progress: number;
  created_at: string;
}

export interface TrainingJob extends Job {
  dataset_id: string;
  epochs_completed?: number;
  final_loss?: number;
  checkpoint_url?: string;
}

export interface ModelArtifact {
  id: string;
  name: string;
  version: string;
  checkpoint_url: string;
  embedding_dim: number;
  num_classes: number;
  final_loss: number;
  is_active: boolean;
  created_at: string;
}

export interface TrainingConfig {
  dataset_id: string;
  model_name: "facebook/dinov2-large" | "facebook/dinov2-base";
  proj_dim: number;
  label_smoothing: number;
  epochs: number;
  batch_size: number;
  lr: number;
  weight_decay: number;
  llrd_decay: number;
  warmup_epochs: number;
  grad_clip: number;
  domain_aware_ratio: number;
  hard_negative_pool_size: number;
  use_hardest_negatives: boolean;
  use_mixed_precision: boolean;
  image_size: number;
  num_workers: number;
  // Dataset split
  train_ratio: number;
  valid_ratio: number;
  test_ratio: number;
  split_seed: number;
  // Optional
  resume_checkpoint?: string;
}
```

**Acceptance Criteria:**
- [ ] apiClient import edilebiliyor
- [ ] TypeScript types tanımlı

---

## Phase 1: Backend Foundation

### TASK-010: Setup FastAPI Backend
**Prerequisites:** TASK-001
**Estimated Complexity:** Medium

**Description:**
FastAPI backend uygulamasını oluştur.

**Steps:**
```bash
cd apps/api

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "buybuddy-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "httpx>=0.27.0",
    "supabase>=2.5.0",
    "python-multipart>=0.0.9",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
EOF

# Create directory structure
mkdir -p src/api/v1 src/services src/schemas src/models

# Create main.py
cat > src/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.router import api_router
from config import settings

app = FastAPI(
    title="Buybuddy AI API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF
```

**File:** `apps/api/src/config.py`
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""

    # Runpod
    runpod_api_key: str = ""
    runpod_endpoint_video: str = ""
    runpod_endpoint_augmentation: str = ""
    runpod_endpoint_training: str = ""
    runpod_endpoint_embedding: str = ""

    # External APIs
    gemini_api_key: str = ""
    hf_token: str = ""

    # Buybuddy Legacy API
    buybuddy_api_url: str = "https://api-legacy.buybuddy.co/api/v1"
    buybuddy_username: str = ""
    buybuddy_password: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

**Acceptance Criteria:**
- [ ] `uvicorn src.main:app --reload` çalışıyor
- [ ] http://localhost:8000/docs Swagger UI açılıyor
- [ ] http://localhost:8000/health 200 dönüyor

---

### TASK-011: Create Products API
**Prerequisites:** TASK-010
**Estimated Complexity:** Medium

**Description:**
Products CRUD API endpoint'lerini oluştur.

**File:** `apps/api/src/api/v1/products.py`

```python
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# Schemas
class ProductBase(BaseModel):
    barcode: str
    brand_name: Optional[str] = None
    sub_brand: Optional[str] = None
    product_name: Optional[str] = None
    variant_flavor: Optional[str] = None
    category: Optional[str] = None
    container_type: Optional[str] = None
    net_quantity: Optional[str] = None

class ProductCreate(ProductBase):
    video_id: Optional[int] = None
    video_url: Optional[str] = None

class ProductUpdate(BaseModel):
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    # ... diğer alanlar

class Product(ProductBase):
    id: str
    video_id: Optional[int]
    status: str
    frame_count: int
    frames_path: Optional[str]
    primary_image_url: Optional[str]
    created_at: datetime
    updated_at: datetime

class ProductsResponse(BaseModel):
    items: list[Product]
    total: int
    page: int
    limit: int

# Mock data (will be replaced with Supabase)
MOCK_PRODUCTS = []

@router.get("", response_model=ProductsResponse)
async def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
):
    """List products with pagination and filters."""
    # TODO: Replace with Supabase query
    items = MOCK_PRODUCTS

    # Apply filters
    if search:
        items = [p for p in items if search.lower() in p.get("barcode", "").lower()
                 or search.lower() in p.get("product_name", "").lower()]
    if status:
        items = [p for p in items if p.get("status") == status]

    total = len(items)
    start = (page - 1) * limit
    end = start + limit

    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "limit": limit,
    }

@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: str):
    """Get product details."""
    # TODO: Replace with Supabase query
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@router.patch("/{product_id}", response_model=Product)
async def update_product(product_id: str, data: ProductUpdate):
    """Update product metadata."""
    # TODO: Replace with Supabase update
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = data.model_dump(exclude_unset=True)
    product.update(update_data)
    return product

@router.delete("/{product_id}")
async def delete_product(product_id: str):
    """Delete a product."""
    # TODO: Replace with Supabase delete
    global MOCK_PRODUCTS
    MOCK_PRODUCTS = [p for p in MOCK_PRODUCTS if p["id"] != product_id]
    return {"status": "deleted"}

@router.get("/{product_id}/frames")
async def get_product_frames(product_id: str):
    """Get product frames."""
    # TODO: Return frame URLs from Supabase Storage
    return {"frames": []}

# ============================================
# Download & Export Endpoints
# ============================================

from fastapi.responses import StreamingResponse
from io import BytesIO
import zipfile
import csv
import json as json_lib

class DownloadRequest(BaseModel):
    product_ids: Optional[list[str]] = None

@router.get("/categories")
async def get_product_categories():
    """Get unique product categories."""
    # TODO: Replace with Supabase distinct query
    categories = list(set(p.get("category") for p in MOCK_PRODUCTS if p.get("category")))
    return sorted(categories)

@router.post("/download")
async def download_products(request: DownloadRequest):
    """Download selected products as ZIP with frames and metadata."""
    products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]

    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    # Create ZIP in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for product in products:
            folder_name = f"{product['barcode']}"

            # Add metadata.json
            metadata = {
                "id": product["id"],
                "barcode": product["barcode"],
                "brand_name": product.get("brand_name"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "status": product.get("status"),
            }
            zf.writestr(f"{folder_name}/metadata.json", json_lib.dumps(metadata, indent=2))

            # TODO: Add frame images from Supabase Storage
            # for frame_url in product.get("frames", []):
            #     frame_data = download_from_storage(frame_url)
            #     zf.writestr(f"{folder_name}/{frame_name}", frame_data)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=products_{len(products)}.zip"}
    )

@router.get("/download/all")
async def download_all_products(
    status: Optional[str] = None,
    category: Optional[str] = None,
):
    """Download ALL products as ZIP (filtered by status/category)."""
    products = MOCK_PRODUCTS.copy()

    if status:
        products = [p for p in products if p.get("status") == status]
    if category:
        products = [p for p in products if p.get("category") == category]

    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    # Create ZIP in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for product in products:
            folder_name = f"{product['barcode']}"
            metadata = {
                "id": product["id"],
                "barcode": product["barcode"],
                "brand_name": product.get("brand_name"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "status": product.get("status"),
            }
            zf.writestr(f"{folder_name}/metadata.json", json_lib.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=all_products_{len(products)}.zip"}
    )

@router.get("/{product_id}/download")
async def download_single_product(product_id: str):
    """Download single product as ZIP with all frames."""
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        metadata = {
            "id": product["id"],
            "barcode": product["barcode"],
            "brand_name": product.get("brand_name"),
            "product_name": product.get("product_name"),
            "category": product.get("category"),
            "status": product.get("status"),
        }
        zf.writestr("metadata.json", json_lib.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={product['barcode']}.zip"}
    )

@router.post("/export/csv")
async def export_products_csv(request: DownloadRequest):
    """Export products as CSV file."""
    if request.product_ids:
        products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]
    else:
        products = MOCK_PRODUCTS

    output = BytesIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["id", "barcode", "brand_name", "product_name", "category", "status", "frame_count", "created_at"],
        extrasaction="ignore"
    )

    # Write header
    output.write(b'\xef\xbb\xbf')  # UTF-8 BOM for Excel
    writer.writeheader()
    for product in products:
        writer.writerow(product)

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=products.csv"}
    )

@router.post("/export/json")
async def export_products_json(request: DownloadRequest):
    """Export products as JSON file."""
    if request.product_ids:
        products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]
    else:
        products = MOCK_PRODUCTS

    output = BytesIO()
    output.write(json_lib.dumps(products, indent=2, ensure_ascii=False).encode("utf-8"))
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=products.json"}
    )

@router.post("/bulk-delete")
async def bulk_delete_products(request: DownloadRequest):
    """Delete multiple products at once."""
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    global MOCK_PRODUCTS
    deleted_count = len([p for p in MOCK_PRODUCTS if p["id"] in request.product_ids])
    MOCK_PRODUCTS = [p for p in MOCK_PRODUCTS if p["id"] not in request.product_ids]

    return {"deleted_count": deleted_count}
```

**Acceptance Criteria:**
- [ ] GET /api/v1/products çalışıyor
- [ ] GET /api/v1/products/{id} çalışıyor
- [ ] PATCH /api/v1/products/{id} çalışıyor
- [ ] DELETE /api/v1/products/{id} çalışıyor
- [ ] GET /api/v1/products/categories çalışıyor
- [ ] POST /api/v1/products/download çalışıyor (selected products ZIP)
- [ ] GET /api/v1/products/download/all çalışıyor (all products ZIP)
- [ ] GET /api/v1/products/{id}/download çalışıyor (single product ZIP)
- [ ] POST /api/v1/products/export/csv çalışıyor
- [ ] POST /api/v1/products/export/json çalışıyor
- [ ] POST /api/v1/products/bulk-delete çalışıyor
- [ ] Swagger'da endpoint'ler görünüyor

---

### TASK-012: Create Supabase Service
**Prerequisites:** TASK-010
**Estimated Complexity:** Medium

**Description:**
Supabase client wrapper'ını oluştur.

**File:** `apps/api/src/services/supabase.py`

```python
from supabase import create_client, Client
from config import settings
from functools import lru_cache
from typing import Optional, Any
import json

@lru_cache
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    return create_client(settings.supabase_url, settings.supabase_key)

class SupabaseService:
    def __init__(self):
        self.client = get_supabase_client()

    # Products
    async def get_products(
        self,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """Get products with pagination."""
        query = self.client.table("products").select("*", count="exact")

        if search:
            query = query.or_(f"barcode.ilike.%{search}%,product_name.ilike.%{search}%")
        if status:
            query = query.eq("status", status)

        # Pagination
        start = (page - 1) * limit
        end = start + limit - 1
        query = query.range(start, end).order("created_at", desc=True)

        response = query.execute()
        return {
            "items": response.data,
            "total": response.count,
            "page": page,
            "limit": limit,
        }

    async def get_product(self, product_id: str) -> Optional[dict]:
        """Get single product."""
        response = self.client.table("products").select("*").eq("id", product_id).single().execute()
        return response.data

    async def create_product(self, data: dict) -> dict:
        """Create new product."""
        response = self.client.table("products").insert(data).execute()
        return response.data[0]

    async def update_product(self, product_id: str, data: dict) -> dict:
        """Update product."""
        response = self.client.table("products").update(data).eq("id", product_id).execute()
        return response.data[0]

    async def delete_product(self, product_id: str) -> None:
        """Delete product."""
        self.client.table("products").delete().eq("id", product_id).execute()

    # Datasets
    async def get_datasets(self) -> list:
        """Get all datasets."""
        response = self.client.table("datasets").select("*").order("created_at", desc=True).execute()
        return response.data

    async def create_dataset(self, data: dict) -> dict:
        """Create new dataset."""
        response = self.client.table("datasets").insert(data).execute()
        return response.data[0]

    async def add_products_to_dataset(self, dataset_id: str, product_ids: list[str]) -> None:
        """Add products to dataset."""
        records = [{"dataset_id": dataset_id, "product_id": pid} for pid in product_ids]
        self.client.table("dataset_products").insert(records).execute()

    # Jobs
    async def create_job(self, data: dict) -> dict:
        """Create new job."""
        response = self.client.table("jobs").insert(data).execute()
        return response.data[0]

    async def update_job(self, job_id: str, data: dict) -> dict:
        """Update job status."""
        response = self.client.table("jobs").update(data).eq("id", job_id).execute()
        return response.data[0]

    # Storage
    async def upload_file(self, bucket: str, path: str, file_data: bytes) -> str:
        """Upload file to storage."""
        self.client.storage.from_(bucket).upload(path, file_data)
        return f"{settings.supabase_url}/storage/v1/object/public/{bucket}/{path}"

    async def get_file_url(self, bucket: str, path: str) -> str:
        """Get public URL for file."""
        return f"{settings.supabase_url}/storage/v1/object/public/{bucket}/{path}"

# Singleton
supabase_service = SupabaseService()
```

**Acceptance Criteria:**
- [ ] Supabase'e bağlanabiliyor
- [ ] CRUD işlemleri çalışıyor
- [ ] Storage işlemleri çalışıyor

---

### TASK-013: Create Supabase Database Schema
**Prerequisites:** Supabase project created
**Estimated Complexity:** Medium

**Description:**
Supabase'de database tablolarını oluştur.

**File:** `infra/supabase/migrations/001_initial.sql`

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- PRODUCTS TABLE
-- ============================================
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identifiers
    video_id INTEGER,
    video_url TEXT,

    -- Metadata (from Gemini)
    brand_name TEXT,
    sub_brand TEXT,
    product_name TEXT,
    variant_flavor TEXT,
    category TEXT,
    container_type TEXT,
    net_quantity TEXT,
    nutrition_facts JSONB,
    claims TEXT[],
    grounding_prompt TEXT,
    visibility_score INTEGER,

    -- Frames
    frame_count INTEGER DEFAULT 0,
    frames_path TEXT,
    primary_image_url TEXT,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'needs_matching', 'ready', 'rejected')),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_brand ON products(brand_name);
CREATE INDEX idx_products_category ON products(category);

-- ============================================
-- PRODUCT BARCODES (multiple per product)
-- ============================================
CREATE TABLE product_barcodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    barcode TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(barcode)
);

CREATE INDEX idx_barcodes_product ON product_barcodes(product_id);
CREATE INDEX idx_barcodes_barcode ON product_barcodes(barcode);

-- ============================================
-- PRODUCT IMAGES
-- ============================================
CREATE TYPE image_type AS ENUM ('synthetic', 'real', 'augmented');
CREATE TYPE image_source AS ENUM ('video_frame', 'matching', 'augmentation');

CREATE TABLE product_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,

    image_path TEXT NOT NULL,
    image_url TEXT,
    image_type image_type NOT NULL,
    source image_source NOT NULL,

    -- For ordering frames
    frame_index INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_images_product ON product_images(product_id);
CREATE INDEX idx_images_type ON product_images(image_type);

-- ============================================
-- JOBS TABLE
-- ============================================
CREATE TYPE job_type AS ENUM (
    'video_processing',
    'augmentation',
    'training',
    'embedding_extraction',
    'matching'
);

CREATE TYPE job_status AS ENUM (
    'pending',
    'queued',
    'processing',
    'completed',
    'failed',
    'cancelled'
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type job_type NOT NULL,
    status job_status DEFAULT 'pending',

    -- Config
    config JSONB NOT NULL DEFAULT '{}',

    -- Progress
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    current_step TEXT,

    -- Results
    result JSONB,
    error_message TEXT,

    -- Runpod
    runpod_job_id TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(type);

-- ============================================
-- DATASETS
-- ============================================
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    product_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE dataset_products (
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dataset_id, product_id)
);

-- ============================================
-- MODEL ARTIFACTS
-- ============================================
CREATE TABLE model_artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id),

    name TEXT NOT NULL,
    version TEXT NOT NULL,
    checkpoint_url TEXT NOT NULL,

    embedding_dim INTEGER NOT NULL DEFAULT 1024,
    num_classes INTEGER,

    -- Metrics
    final_loss FLOAT,
    accuracy FLOAT,

    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- EMBEDDING INDEXES
-- ============================================
CREATE TABLE embedding_indexes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    model_artifact_id UUID REFERENCES model_artifacts(id),

    vector_count INTEGER DEFAULT 0,
    index_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- TRIGGERS
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- STORAGE BUCKETS (run in Supabase dashboard)
-- ============================================
-- Create buckets: frames, models, embeddings
```

**Acceptance Criteria:**
- [ ] Tüm tablolar oluşturuldu
- [ ] Index'ler aktif
- [ ] Trigger'lar çalışıyor
- [ ] Storage bucket'ları oluşturuldu

---

## Phase 2: Core Pages

### TASK-020: Create Products List Page
**Prerequisites:** TASK-005, TASK-011
**Estimated Complexity:** High

**Description:**
Products list sayfasını data table ile oluştur. Bulk selection, download, export özellikleri dahil.

**File:** `apps/web/src/app/products/page.tsx`

```typescript
"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Plus,
  MoreHorizontal,
  Download,
  FileJson,
  FileSpreadsheet,
  Trash2,
  FolderPlus,
  Loader2,
  Package,
} from "lucide-react";
import Link from "next/link";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/components/ui/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export default function ProductsPage() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isDownloading, setIsDownloading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const { toast } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["products", { page, search, statusFilter, categoryFilter }],
    queryFn: () =>
      apiClient.getProducts({
        page,
        limit: 20,
        search,
        status: statusFilter !== "all" ? statusFilter : undefined,
        category: categoryFilter !== "all" ? categoryFilter : undefined,
      }),
  });

  const { data: categories } = useQuery({
    queryKey: ["product-categories"],
    queryFn: () => apiClient.getProductCategories(),
  });

  // Bulk selection
  const allSelected = useMemo(() => {
    if (!data?.items.length) return false;
    return data.items.every((p) => selectedIds.has(p.id));
  }, [data?.items, selectedIds]);

  const someSelected = useMemo(() => {
    if (!data?.items.length) return false;
    return data.items.some((p) => selectedIds.has(p.id)) && !allSelected;
  }, [data?.items, selectedIds, allSelected]);

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(data?.items.map((p) => p.id) || []));
    }
  };

  const toggleOne = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  // Download selected products
  const handleDownloadSelected = async () => {
    if (selectedIds.size === 0) return;

    setIsDownloading(true);
    try {
      const blob = await apiClient.downloadProducts(Array.from(selectedIds));
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast({
        title: "Download Started",
        description: `${selectedIds.size} ürün indiriliyor...`,
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "İndirme başarısız oldu",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  };

  // Download ALL products
  const handleDownloadAll = async () => {
    setIsDownloading(true);
    try {
      const blob = await apiClient.downloadAllProducts({
        status: statusFilter !== "all" ? statusFilter : undefined,
        category: categoryFilter !== "all" ? categoryFilter : undefined,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `all_products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast({
        title: "Download Started",
        description: `Tüm ürünler indiriliyor...`,
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "İndirme başarısız oldu",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  };

  // Export to CSV
  const handleExportCSV = async () => {
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
    const blob = await apiClient.exportProductsCSV(ids);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `products_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export to JSON
  const handleExportJSON = async () => {
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
    const blob = await apiClient.exportProductsJSON(ids);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `products_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Bulk delete
  const deleteMutation = useMutation({
    mutationFn: (ids: string[]) => apiClient.deleteProducts(ids),
    onSuccess: () => {
      toast({ title: "Deleted", description: `${selectedIds.size} ürün silindi` });
      setSelectedIds(new Set());
      refetch();
    },
  });

  // Add to dataset
  const addToDatasetMutation = useMutation({
    mutationFn: ({ datasetId, productIds }: { datasetId: string; productIds: string[] }) =>
      apiClient.addProductsToDataset(datasetId, productIds),
    onSuccess: () => {
      toast({ title: "Added", description: `${selectedIds.size} ürün dataset'e eklendi` });
      setSelectedIds(new Set());
    },
  });

  const statusColors: Record<string, string> = {
    pending: "bg-yellow-100 text-yellow-800",
    processing: "bg-blue-100 text-blue-800",
    needs_matching: "bg-purple-100 text-purple-800",
    ready: "bg-green-100 text-green-800",
    rejected: "bg-red-100 text-red-800",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Products</h1>
          <p className="text-gray-500">{data?.total || 0} products in directory</p>
        </div>

        {/* Download & Export Buttons */}
        <div className="flex gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" disabled={isDownloading}>
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Download
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={handleDownloadSelected}
                disabled={selectedIds.size === 0}
              >
                <Package className="h-4 w-4 mr-2" />
                Download Selected ({selectedIds.size})
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleDownloadAll}>
                <Download className="h-4 w-4 mr-2" />
                Download All Products (ZIP)
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleExportCSV}>
                <FileSpreadsheet className="h-4 w-4 mr-2" />
                Export to CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportJSON}>
                <FileJson className="h-4 w-4 mr-2" />
                Export to JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Add to Dataset
          </Button>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="flex gap-4 flex-wrap">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by barcode or name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>

        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="processing">Processing</SelectItem>
            <SelectItem value="needs_matching">Needs Matching</SelectItem>
            <SelectItem value="ready">Ready</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
          </SelectContent>
        </Select>

        <Select value={categoryFilter} onValueChange={setCategoryFilter}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {categories?.map((cat: string) => (
              <SelectItem key={cat} value={cat}>
                {cat}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Bulk Actions Bar */}
      {selectedIds.size > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center justify-between">
          <span className="text-sm text-blue-700">
            {selectedIds.size} ürün seçildi
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadSelected}
              disabled={isDownloading}
            >
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
            <Button variant="outline" size="sm">
              <FolderPlus className="h-4 w-4 mr-1" />
              Add to Dataset
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-red-600 hover:text-red-700"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">
                <Checkbox
                  checked={allSelected}
                  indeterminate={someSelected}
                  onCheckedChange={toggleAll}
                />
              </TableHead>
              <TableHead>Image</TableHead>
              <TableHead>Barcode</TableHead>
              <TableHead>Brand</TableHead>
              <TableHead>Product</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Frames</TableHead>
              <TableHead className="w-12"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center py-8">
                  Loading...
                </TableCell>
              </TableRow>
            ) : data?.items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center py-8">
                  No products found
                </TableCell>
              </TableRow>
            ) : (
              data?.items.map((product) => (
                <TableRow
                  key={product.id}
                  className={selectedIds.has(product.id) ? "bg-blue-50" : ""}
                >
                  <TableCell>
                    <Checkbox
                      checked={selectedIds.has(product.id)}
                      onCheckedChange={() => toggleOne(product.id)}
                    />
                  </TableCell>
                  <TableCell>
                    {product.primary_image_url ? (
                      <img
                        src={product.primary_image_url}
                        alt=""
                        className="w-10 h-10 object-cover rounded"
                      />
                    ) : (
                      <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                        <Package className="h-5 w-5 text-gray-400" />
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {product.barcode}
                  </TableCell>
                  <TableCell>{product.brand_name || "-"}</TableCell>
                  <TableCell className="max-w-[200px] truncate">
                    {product.product_name || "-"}
                  </TableCell>
                  <TableCell>{product.category || "-"}</TableCell>
                  <TableCell>
                    <Badge className={statusColors[product.status] || ""}>
                      {product.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{product.frame_count}</TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem asChild>
                          <Link href={`/products/${product.id}`}>View Details</Link>
                        </DropdownMenuItem>
                        <DropdownMenuItem asChild>
                          <Link href={`/products/${product.id}?edit=true`}>Edit</Link>
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => apiClient.downloadProduct(product.id)}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download Frames
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem className="text-red-600">
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex justify-between items-center">
        <p className="text-sm text-gray-500">
          Showing {(page - 1) * 20 + 1} to {Math.min(page * 20, data?.total || 0)} of{" "}
          {data?.total || 0}
        </p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={page === 1}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            disabled={!data || page * 20 >= data.total}
            onClick={() => setPage(page + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Ürünleri Sil</AlertDialogTitle>
            <AlertDialogDescription>
              {selectedIds.size} ürünü silmek istediğinizden emin misiniz? Bu işlem geri
              alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>İptal</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => deleteMutation.mutate(Array.from(selectedIds))}
            >
              Sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Products listesi görünüyor
- [ ] Search çalışıyor
- [ ] Status filter çalışıyor
- [ ] Category filter çalışıyor
- [ ] Pagination çalışıyor
- [ ] Status badge'leri doğru renkte
- [ ] Bulk selection (checkbox) çalışıyor
- [ ] Download Selected butonu ZIP indiriyor
- [ ] Download All butonu tüm ürünleri ZIP olarak indiriyor
- [ ] Export CSV çalışıyor
- [ ] Export JSON çalışıyor
- [ ] Bulk delete çalışıyor
- [ ] Add to Dataset çalışıyor
- [ ] Row actions dropdown çalışıyor

---

### TASK-021: Create Product Detail Page
**Prerequisites:** TASK-020
**Estimated Complexity:** High

**Description:**
Product detail sayfasını tabs ile oluştur. Video player, frame gallery, metadata edit.

**File:** `apps/web/src/app/products/[id]/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Save, Play, Image as ImageIcon } from "lucide-react";
import Link from "next/link";

export default function ProductDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState<Record<string, any>>({});

  const { data: product, isLoading } = useQuery({
    queryKey: ["product", id],
    queryFn: () => apiClient.getProduct(id),
  });

  const { data: frames } = useQuery({
    queryKey: ["product-frames", id],
    queryFn: () => apiClient.getProductFrames(id),
    enabled: !!product,
  });

  const updateMutation = useMutation({
    mutationFn: (data: Record<string, any>) => apiClient.updateProduct(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["product", id] });
      setIsEditing(false);
    },
  });

  if (isLoading) return <div>Loading...</div>;
  if (!product) return <div>Product not found</div>;

  const handleSave = () => {
    updateMutation.mutate(editData);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/products">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">
              {product.brand_name} {product.product_name}
            </h1>
            <p className="text-gray-500 font-mono">{product.barcode}</p>
          </div>
        </div>
        <div className="flex gap-2">
          {isEditing ? (
            <>
              <Button variant="outline" onClick={() => setIsEditing(false)}>Cancel</Button>
              <Button onClick={handleSave} disabled={updateMutation.isPending}>
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
            </>
          ) : (
            <Button onClick={() => {
              setEditData(product);
              setIsEditing(true);
            }}>
              Edit
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="grid grid-cols-3 gap-6">
        {/* Left: Video & Primary Image */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-4 w-4" />
                Source Video
              </CardTitle>
            </CardHeader>
            <CardContent>
              {product.video_url ? (
                <video
                  src={product.video_url}
                  controls
                  className="w-full rounded-lg"
                />
              ) : (
                <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  No video
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Status</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge>{product.status}</Badge>
              <p className="text-sm text-gray-500 mt-2">
                {product.frame_count} frames extracted
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Right: Tabs */}
        <div className="col-span-2">
          <Tabs defaultValue="metadata">
            <TabsList>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
              <TabsTrigger value="synthetic">Synthetic ({product.frame_count})</TabsTrigger>
              <TabsTrigger value="real">Real Images</TabsTrigger>
              <TabsTrigger value="augmented">Augmented</TabsTrigger>
            </TabsList>

            <TabsContent value="metadata" className="mt-4">
              <Card>
                <CardContent className="pt-6 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Brand Name</Label>
                      {isEditing ? (
                        <Input
                          value={editData.brand_name || ""}
                          onChange={(e) => setEditData({ ...editData, brand_name: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.brand_name || "-"}</p>
                      )}
                    </div>
                    <div>
                      <Label>Sub Brand</Label>
                      {isEditing ? (
                        <Input
                          value={editData.sub_brand || ""}
                          onChange={(e) => setEditData({ ...editData, sub_brand: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.sub_brand || "-"}</p>
                      )}
                    </div>
                    <div>
                      <Label>Product Name</Label>
                      {isEditing ? (
                        <Input
                          value={editData.product_name || ""}
                          onChange={(e) => setEditData({ ...editData, product_name: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.product_name || "-"}</p>
                      )}
                    </div>
                    <div>
                      <Label>Category</Label>
                      {isEditing ? (
                        <Input
                          value={editData.category || ""}
                          onChange={(e) => setEditData({ ...editData, category: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.category || "-"}</p>
                      )}
                    </div>
                    <div>
                      <Label>Container Type</Label>
                      {isEditing ? (
                        <Input
                          value={editData.container_type || ""}
                          onChange={(e) => setEditData({ ...editData, container_type: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.container_type || "-"}</p>
                      )}
                    </div>
                    <div>
                      <Label>Net Quantity</Label>
                      {isEditing ? (
                        <Input
                          value={editData.net_quantity || ""}
                          onChange={(e) => setEditData({ ...editData, net_quantity: e.target.value })}
                        />
                      ) : (
                        <p className="mt-1">{product.net_quantity || "-"}</p>
                      )}
                    </div>
                  </div>

                  {/* Grounding Prompt */}
                  <div>
                    <Label>Grounding Prompt (for SAM3)</Label>
                    {isEditing ? (
                      <Input
                        value={editData.grounding_prompt || ""}
                        onChange={(e) => setEditData({ ...editData, grounding_prompt: e.target.value })}
                      />
                    ) : (
                      <p className="mt-1 font-mono text-sm bg-gray-50 p-2 rounded">
                        {product.grounding_prompt || "-"}
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="synthetic" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="grid grid-cols-6 gap-2">
                    {frames?.synthetic?.map((frame: any, i: number) => (
                      <div key={i} className="aspect-square bg-gray-100 rounded overflow-hidden">
                        <img src={frame.url} alt={`Frame ${i}`} className="w-full h-full object-cover" />
                      </div>
                    )) || (
                      <p className="col-span-6 text-center text-gray-500 py-8">
                        No synthetic frames
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="real" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="grid grid-cols-4 gap-4">
                    {frames?.real?.length > 0 ? (
                      frames.real.map((img: any, i: number) => (
                        <div key={i} className="aspect-square bg-gray-100 rounded overflow-hidden">
                          <img src={img.url} alt={`Real ${i}`} className="w-full h-full object-cover" />
                        </div>
                      ))
                    ) : (
                      <div className="col-span-4 text-center py-8">
                        <ImageIcon className="h-12 w-12 mx-auto text-gray-300" />
                        <p className="text-gray-500 mt-2">No real images matched yet</p>
                        <Link href="/matching">
                          <Button variant="outline" className="mt-4">
                            Go to Matching
                          </Button>
                        </Link>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="augmented" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="grid grid-cols-6 gap-2">
                    {frames?.augmented?.map((frame: any, i: number) => (
                      <div key={i} className="aspect-square bg-gray-100 rounded overflow-hidden">
                        <img src={frame.url} alt={`Augmented ${i}`} className="w-full h-full object-cover" />
                      </div>
                    )) || (
                      <p className="col-span-6 text-center text-gray-500 py-8">
                        No augmented images yet. Run augmentation on a dataset first.
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
```

**Add API client method:** `apps/web/src/lib/api-client.ts`
```typescript
// Add to ApiClient class
async getProductFrames(productId: string) {
  return this.request<{
    synthetic: { url: string; frame_index: number }[];
    real: { url: string; source: string }[];
    augmented: { url: string }[];
  }>(`/api/v1/products/${productId}/frames`);
}
```

**Acceptance Criteria:**
- [ ] Product detail görünüyor
- [ ] Tabs arası geçiş çalışıyor
- [ ] Edit mode çalışıyor (form submission)
- [ ] Video player çalışıyor
- [ ] Frame gallery görünüyor (synthetic, real, augmented)

---

### TASK-022: Create Videos Page
**Prerequisites:** TASK-005
**Estimated Complexity:** High

**Description:**
Buybuddy API'den video sync ve processing queue sayfası.

**File:** `apps/web/src/app/videos/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Play, CheckCircle, XCircle, Clock, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface VideoItem {
  id: string;
  video_id: number;
  barcode: string;
  video_url: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress?: number;
  product_name?: string;
}

export default function VideosPage() {
  const queryClient = useQueryClient();
  const [selectedVideos, setSelectedVideos] = useState<string[]>([]);

  // Fetch unprocessed videos from Buybuddy API
  const { data: buybuddyVideos, isLoading: isSyncing, refetch: syncVideos } = useQuery({
    queryKey: ["buybuddy-videos"],
    queryFn: () => apiClient.getBuybuddyVideos(),
    enabled: false, // Manual trigger
  });

  // Fetch processing queue
  const { data: queue } = useQuery({
    queryKey: ["video-queue"],
    queryFn: () => apiClient.getVideoQueue(),
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Sync mutation
  const syncMutation = useMutation({
    mutationFn: () => apiClient.syncVideos(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["video-queue"] });
    },
  });

  // Process mutation
  const processMutation = useMutation({
    mutationFn: (videoIds: string[]) => apiClient.processVideos(videoIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["video-queue"] });
      setSelectedVideos([]);
    },
  });

  const statusIcons = {
    pending: <Clock className="h-4 w-4 text-yellow-500" />,
    processing: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
    completed: <CheckCircle className="h-4 w-4 text-green-500" />,
    failed: <XCircle className="h-4 w-4 text-red-500" />,
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedVideos(queue?.pending?.map((v: VideoItem) => v.id) || []);
    } else {
      setSelectedVideos([]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Videos</h1>
          <p className="text-gray-500">Sync and process videos from Buybuddy</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => syncMutation.mutate()}
            disabled={syncMutation.isPending}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${syncMutation.isPending ? "animate-spin" : ""}`} />
            Sync from Buybuddy
          </Button>
          <Button
            onClick={() => processMutation.mutate(selectedVideos)}
            disabled={selectedVideos.length === 0 || processMutation.isPending}
          >
            <Play className="h-4 w-4 mr-2" />
            Process Selected ({selectedVideos.length})
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Pending</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{queue?.stats?.pending || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{queue?.stats?.processing || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Completed Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{queue?.stats?.completed_today || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-600">{queue?.stats?.failed || 0}</p>
          </CardContent>
        </Card>
      </div>

      {/* Processing Queue */}
      {queue?.processing?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Currently Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {queue.processing.map((video: VideoItem) => (
                <div key={video.id} className="flex items-center gap-4">
                  <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                  <div className="flex-1">
                    <p className="font-medium">{video.barcode}</p>
                    <Progress value={video.progress || 0} className="h-2 mt-1" />
                  </div>
                  <span className="text-sm text-gray-500">{video.progress || 0}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pending Queue */}
      <Card>
        <CardHeader>
          <CardTitle>Pending Videos</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">
                  <input
                    type="checkbox"
                    onChange={(e) => handleSelectAll(e.target.checked)}
                    checked={selectedVideos.length === queue?.pending?.length && queue?.pending?.length > 0}
                  />
                </TableHead>
                <TableHead>Video ID</TableHead>
                <TableHead>Barcode</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Added</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {queue?.pending?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                    No pending videos. Click "Sync from Buybuddy" to fetch new videos.
                  </TableCell>
                </TableRow>
              ) : (
                queue?.pending?.map((video: VideoItem) => (
                  <TableRow key={video.id}>
                    <TableCell>
                      <input
                        type="checkbox"
                        checked={selectedVideos.includes(video.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedVideos([...selectedVideos, video.id]);
                          } else {
                            setSelectedVideos(selectedVideos.filter((id) => id !== video.id));
                          }
                        }}
                      />
                    </TableCell>
                    <TableCell>{video.video_id}</TableCell>
                    <TableCell className="font-mono">{video.barcode}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="gap-1">
                        {statusIcons[video.status]}
                        {video.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-gray-500 text-sm">
                      {new Date(video.created_at).toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
```

**Backend API:** `apps/api/src/api/v1/videos.py`
```python
from fastapi import APIRouter
from services.buybuddy_api import BuybuddyApiService
from services.runpod import RunpodService

router = APIRouter()
buybuddy_service = BuybuddyApiService()
runpod_service = RunpodService()

@router.post("/sync")
async def sync_videos():
    """Fetch unprocessed videos from Buybuddy API and store in DB."""
    videos = await buybuddy_service.get_unprocessed_videos()
    # Store in Supabase
    # Return count
    return {"synced": len(videos)}

@router.get("/queue")
async def get_video_queue():
    """Get video processing queue status."""
    # Return pending, processing, completed stats
    return {
        "stats": {"pending": 0, "processing": 0, "completed_today": 0, "failed": 0},
        "pending": [],
        "processing": [],
    }

@router.post("/process")
async def process_videos(video_ids: list[str]):
    """Queue videos for processing via Runpod."""
    jobs = []
    for vid in video_ids:
        job = await runpod_service.submit_video_job(vid)
        jobs.append(job)
    return {"jobs": jobs}
```

**Acceptance Criteria:**
- [ ] Buybuddy sync çalışıyor
- [ ] Processing queue görünüyor
- [ ] Progress bar real-time güncelleniyor
- [ ] Batch selection çalışıyor
- [ ] Process butonu işleri queue'ya ekliyor

---

### TASK-023: Create Datasets Page
**Prerequisites:** TASK-005
**Estimated Complexity:** Medium

**Description:**
Dataset listesi ve oluşturma sayfası.

**File:** `apps/web/src/app/datasets/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, FolderOpen, MoreHorizontal, Trash2 } from "lucide-react";
import Link from "next/link";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function DatasetsPage() {
  const queryClient = useQueryClient();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [newDataset, setNewDataset] = useState({ name: "", description: "" });

  const { data: datasets, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description: string }) =>
      apiClient.createDataset(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setIsCreateOpen(false);
      setNewDataset({ name: "", description: "" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Datasets</h1>
          <p className="text-gray-500">Manage product datasets for training and augmentation</p>
        </div>
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Dataset
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Dataset</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 pt-4">
              <div>
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={newDataset.name}
                  onChange={(e) => setNewDataset({ ...newDataset, name: e.target.value })}
                  placeholder="e.g., Beverages v1"
                />
              </div>
              <div>
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  value={newDataset.description}
                  onChange={(e) => setNewDataset({ ...newDataset, description: e.target.value })}
                  placeholder="What products are in this dataset?"
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={() => createMutation.mutate(newDataset)}
                  disabled={!newDataset.name || createMutation.isPending}
                >
                  Create
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Dataset Grid */}
      {isLoading ? (
        <div className="text-center py-12">Loading...</div>
      ) : datasets?.length === 0 ? (
        <Card className="py-12">
          <CardContent className="text-center">
            <FolderOpen className="h-12 w-12 mx-auto text-gray-300" />
            <h3 className="mt-4 text-lg font-medium">No datasets yet</h3>
            <p className="text-gray-500 mt-1">Create your first dataset to get started</p>
            <Button className="mt-4" onClick={() => setIsCreateOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets?.map((dataset) => (
            <Card key={dataset.id} className="hover:border-slate-400 transition-colors">
              <CardHeader className="flex flex-row items-start justify-between">
                <div>
                  <CardTitle>
                    <Link href={`/datasets/${dataset.id}`} className="hover:underline">
                      {dataset.name}
                    </Link>
                  </CardTitle>
                  <CardDescription className="mt-1">
                    {dataset.description || "No description"}
                  </CardDescription>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem asChild>
                      <Link href={`/datasets/${dataset.id}`}>View Details</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="text-red-600"
                      onClick={() => {
                        if (confirm("Delete this dataset?")) {
                          deleteMutation.mutate(dataset.id);
                        }
                      }}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Products</span>
                  <span className="font-medium">{dataset.product_count}</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-gray-500">Created</span>
                  <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Dataset listesi görünüyor
- [ ] Create dialog çalışıyor
- [ ] Delete confirmation var
- [ ] Dataset card'ları tıklanabilir (detail page'e gidiyor)

---

### TASK-024: Create Dataset Detail Page
**Prerequisites:** TASK-023
**Estimated Complexity:** High

**Description:**
Dataset detail sayfası - ürün ekleme, actions (augment, train, extract).

**File:** `apps/web/src/app/datasets/[id]/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ArrowLeft, Plus, Trash2, Sparkles, GraduationCap, Database, Search } from "lucide-react";
import Link from "next/link";

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProducts, setSelectedProducts] = useState<string[]>([]);

  const { data: dataset, isLoading } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.getDataset(id),
  });

  const { data: searchResults } = useQuery({
    queryKey: ["product-search", searchQuery],
    queryFn: () => apiClient.getProducts({ search: searchQuery, limit: 10 }),
    enabled: searchQuery.length > 2,
  });

  const addProductsMutation = useMutation({
    mutationFn: (productIds: string[]) => apiClient.addProductsToDataset(id, productIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      setIsAddOpen(false);
      setSelectedProducts([]);
    },
  });

  const removeProductMutation = useMutation({
    mutationFn: (productId: string) => apiClient.removeProductFromDataset(id, productId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
    },
  });

  // Action mutations
  const augmentMutation = useMutation({
    mutationFn: () => apiClient.startAugmentation(id),
  });

  const trainMutation = useMutation({
    mutationFn: () => apiClient.startTraining(id),
  });

  const extractMutation = useMutation({
    mutationFn: () => apiClient.startEmbeddingExtraction(id),
  });

  if (isLoading) return <div>Loading...</div>;
  if (!dataset) return <div>Dataset not found</div>;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/datasets">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{dataset.name}</h1>
            <p className="text-gray-500">{dataset.description || "No description"}</p>
          </div>
        </div>
      </div>

      {/* Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button
              onClick={() => augmentMutation.mutate()}
              disabled={augmentMutation.isPending || dataset.product_count === 0}
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Run Augmentation
            </Button>
            <Button
              variant="outline"
              onClick={() => trainMutation.mutate()}
              disabled={trainMutation.isPending || dataset.product_count === 0}
            >
              <GraduationCap className="h-4 w-4 mr-2" />
              Start Training
            </Button>
            <Button
              variant="outline"
              onClick={() => extractMutation.mutate()}
              disabled={extractMutation.isPending || dataset.product_count === 0}
            >
              <Database className="h-4 w-4 mr-2" />
              Extract Embeddings
            </Button>
          </div>
          {dataset.product_count === 0 && (
            <p className="text-sm text-yellow-600 mt-2">
              Add products to this dataset before running actions.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Products */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Products ({dataset.product_count})</CardTitle>
          <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Products
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Add Products to Dataset</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search by barcode or name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <div className="max-h-64 overflow-auto border rounded-lg">
                  {searchResults?.items?.map((product) => (
                    <div
                      key={product.id}
                      className={`p-3 flex items-center justify-between border-b last:border-0 cursor-pointer hover:bg-gray-50 ${
                        selectedProducts.includes(product.id) ? "bg-blue-50" : ""
                      }`}
                      onClick={() => {
                        if (selectedProducts.includes(product.id)) {
                          setSelectedProducts(selectedProducts.filter((id) => id !== product.id));
                        } else {
                          setSelectedProducts([...selectedProducts, product.id]);
                        }
                      }}
                    >
                      <div>
                        <p className="font-medium">{product.brand_name} {product.product_name}</p>
                        <p className="text-sm text-gray-500 font-mono">{product.barcode}</p>
                      </div>
                      <input
                        type="checkbox"
                        checked={selectedProducts.includes(product.id)}
                        onChange={() => {}}
                      />
                    </div>
                  ))}
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setIsAddOpen(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={() => addProductsMutation.mutate(selectedProducts)}
                    disabled={selectedProducts.length === 0 || addProductsMutation.isPending}
                  >
                    Add {selectedProducts.length} Products
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Image</TableHead>
                <TableHead>Barcode</TableHead>
                <TableHead>Brand</TableHead>
                <TableHead>Product</TableHead>
                <TableHead>Images</TableHead>
                <TableHead className="w-12"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {dataset.products?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                    No products in this dataset
                  </TableCell>
                </TableRow>
              ) : (
                dataset.products?.map((product) => (
                  <TableRow key={product.id}>
                    <TableCell>
                      <div className="w-10 h-10 bg-gray-100 rounded" />
                    </TableCell>
                    <TableCell className="font-mono text-sm">{product.barcode}</TableCell>
                    <TableCell>{product.brand_name || "-"}</TableCell>
                    <TableCell>{product.product_name || "-"}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Badge variant="outline">S: {product.synthetic_count || 0}</Badge>
                        <Badge variant="outline">R: {product.real_count || 0}</Badge>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-red-500 hover:text-red-600"
                        onClick={() => removeProductMutation.mutate(product.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Dataset detail görünüyor
- [ ] Products listesi görünüyor
- [ ] Add Products dialog çalışıyor (search + select)
- [ ] Remove product çalışıyor
- [ ] Action butonları (Augment, Train, Extract) tıklanabilir
- [ ] Empty state için uyarı mesajı

---

### TASK-025: Create Matching Page
**Prerequisites:** TASK-021
**Estimated Complexity:** Very High

**Description:**
Product matching UI - iki panel layout. Sol: ürün seçimi (dropdown + search). Sağ: FAISS ile bulunan adaylar. Approve/reject workflow.

**Reference:** `Eski kodlar/urun_temizleme_esleme_ui_custom.py`

**File:** `apps/web/src/app/matching/page.tsx`

```typescript
"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Check,
  X,
  SkipForward,
  Lock,
  Unlock,
  ZoomIn,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Candidate {
  id: string;
  image_url: string;
  similarity: number;
  source: string;
  is_approved?: boolean;
  is_rejected?: boolean;
}

interface Product {
  id: string;
  barcode: string;
  brand_name: string;
  product_name: string;
  primary_image_url: string;
  matched_count: number;
}

export default function MatchingPage() {
  const queryClient = useQueryClient();

  // State
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [isLocked, setIsLocked] = useState(false);
  const [selectedCandidates, setSelectedCandidates] = useState<Set<string>>(new Set());

  // Products needing matching
  const { data: products } = useQuery({
    queryKey: ["products-for-matching"],
    queryFn: () => apiClient.getProducts({ status: "needs_matching", limit: 100 }),
  });

  // Search products
  const { data: searchResults } = useQuery({
    queryKey: ["product-search", searchQuery],
    queryFn: () => apiClient.getProducts({ search: searchQuery, limit: 10 }),
    enabled: searchQuery.length > 2,
  });

  // Candidates for selected product
  const { data: candidates, isLoading: isLoadingCandidates, refetch: refetchCandidates } = useQuery({
    queryKey: ["matching-candidates", selectedProduct?.id, similarityThreshold],
    queryFn: () =>
      apiClient.getMatchingCandidates(selectedProduct!.id, { threshold: similarityThreshold }),
    enabled: !!selectedProduct,
  });

  // Mutations
  const approveMutation = useMutation({
    mutationFn: (candidateIds: string[]) =>
      apiClient.approveMatches(selectedProduct!.id, candidateIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matching-candidates"] });
      setSelectedCandidates(new Set());
    },
  });

  const rejectMutation = useMutation({
    mutationFn: (candidateIds: string[]) =>
      apiClient.rejectMatches(selectedProduct!.id, candidateIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matching-candidates"] });
      setSelectedCandidates(new Set());
    },
  });

  const skipMutation = useMutation({
    mutationFn: () => apiClient.skipProduct(selectedProduct!.id),
    onSuccess: () => {
      // Move to next product
      const currentIndex = products?.items?.findIndex((p) => p.id === selectedProduct?.id) || 0;
      const nextProduct = products?.items?.[currentIndex + 1];
      if (nextProduct) {
        setSelectedProduct(nextProduct);
      } else {
        setSelectedProduct(null);
      }
    },
  });

  // Lock mechanism for multi-user
  useEffect(() => {
    if (selectedProduct && !isLocked) {
      // Attempt to lock
      apiClient.lockProduct(selectedProduct.id).then((locked) => {
        setIsLocked(locked);
      });
    }
    return () => {
      if (selectedProduct && isLocked) {
        apiClient.unlockProduct(selectedProduct.id);
      }
    };
  }, [selectedProduct?.id]);

  const handleSelectCandidate = (candidateId: string) => {
    const newSelected = new Set(selectedCandidates);
    if (newSelected.has(candidateId)) {
      newSelected.delete(candidateId);
    } else {
      newSelected.add(candidateId);
    }
    setSelectedCandidates(newSelected);
  };

  const handleApproveSelected = () => {
    approveMutation.mutate(Array.from(selectedCandidates));
  };

  const handleRejectSelected = () => {
    rejectMutation.mutate(Array.from(selectedCandidates));
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-6">
      {/* Left Panel: Product Selection */}
      <div className="w-80 flex flex-col gap-4">
        <Card className="flex-1 flex flex-col">
          <CardHeader>
            <CardTitle>Select Product</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by barcode..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Dropdown for quick select */}
            <Select
              value={selectedProduct?.id || ""}
              onValueChange={(value) => {
                const product = products?.items?.find((p) => p.id === value);
                if (product) setSelectedProduct(product);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Or select from queue..." />
              </SelectTrigger>
              <SelectContent>
                {products?.items?.map((product) => (
                  <SelectItem key={product.id} value={product.id}>
                    {product.barcode} - {product.brand_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Search Results */}
            {searchResults?.items && searchResults.items.length > 0 && (
              <div className="border rounded-lg max-h-40 overflow-auto">
                {searchResults.items.map((product) => (
                  <div
                    key={product.id}
                    className="p-2 hover:bg-gray-50 cursor-pointer border-b last:border-0"
                    onClick={() => {
                      setSelectedProduct(product);
                      setSearchQuery("");
                    }}
                  >
                    <p className="font-mono text-sm">{product.barcode}</p>
                    <p className="text-xs text-gray-500">
                      {product.brand_name} {product.product_name}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {/* Selected Product Preview */}
            {selectedProduct && (
              <div className="mt-auto border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  {isLocked ? (
                    <Badge variant="default" className="gap-1">
                      <Lock className="h-3 w-3" />
                      Locked
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="gap-1">
                      <Unlock className="h-3 w-3" />
                      Unlocked
                    </Badge>
                  )}
                </div>
                <div className="aspect-square bg-gray-100 rounded-lg mb-2 overflow-hidden">
                  {selectedProduct.primary_image_url ? (
                    <img
                      src={selectedProduct.primary_image_url}
                      alt="Product"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      No image
                    </div>
                  )}
                </div>
                <p className="font-mono text-sm">{selectedProduct.barcode}</p>
                <p className="text-sm">
                  {selectedProduct.brand_name} {selectedProduct.product_name}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {selectedProduct.matched_count} matches approved
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Right Panel: Candidates */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Controls */}
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-6">
              {/* Similarity Threshold */}
              <div className="flex-1">
                <Label className="text-xs text-gray-500">
                  Similarity Threshold: {similarityThreshold.toFixed(2)}
                </Label>
                <Slider
                  value={[similarityThreshold]}
                  onValueChange={([value]) => setSimilarityThreshold(value)}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  className="mt-2"
                />
              </div>

              {/* Refresh */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchCandidates()}
                disabled={isLoadingCandidates}
              >
                <RefreshCw className={cn("h-4 w-4", isLoadingCandidates && "animate-spin")} />
              </Button>

              {/* Actions */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRejectSelected}
                  disabled={selectedCandidates.size === 0}
                  className="text-red-600"
                >
                  <X className="h-4 w-4 mr-1" />
                  Reject ({selectedCandidates.size})
                </Button>
                <Button
                  size="sm"
                  onClick={handleApproveSelected}
                  disabled={selectedCandidates.size === 0}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <Check className="h-4 w-4 mr-1" />
                  Approve ({selectedCandidates.size})
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => skipMutation.mutate()}
                  disabled={!selectedProduct}
                >
                  <SkipForward className="h-4 w-4 mr-1" />
                  Skip
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Candidate Grid */}
        <Card className="flex-1 overflow-hidden">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">
              Candidates ({candidates?.length || 0})
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-auto h-[calc(100%-3rem)]">
            {!selectedProduct ? (
              <div className="text-center py-12 text-gray-500">
                Select a product to see matching candidates
              </div>
            ) : isLoadingCandidates ? (
              <div className="text-center py-12">Loading candidates...</div>
            ) : candidates?.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                No candidates found above threshold
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-3">
                {candidates?.map((candidate: Candidate) => (
                  <div
                    key={candidate.id}
                    className={cn(
                      "relative aspect-square rounded-lg overflow-hidden border-2 cursor-pointer transition-all",
                      selectedCandidates.has(candidate.id)
                        ? "border-blue-500 ring-2 ring-blue-200"
                        : "border-transparent hover:border-gray-300",
                      candidate.is_approved && "border-green-500",
                      candidate.is_rejected && "opacity-50"
                    )}
                    onClick={() => handleSelectCandidate(candidate.id)}
                  >
                    <img
                      src={candidate.image_url}
                      alt="Candidate"
                      className="w-full h-full object-cover"
                    />

                    {/* Similarity Badge */}
                    <div className="absolute top-1 left-1">
                      <Badge
                        variant={candidate.similarity > 0.85 ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {(candidate.similarity * 100).toFixed(0)}%
                      </Badge>
                    </div>

                    {/* Source */}
                    <div className="absolute bottom-1 right-1">
                      <Badge variant="outline" className="text-xs bg-white/80">
                        {candidate.source}
                      </Badge>
                    </div>

                    {/* Selection Indicator */}
                    {selectedCandidates.has(candidate.id) && (
                      <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center">
                        <Check className="h-8 w-8 text-blue-600" />
                      </div>
                    )}

                    {/* Zoom on hover */}
                    <div className="absolute top-1 right-1 opacity-0 hover:opacity-100 transition-opacity">
                      <Button size="icon" variant="secondary" className="h-6 w-6">
                        <ZoomIn className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

**Backend API:** `apps/api/src/api/v1/matching.py`
```python
from fastapi import APIRouter, HTTPException
from services.faiss_index import FaissIndexService

router = APIRouter()
faiss_service = FaissIndexService()

@router.get("/candidates/{product_id}")
async def get_matching_candidates(
    product_id: str,
    threshold: float = 0.7,
    limit: int = 50,
):
    """Get matching candidates for a product using FAISS."""
    # 1. Get product embedding
    # 2. Search FAISS index
    # 3. Return sorted candidates
    candidates = await faiss_service.search(product_id, threshold, limit)
    return candidates

@router.post("/approve")
async def approve_matches(product_id: str, candidate_ids: list[str]):
    """Approve selected matches."""
    # Add to product_images table with type='real'
    pass

@router.post("/reject")
async def reject_matches(product_id: str, candidate_ids: list[str]):
    """Reject matches (won't show again)."""
    pass

@router.post("/lock/{product_id}")
async def lock_product(product_id: str):
    """Lock product for current user (multi-user support)."""
    # Use Redis or DB for locking
    return True

@router.post("/unlock/{product_id}")
async def unlock_product(product_id: str):
    """Unlock product."""
    pass
```

**FAISS Service:** `apps/api/src/services/faiss_index.py`
```python
import faiss
import numpy as np
from pathlib import Path

class FaissIndexService:
    def __init__(self):
        self.index = None
        self.id_map = {}  # Maps FAISS index to image IDs

    def load_index(self, path: Path):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(str(path))
        # Load id_map from JSON

    async def search(
        self,
        product_id: str,
        threshold: float,
        limit: int,
    ) -> list[dict]:
        """Search for similar images."""
        # 1. Get product embedding from cache or extract
        # 2. Search FAISS
        # 3. Filter by threshold
        # 4. Return with metadata
        pass

    def add_embeddings(self, embeddings: np.ndarray, ids: list[str]):
        """Add new embeddings to index."""
        self.index.add(embeddings)
        # Update id_map
```

**Acceptance Criteria:**
- [ ] Two-panel layout çalışıyor
- [ ] Product search + dropdown çalışıyor
- [ ] Candidate grid görünüyor
- [ ] Similarity threshold slider çalışıyor
- [ ] Multi-select (checkbox) çalışıyor
- [ ] Approve/Reject butonları çalışıyor
- [ ] Skip butonu sonraki ürüne geçiyor
- [ ] Lock mechanism görünüyor (badge)

---

### TASK-025B: Create Training API Router
**Prerequisites:** TASK-010
**Estimated Complexity:** Medium

**Description:**
Training API endpoint'lerini oluştur - job başlatma, model listesi, model aktivasyonu.

**File:** `apps/api/src/api/v1/training.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from services.supabase import supabase_service
from services.runpod import runpod_service

router = APIRouter()


# ========================================
# SCHEMAS
# ========================================

class ModelName(str, Enum):
    DINOV2_LARGE = "facebook/dinov2-large"
    DINOV2_BASE = "facebook/dinov2-base"


class TrainingConfigRequest(BaseModel):
    """Training configuration matching shell script parameters."""
    dataset_id: str
    model_name: ModelName = ModelName.DINOV2_LARGE

    # Model architecture
    proj_dim: int = Field(default=1024, description="Projection dimension")
    label_smoothing: float = Field(default=0.2, ge=0, le=1)

    # Training
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    llrd_decay: float = Field(default=0.9, ge=0, le=1, description="Layer-wise LR decay")
    warmup_epochs: int = Field(default=5, ge=0)
    grad_clip: float = Field(default=0.5, ge=0)

    # ArcFace & Domain Adaptation
    domain_aware_ratio: float = Field(default=0.57, ge=0, le=1)
    hard_negative_pool_size: int = Field(default=5, ge=1)
    use_hardest_negatives: bool = True

    # Performance
    use_mixed_precision: bool = True
    image_size: int = Field(default=518, ge=224)
    num_workers: int = Field(default=4, ge=0)

    # Dataset split
    train_ratio: float = Field(default=0.80, ge=0, le=1)
    valid_ratio: float = Field(default=0.10, ge=0, le=1)
    test_ratio: float = Field(default=0.10, ge=0, le=1)
    split_seed: int = Field(default=42)

    # Optional
    resume_checkpoint: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: str
    dataset_id: str
    status: str
    progress: int
    config: dict
    epochs_completed: Optional[int] = None
    final_loss: Optional[float] = None
    checkpoint_url: Optional[str] = None
    created_at: str


class ModelArtifactResponse(BaseModel):
    id: str
    name: str
    version: str
    checkpoint_url: str
    embedding_dim: int
    num_classes: int
    final_loss: float
    is_active: bool
    created_at: str


# ========================================
# ENDPOINTS
# ========================================

@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
):
    """List all training jobs."""
    query = supabase_service.client.table("jobs").select("*").eq("type", "training")

    if status:
        query = query.eq("status", status)

    result = query.order("created_at", desc=True).limit(limit).execute()
    return result.data


@router.post("/start", response_model=TrainingJobResponse)
async def start_training_job(
    config: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new training job."""
    # Validate dataset exists
    dataset = await supabase_service.get_dataset(config.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate split ratios
    total_ratio = config.train_ratio + config.valid_ratio + config.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise HTTPException(
            status_code=400,
            detail=f"Split ratios must sum to 1.0, got {total_ratio}"
        )

    # Create job record
    job = await supabase_service.create_job({
        "type": "training",
        "status": "pending",
        "progress": 0,
        "config": config.model_dump(),
    })

    # Submit to Runpod in background
    background_tasks.add_task(
        submit_training_to_runpod,
        job["id"],
        config.model_dump(),
    )

    return job


async def submit_training_to_runpod(job_id: str, config: dict):
    """Submit training job to Runpod."""
    try:
        await supabase_service.update_job(job_id, {"status": "submitted"})

        runpod_job = await runpod_service.submit_training(config)

        await supabase_service.update_job(job_id, {
            "status": "running",
            "runpod_job_id": runpod_job["id"],
        })

    except Exception as e:
        await supabase_service.update_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get training job details."""
    job = await supabase_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    job = await supabase_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["pending", "submitted", "running"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    # Cancel on Runpod if running
    if job.get("runpod_job_id"):
        await runpod_service.cancel_job(job["runpod_job_id"])

    await supabase_service.update_job(job_id, {"status": "cancelled"})
    return {"status": "cancelled"}


@router.get("/models", response_model=List[ModelArtifactResponse])
async def list_models():
    """List all trained models."""
    result = supabase_service.client.table("model_artifacts").select("*").order(
        "created_at", desc=True
    ).execute()
    return result.data


@router.get("/models/{model_id}", response_model=ModelArtifactResponse)
async def get_model(model_id: str):
    """Get model details."""
    result = supabase_service.client.table("model_artifacts").select("*").eq(
        "id", model_id
    ).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return result.data


@router.post("/models/{model_id}/activate", response_model=ModelArtifactResponse)
async def activate_model(model_id: str):
    """Set a model as the active model for embedding extraction."""
    # Deactivate all other models
    supabase_service.client.table("model_artifacts").update(
        {"is_active": False}
    ).neq("id", model_id).execute()

    # Activate this model
    result = supabase_service.client.table("model_artifacts").update(
        {"is_active": True}
    ).eq("id", model_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return result.data[0]
```

**Register in Router:** `apps/api/src/api/v1/router.py`
```python
from fastapi import APIRouter
from . import products, videos, datasets, matching, training, webhooks

api_router = APIRouter()

api_router.include_router(products.router, prefix="/products", tags=["products"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(matching.router, prefix="/matching", tags=["matching"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
```

**Acceptance Criteria:**
- [ ] POST /api/v1/training/start accepts full config
- [ ] GET /api/v1/training/jobs returns job list
- [ ] GET /api/v1/training/models returns model list
- [ ] POST /api/v1/training/models/{id}/activate works
- [ ] Runpod job submission works
- [ ] Job cancellation works

---

### TASK-026: Create Training Page
**Prerequisites:** TASK-024
**Estimated Complexity:** High

**Description:**
Model training sayfası - dataset seçimi, tam training config (shell script ile uyumlu), job monitoring.

**Reference:**
- `Eski kodlar/train_optimized_v14.py`
- `Eski kodlar/train_optimized_v14 (1) (1).sh`
- `Eski kodlar/split_dataset_to_train_valid_test (1).py`

**File:** `apps/web/src/app/training/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Separator } from "@/components/ui/separator";
import { Play, Loader2, CheckCircle, XCircle, Download, Settings2, Zap, Brain, Gauge } from "lucide-react";

interface TrainingJob {
  id: string;
  dataset_id: string;
  dataset_name: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_loss: number;
  best_accuracy: number;
  started_at: string;
  completed_at?: string;
  checkpoint_url?: string;
}

// ========================================
// FULL TRAINING CONFIG - Matches train_optimized_v14.sh
// ========================================
interface TrainingConfig {
  // Required
  dataset_id: string;

  // Model Configuration
  model_name: "facebook/dinov2-large" | "facebook/dinov2-base";
  proj_dim: number;         // 1024 for LARGE, 768 for base
  label_smoothing: number;  // 0.2 default

  // Training Configuration
  epochs: number;           // 30 default
  batch_size: number;       // 16 for LARGE, 32 for base
  lr: number;               // 2e-5 for LARGE
  weight_decay: number;     // 0.01
  llrd_decay: number;       // 0.9 (Layer-wise LR Decay)
  warmup_epochs: number;    // 5 for LARGE
  grad_clip: number;        // 0.5 for stability

  // Domain Adaptation
  domain_aware_ratio: number;      // 0.57
  hard_negative_pool_size: number; // 5
  use_hardest_negatives: boolean;  // true

  // Stability Settings
  use_mixed_precision: boolean;    // false for stability, true for speed
  image_size: number;              // 384
  num_workers: number;             // 2 for memory savings

  // Dataset Split (train/valid/test)
  train_ratio: number;     // 0.80
  valid_ratio: number;     // 0.10
  test_ratio: number;      // 0.10
  split_seed: number;      // 42

  // Resume (optional)
  resume_checkpoint?: string;
}

// Default config based on shell script
const DEFAULT_CONFIG: TrainingConfig = {
  dataset_id: "",
  model_name: "facebook/dinov2-large",
  proj_dim: 1024,
  label_smoothing: 0.2,
  epochs: 30,
  batch_size: 16,
  lr: 0.00002,  // 2e-5
  weight_decay: 0.01,
  llrd_decay: 0.9,
  warmup_epochs: 5,
  grad_clip: 0.5,
  domain_aware_ratio: 0.57,
  hard_negative_pool_size: 5,
  use_hardest_negatives: true,
  use_mixed_precision: false,  // Stability first
  image_size: 384,
  num_workers: 2,
  train_ratio: 0.80,
  valid_ratio: 0.10,
  test_ratio: 0.10,
  split_seed: 42,
};

export default function TrainingPage() {
  const queryClient = useQueryClient();

  // State
  const [activeTab, setActiveTab] = useState("new");
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);

  // Auto-adjust config when model changes
  const handleModelChange = (modelName: typeof config.model_name) => {
    if (modelName === "facebook/dinov2-large") {
      setConfig({
        ...config,
        model_name: modelName,
        proj_dim: 1024,
        batch_size: 16,
        lr: 0.00002,
        warmup_epochs: 5,
        grad_clip: 0.5,
      });
    } else {
      setConfig({
        ...config,
        model_name: modelName,
        proj_dim: 768,
        batch_size: 32,
        lr: 0.0001,
        warmup_epochs: 3,
        grad_clip: 1.0,
      });
    }
  };

  // Queries
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  const { data: jobs, isLoading: isLoadingJobs } = useQuery({
    queryKey: ["training-jobs"],
    queryFn: () => apiClient.getTrainingJobs(),
    refetchInterval: 5000,
  });

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: () => apiClient.getModels(),
  });

  // Mutations
  const startTrainingMutation = useMutation({
    mutationFn: (config: TrainingConfig) => apiClient.startTrainingJob(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["training-jobs"] });
      setActiveTab("jobs");
    },
  });

  const activateModelMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.activateModel(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const statusIcons = {
    pending: <Loader2 className="h-4 w-4 text-gray-400" />,
    running: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
    completed: <CheckCircle className="h-4 w-4 text-green-500" />,
    failed: <XCircle className="h-4 w-4 text-red-500" />,
  };

  // Validate split ratios
  const splitSum = config.train_ratio + config.valid_ratio + config.test_ratio;
  const isSplitValid = Math.abs(splitSum - 1.0) < 0.01;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Training</h1>
        <p className="text-gray-500">Train DINOv2 + ArcFace models on your datasets</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="new">New Training</TabsTrigger>
          <TabsTrigger value="jobs">Jobs ({jobs?.filter((j: TrainingJob) => j.status === "running").length || 0} running)</TabsTrigger>
          <TabsTrigger value="models">Models ({models?.length || 0})</TabsTrigger>
        </TabsList>

        {/* New Training Tab */}
        <TabsContent value="new" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              {/* Basic Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Model Configuration
                  </CardTitle>
                  <CardDescription>
                    Select model and configure training parameters
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Dataset Selection */}
                  <div>
                    <Label>Dataset</Label>
                    <Select
                      value={config.dataset_id}
                      onValueChange={(value) => setConfig({ ...config, dataset_id: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a dataset..." />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets?.map((dataset) => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            {dataset.name} ({dataset.product_count} products)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Model Selection */}
                  <div>
                    <Label>Model</Label>
                    <Select
                      value={config.model_name}
                      onValueChange={(value) => handleModelChange(value as typeof config.model_name)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="facebook/dinov2-large">
                          DINOv2-LARGE (304M params) - Recommended
                        </SelectItem>
                        <SelectItem value="facebook/dinov2-base">
                          DINOv2-BASE (86M params) - Faster
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-500 mt-1">
                      LARGE: +3-5% accuracy, 2x slower | BASE: Faster training
                    </p>
                  </div>

                  {/* Epochs */}
                  <div>
                    <Label>Epochs: {config.epochs}</Label>
                    <Slider
                      value={[config.epochs]}
                      onValueChange={([value]) => setConfig({ ...config, epochs: value })}
                      min={5}
                      max={100}
                      step={5}
                      className="mt-2"
                    />
                  </div>

                  {/* Batch Size */}
                  <div>
                    <Label>Batch Size: {config.batch_size}</Label>
                    <Slider
                      value={[config.batch_size]}
                      onValueChange={([value]) => setConfig({ ...config, batch_size: value })}
                      min={8}
                      max={64}
                      step={8}
                      className="mt-2"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Lower = more stable, less GPU memory
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Dataset Split Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="h-5 w-5" />
                    Dataset Split
                  </CardTitle>
                  <CardDescription>
                    Configure train/validation/test split ratios
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Train Ratio: {(config.train_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[config.train_ratio]}
                      onValueChange={([value]) => setConfig({ ...config, train_ratio: value })}
                      min={0.5}
                      max={0.95}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label>Validation Ratio: {(config.valid_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[config.valid_ratio]}
                      onValueChange={([value]) => setConfig({ ...config, valid_ratio: value })}
                      min={0.05}
                      max={0.3}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label>Test Ratio: {(config.test_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[config.test_ratio]}
                      onValueChange={([value]) => setConfig({ ...config, test_ratio: value })}
                      min={0.05}
                      max={0.3}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                  {!isSplitValid && (
                    <p className="text-sm text-red-500">
                      Split ratios must sum to 100% (current: {(splitSum * 100).toFixed(0)}%)
                    </p>
                  )}
                  <Separator />
                  <div>
                    <Label>Random Seed</Label>
                    <Input
                      type="number"
                      value={config.split_seed}
                      onChange={(e) => setConfig({ ...config, split_seed: Number(e.target.value) })}
                      className="mt-1"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      For reproducible splits
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="space-y-6">
              {/* Advanced Settings */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="h-5 w-5" />
                    Advanced Settings
                  </CardTitle>
                  <CardDescription>
                    Fine-tune training hyperparameters
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Accordion type="multiple" className="w-full">
                    {/* Learning Rate & Optimization */}
                    <AccordionItem value="lr">
                      <AccordionTrigger>Learning Rate & Optimization</AccordionTrigger>
                      <AccordionContent className="space-y-4 pt-2">
                        <div>
                          <Label>Learning Rate</Label>
                          <Select
                            value={String(config.lr)}
                            onValueChange={(value) => setConfig({ ...config, lr: Number(value) })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="0.0001">1e-4 (BASE)</SelectItem>
                              <SelectItem value="0.00005">5e-5</SelectItem>
                              <SelectItem value="0.00002">2e-5 (LARGE - recommended)</SelectItem>
                              <SelectItem value="0.00001">1e-5</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label>Weight Decay: {config.weight_decay}</Label>
                          <Slider
                            value={[config.weight_decay]}
                            onValueChange={([value]) => setConfig({ ...config, weight_decay: value })}
                            min={0.001}
                            max={0.1}
                            step={0.001}
                            className="mt-2"
                          />
                        </div>
                        <div>
                          <Label>LLRD Decay: {config.llrd_decay}</Label>
                          <Slider
                            value={[config.llrd_decay]}
                            onValueChange={([value]) => setConfig({ ...config, llrd_decay: value })}
                            min={0.7}
                            max={1.0}
                            step={0.05}
                            className="mt-2"
                          />
                          <p className="text-xs text-gray-500 mt-1">
                            Layer-wise Learning Rate Decay
                          </p>
                        </div>
                        <div>
                          <Label>Warmup Epochs: {config.warmup_epochs}</Label>
                          <Slider
                            value={[config.warmup_epochs]}
                            onValueChange={([value]) => setConfig({ ...config, warmup_epochs: value })}
                            min={1}
                            max={10}
                            step={1}
                            className="mt-2"
                          />
                        </div>
                        <div>
                          <Label>Gradient Clipping: {config.grad_clip}</Label>
                          <Slider
                            value={[config.grad_clip]}
                            onValueChange={([value]) => setConfig({ ...config, grad_clip: value })}
                            min={0.1}
                            max={2.0}
                            step={0.1}
                            className="mt-2"
                          />
                          <p className="text-xs text-gray-500 mt-1">
                            Lower = more stable, prevents NaN
                          </p>
                        </div>
                      </AccordionContent>
                    </AccordionItem>

                    {/* ArcFace & Domain Adaptation */}
                    <AccordionItem value="arcface">
                      <AccordionTrigger>ArcFace & Domain Adaptation</AccordionTrigger>
                      <AccordionContent className="space-y-4 pt-2">
                        <div>
                          <Label>Projection Dimension: {config.proj_dim}</Label>
                          <Select
                            value={String(config.proj_dim)}
                            onValueChange={(value) => setConfig({ ...config, proj_dim: Number(value) })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="1024">1024 (LARGE)</SelectItem>
                              <SelectItem value="768">768 (BASE)</SelectItem>
                              <SelectItem value="512">512 (compact)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label>Label Smoothing: {config.label_smoothing}</Label>
                          <Slider
                            value={[config.label_smoothing]}
                            onValueChange={([value]) => setConfig({ ...config, label_smoothing: value })}
                            min={0}
                            max={0.3}
                            step={0.05}
                            className="mt-2"
                          />
                        </div>
                        <div>
                          <Label>Domain-Aware Ratio: {(config.domain_aware_ratio * 100).toFixed(0)}%</Label>
                          <Slider
                            value={[config.domain_aware_ratio]}
                            onValueChange={([value]) => setConfig({ ...config, domain_aware_ratio: value })}
                            min={0.3}
                            max={0.8}
                            step={0.05}
                            className="mt-2"
                          />
                          <p className="text-xs text-gray-500 mt-1">
                            % of triplets with synthetic-real pairs
                          </p>
                        </div>
                        <div>
                          <Label>Hard Negative Pool Size: {config.hard_negative_pool_size}</Label>
                          <Slider
                            value={[config.hard_negative_pool_size]}
                            onValueChange={([value]) => setConfig({ ...config, hard_negative_pool_size: value })}
                            min={3}
                            max={20}
                            step={1}
                            className="mt-2"
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <Label>Use Hardest Negatives</Label>
                          <Switch
                            checked={config.use_hardest_negatives}
                            onCheckedChange={(checked) => setConfig({ ...config, use_hardest_negatives: checked })}
                          />
                        </div>
                      </AccordionContent>
                    </AccordionItem>

                    {/* Performance Settings */}
                    <AccordionItem value="performance">
                      <AccordionTrigger>Performance & Stability</AccordionTrigger>
                      <AccordionContent className="space-y-4 pt-2">
                        <div className="flex items-center justify-between">
                          <div>
                            <Label>Mixed Precision (AMP)</Label>
                            <p className="text-xs text-gray-500">2-3x faster, may be less stable</p>
                          </div>
                          <Switch
                            checked={config.use_mixed_precision}
                            onCheckedChange={(checked) => setConfig({ ...config, use_mixed_precision: checked })}
                          />
                        </div>
                        <div>
                          <Label>Image Size: {config.image_size}px</Label>
                          <Select
                            value={String(config.image_size)}
                            onValueChange={(value) => setConfig({ ...config, image_size: Number(value) })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="224">224px (fast)</SelectItem>
                              <SelectItem value="384">384px (recommended)</SelectItem>
                              <SelectItem value="518">518px (high quality)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label>DataLoader Workers: {config.num_workers}</Label>
                          <Slider
                            value={[config.num_workers]}
                            onValueChange={([value]) => setConfig({ ...config, num_workers: value })}
                            min={0}
                            max={8}
                            step={1}
                            className="mt-2"
                          />
                          <p className="text-xs text-gray-500 mt-1">
                            Lower = less memory, slower data loading
                          </p>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
              </Card>

              {/* Training Summary & Start */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Training Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Model:</span>
                      <p className="font-medium">{config.model_name.split("/")[1]}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Embedding Dim:</span>
                      <p className="font-medium">{config.proj_dim}D</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Epochs:</span>
                      <p className="font-medium">{config.epochs}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Batch Size:</span>
                      <p className="font-medium">{config.batch_size}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Mixed Precision:</span>
                      <p className="font-medium">{config.use_mixed_precision ? "Enabled" : "Disabled"}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Est. Time:</span>
                      <p className="font-medium">
                        ~{Math.ceil(config.epochs * (config.use_mixed_precision ? 1 : 2))} min (A100)
                      </p>
                    </div>
                  </div>

                  <Separator />

                  <Button
                    className="w-full"
                    size="lg"
                    onClick={() => startTrainingMutation.mutate(config)}
                    disabled={!config.dataset_id || !isSplitValid || startTrainingMutation.isPending}
                  >
                    {startTrainingMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Start Training
                  </Button>

                  {startTrainingMutation.isPending && (
                    <p className="text-sm text-center text-gray-500">
                      Submitting job to Runpod...
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Jobs Tab */}
        <TabsContent value="jobs" className="mt-6">
          <div className="space-y-4">
            {isLoadingJobs ? (
              <div className="text-center py-12">Loading...</div>
            ) : jobs?.length === 0 ? (
              <Card className="py-12">
                <CardContent className="text-center">
                  <p className="text-gray-500">No training jobs yet</p>
                </CardContent>
              </Card>
            ) : (
              jobs?.map((job: TrainingJob) => (
                <Card key={job.id}>
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        {statusIcons[job.status]}
                        <div>
                          <p className="font-medium">{job.dataset_name}</p>
                          <p className="text-sm text-gray-500">
                            Started {new Date(job.started_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        {job.status === "running" && (
                          <div className="text-right">
                            <p className="text-sm">
                              Epoch {job.current_epoch}/{job.total_epochs}
                            </p>
                            <p className="text-xs text-gray-500">
                              Loss: {job.current_loss.toFixed(4)}
                            </p>
                          </div>
                        )}
                        {job.status === "completed" && (
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">
                              Acc: {(job.best_accuracy * 100).toFixed(1)}%
                            </Badge>
                            <Button variant="outline" size="sm" asChild>
                              <a href={job.checkpoint_url} download>
                                <Download className="h-4 w-4 mr-1" />
                                Download
                              </a>
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                    {job.status === "running" && (
                      <Progress value={job.progress} className="mt-4 h-2" />
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models" className="mt-6">
          <div className="grid grid-cols-3 gap-4">
            {models?.map((model) => (
              <Card key={model.id} className={model.is_active ? "border-green-500" : ""}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-base">{model.name}</CardTitle>
                    {model.is_active && (
                      <Badge className="bg-green-500">Active</Badge>
                    )}
                  </div>
                  <CardDescription>{model.version}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Accuracy</span>
                      <span>{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Embedding Dim</span>
                      <span>{model.embedding_dim}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Classes</span>
                      <span>{model.num_classes}</span>
                    </div>
                  </div>
                  {!model.is_active && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full mt-4"
                      onClick={() => activateModelMutation.mutate(model.id)}
                    >
                      Set as Active
                    </Button>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Dataset selection çalışıyor
- [ ] Training config form çalışıyor
- [ ] Start Training butonu job oluşturuyor
- [ ] Jobs listesi görünüyor
- [ ] Running job progress bar görünüyor
- [ ] Completed job'da download butonu var
- [ ] Models listesi görünüyor
- [ ] "Set as Active" butonu çalışıyor

---

### TASK-027: Create Embeddings Page
**Prerequisites:** TASK-026
**Estimated Complexity:** Medium

**Description:**
Embedding extraction ve index management sayfası.

**Reference:** `Eski kodlar/extract_embeddings_large.py`

**File:** `apps/web/src/app/embeddings/page.tsx`

```typescript
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Database, Play, Loader2, CheckCircle, Plus, Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface EmbeddingJob {
  id: string;
  dataset_name: string;
  model_name: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  total_images: number;
  processed_images: number;
  started_at: string;
}

interface EmbeddingIndex {
  id: string;
  name: string;
  model_name: string;
  vector_count: number;
  created_at: string;
}

export default function EmbeddingsPage() {
  const queryClient = useQueryClient();

  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedIndex, setSelectedIndex] = useState("");

  // Queries
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: () => apiClient.getModels(),
  });

  const { data: jobs } = useQuery({
    queryKey: ["embedding-jobs"],
    queryFn: () => apiClient.getEmbeddingJobs(),
    refetchInterval: 5000,
  });

  const { data: indexes } = useQuery({
    queryKey: ["embedding-indexes"],
    queryFn: () => apiClient.getEmbeddingIndexes(),
  });

  // Mutations
  const extractMutation = useMutation({
    mutationFn: () =>
      apiClient.startEmbeddingExtraction({
        dataset_id: selectedDataset,
        model_id: selectedModel,
        index_id: selectedIndex || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
    },
  });

  const createIndexMutation = useMutation({
    mutationFn: (name: string) =>
      apiClient.createEmbeddingIndex({
        name,
        model_id: selectedModel,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["embedding-indexes"] });
    },
  });

  const deleteIndexMutation = useMutation({
    mutationFn: (indexId: string) => apiClient.deleteEmbeddingIndex(indexId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["embedding-indexes"] });
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Embeddings</h1>
        <p className="text-gray-500">Extract embeddings and manage FAISS indexes</p>
      </div>

      <Tabs defaultValue="extract">
        <TabsList>
          <TabsTrigger value="extract">Extract</TabsTrigger>
          <TabsTrigger value="indexes">Indexes ({indexes?.length || 0})</TabsTrigger>
          <TabsTrigger value="jobs">Jobs</TabsTrigger>
        </TabsList>

        {/* Extract Tab */}
        <TabsContent value="extract" className="mt-6">
          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Extract Embeddings</CardTitle>
                <CardDescription>
                  Extract embeddings from a dataset using a trained model
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Dataset */}
                <div>
                  <Label>Dataset</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select dataset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets?.map((d) => (
                        <SelectItem key={d.id} value={d.id}>
                          {d.name} ({d.product_count} products)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Model */}
                <div>
                  <Label>Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select model..." />
                    </SelectTrigger>
                    <SelectContent>
                      {models?.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name} {m.is_active && "(Active)"}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Target Index */}
                <div>
                  <Label>Add to Index (optional)</Label>
                  <Select value={selectedIndex} onValueChange={setSelectedIndex}>
                    <SelectTrigger>
                      <SelectValue placeholder="Create new or select existing..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Create new index</SelectItem>
                      {indexes?.map((idx) => (
                        <SelectItem key={idx.id} value={idx.id}>
                          {idx.name} ({idx.vector_count} vectors)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  className="w-full"
                  onClick={() => extractMutation.mutate()}
                  disabled={!selectedDataset || !selectedModel || extractMutation.isPending}
                >
                  <Database className="h-4 w-4 mr-2" />
                  Start Extraction
                </Button>
              </CardContent>
            </Card>

            {/* Info */}
            <Card>
              <CardHeader>
                <CardTitle>Extraction Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div>
                  <h4 className="font-medium">Process</h4>
                  <ol className="list-decimal list-inside text-gray-600 mt-1 space-y-1">
                    <li>Load trained model checkpoint</li>
                    <li>Process images in batches (GPU)</li>
                    <li>Extract 1024D embeddings</li>
                    <li>Add to FAISS index</li>
                    <li>Save index to storage</li>
                  </ol>
                </div>
                <div>
                  <h4 className="font-medium">Image Types</h4>
                  <p className="text-gray-600 mt-1">
                    Extracts embeddings from both synthetic (video frames) and
                    real (matched) images for domain-aware retrieval.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Indexes Tab */}
        <TabsContent value="indexes" className="mt-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>FAISS Indexes</CardTitle>
              <Button
                onClick={() => {
                  const name = prompt("Enter index name:");
                  if (name) createIndexMutation.mutate(name);
                }}
              >
                <Plus className="h-4 w-4 mr-2" />
                New Index
              </Button>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Vectors</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="w-12"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {indexes?.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                        No indexes yet
                      </TableCell>
                    </TableRow>
                  ) : (
                    indexes?.map((idx: EmbeddingIndex) => (
                      <TableRow key={idx.id}>
                        <TableCell className="font-medium">{idx.name}</TableCell>
                        <TableCell>{idx.model_name}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{idx.vector_count.toLocaleString()}</Badge>
                        </TableCell>
                        <TableCell className="text-gray-500">
                          {new Date(idx.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="text-red-500"
                            onClick={() => {
                              if (confirm("Delete this index?")) {
                                deleteIndexMutation.mutate(idx.id);
                              }
                            }}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Jobs Tab */}
        <TabsContent value="jobs" className="mt-6">
          <div className="space-y-4">
            {jobs?.map((job: EmbeddingJob) => (
              <Card key={job.id}>
                <CardContent className="py-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      {job.status === "running" ? (
                        <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                      ) : job.status === "completed" ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <Database className="h-5 w-5 text-gray-400" />
                      )}
                      <div>
                        <p className="font-medium">{job.dataset_name}</p>
                        <p className="text-sm text-gray-500">
                          Model: {job.model_name}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm">
                        {job.processed_images} / {job.total_images} images
                      </p>
                      <p className="text-xs text-gray-500">
                        {new Date(job.started_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  {job.status === "running" && (
                    <Progress value={job.progress} className="mt-4 h-2" />
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

**Acceptance Criteria:**
- [ ] Dataset ve model selection çalışıyor
- [ ] Extract butonu job oluşturuyor
- [ ] Indexes listesi görünüyor
- [ ] Create/Delete index çalışıyor
- [ ] Jobs progress görünüyor

---

## Phase 3: GPU Workers

### TASK-030: Migrate Video Segmentation Worker
**Prerequisites:** TASK-013
**Estimated Complexity:** High

**Description:**
Mevcut video segmentation pipeline'ını Runpod worker olarak yapılandır.

**Source:** `worker/src/pipeline.py`, `worker/src/handler.py`

**Target:** `workers/video-segmentation/`

**Structure:**
```
workers/video-segmentation/
├── Dockerfile
├── requirements.txt
└── src/
    ├── handler.py      # Runpod entrypoint
    ├── pipeline.py     # Main pipeline (migrated)
    ├── gemini_extractor.py
    └── sam3_segmenter.py
```

**File:** `workers/video-segmentation/Dockerfile`
```dockerfile
FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM3 (assuming it's available)
RUN pip install git+https://github.com/facebookresearch/sam3.git

# Copy source
COPY src/ ./src/

# Set HF token for model download
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

CMD ["python", "-u", "src/handler.py"]
```

**File:** `workers/video-segmentation/requirements.txt`
```
runpod==1.6.0
torch>=2.2.0
torchvision
numpy
opencv-python-headless
Pillow
requests
tqdm
google-generativeai>=0.4.0
supabase>=2.0.0
```

**File:** `workers/video-segmentation/src/handler.py`
```python
"""Runpod Serverless Handler for Video Segmentation."""

import runpod
import traceback
import os
from pipeline import ProductPipeline

# Pipeline singleton
pipeline = None

def get_pipeline():
    """Get or create pipeline singleton."""
    global pipeline
    if pipeline is None:
        print("=" * 60)
        print("COLD START - Loading pipeline...")
        print("=" * 60)
        pipeline = ProductPipeline()
        print("Pipeline ready!")
    return pipeline

def handler(job):
    """Main handler for Runpod serverless."""
    try:
        job_input = job.get("input", {})

        video_url = job_input.get("video_url")
        if not video_url:
            return {"status": "error", "error": "video_url is required"}

        barcode = job_input.get("barcode", "unknown")
        video_id = job_input.get("video_id")
        product_id = job_input.get("product_id")  # UUID from our system

        print(f"\nProcessing: {barcode}")
        print(f"Video URL: {video_url[:80]}...")

        pipe = get_pipeline()
        result = pipe.process(
            video_url=video_url,
            barcode=barcode,
            video_id=video_id,
            product_id=product_id,
        )

        return {
            "status": "success",
            "barcode": barcode,
            "product_id": product_id,
            "metadata": result["metadata"],
            "frame_count": result["frame_count"],
            "frames_url": result.get("frames_url"),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

if __name__ == "__main__":
    print("Starting Video Segmentation Worker...")
    runpod.serverless.start({"handler": handler})
```

**Webhook Callback:**
Add webhook callback to notify backend when job completes.

```python
# In pipeline.py, after processing
def _send_callback(self, product_id: str, result: dict):
    """Send completion callback to backend."""
    callback_url = os.environ.get("CALLBACK_URL")
    if callback_url:
        import requests
        requests.post(
            f"{callback_url}/api/v1/webhooks/runpod",
            json={
                "type": "video_processing",
                "product_id": product_id,
                "result": result,
            },
            timeout=10,
        )
```

**Acceptance Criteria:**
- [ ] Dockerfile build ediyor
- [ ] Handler Runpod'da çalışıyor
- [ ] Pipeline video işliyor
- [ ] Frames Supabase Storage'a yükleniyor
- [ ] Webhook callback gönderiliyor

---

### TASK-031: Migrate Augmentation Worker
**Prerequisites:** TASK-030
**Estimated Complexity:** High

**Description:**
Augmentation pipeline'ını Runpod worker olarak yapılandır. Eski koddan tüm optimizasyonlar korunmalı.

**Reference:** `Eski kodlar/final_augmentor_v3.py`

**Target:** `workers/augmentation/`

**Key Optimizations (from original code):**
- Thread limiting (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`)
- BiRefNet with GPU half-precision for memory efficiency
- 3 different augmentation pipelines (light, heavy, real)
- Idempotent top-up (only generates missing images)
- Per-source quota distribution
- Background composition with shadow effects
- Border detection for resize decision
- JSON progress reporting

---

**File:** `workers/augmentation/src/supabase_client.py`
```python
"""Supabase client for worker - downloads dataset and uploads results."""

import os
import json
from pathlib import Path
from supabase import create_client, Client
from typing import Optional
import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")


def get_supabase() -> Client:
    """Get Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class DatasetDownloader:
    """Downloads dataset from Supabase Storage for processing."""

    def __init__(self, local_base: Path = Path("/tmp/datasets")):
        self.client = get_supabase()
        self.local_base = local_base
        self.local_base.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str) -> Path:
        """
        Download all product images for a dataset.

        Structure created:
        /tmp/datasets/{dataset_id}/
          train/
            {product_id_1}/
              frame_0000.png
              frame_0001.png
              real/
                real_image_1.jpg
            {product_id_2}/
              ...
        """
        print(f"\n📥 Downloading dataset: {dataset_id}")

        # 1. Get dataset products from DB
        response = self.client.table("dataset_products").select(
            "product_id, products(id, barcode, frames_path)"
        ).eq("dataset_id", dataset_id).execute()

        products = response.data
        if not products:
            raise ValueError(f"Dataset {dataset_id} has no products")

        print(f"   Found {len(products)} products")

        # 2. Create local directory structure
        dataset_dir = self.local_base / dataset_id / "train"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 3. Download each product's frames
        for item in products:
            product = item.get("products", {})
            product_id = product.get("id")
            barcode = product.get("barcode", product_id)
            frames_path = product.get("frames_path")

            if not frames_path:
                print(f"   ⚠️ No frames for product {barcode}")
                continue

            # Create product directory (use barcode as folder name for compatibility)
            product_dir = dataset_dir / barcode
            product_dir.mkdir(parents=True, exist_ok=True)

            # Download frames from storage
            self._download_product_frames(frames_path, product_dir)

            # Download real images if exist
            real_dir = product_dir / "real"
            real_dir.mkdir(exist_ok=True)
            self._download_real_images(product_id, real_dir)

        print(f"   ✅ Dataset downloaded to: {dataset_dir.parent}")
        return dataset_dir.parent

    def _download_product_frames(self, frames_path: str, target_dir: Path):
        """Download frames from Supabase Storage."""
        try:
            # frames_path format: "{supabase_url}/storage/v1/object/public/frames/{barcode}/"
            # or just "{barcode}/" if relative
            bucket = "frames"

            # Extract barcode from path
            if "/" in frames_path:
                barcode = frames_path.rstrip("/").split("/")[-1]
            else:
                barcode = frames_path.rstrip("/")

            # List files in bucket
            files = self.client.storage.from_(bucket).list(barcode)

            for f in files:
                if f.get("name", "").endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    file_path = f"{barcode}/{f['name']}"
                    data = self.client.storage.from_(bucket).download(file_path)

                    local_path = target_dir / f['name']
                    with open(local_path, 'wb') as fp:
                        fp.write(data)

        except Exception as e:
            print(f"   ⚠️ Frame download error: {e}")

    def _download_real_images(self, product_id: str, target_dir: Path):
        """Download matched real images for a product."""
        try:
            # Get real images from product_images table
            response = self.client.table("product_images").select("*").eq(
                "product_id", product_id
            ).eq("image_type", "real").execute()

            for img in response.data:
                image_path = img.get("image_path")
                if not image_path:
                    continue

                # Download from storage
                bucket = "real-images"
                try:
                    data = self.client.storage.from_(bucket).download(image_path)
                    local_path = target_dir / Path(image_path).name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                except Exception:
                    pass

        except Exception as e:
            print(f"   ⚠️ Real images download error: {e}")


class ResultUploader:
    """Uploads augmented images back to Supabase Storage."""

    def __init__(self):
        self.client = get_supabase()

    def upload_augmented(self, dataset_dir: Path, dataset_id: str) -> dict:
        """
        Upload augmented images to Supabase Storage.

        Returns upload statistics.
        """
        print(f"\n📤 Uploading augmented images...")

        bucket = "augmented-images"
        stats = {"syn_uploaded": 0, "real_uploaded": 0}

        # Walk through dataset directory
        train_dir = dataset_dir / "train"
        if not train_dir.exists():
            train_dir = dataset_dir

        for product_dir in train_dir.iterdir():
            if not product_dir.is_dir():
                continue

            barcode = product_dir.name

            # Upload synthetic augmented images (syn_*.jpg)
            for img in product_dir.glob("syn_*.jpg"):
                storage_path = f"{dataset_id}/{barcode}/{img.name}"
                try:
                    with open(img, 'rb') as f:
                        self.client.storage.from_(bucket).upload(storage_path, f.read())
                    stats["syn_uploaded"] += 1
                except Exception:
                    pass  # May already exist

            # Upload real augmented images (real/*_aug_*.jpg)
            real_dir = product_dir / "real"
            if real_dir.exists():
                for img in real_dir.glob("*_aug_*.jpg"):
                    storage_path = f"{dataset_id}/{barcode}/real/{img.name}"
                    try:
                        with open(img, 'rb') as f:
                            self.client.storage.from_(bucket).upload(storage_path, f.read())
                        stats["real_uploaded"] += 1
                    except Exception:
                        pass

        print(f"   ✅ Uploaded: {stats['syn_uploaded']} syn + {stats['real_uploaded']} real")
        return stats

    def update_job_progress(self, job_id: str, progress: int, current_step: str):
        """Update job progress in database."""
        try:
            self.client.table("jobs").update({
                "progress": progress,
                "current_step": current_step,
            }).eq("runpod_job_id", job_id).execute()
        except Exception as e:
            print(f"   ⚠️ Progress update error: {e}")

    def send_callback(self, result: dict):
        """Send completion callback to backend."""
        if not CALLBACK_URL:
            return
        try:
            httpx.post(
                f"{CALLBACK_URL}/api/v1/webhooks/runpod",
                json=result,
                timeout=30,
            )
        except Exception as e:
            print(f"   ⚠️ Callback error: {e}")
```

---

**File:** `workers/augmentation/src/handler.py`
```python
"""
Runpod Serverless Handler for Augmentation.
Based on: final_augmentor_v3.py

INTEGRATED WITH SUPABASE:
- Receives dataset_id from API
- Downloads images from Supabase Storage
- Processes locally with all optimizations
- Uploads results back to Supabase Storage
"""

import os
# CRITICAL: Thread limiting for stability
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import runpod
import json
import traceback
import shutil
from pathlib import Path
from augmentor import ProductAugmentor
from supabase_client import DatasetDownloader, ResultUploader

# Singletons - loaded once on cold start
augmentor = None
downloader = None
uploader = None


def get_augmentor():
    """Get or create augmentor singleton."""
    global augmentor
    if augmentor is None:
        print("=" * 60)
        print("COLD START - Loading Augmentor...")
        print("=" * 60)
        augmentor = ProductAugmentor()
        print("Augmentor ready!")
        print("=" * 60)
    return augmentor


def get_downloader():
    """Get or create downloader singleton."""
    global downloader
    if downloader is None:
        downloader = DatasetDownloader()
    return downloader


def get_uploader():
    """Get or create uploader singleton."""
    global uploader
    if uploader is None:
        uploader = ResultUploader()
    return uploader


def handler(job):
    """Main handler for Runpod serverless."""
    job_id = job.get("id", "unknown")
    local_dataset_path = None

    try:
        job_input = job.get("input", {})

        # ========================================
        # INPUT PARAMETERS
        # ========================================
        # Primary: dataset_id for Supabase integration
        dataset_id = job_input.get("dataset_id")

        # Fallback: direct dataset_path for local testing
        dataset_path = job_input.get("dataset_path")

        if not dataset_id and not dataset_path:
            return {"status": "error", "error": "dataset_id or dataset_path is required"}

        # Optional parameters with defaults (from original code)
        syn_target = job_input.get("syn_target", 600)
        real_target = job_input.get("real_target", 400)
        backgrounds_path = job_input.get("backgrounds_path")

        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job_id}")
        print(f"Dataset ID: {dataset_id or 'N/A (using local path)'}")
        print(f"SYN target: {syn_target}, REAL target: {real_target}")
        print(f"{'=' * 60}\n")

        # ========================================
        # STEP 1: DOWNLOAD FROM SUPABASE (if needed)
        # ========================================
        if dataset_id:
            dl = get_downloader()
            up = get_uploader()

            up.update_job_progress(job_id, 10, "Downloading dataset from Supabase")
            local_dataset_path = dl.download_dataset(dataset_id)
        else:
            local_dataset_path = Path(dataset_path)

        # ========================================
        # STEP 2: PROCESS DATASET
        # ========================================
        aug = get_augmentor()

        if dataset_id:
            get_uploader().update_job_progress(job_id, 30, "Augmenting images")

        result = aug.process_dataset(
            dataset_path=str(local_dataset_path),
            syn_target=syn_target,
            real_target=real_target,
            backgrounds_path=backgrounds_path,
        )

        # ========================================
        # STEP 3: UPLOAD RESULTS (if using Supabase)
        # ========================================
        if dataset_id:
            up = get_uploader()
            up.update_job_progress(job_id, 80, "Uploading augmented images")
            upload_stats = up.upload_augmented(local_dataset_path, dataset_id)
            result["upload_stats"] = upload_stats

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: {result['syn_produced']} syn + {result['real_produced']} real generated")
        print(f"{'=' * 60}\n")

        final_result = {
            "status": "success",
            "type": "augmentation",
            "dataset_id": dataset_id,
            "syn_produced": result["syn_produced"],
            "real_produced": result["real_produced"],
            "report": result["report"],
        }

        # Send callback if configured
        if dataset_id:
            get_uploader().send_callback(final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(error_trace)

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }

    finally:
        # Cleanup downloaded files to save disk space
        if local_dataset_path and dataset_id:
            try:
                shutil.rmtree(local_dataset_path.parent, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    print("Starting Augmentation Worker...")
    runpod.serverless.start({"handler": handler})
```

---

**File:** `workers/augmentation/src/augmentor.py`
```python
"""
Product Augmentor using BiRefNet + Albumentations.
Based on: final_augmentor_v3.py - IN-PLACE TOP-UP (IDEMPOTENT)

Features:
- BiRefNet segmentation with GPU half-precision
- 3 augmentation pipelines (light, heavy, real)
- Idempotent: only generates missing images
- Background composition with shadows
- Border detection for resize decision
"""

import os
import re
import cv2
import math
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import albumentations as A
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm
from collections import defaultdict
import json

# ==============================
# CONFIGURATION
# ==============================
TARGET_SIZE = (384, 384)
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']

# Regex patterns for output detection (idempotent)
RE_SYN_OUT = re.compile(r"^syn_(.+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)
RE_REAL_OUT_ANY = re.compile(r"^.+_aug_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def RE_REAL_OUT_FOR_STEM(stem):
    return re.compile(rf"^{re.escape(stem)}_aug_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def list_images(directory: Path):
    """List all images in directory."""
    imgs = []
    if directory.exists():
        for ext in IMAGE_EXTENSIONS:
            imgs.extend(directory.glob(ext))
    return imgs


def ceil_div(a, b):
    return math.ceil(a / b) if b > 0 else 0


class ProductAugmentor:
    """Production-grade augmentor with all optimizations."""

    def __init__(self):
        # Device selection with fallback
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("⚠️ CUDA not available, using CPU (slower)")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Load BiRefNet for high-quality segmentation
        print("\n📦 Loading BiRefNet...")
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet', trust_remote_code=True
        )
        self.birefnet.to(self.device).eval()

        # Use half precision on GPU for memory efficiency
        if self.device == 'cuda':
            self.birefnet.half()
        print("✅ BiRefNet loaded")

        # Initialize transforms
        self.transforms = self._get_transforms()
        print("✅ Transforms initialized")

        # Background images (loaded later)
        self.backgrounds = []

    def _get_transforms(self):
        """Get augmentation transforms - 3 different pipelines."""
        # Light transforms for synthetic
        light = A.Compose([
            A.Affine(shear=(-3, 3), rotate=(-3, 3), p=0.4),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
        ], is_check_shapes=False)

        # Heavy transforms for synthetic
        heavy = A.Compose([
            A.Perspective(scale=(0.01, 0.05), p=0.5),
            A.GridDistortion(p=0.2, distort_limit=0.2),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            A.Affine(shear=(-7, 7), rotate=(-7, 7), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7))
            ], p=0.5),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
        ], is_check_shapes=False)

        # Real image transforms
        real = A.Compose([
            A.OneOf([
                A.Affine(shear=(-5, 5), rotate=(-5, 5), p=0.6),
                A.Perspective(scale=(0.01, 0.06), p=0.4),
                A.GridDistortion(p=0.2, distort_limit=0.2),
                A.OpticalDistortion(distort_limit=0.2, p=0.2),
            ], p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.15, p=0.7),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.03, p=0.6),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.4),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7))
            ], p=0.5),
            A.CoarseDropout(
                max_holes=6, max_height=30, max_width=30,
                min_holes=1, fill_value=220, p=0.3
            ),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
        ], is_check_shapes=False)

        # Real without resize (for already-resized images)
        real_transforms_list = [t for t in real.transforms if not isinstance(t, A.Resize)]
        real_no_resize = A.Compose(real_transforms_list, is_check_shapes=False)

        return {
            'light': light,
            'heavy': heavy,
            'real': real,
            'real_no_resize': real_no_resize
        }

    def detect_resize_by_border(self, image, border_thickness=10, black_threshold=10):
        """Detect if image needs resize by checking black borders."""
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            img = np.array(image)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

            tb = gray[0:border_thickness, :]
            bb = gray[h - border_thickness:h, :]
            lb = gray[:, 0:border_thickness]
            rb = gray[:, w - border_thickness:w]

            all_black = (np.mean(tb) < black_threshold and np.mean(bb) < black_threshold
                         and np.mean(lb) < black_threshold and np.mean(rb) < black_threshold)

            return {'needs_resize': all_black, 'is_resized': not all_black}
        except Exception as e:
            print(f"  Resize detection error: {e}")
            return {'needs_resize': False, 'is_resized': False}

    def segment_with_birefnet(self, img_pil):
        """BiRefNet segmentation with GPU half-precision."""
        try:
            tfm = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            with torch.no_grad():
                inp = tfm(img_pil.copy()).unsqueeze(0).to(self.device)
                if self.device == 'cuda':
                    inp = inp.half()
                preds = self.birefnet(inp)[-1].sigmoid()
                mask = (preds[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

            mask_pil = Image.fromarray(mask).resize(img_pil.size, Image.BILINEAR)
            rgba = np.array(img_pil.convert("RGBA"))
            rgba[:, :, 3] = np.array(mask_pil)
            return Image.fromarray(rgba, 'RGBA')
        except Exception as e:
            print(f"  Segmentation error: {e}")
            return None

    def augment_real_image(self, img_path):
        """Real image augmentation with resize detection."""
        try:
            img = Image.open(img_path).convert("RGB")
            resize_info = self.detect_resize_by_border(img)

            if resize_info['needs_resize']:
                # Raw image - segment + crop + augment
                rgba = self.segment_with_birefnet(img)
                if rgba is None:
                    return None

                bbox = rgba.getbbox()
                if bbox:
                    cropped = rgba.crop(bbox).convert("RGB")
                    augmented = self.transforms['real'](image=np.array(cropped))['image']
                else:
                    augmented = self.transforms['real'](image=np.array(img))['image']
            else:
                # Clean image - direct augment (no resize)
                augmented = self.transforms['real_no_resize'](image=np.array(img))['image']

            return Image.fromarray(augmented)
        except Exception as e:
            print(f"  Real aug error for {img_path.name}: {e}")
            return None

    def augment_synthetic_image(self, img_path, bg_image):
        """Synthetic image augmentation with background composition and shadows."""
        try:
            img = Image.open(img_path).convert("RGB")

            # Select and apply transform
            transform = self.transforms['heavy'] if random.random() < 0.5 else self.transforms['light']
            augmented = Image.fromarray(transform(image=np.array(img))['image'])

            # Segment + crop
            rgba = self.segment_with_birefnet(augmented)
            if rgba is None:
                return None

            bbox = rgba.getbbox()
            if not bbox:
                return None

            product = rgba.crop(bbox)  # RGBA

            # Background composition
            composed = bg_image.resize(product.size, Image.LANCZOS).convert("RGB")
            composed_rgba = composed.convert("RGBA")

            # Shadow effect
            if random.random() < 0.7:
                product_alpha = product.getchannel('A')
                shadow = Image.new('RGBA', product.size, (0, 0, 0, random.randint(80, 140)))
                shadow.putalpha(product_alpha)
                for _ in range(random.randint(5, 8)):
                    shadow = shadow.filter(ImageFilter.BLUR)
                temp = Image.new('RGBA', composed.size, (0, 0, 0, 0))
                temp.paste(shadow, (5, 5), shadow)
                composed_rgba = Image.alpha_composite(composed_rgba, temp)

            # Paste product
            temp_product = Image.new('RGBA', composed.size, (0, 0, 0, 0))
            temp_product.paste(product, (0, 0), product)
            composed_rgba = Image.alpha_composite(composed_rgba, temp_product)

            final = composed_rgba.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            return final
        except Exception as e:
            print(f"  Synthetic aug error for {img_path.name}: {e}")
            return None

    def load_backgrounds(self, backgrounds_path):
        """Load background images for synthetic augmentation."""
        self.backgrounds = []
        if backgrounds_path:
            bg_dir = Path(backgrounds_path)
            if bg_dir.exists():
                for ext in IMAGE_EXTENSIONS:
                    for p in bg_dir.glob(ext):
                        try:
                            self.backgrounds.append(Image.open(p).convert("RGB"))
                        except Exception:
                            pass
        print(f"🖼️ Background images loaded: {len(self.backgrounds)}")

    def get_random_background(self):
        """Get random background or generate solid color."""
        if self.backgrounds:
            return random.choice(self.backgrounds)
        return Image.new('RGB', (512, 512), (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        ))

    def max_index_for_syn_stem(self, upc_root, stem):
        """Find max index for syn_{stem}_{idx}.ext files."""
        max_idx = -1
        for p in list_images(upc_root):
            m = re.match(rf"^syn_{re.escape(stem)}_(\d+)\.(jpg|jpeg|png|webp)$", p.name, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx

    def max_index_for_real_stem(self, real_dir, stem):
        """Find max index for {stem}_aug_{idx}.ext files."""
        pat = RE_REAL_OUT_FOR_STEM(stem)
        max_idx = -1
        for p in list_images(real_dir):
            m = pat.match(p.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx

    def process_upc(self, upc_dir, syn_target, real_target):
        """
        Process single UPC - IDEMPOTENT TOP-UP.
        Only generates missing images to reach target.
        """
        upc_root = Path(upc_dir)
        real_dir = upc_root / "real"
        real_dir.mkdir(exist_ok=True)

        # --- SYN SOURCE & COUNT ---
        root_imgs_all = [p for p in list_images(upc_root) if p.parent == upc_root]
        syn_sources = [p for p in root_imgs_all if not p.name.lower().startswith('syn_')]
        current_syn_total = len(root_imgs_all)
        missing_syn = max(0, syn_target - current_syn_total)

        # --- REAL SOURCE & COUNT ---
        real_imgs_all = list_images(real_dir)
        real_sources = [p for p in real_imgs_all if not RE_REAL_OUT_ANY.match(p.name)]
        current_real_total = len(real_imgs_all)
        missing_real = max(0, real_target - current_real_total)

        produced_syn = 0
        produced_real = 0

        # ========== SYN TOP-UP ==========
        if missing_syn > 0 and len(syn_sources) > 0:
            per_src = ceil_div(missing_syn, len(syn_sources))
            for src in syn_sources:
                if produced_syn >= missing_syn:
                    break
                bg = self.get_random_background()
                stem = src.stem
                start_idx = self.max_index_for_syn_stem(upc_root, stem) + 1
                quota = min(per_src, missing_syn - produced_syn)

                for k in range(quota):
                    aug = self.augment_synthetic_image(src, bg)
                    if aug is None:
                        continue
                    out_name = f"syn_{stem}_{start_idx + k:03d}.jpg"
                    out_path = upc_root / out_name
                    try:
                        aug.save(out_path, quality=95)
                        produced_syn += 1
                    except Exception as e:
                        print(f"  ⚠️ Syn save error {out_name}: {e}")
                    if produced_syn >= missing_syn:
                        break

        # ========== REAL TOP-UP ==========
        if missing_real > 0 and len(real_sources) > 0:
            per_src = ceil_div(missing_real, len(real_sources))
            for src in real_sources:
                if produced_real >= missing_real:
                    break
                stem = src.stem
                start_idx = self.max_index_for_real_stem(real_dir, stem) + 1
                quota = min(per_src, missing_real - produced_real)

                for k in range(quota):
                    aug = self.augment_real_image(src)
                    if aug is None:
                        continue
                    out_name = f"{stem}_aug_{start_idx + k:03d}.jpg"
                    out_path = real_dir / out_name
                    try:
                        aug.save(out_path, quality=95)
                        produced_real += 1
                    except Exception as e:
                        print(f"  ⚠️ Real save error {out_name}: {e}")
                    if produced_real >= missing_real:
                        break

        return {
            "syn_sources": len(syn_sources),
            "real_sources": len(real_sources),
            "current_syn": current_syn_total,
            "current_real": current_real_total,
            "produced_syn": produced_syn,
            "produced_real": produced_real,
            "final_syn": current_syn_total + produced_syn,
            "final_real": current_real_total + produced_real,
        }

    def process_dataset(self, dataset_path, syn_target, real_target, backgrounds_path=None):
        """Process entire dataset - all UPCs."""
        base = Path(dataset_path)
        if not base.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Load backgrounds
        self.load_backgrounds(backgrounds_path)

        totals = defaultdict(int)
        per_upc_logs = []

        # Find all splits (train, test, valid)
        splits = [d for d in base.iterdir() if d.is_dir() and d.name in ('train', 'test', 'valid')]

        for split_dir in splits:
            print(f"\n{'=' * 28}  {split_dir.name.upper()}  {'=' * 28}")

            # Find all UPC directories
            upc_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

            for upc in tqdm(upc_dirs, desc=f"{split_dir.name} UPC"):
                try:
                    stats = self.process_upc(upc, syn_target, real_target)
                    totals[f"{split_dir.name}_syn_produced"] += stats["produced_syn"]
                    totals[f"{split_dir.name}_real_produced"] += stats["produced_real"]
                    totals["syn_produced"] += stats["produced_syn"]
                    totals["real_produced"] += stats["produced_real"]

                    per_upc_logs.append({
                        "split": split_dir.name,
                        "upc": upc.name,
                        **stats
                    })
                except Exception as e:
                    print(f"❌ UPC error: {upc.name} -> {e}")

        # Generate report
        report = {
            "totals": dict(totals),
            "items": per_upc_logs,
            "target_syn_per_upc": syn_target,
            "target_real_per_upc": real_target
        }

        # Save report
        report_path = base / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Report saved: {report_path}")

        return {
            "syn_produced": totals["syn_produced"],
            "real_produced": totals["real_produced"],
            "report": report,
        }
```

---

**File:** `workers/augmentation/Dockerfile`
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables for thread limiting
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Supabase configuration (set at runtime)
# ENV SUPABASE_URL=
# ENV SUPABASE_SERVICE_KEY=
# ENV CALLBACK_URL=

WORKDIR /app/src
CMD ["python", "handler.py"]
```

---

**File:** `workers/augmentation/requirements.txt`
```
runpod
torch>=2.0.0
torchvision
transformers
albumentations>=1.3.0
opencv-python-headless
pillow
numpy
tqdm
supabase>=2.0.0
httpx>=0.25.0
```

---

**Acceptance Criteria:**
- [ ] Thread limiting (`OMP_NUM_THREADS=1`) active
- [ ] BiRefNet loads with GPU half-precision
- [ ] 3 transform pipelines working (light, heavy, real)
- [ ] Idempotent: running twice produces no extra images
- [ ] Border detection correctly identifies resize needs
- [ ] Synthetic images have shadow effects
- [ ] Background composition working
- [ ] JSON report generated after processing
- [ ] Per-UPC quota distribution correct

---

### TASK-032: Migrate Training Worker
**Prerequisites:** TASK-030
**Estimated Complexity:** Very High

**Description:**
Training pipeline'ını Runpod worker olarak yapılandır. Eski koddan tüm optimizasyonlar korunmalı.

**Reference:** `Eski kodlar/train_optimized_v14.py`

**Target:** `workers/training/`

**Key Optimizations (from original code):**
- Mixed Precision Training (AMP) - 2-3x faster
- Memory optimization (cleanup_memory, PYTORCH_CUDA_ALLOC_CONF)
- Cached processor for efficiency
- Enhanced ArcFace with label smoothing
- Layer-wise Learning Rate Decay (LLRD)
- Loss tracking for plateau/divergence detection
- Domain-aware triplet sampler with hard negative mining
- EMA (Exponential Moving Average)
- NaN explosion detection + emergency checkpoints
- Cosine annealing with warmup
- TensorBoard logging
- Disk space monitoring

---

**File:** `workers/training/src/supabase_client.py`
```python
"""Supabase client for training worker - downloads dataset and uploads checkpoints."""

import os
import json
from pathlib import Path
from supabase import create_client, Client
import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")


def get_supabase() -> Client:
    """Get Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class TrainingDataDownloader:
    """Downloads augmented dataset from Supabase for training."""

    def __init__(self, local_base: Path = Path("/tmp/training_data")):
        self.client = get_supabase()
        self.local_base = local_base
        self.local_base.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str) -> Path:
        """
        Download augmented images for a dataset.

        Structure created:
        /tmp/training_data/{dataset_id}/
          train/
            {barcode_1}/
              frame_0000.png
              syn_frame_0000_001.jpg
              ...
              real/
                real_image_1.jpg
                real_image_1_aug_001.jpg
        """
        print(f"\n📥 Downloading training data for dataset: {dataset_id}")

        # Get dataset products
        response = self.client.table("dataset_products").select(
            "product_id, products(id, barcode, frames_path)"
        ).eq("dataset_id", dataset_id).execute()

        products = response.data
        if not products:
            raise ValueError(f"Dataset {dataset_id} has no products")

        print(f"   Found {len(products)} products")

        dataset_dir = self.local_base / dataset_id / "train"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for item in products:
            product = item.get("products", {})
            barcode = product.get("barcode", product.get("id"))
            product_dir = dataset_dir / barcode
            product_dir.mkdir(parents=True, exist_ok=True)
            real_dir = product_dir / "real"
            real_dir.mkdir(exist_ok=True)

            # Download original frames
            self._download_frames(barcode, product_dir)

            # Download augmented synthetic images
            self._download_augmented(dataset_id, barcode, product_dir)

            # Download real images (including augmented)
            self._download_real_images(product.get("id"), real_dir)
            self._download_augmented_real(dataset_id, barcode, real_dir)

        print(f"   ✅ Training data downloaded to: {dataset_dir.parent}")
        return dataset_dir.parent

    def _download_frames(self, barcode: str, target_dir: Path):
        """Download original frames from frames bucket."""
        try:
            bucket = "frames"
            files = self.client.storage.from_(bucket).list(barcode)
            for f in files:
                if f.get("name", "").endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    data = self.client.storage.from_(bucket).download(f"{barcode}/{f['name']}")
                    with open(target_dir / f['name'], 'wb') as fp:
                        fp.write(data)
        except Exception as e:
            print(f"   ⚠️ Frame download error for {barcode}: {e}")

    def _download_augmented(self, dataset_id: str, barcode: str, target_dir: Path):
        """Download augmented synthetic images."""
        try:
            bucket = "augmented-images"
            path = f"{dataset_id}/{barcode}"
            files = self.client.storage.from_(bucket).list(path)
            for f in files:
                name = f.get("name", "")
                if name.endswith(('.jpg', '.jpeg', '.png')) and not f.get("metadata", {}).get("isFolder"):
                    data = self.client.storage.from_(bucket).download(f"{path}/{name}")
                    with open(target_dir / name, 'wb') as fp:
                        fp.write(data)
        except Exception as e:
            print(f"   ⚠️ Augmented download error: {e}")

    def _download_real_images(self, product_id: str, target_dir: Path):
        """Download matched real images."""
        try:
            response = self.client.table("product_images").select("*").eq(
                "product_id", product_id
            ).eq("image_type", "real").execute()

            bucket = "real-images"
            for img in response.data:
                image_path = img.get("image_path")
                if not image_path:
                    continue
                try:
                    data = self.client.storage.from_(bucket).download(image_path)
                    with open(target_dir / Path(image_path).name, 'wb') as fp:
                        fp.write(data)
                except Exception:
                    pass
        except Exception as e:
            print(f"   ⚠️ Real images download error: {e}")

    def _download_augmented_real(self, dataset_id: str, barcode: str, target_dir: Path):
        """Download augmented real images."""
        try:
            bucket = "augmented-images"
            path = f"{dataset_id}/{barcode}/real"
            files = self.client.storage.from_(bucket).list(path)
            for f in files:
                name = f.get("name", "")
                if name.endswith(('.jpg', '.jpeg', '.png')):
                    data = self.client.storage.from_(bucket).download(f"{path}/{name}")
                    with open(target_dir / name, 'wb') as fp:
                        fp.write(data)
        except Exception:
            pass  # May not exist


class CheckpointUploader:
    """Uploads model checkpoints to Supabase Storage."""

    def __init__(self):
        self.client = get_supabase()

    def upload_checkpoint(self, checkpoint_path: Path, dataset_id: str, job_id: str) -> str:
        """Upload model checkpoint to Supabase Storage."""
        print(f"\n📤 Uploading checkpoint...")

        bucket = "models"
        storage_path = f"{dataset_id}/{job_id}/{checkpoint_path.name}"

        with open(checkpoint_path, 'rb') as f:
            self.client.storage.from_(bucket).upload(storage_path, f.read())

        url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"
        print(f"   ✅ Checkpoint uploaded: {url}")
        return url

    def upload_tensorboard(self, tb_dir: Path, dataset_id: str, job_id: str):
        """Upload TensorBoard logs."""
        bucket = "models"
        for f in tb_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(tb_dir)
                storage_path = f"{dataset_id}/{job_id}/tensorboard/{rel_path}"
                try:
                    with open(f, 'rb') as fp:
                        self.client.storage.from_(bucket).upload(storage_path, fp.read())
                except Exception:
                    pass

    def update_job_progress(self, job_id: str, progress: int, current_step: str):
        """Update job progress."""
        try:
            self.client.table("jobs").update({
                "progress": progress,
                "current_step": current_step,
            }).eq("runpod_job_id", job_id).execute()
        except Exception as e:
            print(f"   ⚠️ Progress update error: {e}")

    def create_model_artifact(self, job_id: str, checkpoint_url: str, metrics: dict):
        """Create model artifact record in database."""
        try:
            self.client.table("model_artifacts").insert({
                "job_id": job_id,
                "name": f"model_{job_id[:8]}",
                "version": "v1",
                "checkpoint_url": checkpoint_url,
                "embedding_dim": metrics.get("proj_dim", 1024),
                "num_classes": metrics.get("num_classes"),
                "final_loss": metrics.get("final_loss"),
                "is_active": False,
            }).execute()
        except Exception as e:
            print(f"   ⚠️ Model artifact creation error: {e}")

    def send_callback(self, result: dict):
        """Send completion callback to backend."""
        if not CALLBACK_URL:
            return
        try:
            httpx.post(
                f"{CALLBACK_URL}/api/v1/webhooks/runpod",
                json=result,
                timeout=30,
            )
        except Exception as e:
            print(f"   ⚠️ Callback error: {e}")
```

---

**File:** `workers/training/src/dataset_splitter.py`
```python
"""
Dataset Splitter for Training Worker.
Based on: split_dataset_to_train_valid_test.py

Splits dataset into train/valid/test sets while preserving UPC (barcode) integrity.
Each UPC folder stays together - not split across sets.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple


class DatasetSplitter:
    """
    Splits a dataset directory into train/valid/test sets.

    Preserves UPC integrity: all images for a product stay in the same split.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        valid_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        """
        Initialize splitter with split ratios.

        Args:
            train_ratio: Fraction for training set (default 0.80)
            valid_ratio: Fraction for validation set (default 0.10)
            test_ratio: Fraction for test set (default 0.10)
            seed: Random seed for reproducibility (default 42)
        """
        # Validate ratios sum to 1.0
        total = train_ratio + valid_ratio + test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, source_dir: Path, output_dir: Path) -> Path:
        """
        Split dataset from source_dir into train/valid/test in output_dir.

        Args:
            source_dir: Directory containing UPC folders (barcode folders with images)
            output_dir: Directory to create train/valid/test splits

        Returns:
            Path to output_dir containing train/valid/test subdirectories
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)

        if not source_dir.is_dir():
            raise ValueError(f"Source directory not found: {source_dir}")

        # Find all UPC folders (barcode directories)
        upc_folders = [d for d in source_dir.iterdir() if d.is_dir()]

        if not upc_folders:
            raise ValueError(f"No UPC folders found in {source_dir}")

        print(f"\n📊 Dataset Split")
        print(f"   Source: {source_dir}")
        print(f"   Total UPCs: {len(upc_folders)}")
        print(f"   Ratios: train={self.train_ratio:.0%}, valid={self.valid_ratio:.0%}, test={self.test_ratio:.0%}")

        # Shuffle UPCs with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(upc_folders)

        # Calculate split indices
        total_upcs = len(upc_folders)
        train_end = int(total_upcs * self.train_ratio)
        valid_end = train_end + int(total_upcs * self.valid_ratio)

        train_upcs = upc_folders[:train_end]
        valid_upcs = upc_folders[train_end:valid_end]
        test_upcs = upc_folders[valid_end:]

        # Handle rounding - assign any remaining to test
        all_assigned = set(train_upcs + valid_upcs + test_upcs)
        unassigned = [upc for upc in upc_folders if upc not in all_assigned]
        test_upcs.extend(unassigned)

        print(f"   Split: train={len(train_upcs)}, valid={len(valid_upcs)}, test={len(test_upcs)}")

        # Create output directories and copy files
        split_map = {
            'train': train_upcs,
            'valid': valid_upcs,
            'test': test_upcs,
        }

        for split_name, upc_list in split_map.items():
            if not upc_list:
                continue

            split_path = output_dir / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            for upc_path in upc_list:
                dest = split_path / upc_path.name
                shutil.copytree(str(upc_path), str(dest), dirs_exist_ok=True)

        # Log statistics
        stats = self._get_split_stats(output_dir)
        print(f"   Images: train={stats['train']}, valid={stats['valid']}, test={stats['test']}")
        print(f"   ✅ Split complete: {output_dir}")

        return output_dir

    def _get_split_stats(self, output_dir: Path) -> dict:
        """Count images in each split."""
        stats = {}
        for split in ['train', 'valid', 'test']:
            split_path = output_dir / split
            if split_path.exists():
                count = sum(1 for f in split_path.rglob('*')
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])
                stats[split] = count
            else:
                stats[split] = 0
        return stats
```

---

**File:** `workers/training/src/handler.py`
```python
"""
Runpod Serverless Handler for Training.
Based on: train_optimized_v14.py

INTEGRATED WITH SUPABASE:
- Receives dataset_id from API
- Downloads augmented images from Supabase Storage
- Generates embeddings and trains model
- Uploads checkpoint back to Supabase Storage
"""

import os
import sys
import shutil

# CRITICAL: Environment setup BEFORE imports
def setup_environment():
    """Production environment setup."""
    import tempfile
    from pathlib import Path

    workspace_tmp = Path("/workspace/tmp")
    workspace_tmp.mkdir(parents=True, exist_ok=True)

    env_vars = {
        'TMPDIR': str(workspace_tmp),
        'TEMP': str(workspace_tmp),
        'TMP': str(workspace_tmp),
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    tempfile.tempdir = str(workspace_tmp)
    return workspace_tmp


TEMP_DIR = setup_environment()

import runpod
import traceback
from pathlib import Path
from trainer import DomainAdaptationTrainer
from supabase_client import TrainingDataDownloader, CheckpointUploader


trainer = None
downloader = None
uploader = None


def get_trainer():
    """Get or create trainer singleton."""
    global trainer
    if trainer is None:
        print("=" * 60)
        print("COLD START - Loading Trainer...")
        print("=" * 60)
        trainer = DomainAdaptationTrainer()
        print("Trainer ready!")
        print("=" * 60)
    return trainer


def get_downloader():
    global downloader
    if downloader is None:
        downloader = TrainingDataDownloader()
    return downloader


def get_uploader():
    global uploader
    if uploader is None:
        uploader = CheckpointUploader()
    return uploader


def handler(job):
    """Main handler for Runpod serverless."""
    job_id = job.get("id", "unknown")
    local_data_path = None

    try:
        job_input = job.get("input", {})

        # ========================================
        # INPUT PARAMETERS
        # ========================================
        # Primary: dataset_id for Supabase integration
        dataset_id = job_input.get("dataset_id")

        # Fallback: direct embeddings_path for local testing
        embeddings_path = job_input.get("embeddings_path")

        if not dataset_id and not embeddings_path:
            return {"status": "error", "error": "dataset_id or embeddings_path is required"}

        output_dir = Path(job_input.get("output_dir", "/workspace/outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training config (with defaults from v14)
        config = {
            "model_name": job_input.get("model_name", "facebook/dinov2-large"),
            "proj_dim": job_input.get("proj_dim", 1024),
            "epochs": job_input.get("epochs", 30),
            "batch_size": job_input.get("batch_size", 16),
            "lr": job_input.get("lr", 2e-5),
            "weight_decay": job_input.get("weight_decay", 0.01),
            "llrd_decay": job_input.get("llrd_decay", 0.9),
            "warmup_epochs": job_input.get("warmup_epochs", 3),
            "grad_clip": job_input.get("grad_clip", 1.0),
            "label_smoothing": job_input.get("label_smoothing", 0.2),
            "domain_aware_ratio": job_input.get("domain_aware_ratio", 0.57),
            "hard_negative_pool_size": job_input.get("hard_negative_pool_size", 5),
            "use_hardest_negatives": job_input.get("use_hardest_negatives", True),
            "image_size": job_input.get("image_size", 518),  # DINOv2 default
            "num_workers": job_input.get("num_workers", 4),
            "use_mixed_precision": job_input.get("use_mixed_precision", True),
        }

        # Dataset split config
        split_config = {
            "train_ratio": job_input.get("train_ratio", 0.80),
            "valid_ratio": job_input.get("valid_ratio", 0.10),
            "test_ratio": job_input.get("test_ratio", 0.10),
            "split_seed": job_input.get("split_seed", 42),
        }

        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job_id}")
        print(f"Dataset ID: {dataset_id or 'N/A (using local path)'}")
        print(f"Config: {config}")
        print(f"Split: train={split_config['train_ratio']:.0%}, valid={split_config['valid_ratio']:.0%}, test={split_config['test_ratio']:.0%}")
        print(f"{'=' * 60}\n")

        # ========================================
        # STEP 1: DOWNLOAD FROM SUPABASE (if needed)
        # ========================================
        if dataset_id:
            dl = get_downloader()
            up = get_uploader()

            up.update_job_progress(job_id, 5, "Downloading training data from Supabase")
            local_data_path = dl.download_dataset(dataset_id)

            # ========================================
            # STEP 2: SPLIT DATASET INTO TRAIN/VALID/TEST
            # ========================================
            from dataset_splitter import DatasetSplitter

            up.update_job_progress(job_id, 10, "Splitting dataset into train/valid/test")
            splitter = DatasetSplitter(
                train_ratio=split_config["train_ratio"],
                valid_ratio=split_config["valid_ratio"],
                test_ratio=split_config["test_ratio"],
                seed=split_config["split_seed"],
            )
            split_dir = splitter.split(local_data_path / "train", output_dir / "split")
            print(f"   ✅ Dataset split completed: {split_dir}")

            # ========================================
            # STEP 3: GENERATE EMBEDDINGS
            # ========================================
            up.update_job_progress(job_id, 20, "Generating embeddings")
            embeddings_path = get_trainer().generate_embeddings(
                dataset_path=str(split_dir),  # Use split directory
                output_path=output_dir / "embeddings.pt",
                model_name=config["model_name"],
                image_size=config["image_size"],
            )
            up.update_job_progress(job_id, 40, "Starting training")

        # ========================================
        # STEP 4: TRAIN MODEL
        # ========================================
        t = get_trainer()
        result = t.train(
            embeddings_path=str(embeddings_path),
            output_dir=str(output_dir),
            **config
        )

        # ========================================
        # STEP 5: UPLOAD RESULTS (if using Supabase)
        # ========================================
        checkpoint_url = None
        if dataset_id:
            up = get_uploader()
            up.update_job_progress(job_id, 90, "Uploading checkpoint")

            checkpoint_path = Path(result["checkpoint_path"])
            checkpoint_url = up.upload_checkpoint(checkpoint_path, dataset_id, job_id)

            # Upload TensorBoard logs
            tb_dir = Path(result["metrics_path"])
            if tb_dir.exists():
                up.upload_tensorboard(tb_dir, dataset_id, job_id)

            # Create model artifact
            up.create_model_artifact(job_id, checkpoint_url, {
                "proj_dim": config["proj_dim"],
                "num_classes": result.get("num_classes"),
                "final_loss": result["final_loss"],
            })

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Training completed in {result['epochs_completed']} epochs")
        print(f"Final loss: {result['final_loss']:.6f}")
        print(f"{'=' * 60}\n")

        final_result = {
            "status": "success",
            "type": "training",
            "dataset_id": dataset_id,
            "epochs_completed": result["epochs_completed"],
            "final_loss": result["final_loss"],
            "checkpoint_url": checkpoint_url or result["checkpoint_path"],
            "metrics_path": result["metrics_path"],
        }

        # Send callback if configured
        if dataset_id:
            get_uploader().send_callback(final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(error_trace)

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }

    finally:
        # Cleanup downloaded files
        if local_data_path and dataset_id:
            try:
                shutil.rmtree(local_data_path.parent, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    print("Starting Training Worker...")
    runpod.serverless.start({"handler": handler})
```

---

**File:** `workers/training/src/trainer.py`
```python
"""
Domain Adaptation Trainer.
Based on: train_optimized_v14.py

OPTIMIZATIONS:
- Mixed Precision Training (AMP) for 2-3x speedup
- Optimized single dropout (0.15)
- Loss tracking with plateau/divergence detection
- Aggressive cosine annealing
- All NaN explosion protections
"""

import os
import gc
import math
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import json
from datetime import datetime

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler, autocast

from torchvision import transforms

# Safe imports
try:
    from torch_ema import ExponentialMovingAverage
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False

# PyTorch optimizations
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global caches
_PROCESSOR_CACHE: Dict[str, Any] = {}


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_disk_space(path: Path, min_gb: float = 10.0) -> bool:
    """Disk space verification."""
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)
        return available_gb >= min_gb
    except Exception:
        return False


def get_processor_cached(model_name: str):
    """Cached processor retrieval."""
    proc = _PROCESSOR_CACHE.get(model_name)
    if proc is None:
        try:
            proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        except TypeError:
            proc = AutoImageProcessor.from_pretrained(model_name)
        _PROCESSOR_CACHE[model_name] = proc
    return proc


class GeMPooling(nn.Module):
    """Generalized Mean Pooling with stability."""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)


class EnhancedArcFaceLoss(nn.Module):
    """Enhanced ArcFace Loss with label smoothing and stability."""
    def __init__(self, in_features, out_features, s=30.0, m=0.30, label_smoothing=0.0):
        super().__init__()
        self.s, self.m = s, m
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, label):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, device=emb.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        out *= self.s

        # Label smoothing
        if self.training and self.label_smoothing > 0:
            n_classes = out.size(1)
            with torch.no_grad():
                smooth_targets = torch.full_like(out, self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, label.unsqueeze(1), 1.0 - self.label_smoothing)

            log_probs = F.log_softmax(out, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1).mean()
            return loss

        return F.cross_entropy(out, label)


class DinoWithArcFaceHead(nn.Module):
    """DINOv2 + ArcFace model with OPTIMIZED SINGLE DROPOUT."""
    def __init__(self, model_name: str, num_classes: int, proj_dim: int = 768, label_smoothing: float = 0.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        emb_dim = getattr(self.backbone.config, "hidden_size", 768)
        self.pool = GeMPooling()

        # Optimized: Single strategic dropout
        self.dropout = nn.Dropout(0.15)

        self.proj = nn.Linear(emb_dim, proj_dim)
        self.head = EnhancedArcFaceLoss(proj_dim, num_classes, label_smoothing=label_smoothing)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, images, labels=None):
        out = self.backbone(pixel_values=images, output_hidden_states=False).last_hidden_state
        patch_tokens = out[:, 1:, :]  # Remove CLS
        B, N, C = patch_tokens.shape

        # GeM pooling with fallback
        expected_grid_size = 27
        if N == expected_grid_size * expected_grid_size:
            H = W = expected_grid_size
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            pooled = self.pool(patch_tokens).flatten(1)
        else:
            pooled = patch_tokens.mean(dim=1)

        pooled = self.dropout(pooled)
        emb = self.proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)

        if labels is not None:
            loss = self.head(emb, labels)
            return emb, loss
        return emb


class LossTracker:
    """Track training loss for plateau and divergence detection."""
    def __init__(self, window_size=10):
        self.losses = []
        self.window_size = window_size

    def add(self, loss):
        self.losses.append(loss)
        if len(self.losses) > 100:
            self.losses.pop(0)

    def is_plateauing(self):
        if len(self.losses) < self.window_size * 2:
            return False
        recent_avg = np.mean(self.losses[-self.window_size:])
        previous_avg = np.mean(self.losses[-self.window_size*2:-self.window_size])
        improvement = (previous_avg - recent_avg) / (previous_avg + 1e-10)
        return improvement < 0.01

    def is_diverging(self):
        if len(self.losses) < 5:
            return False
        recent_trend = np.mean(self.losses[-3:])
        previous_trend = np.mean(self.losses[-6:-3])
        return recent_trend > previous_trend * 1.05


class OptimizedDomainAwareTripletSampler(Sampler):
    """Domain-aware triplet sampler with hard negative mining."""
    def __init__(self, embeddings_data, batch_size: int, domain_aware_ratio=0.57,
                 hard_negative_pool_size=5, use_hardest_negatives=True):
        super().__init__(None)
        self.batch_size = int(batch_size)
        self.triplets_per_batch = self.batch_size // 3

        self.domain_aware_ratio = domain_aware_ratio
        self.hard_negative_pool_size = hard_negative_pool_size
        self.use_hardest_negatives = use_hardest_negatives

        # Load embeddings
        self.emb = F.normalize(
            torch.from_numpy(embeddings_data['embeddings'])
            if isinstance(embeddings_data['embeddings'], np.ndarray)
            else embeddings_data['embeddings'].float(),
            dim=1
        ).cpu()

        self.image_paths = np.array(embeddings_data['image_paths'])
        self.labels = np.array(embeddings_data['labels'])
        self.domains = np.array(embeddings_data['domains'])
        self.unique_labels = np.unique(self.labels)
        self.label_to_indices = {lbl: np.where(self.labels == lbl)[0] for lbl in self.unique_labels}

        # Domain classification
        is_real_mask = np.array([d == 'real' for d in self.domains])
        self.label_to_typed_indices = defaultdict(lambda: {'real': [], 'synthetic': []})

        for i, lbl in enumerate(self.labels):
            domain = 'real' if is_real_mask[i] else 'synthetic'
            self.label_to_typed_indices[lbl][domain].append(i)

        # Find domain-aware capable labels
        self.domain_aware_labels = [
            lbl for lbl, domains in self.label_to_typed_indices.items()
            if len(domains['real']) > 0 and len(domains['synthetic']) > 0
        ]

        self.standard_labels = [
            lbl for lbl, indices in self.label_to_indices.items()
            if len(indices) > 1
        ]

        print(f"🎯 Domain-aware triplet sampler:")
        print(f"   Total UPCs: {len(self.unique_labels)}")
        print(f"   Domain-aware capable: {len(self.domain_aware_labels)}")
        print(f"   Domain-aware ratio: {domain_aware_ratio:.1%}")

        # Build hard negative pools
        self._build_hard_negative_pools()
        self.num_samples = len(self.labels)

    def _build_hard_negative_pools(self):
        """Build hard negative mining pools."""
        with torch.no_grad():
            centroids = []
            order = []
            for lbl in self.unique_labels:
                indices = self.label_to_indices[lbl]
                centroid = self.emb[indices].mean(0)
                centroids.append(F.normalize(centroid.unsqueeze(0), dim=1).squeeze(0))
                order.append(lbl)

            self.centroids = torch.stack(centroids, dim=0)
            self.order = np.array(order)
            similarity_matrix = self.centroids @ self.centroids.T
            similarity_matrix.fill_diagonal_(-1.0)

            k_neighbors = min(50, similarity_matrix.shape[1] - 1)
            topk_indices = torch.topk(similarity_matrix, k=k_neighbors, dim=1).indices.cpu().numpy()

        self.hard_negative_pools = {
            lbl: self.order[topk_indices[i]][:self.hard_negative_pool_size].tolist()
            for i, lbl in enumerate(self.order)
        }

    def __iter__(self):
        all_triplet_indices = []
        num_triplets_needed = (self.num_samples // self.batch_size) * self.triplets_per_batch

        for _ in range(num_triplets_needed):
            if self.domain_aware_labels and random.random() < self.domain_aware_ratio:
                anchor_label = random.choice(self.domain_aware_labels)
                domains = self.label_to_typed_indices[anchor_label]

                if len(domains['real']) > 0 and len(domains['synthetic']) > 0:
                    anchor_idx = random.choice(domains['real'])
                    positive_idx = random.choice(domains['synthetic'])
                else:
                    indices = self.label_to_indices[anchor_label]
                    anchor_idx, positive_idx = random.sample(list(indices), 2)
            else:
                anchor_label = random.choice(self.standard_labels)
                indices = self.label_to_indices[anchor_label]
                anchor_idx, positive_idx = random.sample(list(indices), 2)

            # Hard negative selection
            hard_pool = self.hard_negative_pools.get(anchor_label, [])
            if hard_pool:
                negative_label = hard_pool[0] if self.use_hardest_negatives else random.choice(hard_pool)
            else:
                negative_label = random.choice([l for l in self.standard_labels if l != anchor_label])

            negative_indices = self.label_to_indices[negative_label]
            negative_idx = random.choice(list(negative_indices))

            all_triplet_indices.extend([anchor_idx, positive_idx, negative_idx])

        random.shuffle(all_triplet_indices)
        for i in range(0, len(all_triplet_indices), self.batch_size):
            batch = all_triplet_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        return self.num_samples // self.batch_size


class ImageDataset(Dataset):
    """Production-grade image dataset."""
    def __init__(self, image_paths: List[str], labels: List[int], model_name: str,
                 image_size: int, is_train: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.model_name = model_name
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            x = self._preprocess(image)
            return x, self.labels[idx]
        except Exception:
            return None, None

    def _preprocess(self, image: Image.Image):
        proc = get_processor_cached(self.model_name)

        if self.is_train:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
            ])
            return train_transforms(image)
        else:
            val_transforms = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=proc.image_mean, std=proc.image_std)
            ])
            return val_transforms(image)


def collate_fn(batch):
    """Custom collate with None handling."""
    batch = [(x, y) for (x, y) in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class DomainAdaptationTrainer:
    """Main trainer class with all optimizations."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")

    def train(self, embeddings_path: str, output_dir: str, **config):
        """Main training method."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load embeddings
        emb_data = self._load_embeddings(Path(embeddings_path))
        num_classes = len(np.unique(emb_data['labels']))

        print(f"📊 Training samples: {len(emb_data['labels'])}")
        print(f"   Unique UPCs: {num_classes}")

        # Create model
        model = DinoWithArcFaceHead(
            config["model_name"],
            num_classes,
            proj_dim=config["proj_dim"],
            label_smoothing=config["label_smoothing"]
        ).to(self.device)

        if self.device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)

        # Dataset & Sampler
        dataset = ImageDataset(
            list(emb_data['image_paths']),
            list(emb_data['labels']),
            config["model_name"],
            config["image_size"],
            is_train=True
        )

        sampler = OptimizedDomainAwareTripletSampler(
            emb_data,
            config["batch_size"],
            domain_aware_ratio=config["domain_aware_ratio"],
            hard_negative_pool_size=config["hard_negative_pool_size"],
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # Optimizer with LLRD
        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)

        # Mixed precision
        scaler = GradScaler() if config["use_mixed_precision"] and self.device.type == 'cuda' else None

        # EMA
        ema = ExponentialMovingAverage(model.parameters(), decay=0.995) if EMA_AVAILABLE else None

        # TensorBoard
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

        # Loss tracker
        loss_tracker = LossTracker(window_size=5)

        # Training loop
        print("\n🚀 Starting training...")
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_loss = 0.0
            valid_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{config['epochs']}")

            for batch in pbar:
                if batch is None:
                    continue

                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.device.type == 'cuda':
                    images = images.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                # Forward with AMP
                if scaler:
                    with autocast():
                        _, loss = model(images, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _, loss = model(images, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    optimizer.step()

                # NaN check
                if torch.isnan(loss) or loss.item() > 100.0:
                    torch.save(model.state_dict(), output_dir / "emergency_checkpoint.pth")
                    raise RuntimeError(f"Loss explosion: {loss.item()}")

                if ema:
                    ema.update()

                total_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Epoch stats
            avg_loss = total_loss / max(valid_batches, 1)
            scheduler.step()
            loss_tracker.add(avg_loss)

            print(f"✅ Epoch {epoch}: Loss={avg_loss:.6f}")

            if loss_tracker.is_plateauing():
                print("⚠️ Plateau detected")
            if loss_tracker.is_diverging():
                print("🚨 Divergence detected!")

            # Save checkpoint
            torch.save(model.state_dict(), output_dir / "stage1_last.pth")
            torch.save(model.state_dict(), output_dir / f"stage1_epoch{epoch:03d}.pth")

            writer.add_scalar('Loss/train', avg_loss, epoch)
            cleanup_memory()

        writer.close()

        # Save final
        final_path = output_dir / "stage1_final.pth"
        torch.save(model.state_dict(), final_path)

        return {
            "epochs_completed": config["epochs"],
            "final_loss": avg_loss,
            "checkpoint_path": str(final_path),
            "metrics_path": str(output_dir / "tensorboard"),
        }

    def _load_embeddings(self, path: Path):
        """Load embeddings with PyTorch 2.6 compatibility."""
        raw_data = torch.load(path, map_location='cpu', weights_only=False)

        image_paths = []
        upcs = []
        domains = []
        train_indices = []

        for idx, meta in enumerate(raw_data['metadata']):
            if meta['split'] == 'train':
                image_paths.append(meta['image_path'])
                upcs.append(meta['upc'])
                domains.append(meta['domain'])
                train_indices.append(idx)

        unique_upcs = sorted(list(set(upcs)))
        upc_to_label = {upc: idx for idx, upc in enumerate(unique_upcs)}
        labels = [upc_to_label[upc] for upc in upcs]

        return {
            'embeddings': raw_data['embeddings'][train_indices],
            'image_paths': image_paths,
            'labels': labels,
            'domains': domains,
        }

    def _create_optimizer(self, model, config):
        """Create optimizer with LLRD."""
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        params = [
            {'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
             'weight_decay': config['weight_decay']},
            {'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optim.AdamW(params, lr=config['lr'], eps=1e-8)

    def _create_scheduler(self, optimizer, config):
        """Cosine annealing with warmup."""
        def lr_lambda(epoch):
            if epoch < config['warmup_epochs']:
                return float(epoch + 1) / float(config['warmup_epochs'])
            progress = (epoch - config['warmup_epochs']) / max(1, config['epochs'] - config['warmup_epochs'])
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

---

**File:** `workers/training/Dockerfile`
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Environment
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Supabase configuration (set at runtime)
# ENV SUPABASE_URL=
# ENV SUPABASE_SERVICE_KEY=
# ENV CALLBACK_URL=

WORKDIR /app/src
CMD ["python", "handler.py"]
```

---

**File:** `workers/training/requirements.txt`
```
runpod
torch>=2.0.0
torchvision
transformers
pillow
numpy
tqdm
tensorboard
torch-ema
supabase>=2.0.0
httpx>=0.25.0
```

---

**Acceptance Criteria:**
- [ ] Mixed precision (AMP) working - 2-3x faster
- [ ] Memory cleanup between batches
- [ ] LLRD (Layer-wise LR Decay) applied
- [ ] Enhanced ArcFace with label smoothing
- [ ] Domain-aware triplet sampler with hard negatives
- [ ] Loss plateau/divergence detection working
- [ ] NaN explosion detection + emergency save
- [ ] Cosine annealing with warmup
- [ ] EMA (if available) updating
- [ ] TensorBoard logging active
- [ ] Checkpoints saved every epoch
- [ ] Emergency checkpoint on error

---

### TASK-033: Migrate Embedding Extraction Worker
**Prerequisites:** TASK-032
**Estimated Complexity:** Medium

**Description:**
Embedding extraction pipeline'ını Runpod worker olarak yapılandır.

**Reference:** `Eski kodlar/extract_embeddings_large.py`

**Target:** `workers/embedding-extraction/`

---

**File:** `workers/embedding-extraction/src/supabase_client.py`
```python
"""Supabase client for embedding worker - downloads model and dataset, uploads index."""

import os
import json
from pathlib import Path
from supabase import create_client, Client
import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class EmbeddingDataDownloader:
    """Downloads model checkpoint and dataset images for embedding extraction."""

    def __init__(self, local_base: Path = Path("/tmp/embedding_data")):
        self.client = get_supabase()
        self.local_base = local_base
        self.local_base.mkdir(parents=True, exist_ok=True)

    def download_model(self, model_artifact_id: str) -> Path:
        """Download model checkpoint from Supabase Storage."""
        print(f"\n📥 Downloading model: {model_artifact_id}")

        # Get model artifact info
        response = self.client.table("model_artifacts").select("*").eq(
            "id", model_artifact_id
        ).single().execute()

        artifact = response.data
        if not artifact:
            raise ValueError(f"Model artifact {model_artifact_id} not found")

        checkpoint_url = artifact.get("checkpoint_url")
        if not checkpoint_url:
            raise ValueError("Model has no checkpoint URL")

        # Download from storage
        # URL format: {supabase_url}/storage/v1/object/public/models/{path}
        bucket = "models"
        path = checkpoint_url.split(f"{bucket}/")[-1]

        data = self.client.storage.from_(bucket).download(path)
        local_path = self.local_base / "model.pth"
        with open(local_path, 'wb') as f:
            f.write(data)

        print(f"   ✅ Model downloaded: {local_path}")
        return local_path

    def download_dataset_images(self, dataset_id: str) -> tuple[list[str], list[str], list[str]]:
        """
        Download all images for a dataset.

        Returns:
            (image_paths, product_ids, domains) - lists of local paths, product IDs, and domains
        """
        print(f"\n📥 Downloading dataset images: {dataset_id}")

        response = self.client.table("dataset_products").select(
            "product_id, products(id, barcode, frames_path)"
        ).eq("dataset_id", dataset_id).execute()

        products = response.data
        if not products:
            raise ValueError(f"Dataset {dataset_id} has no products")

        image_paths = []
        product_ids = []
        domains = []

        dataset_dir = self.local_base / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for item in products:
            product = item.get("products", {})
            product_id = product.get("id")
            barcode = product.get("barcode", product_id)

            product_dir = dataset_dir / barcode
            product_dir.mkdir(parents=True, exist_ok=True)

            # Download synthetic frames
            syn_paths = self._download_frames(barcode, product_dir)
            for p in syn_paths:
                image_paths.append(str(p))
                product_ids.append(product_id)
                domains.append("synthetic")

            # Download augmented synthetic
            aug_syn_paths = self._download_augmented(dataset_id, barcode, product_dir)
            for p in aug_syn_paths:
                image_paths.append(str(p))
                product_ids.append(product_id)
                domains.append("synthetic")

            # Download real images
            real_dir = product_dir / "real"
            real_dir.mkdir(exist_ok=True)
            real_paths = self._download_real_images(product_id, real_dir)
            for p in real_paths:
                image_paths.append(str(p))
                product_ids.append(product_id)
                domains.append("real")

        print(f"   ✅ Downloaded {len(image_paths)} images")
        return image_paths, product_ids, domains

    def _download_frames(self, barcode: str, target_dir: Path) -> list[Path]:
        """Download original frames."""
        paths = []
        try:
            bucket = "frames"
            files = self.client.storage.from_(bucket).list(barcode)
            for f in files:
                name = f.get("name", "")
                if name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    data = self.client.storage.from_(bucket).download(f"{barcode}/{name}")
                    local_path = target_dir / name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    paths.append(local_path)
        except Exception as e:
            print(f"   ⚠️ Frame download error: {e}")
        return paths

    def _download_augmented(self, dataset_id: str, barcode: str, target_dir: Path) -> list[Path]:
        """Download augmented synthetic images."""
        paths = []
        try:
            bucket = "augmented-images"
            storage_path = f"{dataset_id}/{barcode}"
            files = self.client.storage.from_(bucket).list(storage_path)
            for f in files:
                name = f.get("name", "")
                if name.endswith(('.jpg', '.jpeg', '.png')):
                    data = self.client.storage.from_(bucket).download(f"{storage_path}/{name}")
                    local_path = target_dir / name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    paths.append(local_path)
        except Exception:
            pass
        return paths

    def _download_real_images(self, product_id: str, target_dir: Path) -> list[Path]:
        """Download real images."""
        paths = []
        try:
            response = self.client.table("product_images").select("*").eq(
                "product_id", product_id
            ).eq("image_type", "real").execute()

            bucket = "real-images"
            for img in response.data:
                image_path = img.get("image_path")
                if not image_path:
                    continue
                try:
                    data = self.client.storage.from_(bucket).download(image_path)
                    local_path = target_dir / Path(image_path).name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    paths.append(local_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"   ⚠️ Real images download error: {e}")
        return paths


class IndexUploader:
    """Uploads FAISS index to Supabase Storage."""

    def __init__(self):
        self.client = get_supabase()

    def upload_index(self, index_dir: Path, index_name: str, model_artifact_id: str) -> str:
        """Upload FAISS index and id_map to Supabase Storage."""
        print(f"\n📤 Uploading index: {index_name}")

        bucket = "embeddings"

        # Upload index.faiss
        index_path = index_dir / "index.faiss"
        storage_path = f"{index_name}/index.faiss"
        with open(index_path, 'rb') as f:
            self.client.storage.from_(bucket).upload(storage_path, f.read())

        # Upload id_map.json
        id_map_path = index_dir / "id_map.json"
        id_map_storage_path = f"{index_name}/id_map.json"
        with open(id_map_path, 'rb') as f:
            self.client.storage.from_(bucket).upload(id_map_storage_path, f.read())

        # Upload metadata.json
        metadata_path = index_dir / "metadata.json"
        if metadata_path.exists():
            metadata_storage_path = f"{index_name}/metadata.json"
            with open(metadata_path, 'rb') as f:
                self.client.storage.from_(bucket).upload(metadata_storage_path, f.read())

        url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{index_name}/"
        print(f"   ✅ Index uploaded: {url}")
        return url

    def create_embedding_index_record(
        self,
        name: str,
        model_artifact_id: str,
        vector_count: int,
        index_path: str,
    ):
        """Create embedding index record in database."""
        try:
            self.client.table("embedding_indexes").insert({
                "name": name,
                "model_artifact_id": model_artifact_id,
                "vector_count": vector_count,
                "index_path": index_path,
            }).execute()
        except Exception as e:
            print(f"   ⚠️ Index record creation error: {e}")

    def update_job_progress(self, job_id: str, progress: int, current_step: str):
        """Update job progress."""
        try:
            self.client.table("jobs").update({
                "progress": progress,
                "current_step": current_step,
            }).eq("runpod_job_id", job_id).execute()
        except Exception as e:
            print(f"   ⚠️ Progress update error: {e}")

    def send_callback(self, result: dict):
        """Send completion callback."""
        if not CALLBACK_URL:
            return
        try:
            httpx.post(
                f"{CALLBACK_URL}/api/v1/webhooks/runpod",
                json=result,
                timeout=30,
            )
        except Exception as e:
            print(f"   ⚠️ Callback error: {e}")
```

---

**File:** `workers/embedding-extraction/src/handler.py`
```python
"""
Runpod Serverless Handler for Embedding Extraction.

INTEGRATED WITH SUPABASE:
- Receives dataset_id and model_artifact_id from API
- Downloads model and images from Supabase Storage
- Extracts embeddings and builds FAISS index
- Uploads index back to Supabase Storage
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import runpod
import traceback
import shutil
from pathlib import Path
from extractor import EmbeddingExtractor
from supabase_client import EmbeddingDataDownloader, IndexUploader


extractor = None
downloader = None
uploader = None


def get_extractor(model_path: str):
    """Get or create extractor with model."""
    global extractor
    if extractor is None or extractor.model_path != model_path:
        print("=" * 60)
        print(f"Loading model: {model_path}")
        print("=" * 60)
        extractor = EmbeddingExtractor(model_path)
        extractor.model_path = model_path
        print("Extractor ready!")
    return extractor


def get_downloader():
    global downloader
    if downloader is None:
        downloader = EmbeddingDataDownloader()
    return downloader


def get_uploader():
    global uploader
    if uploader is None:
        uploader = IndexUploader()
    return uploader


def handler(job):
    """Main handler for Runpod serverless."""
    job_id = job.get("id", "unknown")
    local_data_dir = None

    try:
        job_input = job.get("input", {})

        # ========================================
        # INPUT PARAMETERS
        # ========================================
        dataset_id = job_input.get("dataset_id")
        model_artifact_id = job_input.get("model_artifact_id")
        index_name = job_input.get("index_name", f"index_{job_id[:8]}")

        # Fallback for local testing
        model_path = job_input.get("model_path")
        image_paths = job_input.get("image_paths")

        if not dataset_id and not image_paths:
            return {"status": "error", "error": "dataset_id or image_paths is required"}

        if not model_artifact_id and not model_path:
            return {"status": "error", "error": "model_artifact_id or model_path is required"}

        batch_size = job_input.get("batch_size", 32)
        output_dir = Path(job_input.get("output_dir", "/workspace/outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job_id}")
        print(f"Dataset ID: {dataset_id or 'N/A (using local paths)'}")
        print(f"Model: {model_artifact_id or model_path}")
        print(f"Index name: {index_name}")
        print(f"{'=' * 60}\n")

        # ========================================
        # STEP 1: DOWNLOAD FROM SUPABASE (if needed)
        # ========================================
        product_ids = None
        domains = None

        if dataset_id and model_artifact_id:
            dl = get_downloader()
            up = get_uploader()

            up.update_job_progress(job_id, 10, "Downloading model")
            model_path = dl.download_model(model_artifact_id)

            up.update_job_progress(job_id, 20, "Downloading dataset images")
            image_paths, product_ids, domains = dl.download_dataset_images(dataset_id)
            local_data_dir = dl.local_base / dataset_id

        # ========================================
        # STEP 2: EXTRACT EMBEDDINGS
        # ========================================
        ext = get_extractor(str(model_path))

        if dataset_id:
            get_uploader().update_job_progress(job_id, 40, "Extracting embeddings")

        embeddings, image_ids = ext.extract_dataset(image_paths, batch_size)

        print(f"   Extracted {len(embeddings)} embeddings")

        # ========================================
        # STEP 3: BUILD FAISS INDEX
        # ========================================
        if dataset_id:
            get_uploader().update_job_progress(job_id, 70, "Building FAISS index")

        index = ext.build_faiss_index(embeddings)

        # Save index locally
        index_dir = output_dir / index_name
        index_dir.mkdir(parents=True, exist_ok=True)
        ext.save_index(index, image_ids, index_dir)

        # Save metadata
        import json
        metadata = {
            "dataset_id": dataset_id,
            "model_artifact_id": model_artifact_id,
            "vector_count": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "product_ids": product_ids,
            "domains": domains,
        }
        with open(index_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ========================================
        # STEP 4: UPLOAD INDEX (if using Supabase)
        # ========================================
        index_url = None
        if dataset_id:
            up = get_uploader()
            up.update_job_progress(job_id, 90, "Uploading index")

            index_url = up.upload_index(index_dir, index_name, model_artifact_id)

            # Create index record in DB
            up.create_embedding_index_record(
                name=index_name,
                model_artifact_id=model_artifact_id,
                vector_count=len(embeddings),
                index_path=index_url,
            )

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Extracted {len(embeddings)} embeddings")
        print(f"Index saved: {index_url or index_dir}")
        print(f"{'=' * 60}\n")

        final_result = {
            "status": "success",
            "type": "embedding_extraction",
            "dataset_id": dataset_id,
            "index_name": index_name,
            "vector_count": len(embeddings),
            "index_url": index_url or str(index_dir),
        }

        if dataset_id:
            get_uploader().send_callback(final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(error_trace)

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }

    finally:
        if local_data_dir and dataset_id:
            try:
                shutil.rmtree(local_data_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    print("Starting Embedding Extraction Worker...")
    runpod.serverless.start({"handler": handler})
```

---

**File:** `workers/embedding-extraction/src/extractor.py`
```python
"""Embedding Extractor using trained model."""

import torch
import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from torchvision import transforms


class SimpleImageDataset(Dataset):
    """Simple image dataset for embedding extraction."""

    def __init__(self, image_paths: list, model_name: str = "facebook/dinov2-large", image_size: int = 384):
        self.image_paths = image_paths
        self.image_size = image_size

        # Get processor for normalization values
        try:
            proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        except TypeError:
            proc = AutoImageProcessor.from_pretrained(model_name)

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=proc.image_mean, std=proc.image_std)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            tensor = self.transform(image)
            return tensor, self.image_paths[idx]
        except Exception:
            # Return zeros for failed images
            return torch.zeros(3, self.image_size, self.image_size), self.image_paths[idx]


class EmbeddingExtractor:
    """Extracts embeddings using a trained DINOv2 + ArcFace model."""

    def __init__(self, checkpoint_path: str, model_name: str = "facebook/dinov2-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print(f"   Model loaded on {self.device}")

    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        from transformers import AutoModel
        import torch.nn as nn
        import torch.nn.functional as F

        # Load base DINOv2
        backbone = AutoModel.from_pretrained(self.model_name)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Build model matching training architecture
        class EmbeddingModel(nn.Module):
            def __init__(self, backbone, proj_dim=1024):
                super().__init__()
                self.backbone = backbone
                emb_dim = getattr(backbone.config, "hidden_size", 768)
                self.proj = nn.Linear(emb_dim, proj_dim)

            def forward(self, images):
                out = self.backbone(pixel_values=images).last_hidden_state
                # Use CLS token or mean pooling
                pooled = out[:, 0, :]  # CLS token
                emb = self.proj(pooled)
                return F.normalize(emb, p=2, dim=1)

        model = EmbeddingModel(backbone, proj_dim=checkpoint.get("proj_dim", 1024))

        # Load weights (partial load for compatibility)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        return model.to(self.device)

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        """Extract embeddings for a batch of images."""
        with torch.no_grad():
            if self.device.type == 'cuda':
                images = images.to(self.device, non_blocking=True)
            else:
                images = images.to(self.device)
            embeddings = self.model(images)
        return embeddings.cpu().numpy()

    def extract_dataset(
        self,
        image_paths: list[str],
        batch_size: int = 32,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract embeddings for all images."""
        dataset = SimpleImageDataset(image_paths, self.model_name)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

        all_embeddings = []
        all_ids = []

        for batch_images, batch_paths in tqdm(loader, desc="Extracting"):
            embeddings = self.extract_batch(batch_images)
            all_embeddings.append(embeddings)
            all_ids.extend(batch_paths)

        return np.vstack(all_embeddings), all_ids

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

        # Normalize for cosine similarity
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        index.add(embeddings_normalized)

        print(f"   Built FAISS index with {index.ntotal} vectors")
        return index

    def save_index(self, index: faiss.Index, id_map: list[str], output_path: Path):
        """Save FAISS index and ID mapping."""
        output_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path / "index.faiss"))

        import json
        with open(output_path / "id_map.json", "w") as f:
            json.dump(id_map, f)

        print(f"   Index saved to: {output_path}")
```

---

**File:** `workers/embedding-extraction/Dockerfile`
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Supabase configuration (set at runtime)
# ENV SUPABASE_URL=
# ENV SUPABASE_SERVICE_KEY=
# ENV CALLBACK_URL=

WORKDIR /app/src
CMD ["python", "handler.py"]
```

---

**File:** `workers/embedding-extraction/requirements.txt`
```
runpod
torch>=2.0.0
torchvision
transformers
pillow
numpy
tqdm
faiss-cpu
supabase>=2.0.0
httpx>=0.25.0
```

---

**Acceptance Criteria:**
- [ ] Trained model yükleniyor
- [ ] Batch extraction çalışıyor
- [ ] FAISS index oluşturuluyor
- [ ] Index Supabase Storage'a yükleniyor
- [ ] Supabase integration working (dataset_id + model_artifact_id)

---

## Phase 4: Integration

### TASK-040: Create Runpod Service
**Prerequisites:** TASK-030
**Estimated Complexity:** Medium

**Description:**
Runpod API client ve job orchestration service.

**File:** `apps/api/src/services/runpod.py`
```python
"""Runpod API Service."""

import httpx
from config import settings
from typing import Optional

class RunpodService:
    def __init__(self):
        self.api_key = settings.runpod_api_key
        self.endpoints = {
            "video": settings.runpod_endpoint_video,
            "augmentation": settings.runpod_endpoint_augmentation,
            "training": settings.runpod_endpoint_training,
            "embedding": settings.runpod_endpoint_embedding,
        }
        self.base_url = "https://api.runpod.ai/v2"

    async def submit_job(
        self,
        endpoint_type: str,
        input_data: dict,
        webhook_url: Optional[str] = None,
    ) -> dict:
        """Submit job to Runpod endpoint."""
        endpoint_id = self.endpoints[endpoint_type]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{endpoint_id}/run",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": input_data,
                    "webhook": webhook_url,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    async def get_job_status(self, endpoint_type: str, job_id: str) -> dict:
        """Get job status."""
        endpoint_id = self.endpoints[endpoint_type]

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/{endpoint_id}/status/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    async def cancel_job(self, endpoint_type: str, job_id: str) -> dict:
        """Cancel running job."""
        endpoint_id = self.endpoints[endpoint_type]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{endpoint_id}/cancel/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

runpod_service = RunpodService()
```

**Acceptance Criteria:**
- [ ] Job submit çalışıyor
- [ ] Status check çalışıyor
- [ ] Cancel çalışıyor
- [ ] Webhook URL ekleniyor

---

### TASK-041: Create Webhook Handler
**Prerequisites:** TASK-040
**Estimated Complexity:** Medium

**Description:**
Runpod webhook callback handler.

**File:** `apps/api/src/api/v1/webhooks.py`
```python
"""Webhook handlers for external services."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Optional
from services.supabase import supabase_service

router = APIRouter()

class RunpodWebhook(BaseModel):
    id: str
    status: str
    output: Optional[dict] = None
    error: Optional[str] = None

@router.post("/runpod")
async def handle_runpod_webhook(
    webhook: RunpodWebhook,
    background_tasks: BackgroundTasks,
):
    """Handle Runpod job completion webhooks."""
    job_id = webhook.id

    # Update job status in database
    if webhook.status == "COMPLETED":
        await supabase_service.update_job(
            job_id,
            {
                "status": "completed",
                "result": webhook.output,
                "completed_at": "now()",
            }
        )

        # Process result based on job type
        output = webhook.output or {}
        job_type = output.get("type")

        if job_type == "video_processing":
            background_tasks.add_task(
                process_video_completion,
                output.get("product_id"),
                output,
            )
        elif job_type == "training":
            background_tasks.add_task(
                process_training_completion,
                output.get("job_id"),
                output,
            )

    elif webhook.status == "FAILED":
        await supabase_service.update_job(
            job_id,
            {
                "status": "failed",
                "error_message": webhook.error,
                "completed_at": "now()",
            }
        )

    return {"status": "ok"}

async def process_video_completion(product_id: str, output: dict):
    """Process completed video job."""
    # Update product with extracted metadata
    await supabase_service.update_product(
        product_id,
        {
            "status": "needs_matching",
            "frame_count": output.get("frame_count"),
            "frames_path": output.get("frames_url"),
            **output.get("metadata", {}),
        }
    )

async def process_training_completion(job_id: str, output: dict):
    """Process completed training job."""
    # Create model artifact
    await supabase_service.create_model_artifact({
        "job_id": job_id,
        "checkpoint_url": output.get("checkpoint_url"),
        "accuracy": output.get("accuracy"),
        "final_loss": output.get("final_loss"),
    })
```

**Acceptance Criteria:**
- [ ] Webhook endpoint çalışıyor
- [ ] Job status güncelleniyor
- [ ] Background task'lar çalışıyor
- [ ] Product/Model update'leri yapılıyor

---

### TASK-042: Implement Real-time Updates
**Prerequisites:** TASK-041
**Estimated Complexity:** Medium

**Description:**
Supabase Realtime ile job status güncellemelerini frontend'e aktar.

**Frontend Hook:** `apps/web/src/hooks/use-realtime.ts`
```typescript
"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

export function useRealtimeJobs() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const channel = supabase
      .channel("jobs")
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "jobs" },
        (payload) => {
          // Invalidate queries to refetch
          queryClient.invalidateQueries({ queryKey: ["jobs"] });
          queryClient.invalidateQueries({ queryKey: ["video-queue"] });
          queryClient.invalidateQueries({ queryKey: ["training-jobs"] });
          queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [queryClient]);
}

export function useRealtimeProducts() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const channel = supabase
      .channel("products")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "products" },
        (payload) => {
          queryClient.invalidateQueries({ queryKey: ["products"] });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [queryClient]);
}
```

**Usage in Pages:**
```typescript
// In any page component
import { useRealtimeJobs } from "@/hooks/use-realtime";

export default function VideosPage() {
  useRealtimeJobs(); // Subscribe to job updates

  // Rest of component...
}
```

**Acceptance Criteria:**
- [ ] Supabase Realtime bağlantısı çalışıyor
- [ ] Job güncellemeleri anında görünüyor
- [ ] Product güncellemeleri anında görünüyor
- [ ] Channel cleanup doğru çalışıyor

---

### TASK-043: Multi-User Concurrency Support
**Prerequisites:** TASK-042
**Estimated Complexity:** High

**Description:**
Birden fazla kullanıcının aynı anda platformu kullanabilmesi için gerekli concurrency kontrolleri.

**Key Features:**
1. **Optimistic Locking** - Product/Dataset edit için version kontrolü
2. **Resource Locking** - Matching page için exclusive lock
3. **Job Queue Limits** - Max concurrent GPU job limiti
4. **Presence Tracking** - Kimin neyi düzenlediğini gösterme
5. **Conflict Resolution** - Çakışma durumunda kullanıcı bilgilendirme

---

**Database Schema Updates:** `infra/supabase/migrations/004_multi_user.sql`
```sql
-- Add version column for optimistic locking
ALTER TABLE products ADD COLUMN version INTEGER DEFAULT 1;
ALTER TABLE datasets ADD COLUMN version INTEGER DEFAULT 1;

-- Resource locks table
CREATE TABLE resource_locks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  resource_type TEXT NOT NULL,  -- 'product', 'dataset', 'matching'
  resource_id TEXT NOT NULL,
  user_id UUID NOT NULL REFERENCES auth.users(id),
  user_email TEXT,
  locked_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ DEFAULT (now() + interval '5 minutes'),
  UNIQUE(resource_type, resource_id)
);

-- Auto-expire locks
CREATE INDEX idx_locks_expires ON resource_locks(expires_at);

-- User presence for real-time tracking
CREATE TABLE user_presence (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id),
  current_page TEXT,
  current_resource_id TEXT,
  last_seen TIMESTAMPTZ DEFAULT now()
);

-- Job queue limits
CREATE TABLE job_limits (
  job_type TEXT PRIMARY KEY,
  max_concurrent INTEGER DEFAULT 2,
  current_count INTEGER DEFAULT 0
);

INSERT INTO job_limits (job_type, max_concurrent) VALUES
  ('training', 2),
  ('augmentation', 3),
  ('video_processing', 5),
  ('embedding_extraction', 3);

-- Function to check job limits
CREATE OR REPLACE FUNCTION check_job_limit(p_job_type TEXT)
RETURNS BOOLEAN AS $$
DECLARE
  v_max INTEGER;
  v_current INTEGER;
BEGIN
  SELECT max_concurrent, current_count INTO v_max, v_current
  FROM job_limits WHERE job_type = p_job_type;

  RETURN v_current < v_max;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update version on product/dataset update
CREATE OR REPLACE FUNCTION increment_version()
RETURNS TRIGGER AS $$
BEGIN
  NEW.version = OLD.version + 1;
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER products_version_trigger
  BEFORE UPDATE ON products
  FOR EACH ROW EXECUTE FUNCTION increment_version();

CREATE TRIGGER datasets_version_trigger
  BEFORE UPDATE ON datasets
  FOR EACH ROW EXECUTE FUNCTION increment_version();
```

---

**Backend Service:** `apps/api/src/services/locking.py`
```python
"""Resource locking service for multi-user support."""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException
from services.supabase import supabase_service


class LockingService:
    """Manages resource locks for concurrent access control."""

    LOCK_DURATION_MINUTES = 5

    async def acquire_lock(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        user_email: str,
    ) -> bool:
        """
        Try to acquire a lock on a resource.
        Returns True if lock acquired, False if already locked by another user.
        """
        # Clean up expired locks first
        await self._cleanup_expired_locks()

        # Check existing lock
        existing = supabase_service.client.table("resource_locks").select("*").eq(
            "resource_type", resource_type
        ).eq("resource_id", resource_id).execute()

        if existing.data:
            lock = existing.data[0]
            if lock["user_id"] == user_id:
                # Refresh own lock
                await self._refresh_lock(lock["id"])
                return True
            else:
                # Locked by someone else
                return False

        # Create new lock
        expires_at = datetime.utcnow() + timedelta(minutes=self.LOCK_DURATION_MINUTES)

        try:
            supabase_service.client.table("resource_locks").insert({
                "resource_type": resource_type,
                "resource_id": resource_id,
                "user_id": user_id,
                "user_email": user_email,
                "expires_at": expires_at.isoformat(),
            }).execute()
            return True
        except Exception:
            # Race condition - someone else got the lock
            return False

    async def release_lock(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
    ) -> bool:
        """Release a lock. Only the owner can release."""
        result = supabase_service.client.table("resource_locks").delete().eq(
            "resource_type", resource_type
        ).eq("resource_id", resource_id).eq("user_id", user_id).execute()

        return len(result.data) > 0

    async def get_lock_info(
        self,
        resource_type: str,
        resource_id: str,
    ) -> Optional[dict]:
        """Get current lock info for a resource."""
        await self._cleanup_expired_locks()

        result = supabase_service.client.table("resource_locks").select("*").eq(
            "resource_type", resource_type
        ).eq("resource_id", resource_id).execute()

        return result.data[0] if result.data else None

    async def _refresh_lock(self, lock_id: str):
        """Extend lock expiration."""
        expires_at = datetime.utcnow() + timedelta(minutes=self.LOCK_DURATION_MINUTES)
        supabase_service.client.table("resource_locks").update({
            "expires_at": expires_at.isoformat(),
        }).eq("id", lock_id).execute()

    async def _cleanup_expired_locks(self):
        """Remove expired locks."""
        supabase_service.client.table("resource_locks").delete().lt(
            "expires_at", datetime.utcnow().isoformat()
        ).execute()


class OptimisticLockError(HTTPException):
    """Raised when optimistic lock fails (version mismatch)."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=409,
            detail={
                "error": "CONFLICT",
                "message": f"Bu {resource_type} başka biri tarafından güncellendi. Lütfen sayfayı yenileyin.",
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
        )


async def update_with_version_check(
    table: str,
    record_id: str,
    updates: dict,
    expected_version: int,
) -> dict:
    """
    Update a record with optimistic locking.
    Raises OptimisticLockError if version mismatch.
    """
    # Try to update with version check
    result = supabase_service.client.table(table).update(updates).eq(
        "id", record_id
    ).eq("version", expected_version).execute()

    if not result.data:
        # Version mismatch - someone else updated
        raise OptimisticLockError(table, record_id)

    return result.data[0]


locking_service = LockingService()
```

---

**Backend Service:** `apps/api/src/services/job_queue.py`
```python
"""Job queue management with concurrency limits."""

from fastapi import HTTPException
from services.supabase import supabase_service


class JobQueueService:
    """Manages job queue with concurrency limits per job type."""

    async def can_start_job(self, job_type: str) -> bool:
        """Check if a new job of this type can be started."""
        result = supabase_service.client.rpc(
            "check_job_limit",
            {"p_job_type": job_type}
        ).execute()
        return result.data

    async def increment_job_count(self, job_type: str):
        """Increment running job count."""
        supabase_service.client.table("job_limits").update({
            "current_count": supabase_service.client.table("job_limits")
                .select("current_count")
                .eq("job_type", job_type)
                .single()
                .execute()
                .data["current_count"] + 1
        }).eq("job_type", job_type).execute()

    async def decrement_job_count(self, job_type: str):
        """Decrement running job count."""
        result = supabase_service.client.table("job_limits").select(
            "current_count"
        ).eq("job_type", job_type).single().execute()

        new_count = max(0, result.data["current_count"] - 1)

        supabase_service.client.table("job_limits").update({
            "current_count": new_count
        }).eq("job_type", job_type).execute()

    async def get_queue_status(self) -> list[dict]:
        """Get current queue status for all job types."""
        result = supabase_service.client.table("job_limits").select("*").execute()
        return result.data

    async def require_job_slot(self, job_type: str):
        """
        Check if job can start, raise exception if queue is full.
        Call increment_job_count separately after job is successfully started.
        """
        if not await self.can_start_job(job_type):
            limits = await self.get_queue_status()
            limit_info = next((l for l in limits if l["job_type"] == job_type), None)

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "QUEUE_FULL",
                    "message": f"Maksimum {job_type} job sayısına ulaşıldı. Lütfen bekleyin.",
                    "job_type": job_type,
                    "max_concurrent": limit_info["max_concurrent"] if limit_info else 0,
                    "current_count": limit_info["current_count"] if limit_info else 0,
                }
            )


job_queue_service = JobQueueService()
```

---

**Frontend Hook:** `apps/web/src/hooks/use-resource-lock.ts`
```typescript
"use client";

import { useState, useEffect, useCallback } from "react";
import { apiClient } from "@/lib/api-client";
import { useToast } from "@/components/ui/use-toast";

interface LockInfo {
  user_id: string;
  user_email: string;
  locked_at: string;
  expires_at: string;
}

interface UseResourceLockOptions {
  resourceType: "product" | "dataset" | "matching";
  resourceId: string;
  autoAcquire?: boolean;
  refreshInterval?: number; // ms
}

export function useResourceLock({
  resourceType,
  resourceId,
  autoAcquire = true,
  refreshInterval = 60000, // 1 minute
}: UseResourceLockOptions) {
  const [isLocked, setIsLocked] = useState(false);
  const [lockInfo, setLockInfo] = useState<LockInfo | null>(null);
  const [isOwner, setIsOwner] = useState(false);
  const { toast } = useToast();

  const acquireLock = useCallback(async () => {
    try {
      const result = await apiClient.acquireLock(resourceType, resourceId);
      setIsLocked(true);
      setIsOwner(result.acquired);
      setLockInfo(result.lock_info);

      if (!result.acquired) {
        toast({
          title: "Kaynak Kilitli",
          description: `Bu ${resourceType} şu anda ${result.lock_info?.user_email} tarafından düzenleniyor.`,
          variant: "destructive",
        });
      }

      return result.acquired;
    } catch (error) {
      console.error("Lock acquire error:", error);
      return false;
    }
  }, [resourceType, resourceId, toast]);

  const releaseLock = useCallback(async () => {
    if (!isOwner) return;

    try {
      await apiClient.releaseLock(resourceType, resourceId);
      setIsLocked(false);
      setIsOwner(false);
      setLockInfo(null);
    } catch (error) {
      console.error("Lock release error:", error);
    }
  }, [resourceType, resourceId, isOwner]);

  // Auto-acquire on mount
  useEffect(() => {
    if (autoAcquire && resourceId) {
      acquireLock();
    }

    // Release on unmount
    return () => {
      if (isOwner) {
        releaseLock();
      }
    };
  }, [resourceId, autoAcquire]);

  // Refresh lock periodically
  useEffect(() => {
    if (!isOwner || !refreshInterval) return;

    const interval = setInterval(acquireLock, refreshInterval);
    return () => clearInterval(interval);
  }, [isOwner, refreshInterval, acquireLock]);

  // Release on page unload
  useEffect(() => {
    const handleUnload = () => {
      if (isOwner) {
        // Use sendBeacon for reliable delivery
        navigator.sendBeacon(
          `/api/v1/locks/${resourceType}/${resourceId}/release`,
          JSON.stringify({})
        );
      }
    };

    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [resourceType, resourceId, isOwner]);

  return {
    isLocked,
    lockInfo,
    isOwner,
    acquireLock,
    releaseLock,
    canEdit: isOwner || !isLocked,
  };
}
```

---

**Frontend Hook:** `apps/web/src/hooks/use-optimistic-update.ts`
```typescript
"use client";

import { useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import { useQueryClient } from "@tanstack/react-query";

interface UseOptimisticUpdateOptions<T> {
  queryKey: string[];
  updateFn: (data: T, version: number) => Promise<T>;
  onConflict?: () => void;
}

export function useOptimisticUpdate<T extends { version: number }>({
  queryKey,
  updateFn,
  onConflict,
}: UseOptimisticUpdateOptions<T>) {
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const update = async (data: T) => {
    setIsUpdating(true);
    setError(null);

    try {
      const result = await updateFn(data, data.version);

      // Update cache with new version
      queryClient.setQueryData(queryKey, result);

      return result;
    } catch (err: any) {
      if (err.status === 409) {
        // Conflict - version mismatch
        toast({
          title: "Güncelleme Çakışması",
          description: "Bu kayıt başka biri tarafından güncellendi. Sayfa yenileniyor...",
          variant: "destructive",
        });

        // Invalidate and refetch
        await queryClient.invalidateQueries({ queryKey });

        onConflict?.();
        setError("CONFLICT");
      } else {
        toast({
          title: "Hata",
          description: err.message || "Güncelleme başarısız",
          variant: "destructive",
        });
        setError(err.message);
      }
      throw err;
    } finally {
      setIsUpdating(false);
    }
  };

  return { update, isUpdating, error, isConflict: error === "CONFLICT" };
}
```

---

**Frontend Component:** `apps/web/src/components/shared/lock-indicator.tsx`
```typescript
"use client";

import { Lock, Unlock, User } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatDistanceToNow } from "date-fns";
import { tr } from "date-fns/locale";

interface LockIndicatorProps {
  isLocked: boolean;
  isOwner: boolean;
  lockInfo?: {
    user_email: string;
    locked_at: string;
  } | null;
}

export function LockIndicator({ isLocked, isOwner, lockInfo }: LockIndicatorProps) {
  if (!isLocked) {
    return (
      <Badge variant="outline" className="gap-1">
        <Unlock className="h-3 w-3" />
        Düzenlenebilir
      </Badge>
    );
  }

  if (isOwner) {
    return (
      <Badge variant="default" className="gap-1 bg-green-600">
        <Lock className="h-3 w-3" />
        Siz düzenliyorsunuz
      </Badge>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>
          <Badge variant="destructive" className="gap-1">
            <Lock className="h-3 w-3" />
            Kilitli
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <div className="flex items-center gap-2">
            <User className="h-4 w-4" />
            <div>
              <p className="font-medium">{lockInfo?.user_email}</p>
              <p className="text-xs text-muted-foreground">
                {lockInfo?.locked_at &&
                  formatDistanceToNow(new Date(lockInfo.locked_at), {
                    addSuffix: true,
                    locale: tr,
                  })}
              </p>
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
```

---

**API Client Updates:** `apps/web/src/lib/api-client.ts`
```typescript
// Add to ApiClient class:

  // Locking
  async acquireLock(resourceType: string, resourceId: string) {
    return this.request<{ acquired: boolean; lock_info: any }>(
      `/api/v1/locks/${resourceType}/${resourceId}`,
      { method: "POST" }
    );
  }

  async releaseLock(resourceType: string, resourceId: string) {
    return this.request(`/api/v1/locks/${resourceType}/${resourceId}`, {
      method: "DELETE",
    });
  }

  async getLockInfo(resourceType: string, resourceId: string) {
    return this.request<{ locked: boolean; lock_info: any }>(
      `/api/v1/locks/${resourceType}/${resourceId}`
    );
  }

  // Job Queue
  async getQueueStatus() {
    return this.request<Array<{
      job_type: string;
      max_concurrent: number;
      current_count: number;
    }>>("/api/v1/jobs/queue-status");
  }
```

---

**Backend API:** `apps/api/src/api/v1/locks.py`
```python
from fastapi import APIRouter, Depends
from services.locking import locking_service
from services.auth import get_current_user

router = APIRouter()


@router.post("/{resource_type}/{resource_id}")
async def acquire_lock(
    resource_type: str,
    resource_id: str,
    user = Depends(get_current_user),
):
    """Try to acquire a lock on a resource."""
    acquired = await locking_service.acquire_lock(
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user["id"],
        user_email=user["email"],
    )

    lock_info = await locking_service.get_lock_info(resource_type, resource_id)

    return {
        "acquired": acquired,
        "lock_info": lock_info,
    }


@router.delete("/{resource_type}/{resource_id}")
async def release_lock(
    resource_type: str,
    resource_id: str,
    user = Depends(get_current_user),
):
    """Release a lock on a resource."""
    released = await locking_service.release_lock(
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user["id"],
    )
    return {"released": released}


@router.get("/{resource_type}/{resource_id}")
async def get_lock_info(
    resource_type: str,
    resource_id: str,
):
    """Get lock info for a resource."""
    lock_info = await locking_service.get_lock_info(resource_type, resource_id)
    return {
        "locked": lock_info is not None,
        "lock_info": lock_info,
    }
```

---

**Update Training API to use Job Queue:** `apps/api/src/api/v1/training.py`
```python
# Add to start_training_job function:

from services.job_queue import job_queue_service

@router.post("/start", response_model=TrainingJobResponse)
async def start_training_job(
    config: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new training job."""
    # Check job queue limit
    await job_queue_service.require_job_slot("training")

    # ... rest of the function ...

    # After job is created:
    await job_queue_service.increment_job_count("training")

    return job


# In webhook handler, decrement on completion:
async def process_training_completion(job_id: str, output: dict):
    await job_queue_service.decrement_job_count("training")
    # ... rest of the function ...
```

---

**Acceptance Criteria:**
- [ ] Optimistic locking: Version mismatch 409 hatası
- [ ] Resource locking: Başka kullanıcı düzenlerken uyarı
- [ ] Lock indicator: UI'da kimin düzenlediği görünüyor
- [ ] Job queue: Max limit aşılınca 429 hatası
- [ ] Lock timeout: 5 dakika sonra otomatik unlock
- [ ] Page unload: Sayfa kapanınca lock release
- [ ] Real-time: Başka kullanıcının lock'u anında görünüyor

---

## Phase 5: Testing & Documentation

### TASK-050: End-to-end Testing
**Prerequisites:** All previous tasks
**Estimated Complexity:** High

**Description:**
Tüm pipeline'ı test et.

**Test Scenarios:**

1. **Video Processing Flow:**
   - Sync videos from Buybuddy
   - Process single video
   - Verify frames extracted
   - Verify metadata saved

2. **Product Matching Flow:**
   - Select product
   - Verify candidates loaded
   - Approve/reject matches
   - Verify real images saved

3. **Dataset & Training Flow:**
   - Create dataset
   - Add products
   - Run augmentation
   - Start training
   - Verify model saved

4. **Embedding Flow:**
   - Extract embeddings
   - Verify FAISS index created
   - Test similarity search

**Acceptance Criteria:**
- [ ] Tüm flow'lar manuel test edildi
- [ ] Error handling doğru çalışıyor
- [ ] Edge case'ler ele alındı

---

### TASK-051: Documentation
**Prerequisites:** TASK-050
**Estimated Complexity:** Low

**Description:**
Proje dokümantasyonu.

**Files to Create:**
- `README.md` - Project overview
- `docs/SETUP.md` - Development setup
- `docs/DEPLOYMENT.md` - Runpod deployment
- `docs/API.md` - API endpoints

**README.md Template:**
```markdown
# Buybuddy AI Platform

AI-powered product video processing pipeline.

## Features
- Video segmentation with SAM3
- Metadata extraction with Gemini
- Product matching with FAISS
- Model training with DINOv2 + ArcFace
- Embedding extraction and indexing

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- pnpm
- Docker (for workers)

### Setup
\`\`\`bash
# Clone and install
pnpm install

# Environment
cp .env.example .env
# Edit .env with your API keys

# Run development
pnpm dev
\`\`\`

### Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## License
Private - Buybuddy
```

**Acceptance Criteria:**
- [ ] README açıklayıcı
- [ ] Setup guide çalışıyor
- [ ] API docs güncel
- [ ] Deployment docs hazır

---

## Phase 6: Authentication

### TASK-060: Buybuddy Authentication
**Prerequisites:** TASK-010, TASK-002
**Estimated Complexity:** Medium

**Description:**
Buybuddy API kullanarak authentication sistemi kur. Kullanıcılar mevcut Buybuddy hesapları ile giriş yapabilecek.

**Auth Flow:**
```
1. User enters credentials → Frontend
2. Frontend → POST /api/v1/auth/login → Backend
3. Backend → Buybuddy API (sign_in + token) → Validates
4. Backend → Creates JWT session → Returns to Frontend
5. Frontend → Stores token → Redirects to dashboard
```

**Backend Files:**

**`apps/api/src/api/v1/auth.py`:**
```python
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import jwt
from datetime import datetime, timedelta
from config import settings

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

BUYBUDDY_API = "https://api-legacy.buybuddy.co/api/v1"
JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class TokenData(BaseModel):
    username: str
    buybuddy_token: str
    exp: datetime


async def authenticate_with_buybuddy(username: str, password: str) -> str:
    """Authenticate with Buybuddy API and return token."""
    async with httpx.AsyncClient() as client:
        # Step 1: Get passphrase
        sign_in_response = await client.post(
            f"{BUYBUDDY_API}/user/sign_in",
            json={"user_name": username, "password": password}
        )

        if sign_in_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        passphrase = sign_in_response.json().get("passphrase")
        if not passphrase:
            raise HTTPException(status_code=401, detail="Authentication failed")

        # Step 2: Get token
        token_response = await client.post(
            f"{BUYBUDDY_API}/user/sign_in/token",
            json={"passphrase": passphrase}
        )

        if token_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Token generation failed")

        token = token_response.json().get("token")
        if not token:
            raise HTTPException(status_code=401, detail="Token not received")

        return token


def create_jwt_token(username: str, buybuddy_token: str) -> str:
    """Create JWT token for session."""
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "username": username,
        "buybuddy_token": buybuddy_token,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Verify JWT token and return token data."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login with Buybuddy credentials."""
    # Authenticate with Buybuddy
    buybuddy_token = await authenticate_with_buybuddy(
        request.username,
        request.password
    )

    # Create JWT session
    access_token = create_jwt_token(request.username, buybuddy_token)

    return LoginResponse(
        access_token=access_token,
        user={
            "username": request.username
        }
    )


@router.get("/me")
async def get_current_user(token_data: TokenData = Depends(verify_jwt_token)):
    """Get current user info."""
    return {
        "username": token_data.username,
        "authenticated": True
    }


@router.post("/logout")
async def logout():
    """Logout (client-side token removal)."""
    return {"message": "Logged out successfully"}
```

**`apps/api/src/dependencies.py`:**
```python
from fastapi import Depends
from api.v1.auth import verify_jwt_token, TokenData


def get_current_user(token_data: TokenData = Depends(verify_jwt_token)) -> TokenData:
    """Dependency for protected routes."""
    return token_data


def get_buybuddy_token(token_data: TokenData = Depends(verify_jwt_token)) -> str:
    """Get Buybuddy token for API calls."""
    return token_data.buybuddy_token
```

**Update `apps/api/src/config.py`:**
```python
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    # ... existing settings ...

    # JWT
    jwt_secret: str = secrets.token_hex(32)  # Generate random if not set

    class Config:
        env_file = ".env"


settings = Settings()
```

**Frontend Files:**

**`apps/web/src/lib/auth.ts`:**
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "buybuddy_token";

export interface User {
  username: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  const response = await fetch(`${API_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Login failed");
  }

  const data = await response.json();

  // Store token
  localStorage.setItem(TOKEN_KEY, data.access_token);

  return data;
}

export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
  window.location.href = "/login";
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

export async function getCurrentUser(): Promise<User | null> {
  const token = getToken();
  if (!token) return null;

  try {
    const response = await fetch(`${API_URL}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!response.ok) {
      logout();
      return null;
    }

    return response.json();
  } catch {
    return null;
  }
}
```

**`apps/web/src/app/login/page.tsx`:**
```typescript
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";
import { login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await login(username, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Buybuddy AI</CardTitle>
          <CardDescription>Sign in with your Buybuddy account</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
            </div>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                "Sign in"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

**`apps/web/src/components/auth-provider.tsx`:**
```typescript
"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { useRouter, usePathname } from "next/navigation";
import { getCurrentUser, isAuthenticated, logout, User } from "@/lib/auth";

interface AuthContextType {
  user: User | null;
  loading: boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  logout: () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

const PUBLIC_PATHS = ["/login"];

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const checkAuth = async () => {
      // Skip auth check for public paths
      if (PUBLIC_PATHS.includes(pathname)) {
        setLoading(false);
        return;
      }

      if (!isAuthenticated()) {
        router.push("/login");
        setLoading(false);
        return;
      }

      const currentUser = await getCurrentUser();
      if (!currentUser) {
        router.push("/login");
      } else {
        setUser(currentUser);
      }
      setLoading(false);
    };

    checkAuth();
  }, [pathname, router]);

  // Show nothing while checking auth
  if (loading && !PUBLIC_PATHS.includes(pathname)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900" />
      </div>
    );
  }

  return (
    <AuthContext.Provider value={{ user, loading, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
```

**Update `apps/web/src/app/layout.tsx`:**
```typescript
import { AuthProvider } from "@/components/auth-provider";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
```

**Update API client `apps/web/src/lib/api.ts`:**
```typescript
import { getToken, logout } from "./auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (response.status === 401) {
    logout();
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "API request failed");
  }

  return response.json();
}
```

**Protect Backend Routes:**
```python
# apps/api/src/api/v1/products.py
from fastapi import APIRouter, Depends
from dependencies import get_current_user

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/")
async def list_products(user = Depends(get_current_user)):
    # user is now authenticated
    return {"products": [...]}
```

**Add JWT dependency:**
```bash
# apps/api/pyproject.toml
pip install PyJWT httpx
```

**Environment Variables:**
```bash
# .env
JWT_SECRET=your-super-secret-key-change-in-production
```

**Acceptance Criteria:**
- [ ] `/login` sayfası görüntüleniyor
- [ ] Yanlış credentials'da hata mesajı
- [ ] Doğru credentials'da dashboard'a yönleniyor
- [ ] Token localStorage'da saklanıyor
- [ ] Protected routes'a token olmadan erişilemiyor
- [ ] Logout çalışıyor
- [ ] Backend routes protected

---

## Task Index

| Phase | Task ID | Description | Complexity | Status |
|-------|---------|-------------|------------|--------|
| **Phase 0: Project Init** |
| 0 | TASK-001 | Create Monorepo Structure | Low | ⬜ |
| 0 | TASK-002 | Setup Next.js Frontend | Medium | ⬜ |
| 0 | TASK-003 | Create Sidebar Navigation | Low | ⬜ |
| 0 | TASK-004 | Create Header Component | Low | ⬜ |
| 0 | TASK-005 | Create Dashboard Page | Medium | ⬜ |
| 0 | TASK-006 | Setup React Query Provider | Low | ⬜ |
| 0 | TASK-007 | Create API Client | Low | ⬜ |
| **Phase 1: Backend** |
| 1 | TASK-010 | Setup FastAPI Backend | Medium | ⬜ |
| 1 | TASK-011 | Create Products API | Medium | ⬜ |
| 1 | TASK-012 | Create Supabase Service | Medium | ⬜ |
| 1 | TASK-013 | Create Database Schema | Medium | ⬜ |
| **Phase 2: Core Pages** |
| 2 | TASK-020 | Create Products List Page | High | ⬜ |
| 2 | TASK-021 | Create Product Detail Page | High | ⬜ |
| 2 | TASK-022 | Create Videos Page | High | ⬜ |
| 2 | TASK-023 | Create Datasets Page | Medium | ⬜ |
| 2 | TASK-024 | Create Dataset Detail Page | High | ⬜ |
| 2 | TASK-025 | Create Matching Page | Very High | ⬜ |
| 2 | TASK-025B | Create Training API Router | Medium | ⬜ |
| 2 | TASK-026 | Create Training Page | High | ⬜ |
| 2 | TASK-027 | Create Embeddings Page | Medium | ⬜ |
| **Phase 3: GPU Workers** |
| 3 | TASK-030 | Migrate Video Segmentation Worker | High | ⬜ |
| 3 | TASK-031 | Migrate Augmentation Worker | High | ⬜ |
| 3 | TASK-032 | Migrate Training Worker | Very High | ⬜ |
| 3 | TASK-033 | Migrate Embedding Worker | Medium | ⬜ |
| **Phase 4: Integration** |
| 4 | TASK-040 | Create Runpod Service | Medium | ⬜ |
| 4 | TASK-041 | Create Webhook Handler | Medium | ⬜ |
| 4 | TASK-042 | Implement Real-time Updates | Medium | ⬜ |
| 4 | TASK-043 | Multi-User Concurrency Support | High | ⬜ |
| **Phase 5: Testing & Docs** |
| 5 | TASK-050 | End-to-end Testing | High | ⬜ |
| 5 | TASK-051 | Documentation | Low | ⬜ |
| **Phase 6: Authentication** |
| 6 | TASK-060 | Buybuddy Authentication | Medium | ⬜ |

**Total:** 31 Tasks

---

## Execution Order (Recommended)

```
Phase 0 (Foundation)
├─ TASK-001 → TASK-002 → TASK-003, TASK-004 (parallel)
├─ TASK-005, TASK-006, TASK-007 (parallel)

Phase 1 (Backend)
├─ TASK-010 → TASK-011 → TASK-012
├─ TASK-013 (can run in parallel with TASK-010)

Phase 2 (UI - depends on Phase 0 + 1)
├─ TASK-020 → TASK-021
├─ TASK-022 (parallel with TASK-020)
├─ TASK-023 → TASK-024
├─ TASK-025 (after TASK-021)
├─ TASK-026 (after TASK-024)
├─ TASK-027 (after TASK-026)

Phase 3 (Workers - can start after TASK-013)
├─ TASK-030 → TASK-031 (uses same patterns)
├─ TASK-032 (independent)
├─ TASK-033 (after TASK-032)

Phase 4 (Integration - after Phase 3)
├─ TASK-040 → TASK-041 → TASK-042

Phase 5 (Final)
├─ TASK-050 → TASK-051

Phase 6 (Authentication - can run after TASK-002 + TASK-010)
├─ TASK-060 (Login page + JWT auth)
```

---

## Notes for LLM Agents

1. **Her task bağımsız**: Prerequisite'leri kontrol et, sırayla ilerle
2. **Acceptance criteria**: Her task sonunda kontrol et, tamamlanmadan geçme
3. **Kod örnekleri**: Template olarak kullan, gerekirse projeye göre adapte et
4. **Mock data**: İlk aşamada mock kullan, sonra Supabase'e geç
5. **Type safety**: TypeScript types'ları mutlaka tanımla
6. **Error handling**: Her API call'da hata yönetimi yap
7. **Import paths**: shadcn/ui components `@/components/ui/` altında
8. **API endpoints**: `/api/v1/` prefix'i ile başla
9. **Database**: UUID primary keys kullan (barcode değil)
10. **Product entity**: Bir product birden fazla barcode'a sahip olabilir

---

## Reference Files

| Old File | Purpose | Target Location |
|----------|---------|-----------------|
| `worker/src/pipeline.py` | Video segmentation | `workers/video-segmentation/src/` |
| `worker/src/handler.py` | Runpod handler | `workers/video-segmentation/src/` |
| `Eski kodlar/final_augmentor_v3.py` | Augmentation | `workers/augmentation/src/` |
| `Eski kodlar/train_optimized_v14.py` | Training | `workers/training/src/` |
| `Eski kodlar/extract_embeddings_large.py` | Embedding | `workers/embedding-extraction/src/` |
| `Eski kodlar/urun_temizleme_esleme_ui_custom.py` | Matching UI | `apps/web/src/app/matching/` |
| `Eski kodlar/custom_model_pipeline.py` | FAISS | `apps/api/src/services/faiss_index.py` |

---

## Environment Variables

```bash
# .env.example

# Supabase
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# Runpod
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_VIDEO=
RUNPOD_ENDPOINT_AUGMENTATION=
RUNPOD_ENDPOINT_TRAINING=
RUNPOD_ENDPOINT_EMBEDDING=

# External APIs
GEMINI_API_KEY=
HF_TOKEN=

# Buybuddy Legacy API
BUYBUDDY_API_URL=https://api-legacy.buybuddy.co/api/v1
BUYBUDDY_USERNAME=
BUYBUDDY_PASSWORD=

# Authentication
JWT_SECRET=your-super-secret-key-change-in-production
```
