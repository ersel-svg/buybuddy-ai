"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Image as ImageIcon,
  FolderOpen,
  Tags,
  Cpu,
  Upload,
  Plus,
  ArrowRight,
  Brain,
  RefreshCw,
  Loader2,
  CheckCircle,
  Clock,
  AlertCircle,
  PenTool,
} from "lucide-react";

export default function ClassificationDashboardPage() {
  // Fetch stats
  const { data: stats, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["cls-stats"],
    queryFn: () => apiClient.getCLSStats(),
  });

  // Fetch recent datasets
  const { data: datasets } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
  });

  const recentDatasets = datasets?.slice(0, 3) ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Image Classification</h1>
          <p className="text-muted-foreground">
            Label images, train classification models, and deploy
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Images</CardTitle>
            <ImageIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : stats?.total_images?.toLocaleString() ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              In library
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <FolderOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : stats?.total_datasets ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Active datasets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Labels</CardTitle>
            <PenTool className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : stats?.total_labels?.toLocaleString() ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Image labels
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Classes</CardTitle>
            <Tags className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : stats?.total_classes ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Classification classes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Models</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : stats?.total_models ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Trained models
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Image Status Overview */}
      {stats?.images_by_status && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Image Status Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-orange-100">
                  <AlertCircle className="h-4 w-4 text-orange-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.images_by_status.pending ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Pending</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-blue-100">
                  <Tags className="h-4 w-4 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.images_by_status.labeled ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Labeled</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-yellow-100">
                  <Clock className="h-4 w-4 text-yellow-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.images_by_status.review ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Review</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-green-100">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.images_by_status.completed ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Completed</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-gray-100">
                  <AlertCircle className="h-4 w-4 text-gray-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.images_by_status.skipped ?? 0}</p>
                  <p className="text-xs text-muted-foreground">Skipped</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Get started with common tasks</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-3">
            <Button asChild>
              <Link href="/classification/images">
                <Upload className="h-4 w-4 mr-2" />
                Upload Images
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/classification/datasets">
                <Plus className="h-4 w-4 mr-2" />
                Create Dataset
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/classification/labeling">
                <PenTool className="h-4 w-4 mr-2" />
                Start Labeling
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/classification/training">
                <Brain className="h-4 w-4 mr-2" />
                Train Model
              </Link>
            </Button>
          </CardContent>
        </Card>

        {/* Recent Datasets */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Recent Datasets</CardTitle>
              <CardDescription>Your latest classification projects</CardDescription>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/classification/datasets">
                View All
                <ArrowRight className="h-4 w-4 ml-1" />
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            {recentDatasets.length === 0 ? (
              <div className="text-center py-6 text-muted-foreground">
                <FolderOpen className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No datasets yet</p>
                <Button variant="link" size="sm" asChild>
                  <Link href="/classification/datasets">Create your first dataset</Link>
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {recentDatasets.map((dataset) => {
                  const progress = dataset.image_count > 0
                    ? Math.round((dataset.labeled_image_count / dataset.image_count) * 100)
                    : 0;
                  return (
                    <Link
                      key={dataset.id}
                      href={`/classification/datasets/${dataset.id}`}
                      className="block p-3 rounded-lg border hover:border-primary transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{dataset.name}</span>
                        <Badge variant="secondary">{progress}%</Badge>
                      </div>
                      <Progress value={progress} className="h-1.5 mb-2" />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{dataset.image_count} images</span>
                        <span>{dataset.labeled_image_count} labeled</span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Getting Started */}
      <Card>
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>Follow these steps to train your first classification model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                (stats?.total_images ?? 0) > 0 ? "bg-green-600 text-white" : "bg-primary text-primary-foreground"
              }`}>
                {(stats?.total_images ?? 0) > 0 ? <CheckCircle className="h-4 w-4" /> : "1"}
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Upload Images</h4>
                <p className="text-sm text-muted-foreground">
                  Upload images or import from Products, Cutouts, or OD Images
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/classification/images">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                (stats?.total_datasets ?? 0) > 0 ? "bg-green-600 text-white" : "bg-muted text-muted-foreground"
              }`}>
                {(stats?.total_datasets ?? 0) > 0 ? <CheckCircle className="h-4 w-4" /> : "2"}
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Create Dataset & Classes</h4>
                <p className="text-sm text-muted-foreground">
                  Create a dataset and define classification classes
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/classification/datasets">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                (stats?.total_labels ?? 0) > 0 ? "bg-green-600 text-white" : "bg-muted text-muted-foreground"
              }`}>
                {(stats?.total_labels ?? 0) > 0 ? <CheckCircle className="h-4 w-4" /> : "3"}
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Label Images</h4>
                <p className="text-sm text-muted-foreground">
                  Assign classes to images with keyboard shortcuts
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/classification/labeling">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                (stats?.total_models ?? 0) > 0 ? "bg-green-600 text-white" : "bg-muted text-muted-foreground"
              }`}>
                {(stats?.total_models ?? 0) > 0 ? <CheckCircle className="h-4 w-4" /> : "4"}
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Train Model</h4>
                <p className="text-sm text-muted-foreground">
                  Train ViT, ConvNeXt, EfficientNet, Swin, DINOv2, or CLIP models
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/classification/training">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
