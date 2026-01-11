"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Package,
  Video,
  Database,
  Brain,
  RefreshCw,
  ArrowRight,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import Link from "next/link";
import type { Job, Product, JobStatus } from "@/types";

export default function DashboardPage() {
  const { data: stats, isLoading, refetch } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: () => apiClient.getDashboardStats(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const statusColors: Record<string, string> = {
    pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    processing: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    needs_matching: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
    ready: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    rejected: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  };

  const jobStatusIcons: Record<JobStatus, React.ReactNode> = {
    pending: <Clock className="h-4 w-4 text-yellow-500" />,
    queued: <Clock className="h-4 w-4 text-blue-500" />,
    running: <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />,
    completed: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    failed: <XCircle className="h-4 w-4 text-red-500" />,
    cancelled: <XCircle className="h-4 w-4 text-gray-500" />,
  };

  // Mock data for initial development
  const mockStats = {
    total_products: 1250,
    products_by_status: {
      pending: 45,
      processing: 12,
      needs_matching: 89,
      ready: 1050,
      rejected: 54,
    },
    total_datasets: 8,
    active_jobs: 3,
    completed_jobs_today: 24,
    recent_products: [] as Product[],
    recent_jobs: [] as Job[],
  };

  const displayStats = stats || mockStats;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your product processing pipeline
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Products</CardTitle>
            <Package className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {displayStats.total_products.toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              {displayStats.products_by_status.ready} ready for matching
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{displayStats.total_datasets}</div>
            <p className="text-xs text-muted-foreground">
              Training datasets created
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{displayStats.active_jobs}</div>
            <p className="text-xs text-muted-foreground">
              Currently processing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Completed Today</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {displayStats.completed_jobs_today}
            </div>
            <p className="text-xs text-muted-foreground">Jobs completed today</p>
          </CardContent>
        </Card>
      </div>

      {/* Product Status Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Product Status Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(displayStats.products_by_status).map(([status, count]) => {
              const percentage = (count / displayStats.total_products) * 100;
              return (
                <div key={status} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge className={statusColors[status] || ""}>
                        {status.replace("_", " ")}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        {count.toLocaleString()} products
                      </span>
                    </div>
                    <span className="text-sm font-medium">
                      {percentage.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={percentage} className="h-2" />
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Link href="/videos">
          <Card className="hover:bg-accent transition-colors cursor-pointer">
            <CardContent className="flex items-center gap-4 pt-6">
              <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900">
                <Video className="h-6 w-6 text-blue-600 dark:text-blue-300" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">Process Videos</h3>
                <p className="text-sm text-muted-foreground">
                  Sync and process new videos
                </p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </CardContent>
          </Card>
        </Link>

        <Link href="/datasets/new">
          <Card className="hover:bg-accent transition-colors cursor-pointer">
            <CardContent className="flex items-center gap-4 pt-6">
              <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900">
                <Database className="h-6 w-6 text-purple-600 dark:text-purple-300" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">Create Dataset</h3>
                <p className="text-sm text-muted-foreground">
                  Build a training dataset
                </p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </CardContent>
          </Card>
        </Link>

        <Link href="/training">
          <Card className="hover:bg-accent transition-colors cursor-pointer">
            <CardContent className="flex items-center gap-4 pt-6">
              <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900">
                <Brain className="h-6 w-6 text-green-600 dark:text-green-300" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">Start Training</h3>
                <p className="text-sm text-muted-foreground">
                  Train a new model
                </p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </CardContent>
          </Card>
        </Link>

        <Link href="/matching">
          <Card className="hover:bg-accent transition-colors cursor-pointer">
            <CardContent className="flex items-center gap-4 pt-6">
              <div className="p-3 rounded-lg bg-orange-100 dark:bg-orange-900">
                <Package className="h-6 w-6 text-orange-600 dark:text-orange-300" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">Match Products</h3>
                <p className="text-sm text-muted-foreground">
                  Review and approve matches
                </p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Recent Activity */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent Jobs */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Jobs</CardTitle>
            <Link href="/jobs">
              <Button variant="ghost" size="sm">
                View all
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {displayStats.recent_jobs.length > 0 ? (
              <div className="space-y-4">
                {displayStats.recent_jobs.slice(0, 5).map((job) => (
                  <div
                    key={job.id}
                    className="flex items-center justify-between py-2 border-b last:border-0"
                  >
                    <div className="flex items-center gap-3">
                      {jobStatusIcons[job.status]}
                      <div>
                        <p className="font-medium capitalize">
                          {job.type.replace("_", " ")}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(job.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline">{job.status}</Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No recent jobs
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Products */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Products</CardTitle>
            <Link href="/products">
              <Button variant="ghost" size="sm">
                View all
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {displayStats.recent_products.length > 0 ? (
              <div className="space-y-4">
                {displayStats.recent_products.slice(0, 5).map((product) => (
                  <div
                    key={product.id}
                    className="flex items-center justify-between py-2 border-b last:border-0"
                  >
                    <div>
                      <p className="font-medium">{product.barcode}</p>
                      <p className="text-sm text-muted-foreground">
                        {product.brand_name} - {product.product_name}
                      </p>
                    </div>
                    <Badge className={statusColors[product.status] || ""}>
                      {product.status}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No recent products
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
