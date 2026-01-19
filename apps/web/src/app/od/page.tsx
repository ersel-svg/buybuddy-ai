"use client";

import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Image as ImageIcon,
  FolderOpen,
  PenTool,
  Cpu,
  Upload,
  Plus,
  ArrowRight,
} from "lucide-react";

export default function ODDashboardPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Object Detection</h1>
        <p className="text-muted-foreground">
          Annotate images, train detection models, and deploy
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Images</CardTitle>
            <ImageIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">
              Uploaded images
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <FolderOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">
              Active datasets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Annotations</CardTitle>
            <PenTool className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">
              Total bounding boxes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Models</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0</div>
            <p className="text-xs text-muted-foreground">
              Trained models
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Get started with common tasks</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Button asChild>
            <Link href="/od/images">
              <Upload className="h-4 w-4 mr-2" />
              Upload Images
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/od/datasets">
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/od/annotate">
              <PenTool className="h-4 w-4 mr-2" />
              Start Annotating
            </Link>
          </Button>
        </CardContent>
      </Card>

      {/* Getting Started */}
      <Card>
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>Follow these steps to train your first detection model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                1
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Upload Images</h4>
                <p className="text-sm text-muted-foreground">
                  Upload shelf images or sync from BuyBuddy API
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/od/images">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-muted-foreground text-sm font-medium">
                2
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Create Dataset</h4>
                <p className="text-sm text-muted-foreground">
                  Organize images into datasets for annotation
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/od/datasets">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-muted-foreground text-sm font-medium">
                3
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Annotate</h4>
                <p className="text-sm text-muted-foreground">
                  Draw bounding boxes with AI assistance
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/od/annotate">
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-muted-foreground text-sm font-medium">
                4
              </div>
              <div className="flex-1">
                <h4 className="font-medium">Train Model</h4>
                <p className="text-sm text-muted-foreground">
                  Train RF-DETR, RT-DETR, or YOLO-NAS models
                </p>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/od/training">
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
