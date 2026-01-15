"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Triangle,
  Play,
  Download,
  Loader2,
  Info,
  Database,
} from "lucide-react";

export default function TripletsPage() {
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [hardNegativeThreshold, setHardNegativeThreshold] = useState([0.7]);
  const [positiveThreshold, setPositiveThreshold] = useState([0.9]);

  // Fetch datasets for selection
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  const selectedDatasetInfo = datasets?.find((d) => d.id === selectedDataset);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Hard Triplets Mining</h1>
          <p className="text-muted-foreground">
            Find confusing product pairs to improve model training
          </p>
        </div>
      </div>

      {/* Info Card */}
      <Card className="bg-blue-50 border-blue-200">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-blue-800">
            <Info className="h-4 w-4" />
            What is Hard Triplet Mining?
          </CardTitle>
        </CardHeader>
        <CardContent className="text-blue-700 text-sm">
          Hard triplet mining finds products that look similar but are different (hard negatives).
          Training on these challenging examples significantly improves model accuracy.
          A triplet consists of: <strong>Anchor</strong> (reference product),
          <strong> Positive</strong> (same product, different view), and
          <strong> Hard Negative</strong> (different product that looks similar).
        </CardContent>
      </Card>

      {/* Configuration */}
      <div className="grid grid-cols-2 gap-6">
        {/* Dataset Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Select Dataset</CardTitle>
            <CardDescription>
              Choose a dataset to mine hard triplets from
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select
              value={selectedDataset}
              onValueChange={setSelectedDataset}
              disabled={datasetsLoading}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a dataset..." />
              </SelectTrigger>
              <SelectContent>
                {datasets?.map((dataset) => (
                  <SelectItem key={dataset.id} value={dataset.id}>
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4" />
                      {dataset.name}
                      <Badge variant="secondary" className="ml-2">
                        {dataset.product_count} products
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {selectedDatasetInfo && (
              <div className="rounded-lg bg-muted p-4 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Products:</span>
                  <span className="font-medium">{selectedDatasetInfo.product_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Version:</span>
                  <span className="font-medium">v{selectedDatasetInfo.version}</span>
                </div>
                {selectedDatasetInfo.description && (
                  <p className="text-sm text-muted-foreground pt-2 border-t">
                    {selectedDatasetInfo.description}
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Threshold Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Mining Parameters</CardTitle>
            <CardDescription>
              Configure similarity thresholds for triplet selection
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Hard Negative Threshold</Label>
                <span className="text-sm font-medium">{hardNegativeThreshold[0]}</span>
              </div>
              <Slider
                value={hardNegativeThreshold}
                onValueChange={setHardNegativeThreshold}
                min={0.5}
                max={0.95}
                step={0.05}
              />
              <p className="text-xs text-muted-foreground">
                Products with similarity above this threshold are considered &quot;hard&quot; negatives
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Positive Threshold</Label>
                <span className="text-sm font-medium">{positiveThreshold[0]}</span>
              </div>
              <Slider
                value={positiveThreshold}
                onValueChange={setPositiveThreshold}
                min={0.8}
                max={1.0}
                step={0.02}
              />
              <p className="text-xs text-muted-foreground">
                Same-product pairs must have similarity above this to be positive examples
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <Card>
        <CardHeader>
          <CardTitle>Generate Triplets</CardTitle>
          <CardDescription>
            Mine hard triplets from the selected dataset using the configured thresholds
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button
              disabled={!selectedDataset}
              className="flex-1"
            >
              <Play className="h-4 w-4 mr-2" />
              Start Mining
            </Button>
            <Button
              variant="outline"
              disabled={!selectedDataset}
            >
              <Download className="h-4 w-4 mr-2" />
              Export Triplets
            </Button>
          </div>
          {!selectedDataset && (
            <p className="text-sm text-muted-foreground mt-4 text-center">
              Select a dataset to start mining triplets
            </p>
          )}
        </CardContent>
      </Card>

      {/* Placeholder for Results */}
      <Card>
        <CardHeader>
          <CardTitle>Mining Results</CardTitle>
          <CardDescription>
            Discovered hard triplets will appear here
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Triangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No triplets mined yet</p>
            <p className="text-sm text-muted-foreground mt-1">
              Select a dataset and click &quot;Start Mining&quot; to find hard triplets
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
