"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  RefreshCw,
  Loader2,
  Search,
  ScanLine,
  Layers,
  Cpu,
  Box,
  Star,
  CheckCircle,
  XCircle,
} from "lucide-react";

const categoryConfig: Record<
  string,
  { icon: React.ReactNode; label: string; color: string }
> = {
  detection: {
    icon: <ScanLine className="h-4 w-4" />,
    label: "Detection",
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  },
  classification: {
    icon: <Layers className="h-4 w-4" />,
    label: "Classification",
    color: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  },
  embedding: {
    icon: <Cpu className="h-4 w-4" />,
    label: "Embedding",
    color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  },
  segmentation: {
    icon: <Box className="h-4 w-4" />,
    label: "Segmentation",
    color: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  },
};

export default function WorkflowModelsPage() {
  const queryClient = useQueryClient();

  // State
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [activeTab, setActiveTab] = useState("all");

  // Fetch models (flattened list)
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["workflow-models-list", categoryFilter],
    queryFn: () =>
      apiClient.getWorkflowModelsList({
        model_type: categoryFilter !== "all" ? categoryFilter : undefined,
      }),
  });

  const models = data?.items || [];

  // Filter by search and tab
  const filteredModels = models.filter((model) => {
    const matchesSearch =
      !searchQuery ||
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.model_type.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesTab =
      activeTab === "all" ||
      (activeTab === "pretrained" && model.source === "pretrained") ||
      (activeTab === "trained" && model.source === "trained");

    return matchesSearch && matchesTab;
  });

  // Group models by category for display
  const modelsByCategory = filteredModels.reduce(
    (acc, model) => {
      const category = model.category || "other";
      if (!acc[category]) acc[category] = [];
      acc[category].push(model);
      return acc;
    },
    {} as Record<string, typeof models>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Workflow Models</h1>
          <p className="text-muted-foreground">
            Browse available models for workflow blocks
          </p>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="all">All Models ({models.length})</TabsTrigger>
          <TabsTrigger value="pretrained">
            Pretrained ({models.filter((m) => m.source === "pretrained").length})
          </TabsTrigger>
          <TabsTrigger value="trained">
            Trained ({models.filter((m) => m.source === "trained").length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle>
                    {activeTab === "all"
                      ? "All Models"
                      : activeTab === "pretrained"
                        ? "Pretrained Models"
                        : "Trained Models"}
                  </CardTitle>
                  <CardDescription>
                    {activeTab === "pretrained"
                      ? "Pre-built models ready to use (YOLO, DINOv2, CLIP, etc.)"
                      : activeTab === "trained"
                        ? "Models trained on your data"
                        : "All available models for workflow blocks"}
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    queryClient.invalidateQueries({ queryKey: ["workflow-models"] })
                  }
                  disabled={isFetching}
                >
                  <RefreshCw
                    className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
                  />
                </Button>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center mt-4">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search models..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>
                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="All Categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    <SelectItem value="detection">Detection</SelectItem>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="embedding">Embedding</SelectItem>
                    <SelectItem value="segmentation">Segmentation</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>

            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : filteredModels.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <Cpu className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="font-medium">No models found</p>
                  <p className="text-sm mt-1">
                    {searchQuery || categoryFilter !== "all"
                      ? "Try adjusting your filters"
                      : activeTab === "trained"
                        ? "Train your first model to see it here"
                        : "No models available"}
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {Object.entries(modelsByCategory).map(([category, categoryModels]) => {
                    const config = categoryConfig[category] || {
                      icon: <Box className="h-4 w-4" />,
                      label: category,
                      color: "bg-gray-100 text-gray-800",
                    };

                    return (
                      <div key={category}>
                        <div className="flex items-center gap-2 mb-3">
                          <div className={`p-1.5 rounded ${config.color}`}>
                            {config.icon}
                          </div>
                          <h3 className="font-semibold">{config.label}</h3>
                          <Badge variant="secondary" className="text-xs">
                            {categoryModels.length}
                          </Badge>
                        </div>

                        <div className="rounded-md border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Name</TableHead>
                                <TableHead>Type</TableHead>
                                <TableHead>Source</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Created</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {categoryModels.map((model) => (
                                <TableRow key={model.id}>
                                  <TableCell>
                                    <div className="flex items-center gap-2">
                                      <span className="font-medium">{model.name}</span>
                                      {model.is_default && (
                                        <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
                                      )}
                                    </div>
                                  </TableCell>
                                  <TableCell>
                                    <Badge variant="outline">{model.model_type}</Badge>
                                  </TableCell>
                                  <TableCell>
                                    <Badge
                                      variant={
                                        model.source === "pretrained"
                                          ? "secondary"
                                          : "default"
                                      }
                                    >
                                      {model.source === "pretrained"
                                        ? "Pretrained"
                                        : "Trained"}
                                    </Badge>
                                  </TableCell>
                                  <TableCell>
                                    {model.is_active ? (
                                      <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                        <CheckCircle className="h-3 w-3 mr-1" />
                                        Active
                                      </Badge>
                                    ) : (
                                      <Badge variant="secondary">
                                        <XCircle className="h-3 w-3 mr-1" />
                                        Inactive
                                      </Badge>
                                    )}
                                  </TableCell>
                                  <TableCell className="text-muted-foreground">
                                    {model.created_at ? new Date(model.created_at).toLocaleDateString() : "-"}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Model categories summary */}
      <div className="grid gap-4 md:grid-cols-4">
        {Object.entries(categoryConfig).map(([category, config]) => {
          const count = models.filter((m) => m.category === category).length;
          const activeCount = models.filter(
            (m) => m.category === category && m.is_active
          ).length;

          return (
            <Card key={category}>
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${config.color}`}>
                    {config.icon}
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{count}</p>
                    <p className="text-sm text-muted-foreground">
                      {config.label} ({activeCount} active)
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
