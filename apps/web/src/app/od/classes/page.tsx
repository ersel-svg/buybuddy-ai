"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
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
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  RefreshCw,
  Loader2,
  Tags,
  Search,
  Plus,
  MoreHorizontal,
  Edit,
  Trash2,
  Merge,
  Lock,
  Palette,
  FolderOpen,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Copy,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  X,
  CheckSquare,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface ODClass {
  id: string;
  name: string;
  display_name?: string;
  color: string;
  category?: string;
  annotation_count: number;
  is_system: boolean;
  dataset_id?: string;
}

interface ODDataset {
  id: string;
  name: string;
  image_count: number;
  class_count: number;
}

// Predefined colors for class labels
const PRESET_COLORS = [
  "#ef4444", // red
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#14b8a6", // teal
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#6366f1", // indigo
  "#06b6d4", // cyan
];

export default function ODClassesPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isMergeDialogOpen, setIsMergeDialogOpen] = useState(false);
  const [editingClass, setEditingClass] = useState<ODClass | null>(null);

  // Form states
  const [formName, setFormName] = useState("");
  const [formDisplayName, setFormDisplayName] = useState("");
  const [formColor, setFormColor] = useState(PRESET_COLORS[0]);
  const [formCategory, setFormCategory] = useState("");

  // Merge states
  const [mergeSourceIds, setMergeSourceIds] = useState<string[]>([]);
  const [mergeTargetId, setMergeTargetId] = useState("");

  // Selection states
  const [selectedClassIds, setSelectedClassIds] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<"name" | "annotation_count" | "category">("name");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");

  // Fetch datasets for selector
  const { data: datasets } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
  });

  // Fetch classes based on selected dataset
  const { data: classes, isLoading, isFetching } = useQuery({
    queryKey: ["od-classes", selectedDatasetId, filterCategory],
    queryFn: () =>
      apiClient.getODClasses({
        dataset_id: selectedDatasetId || undefined,
        category: filterCategory !== "all" ? filterCategory : undefined,
        is_active: true,
      }),
    enabled: !!selectedDatasetId,
  });

  // Fetch potential duplicates
  const { data: duplicates } = useQuery({
    queryKey: ["od-class-duplicates", selectedDatasetId],
    queryFn: () => apiClient.getODClassDuplicates(selectedDatasetId),
    enabled: !!selectedDatasetId && (classes?.length ?? 0) > 1,
    staleTime: 30000, // Cache for 30 seconds
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      if (!selectedDatasetId) {
        throw new Error("Please select a dataset first");
      }
      return apiClient.createODDatasetClass(selectedDatasetId, {
        name: formName,
        display_name: formDisplayName || undefined,
        color: formColor,
        category: formCategory || undefined,
      });
    },
    onSuccess: () => {
      toast.success(`Class "${formName}" created`);
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
      setIsCreateDialogOpen(false);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to create class: ${error.message}`);
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: async () => {
      if (!editingClass) return;
      return apiClient.updateODClass(editingClass.id, {
        name: formName,
        display_name: formDisplayName || undefined,
        color: formColor,
        category: formCategory || undefined,
      });
    },
    onSuccess: () => {
      toast.success("Class updated");
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
      setIsEditDialogOpen(false);
      setEditingClass(null);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to update class: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.deleteODClass(id, true);
    },
    onSuccess: () => {
      toast.success("Class deleted");
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete class: ${error.message}`);
    },
  });

  // Merge mutation
  const mergeMutation = useMutation({
    mutationFn: async () => {
      return apiClient.mergeODClasses(mergeSourceIds, mergeTargetId);
    },
    onSuccess: (data) => {
      toast.success(`Merged ${data.merged_count} classes, moved ${data.annotations_moved} annotations`);
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
      setIsMergeDialogOpen(false);
      setMergeSourceIds([]);
      setMergeTargetId("");
    },
    onError: (error) => {
      toast.error(`Failed to merge classes: ${error.message}`);
    },
  });

  const resetForm = () => {
    setFormName("");
    setFormDisplayName("");
    setFormColor(PRESET_COLORS[Math.floor(Math.random() * PRESET_COLORS.length)]);
    setFormCategory("");
  };

  const openEditDialog = (cls: ODClass) => {
    setEditingClass(cls);
    setFormName(cls.name);
    setFormDisplayName(cls.display_name || "");
    setFormColor(cls.color);
    setFormCategory(cls.category || "");
    setIsEditDialogOpen(true);
  };

  const handleDelete = (cls: ODClass) => {
    const warning = cls.is_system ? " (System Class)" : "";
    if (cls.annotation_count > 0) {
      if (!confirm(`This class${warning} has ${cls.annotation_count} annotations. Delete anyway?`)) {
        return;
      }
    } else {
      if (!confirm(`Delete class "${cls.name}"${warning}?`)) {
        return;
      }
    }
    deleteMutation.mutate(cls.id);
  };

  // Filter and sort classes
  const filteredClasses = useMemo(() => {
    let result = classes?.filter((cls) =>
      cls.name.toLowerCase().includes(search.toLowerCase()) ||
      cls.display_name?.toLowerCase().includes(search.toLowerCase())
    ) || [];

    // Sort
    result = [...result].sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "annotation_count":
          comparison = a.annotation_count - b.annotation_count;
          break;
        case "category":
          comparison = (a.category || "").localeCompare(b.category || "");
          break;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

    return result;
  }, [classes, search, sortBy, sortOrder]);

  // Selection helpers
  const selectableClasses = filteredClasses || [];
  const allSelectableSelected = selectableClasses.length > 0 &&
    selectableClasses.every(c => selectedClassIds.has(c.id));
  const someSelected = selectedClassIds.size > 0;

  const toggleSelectAll = () => {
    if (allSelectableSelected) {
      setSelectedClassIds(new Set());
    } else {
      setSelectedClassIds(new Set(selectableClasses.map(c => c.id)));
    }
  };

  const toggleSelect = (id: string) => {
    const newSet = new Set(selectedClassIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedClassIds(newSet);
  };

  const clearSelection = () => setSelectedClassIds(new Set());

  const handleSort = (column: "name" | "annotation_count" | "category") => {
    if (sortBy === column) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortOrder("asc");
    }
  };

  // Bulk delete
  const handleBulkDelete = () => {
    const count = selectedClassIds.size;
    const totalAnnotations = Array.from(selectedClassIds).reduce((sum, id) => {
      const cls = classes?.find(c => c.id === id);
      return sum + (cls?.annotation_count || 0);
    }, 0);

    const message = totalAnnotations > 0
      ? `Delete ${count} class(es) with ${totalAnnotations} total annotations?`
      : `Delete ${count} class(es)?`;

    if (confirm(message)) {
      // Delete one by one
      Array.from(selectedClassIds).forEach(id => {
        deleteMutation.mutate(id);
      });
      clearSelection();
    }
  };

  // Bulk merge
  const handleBulkMerge = () => {
    setMergeSourceIds(Array.from(selectedClassIds));
    setMergeTargetId("");
    setIsMergeDialogOpen(true);
  };

  // Get unique categories
  const categories = [...new Set(classes?.map((c) => c.category).filter(Boolean) || [])];

  // Calculate stats
  const totalAnnotations = classes?.reduce((sum, c) => sum + c.annotation_count, 0) || 0;
  const systemClasses = classes?.filter((c) => c.is_system).length || 0;
  const customClasses = classes?.filter((c) => !c.is_system).length || 0;

  // Get selected dataset name
  const selectedDataset = datasets?.find(d => d.id === selectedDatasetId);

  // Calculate merge preview summary
  const mergePreview = useMemo(() => {
    if (!classes || mergeSourceIds.length === 0) return null;

    const sourceAnnotations = mergeSourceIds.reduce((sum, id) => {
      const cls = classes.find(c => c.id === id);
      return sum + (cls?.annotation_count || 0);
    }, 0);

    const targetClass = classes.find(c => c.id === mergeTargetId);
    const targetAnnotations = targetClass?.annotation_count || 0;

    return {
      sourceAnnotations,
      targetAnnotations,
      totalAfterMerge: sourceAnnotations + targetAnnotations,
      targetName: targetClass?.name || "target",
    };
  }, [classes, mergeSourceIds, mergeTargetId]);

  // Handle duplicate quick merge
  const handleDuplicateQuickMerge = (group: NonNullable<typeof duplicates>["groups"][0]) => {
    setMergeSourceIds(group.suggested_sources);
    setMergeTargetId(group.suggested_target);
    setIsMergeDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Classes</h1>
          <p className="text-muted-foreground">
            Manage detection classes for your datasets
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["od-classes"] })}
            disabled={isFetching || !selectedDatasetId}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            onClick={() => setIsMergeDialogOpen(true)}
            disabled={!classes || classes.length < 2}
          >
            <Merge className="h-4 w-4 mr-2" />
            Merge Classes
          </Button>
          <Button onClick={() => {
            resetForm();
            setIsCreateDialogOpen(true);
          }} disabled={!selectedDatasetId}>
            <Plus className="h-4 w-4 mr-2" />
            New Class
          </Button>
        </div>
      </div>

      {/* Dataset Selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <FolderOpen className="h-4 w-4" />
            Select Dataset
          </CardTitle>
          <CardDescription>
            Classes are now scoped to datasets. Select a dataset to view and manage its classes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
            <SelectTrigger className="w-[400px]">
              <SelectValue placeholder="Choose a dataset..." />
            </SelectTrigger>
            <SelectContent>
              {datasets?.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  <div className="flex items-center gap-2">
                    <span>{dataset.name}</span>
                    <Badge variant="secondary" className="ml-2">
                      {dataset.image_count} images
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {!selectedDatasetId && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No dataset selected</AlertTitle>
          <AlertDescription>
            Please select a dataset above to view and manage its classes. Each dataset has its own set of classes.
          </AlertDescription>
        </Alert>
      )}

      {selectedDatasetId && (
        <>
          {/* Duplicate Detection Banner */}
          {duplicates && duplicates.total_groups > 0 && (
            <Alert className="border-amber-200 bg-amber-50">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <AlertTitle className="text-amber-800">
                Potential Duplicate Classes Detected
              </AlertTitle>
              <AlertDescription className="text-amber-700">
                <p className="mb-3">
                  Found {duplicates.total_groups} group(s) of similar class names that might be duplicates.
                </p>
                <div className="space-y-2">
                  {duplicates.groups.slice(0, 3).map((group, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between bg-white/50 rounded-md p-2"
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        {group.classes.map((cls, i) => (
                          <span key={cls.id} className="flex items-center gap-1">
                            <div
                              className="w-3 h-3 rounded"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className="font-medium">{cls.name}</span>
                            <span className="text-xs text-amber-600">
                              ({cls.annotation_count})
                            </span>
                            {i < group.classes.length - 1 && (
                              <Copy className="h-3 w-3 mx-1 text-amber-400" />
                            )}
                          </span>
                        ))}
                        <span className="text-xs text-amber-500 ml-2">
                          {Math.round(group.max_similarity * 100)}% similar
                        </span>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-amber-300 text-amber-700 hover:bg-amber-100"
                        onClick={() => handleDuplicateQuickMerge(group)}
                      >
                        <Merge className="h-3 w-3 mr-1" />
                        Quick Merge
                      </Button>
                    </div>
                  ))}
                  {duplicates.total_groups > 3 && (
                    <p className="text-xs text-amber-600">
                      ...and {duplicates.total_groups - 3} more group(s)
                    </p>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Total Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{classes?.length ?? "-"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">System Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-blue-600">{systemClasses}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Custom Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-green-600">{customClasses}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Total Annotations</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{totalAnnotations.toLocaleString()}</p>
              </CardContent>
            </Card>
          </div>

          {/* Classes Table */}
          <Card>
            <CardHeader>
              <CardTitle>Class Library - {selectedDataset?.name}</CardTitle>
              <CardDescription>
                {classes?.length ?? 0} detection classes in this dataset
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Filters */}
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search classes..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <Select value={filterCategory} onValueChange={setFilterCategory}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map((cat) => (
                      <SelectItem key={cat} value={cat as string}>
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Bulk Actions Bar */}
              {someSelected && (
                <div className="flex items-center justify-between bg-muted/50 rounded-lg p-3 border">
                  <div className="flex items-center gap-3">
                    <CheckSquare className="h-5 w-5 text-primary" />
                    <span className="font-medium">
                      {selectedClassIds.size} class(es) selected
                    </span>
                    <Button variant="ghost" size="sm" onClick={clearSelection}>
                      <X className="h-4 w-4 mr-1" />
                      Clear
                    </Button>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleBulkMerge}
                      disabled={selectedClassIds.size < 2}
                    >
                      <Merge className="h-4 w-4 mr-2" />
                      Merge Selected
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBulkDelete}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete Selected
                    </Button>
                  </div>
                </div>
              )}

              {/* Table */}
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : filteredClasses?.length === 0 ? (
                <div className="text-center py-12">
                  <Tags className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No classes found</p>
                  <Button className="mt-4" onClick={() => {
                    resetForm();
                    setIsCreateDialogOpen(true);
                  }}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Class
                  </Button>
                </div>
              ) : (
                <div className="border rounded-lg">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/30">
                        <TableHead className="w-[50px]">
                          <Checkbox
                            checked={allSelectableSelected}
                            onCheckedChange={toggleSelectAll}
                            aria-label="Select all"
                          />
                        </TableHead>
                        <TableHead className="w-[60px]">Color</TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleSort("name")}
                          >
                            Name
                            {sortBy === "name" ? (
                              sortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>Display Name</TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleSort("category")}
                          >
                            Category
                            {sortBy === "category" ? (
                              sortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleSort("annotation_count")}
                          >
                            Annotations
                            {sortBy === "annotation_count" ? (
                              sortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead className="w-[50px]"></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredClasses?.map((cls) => (
                        <TableRow
                          key={cls.id}
                          className={selectedClassIds.has(cls.id) ? "bg-primary/5" : ""}
                        >
                          <TableCell>
                            <Checkbox
                              checked={selectedClassIds.has(cls.id)}
                              onCheckedChange={() => toggleSelect(cls.id)}
                              aria-label={`Select ${cls.name}`}
                            />
                          </TableCell>
                          <TableCell>
                            <div
                              className="w-8 h-8 rounded-md border shadow-sm"
                              style={{ backgroundColor: cls.color }}
                            />
                          </TableCell>
                          <TableCell className="font-medium">{cls.name}</TableCell>
                          <TableCell className="text-muted-foreground">
                            {cls.display_name || "-"}
                          </TableCell>
                          <TableCell>
                            {cls.category ? (
                              <Badge variant="secondary">{cls.category}</Badge>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-right tabular-nums">
                            {cls.annotation_count.toLocaleString()}
                          </TableCell>
                          <TableCell>
                            {cls.is_system ? (
                              <Badge variant="outline" className="gap-1">
                                <Lock className="h-3 w-3" />
                                System
                              </Badge>
                            ) : (
                              <Badge variant="secondary">Custom</Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => openEditDialog(cls)}>
                                  <Edit className="h-4 w-4 mr-2" />
                                  Edit
                                </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                className="text-destructive"
                                onClick={() => handleDelete(cls)}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Create Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              New Class
            </DialogTitle>
            <DialogDescription>
              Create a new detection class for {selectedDataset?.name || "the dataset"}.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., product, price_tag, shelf"
                value={formName}
                onChange={(e) => setFormName(e.target.value.toLowerCase().replace(/\s+/g, "_"))}
              />
              <p className="text-xs text-muted-foreground">
                Use lowercase with underscores (e.g., price_tag)
              </p>
            </div>

            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g., Product, Price Tag, Shelf"
                value={formDisplayName}
                onChange={(e) => setFormDisplayName(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Human-readable name shown in the UI
              </p>
            </div>

            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex gap-2 flex-wrap">
                {PRESET_COLORS.map((color) => (
                  <button
                    key={color}
                    type="button"
                    className={`w-8 h-8 rounded-md border-2 transition-all ${
                      formColor === color ? "border-foreground scale-110" : "border-transparent"
                    }`}
                    style={{ backgroundColor: color }}
                    onClick={() => setFormColor(color)}
                  />
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Palette className="h-4 w-4 text-muted-foreground" />
                <Input
                  type="color"
                  value={formColor}
                  onChange={(e) => setFormColor(e.target.value)}
                  className="w-20 h-8 p-1"
                />
                <span className="text-sm text-muted-foreground font-mono">{formColor}</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Category</Label>
              <Input
                placeholder="e.g., retail, shelf_elements"
                value={formCategory}
                onChange={(e) => setFormCategory(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Group related classes together
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!formName || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Class"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit className="h-5 w-5" />
              Edit Class
            </DialogTitle>
            <DialogDescription>
              Update the class details.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., product, price_tag, shelf"
                value={formName}
                onChange={(e) => setFormName(e.target.value.toLowerCase().replace(/\s+/g, "_"))}
              />
            </div>

            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g., Product, Price Tag, Shelf"
                value={formDisplayName}
                onChange={(e) => setFormDisplayName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex gap-2 flex-wrap">
                {PRESET_COLORS.map((color) => (
                  <button
                    key={color}
                    type="button"
                    className={`w-8 h-8 rounded-md border-2 transition-all ${
                      formColor === color ? "border-foreground scale-110" : "border-transparent"
                    }`}
                    style={{ backgroundColor: color }}
                    onClick={() => setFormColor(color)}
                  />
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Palette className="h-4 w-4 text-muted-foreground" />
                <Input
                  type="color"
                  value={formColor}
                  onChange={(e) => setFormColor(e.target.value)}
                  className="w-20 h-8 p-1"
                />
                <span className="text-sm text-muted-foreground font-mono">{formColor}</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Category</Label>
              <Input
                placeholder="e.g., retail, shelf_elements"
                value={formCategory}
                onChange={(e) => setFormCategory(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => updateMutation.mutate()}
              disabled={!formName || updateMutation.isPending}
            >
              {updateMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Changes"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Merge Dialog */}
      <Dialog open={isMergeDialogOpen} onOpenChange={setIsMergeDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Merge className="h-5 w-5" />
              Merge Classes
            </DialogTitle>
            <DialogDescription>
              Combine multiple classes into one. All annotations from source classes will be moved to the target.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Source Classes (to merge from)</Label>
              <div className="border rounded-md p-3 max-h-40 overflow-y-auto space-y-2">
                {classes?.filter((c) => c.id !== mergeTargetId).map((cls) => (
                  <label key={cls.id} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={mergeSourceIds.includes(cls.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setMergeSourceIds([...mergeSourceIds, cls.id]);
                        } else {
                          setMergeSourceIds(mergeSourceIds.filter((id) => id !== cls.id));
                        }
                      }}
                      className="rounded"
                    />
                    <div
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: cls.color }}
                    />
                    <span>{cls.name}</span>
                    <span className="text-muted-foreground text-sm">
                      ({cls.annotation_count} annotations)
                    </span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {mergeSourceIds.length} class(es) selected
              </p>
            </div>

            <div className="space-y-2">
              <Label>Target Class (merge into)</Label>
              <Select value={mergeTargetId} onValueChange={setMergeTargetId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select target class" />
                </SelectTrigger>
                <SelectContent>
                  {classes?.filter((c) => !mergeSourceIds.includes(c.id)).map((cls) => (
                    <SelectItem key={cls.id} value={cls.id}>
                      <div className="flex items-center gap-2">
                        <div
                          className="w-4 h-4 rounded"
                          style={{ backgroundColor: cls.color }}
                        />
                        {cls.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Merge Preview Summary */}
            {mergePreview && mergeTargetId && (
              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <h4 className="font-medium text-sm">Merge Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Annotations to move</p>
                    <p className="text-xl font-bold text-blue-600">
                      {mergePreview.sourceAnnotations.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Target current</p>
                    <p className="text-xl font-bold">
                      {mergePreview.targetAnnotations.toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 pt-2 border-t">
                  <span className="text-sm text-muted-foreground">After merge:</span>
                  <span className="font-mono font-bold">
                    {mergePreview.targetName}
                  </span>
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  <span className="text-lg font-bold text-green-600">
                    {mergePreview.totalAfterMerge.toLocaleString()} annotations
                  </span>
                </div>
              </div>
            )}

            {mergeSourceIds.length > 0 && mergeTargetId && (
              <div className="bg-orange-50 text-orange-800 p-3 rounded-md text-sm">
                <strong>Warning:</strong> This will permanently delete {mergeSourceIds.length} source class(es) and move all their annotations to the target class. This action cannot be undone.
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsMergeDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => mergeMutation.mutate()}
              disabled={mergeSourceIds.length === 0 || !mergeTargetId || mergeMutation.isPending}
            >
              {mergeMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Merging...
                </>
              ) : (
                <>
                  <Merge className="h-4 w-4 mr-2" />
                  Merge {mergeSourceIds.length} Class(es)
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
