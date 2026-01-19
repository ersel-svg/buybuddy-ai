"use client";

import { useState } from "react";
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
} from "lucide-react";

interface ODClass {
  id: string;
  name: string;
  display_name?: string;
  color: string;
  category?: string;
  annotation_count: number;
  is_system: boolean;
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

  // Fetch classes
  const { data: classes, isLoading, isFetching } = useQuery({
    queryKey: ["od-classes", filterCategory],
    queryFn: () =>
      apiClient.getODClasses({
        category: filterCategory !== "all" ? filterCategory : undefined,
        is_active: true,
      }),
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createODClass({
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
    if (cls.is_system) {
      toast.error("Cannot delete system classes");
      return;
    }
    if (cls.annotation_count > 0) {
      if (!confirm(`This class has ${cls.annotation_count} annotations. Delete anyway?`)) {
        return;
      }
    } else {
      if (!confirm(`Delete class "${cls.name}"?`)) {
        return;
      }
    }
    deleteMutation.mutate(cls.id);
  };

  // Filter classes by search
  const filteredClasses = classes?.filter((cls) =>
    cls.name.toLowerCase().includes(search.toLowerCase()) ||
    cls.display_name?.toLowerCase().includes(search.toLowerCase())
  );

  // Get unique categories
  const categories = [...new Set(classes?.map((c) => c.category).filter(Boolean) || [])];

  // Calculate stats
  const totalAnnotations = classes?.reduce((sum, c) => sum + c.annotation_count, 0) || 0;
  const systemClasses = classes?.filter((c) => c.is_system).length || 0;
  const customClasses = classes?.filter((c) => !c.is_system).length || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Classes</h1>
          <p className="text-muted-foreground">
            Manage detection classes and labels
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["od-classes"] })}
            disabled={isFetching}
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
          }}>
            <Plus className="h-4 w-4 mr-2" />
            New Class
          </Button>
        </div>
      </div>

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
          <CardTitle>Class Library</CardTitle>
          <CardDescription>
            {classes?.length ?? 0} detection classes available
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
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[60px]">Color</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Display Name</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead className="text-right">Annotations</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredClasses?.map((cls) => (
                  <TableRow key={cls.id}>
                    <TableCell>
                      <div
                        className="w-8 h-8 rounded-md border"
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
                    <TableCell className="text-right font-mono">
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
                            disabled={cls.is_system}
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
          )}
        </CardContent>
      </Card>

      {/* Create Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              New Class
            </DialogTitle>
            <DialogDescription>
              Create a new detection class for annotating objects.
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
              {editingClass?.is_system && (
                <span className="block mt-1 text-orange-600">
                  Note: System class names cannot be changed.
                </span>
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., product, price_tag, shelf"
                value={formName}
                onChange={(e) => setFormName(e.target.value.toLowerCase().replace(/\s+/g, "_"))}
                disabled={editingClass?.is_system}
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
                {classes?.filter((c) => !c.is_system && c.id !== mergeTargetId).map((cls) => (
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
                {mergeSourceIds.length} class(es) selected. System classes cannot be merged.
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

            {mergeSourceIds.length > 0 && mergeTargetId && (
              <div className="bg-orange-50 text-orange-800 p-3 rounded-md text-sm">
                <strong>Warning:</strong> This will permanently delete the source classes and move all their annotations to the target class. This action cannot be undone.
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
