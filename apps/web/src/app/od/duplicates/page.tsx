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
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import {
  RefreshCw,
  Loader2,
  ImageIcon,
  Trash2,
  CheckCircle,
  Copy,
  ArrowRight,
  Eye,
  AlertTriangle,
  Layers,
} from "lucide-react";
import Image from "next/image";

interface DuplicateGroup {
  images: Array<{
    id: string;
    filename: string;
    image_url: string;
    similarity?: number;
  }>;
  max_similarity: number;
}

export default function ODDuplicatesPage() {
  const queryClient = useQueryClient();
  const [threshold, setThreshold] = useState(10);
  const [selectedGroup, setSelectedGroup] = useState<DuplicateGroup | null>(null);
  const [keepImageId, setKeepImageId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Fetch duplicate groups
  const {
    data: duplicatesData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ["od-duplicates", threshold],
    queryFn: () => apiClient.getODImageDuplicateGroups(threshold),
  });

  // Resolve duplicates mutation
  const resolveMutation = useMutation({
    mutationFn: async ({
      keepId,
      deleteIds,
    }: {
      keepId: string;
      deleteIds: string[];
    }) => {
      return apiClient.resolveODImageDuplicates(keepId, deleteIds);
    },
    onSuccess: (data) => {
      toast.success(`Deleted ${data.deleted} duplicate images`);
      queryClient.invalidateQueries({ queryKey: ["od-duplicates"] });
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setSelectedGroup(null);
      setKeepImageId(null);
      setConfirmDelete(false);
    },
    onError: (error) => {
      toast.error(`Failed to resolve duplicates: ${error}`);
    },
  });

  const handleResolve = () => {
    if (!selectedGroup || !keepImageId) return;

    const deleteIds = selectedGroup.images
      .filter((img) => img.id !== keepImageId)
      .map((img) => img.id);

    resolveMutation.mutate({ keepId: keepImageId, deleteIds });
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 95) return "text-red-600 bg-red-50";
    if (similarity >= 85) return "text-orange-600 bg-orange-50";
    return "text-yellow-600 bg-yellow-50";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Duplicate Detection</h1>
          <p className="text-muted-foreground">
            Find and resolve duplicate images using perceptual hash comparison
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Threshold Control */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Similarity Threshold</CardTitle>
          <CardDescription>
            Lower values find more duplicates (stricter matching)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Slider
                value={[threshold]}
                onValueChange={(v) => setThreshold(v[0])}
                min={0}
                max={20}
                step={1}
              />
            </div>
            <div className="w-20 text-center">
              <Badge variant="outline" className="text-lg px-3 py-1">
                {threshold}
              </Badge>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Hamming distance: 0 = exact match, 5 = very similar, 10 = similar, 15+ = different
          </p>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Duplicate Groups</CardTitle>
              <CardDescription>
                {duplicatesData?.total_groups ?? 0} groups of potential duplicates found
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : !duplicatesData?.groups || duplicatesData.groups.length === 0 ? (
            <div className="text-center py-24">
              <CheckCircle className="h-16 w-16 mx-auto text-green-500 mb-4" />
              <h3 className="text-lg font-medium">No duplicates found</h3>
              <p className="text-muted-foreground mt-1">
                Your image library is clean! Try adjusting the threshold to find similar images.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {duplicatesData.groups.map((group, index) => (
                <div
                  key={index}
                  className="border rounded-lg p-4 hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Copy className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">
                        {group.images.length} similar images
                      </span>
                      <Badge
                        variant="outline"
                        className={getSimilarityColor(group.max_similarity)}
                      >
                        {group.max_similarity.toFixed(0)}% similar
                      </Badge>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSelectedGroup(group);
                        setKeepImageId(group.images[0]?.id || null);
                      }}
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      Review
                    </Button>
                  </div>

                  {/* Preview thumbnails */}
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {group.images.slice(0, 6).map((image, imgIndex) => (
                      <div
                        key={image.id}
                        className="relative flex-shrink-0 w-20 h-20 rounded-md overflow-hidden bg-muted"
                      >
                        <Image
                          src={image.image_url}
                          alt={image.filename}
                          fill
                          className="object-cover"
                          unoptimized
                        />
                        {imgIndex > 0 && (
                          <div className="absolute inset-0 bg-red-500/20 flex items-center justify-center">
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                          </div>
                        )}
                      </div>
                    ))}
                    {group.images.length > 6 && (
                      <div className="flex-shrink-0 w-20 h-20 rounded-md bg-muted flex items-center justify-center">
                        <span className="text-sm text-muted-foreground">
                          +{group.images.length - 6}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Review Dialog */}
      <Dialog open={!!selectedGroup} onOpenChange={(open) => !open && setSelectedGroup(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Review Duplicate Group
            </DialogTitle>
            <DialogDescription>
              Select the image to keep. All other images will be deleted.
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-auto py-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {selectedGroup?.images.map((image, index) => (
                <div
                  key={image.id}
                  className={`relative rounded-lg overflow-hidden border-2 cursor-pointer transition-all ${
                    keepImageId === image.id
                      ? "border-green-500 ring-2 ring-green-500/20"
                      : "border-transparent hover:border-muted-foreground/30"
                  }`}
                  onClick={() => setKeepImageId(image.id)}
                >
                  <div className="aspect-square relative bg-muted">
                    <Image
                      src={image.image_url}
                      alt={image.filename}
                      fill
                      className="object-cover"
                      unoptimized
                    />
                    {keepImageId === image.id ? (
                      <div className="absolute inset-0 bg-green-500/20 flex items-center justify-center">
                        <Badge className="bg-green-600 text-white">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Keep
                        </Badge>
                      </div>
                    ) : (
                      <div className="absolute inset-0 bg-red-500/10 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                        <Badge variant="destructive">
                          <Trash2 className="h-3 w-3 mr-1" />
                          Will Delete
                        </Badge>
                      </div>
                    )}
                  </div>
                  <div className="p-2 bg-background">
                    <p className="text-xs font-medium truncate">{image.filename}</p>
                    {image.similarity !== undefined && (
                      <p className="text-xs text-muted-foreground">
                        {image.similarity.toFixed(0)}% similar
                      </p>
                    )}
                  </div>
                  {index === 0 && (
                    <Badge
                      variant="outline"
                      className="absolute top-2 right-2 bg-background"
                    >
                      Original
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </div>

          <DialogFooter className="border-t pt-4">
            <div className="flex items-center justify-between w-full">
              <div className="text-sm text-muted-foreground">
                {keepImageId && selectedGroup && (
                  <>
                    <span className="text-green-600 font-medium">1</span> to keep,{" "}
                    <span className="text-red-600 font-medium">
                      {selectedGroup.images.length - 1}
                    </span>{" "}
                    to delete
                  </>
                )}
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setSelectedGroup(null)}>
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={() => setConfirmDelete(true)}
                  disabled={!keepImageId || resolveMutation.isPending}
                >
                  {resolveMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Resolving...
                    </>
                  ) : (
                    <>
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete Duplicates
                    </>
                  )}
                </Button>
              </div>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Duplicate Images?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete{" "}
              {selectedGroup ? selectedGroup.images.length - 1 : 0} duplicate
              images. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleResolve}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
