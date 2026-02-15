"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useStoreMaps, useDeleteStoreMap } from "@/hooks/store-maps/use-map-queries";
import { CreateMapDialog } from "@/components/store-maps/create-map-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Search,
  Plus,
  Map,
  Layers,
  MoreHorizontal,
  Trash2,
  Edit,
  Loader2,
  FolderOpen,
} from "lucide-react";

export default function StoreMapsPage() {
  const router = useRouter();
  const { data: maps, isLoading } = useStoreMaps();
  const deleteMap = useDeleteStoreMap();

  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);

  const filteredMaps = (maps ?? []).filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase())
  );

  const handleMapCreated = (mapId: number) => {
    router.push(`/store-maps/editor?id=${mapId}`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Store Maps</h1>
          <p className="text-muted-foreground">
            Create and manage store layout maps with areas, products, and
            planograms.
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Map
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total Maps
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{maps?.length ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total Floors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {maps?.reduce((acc, m) => acc + (m.floor_count ?? 0), 0) ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total Areas
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {maps?.reduce((acc, m) => acc + (m.area_count ?? 0), 0) ?? 0}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search maps..."
            className="pl-10"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center py-24">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : filteredMaps.length === 0 ? (
        <div className="text-center py-24">
          <FolderOpen className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">
            {search ? "No maps found" : "No store maps yet"}
          </h3>
          <p className="text-muted-foreground mt-1">
            {search
              ? "Try adjusting your search."
              : "Create your first store map to get started."}
          </p>
          {!search && (
            <Button className="mt-4" onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Map
            </Button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredMaps.map((map) => (
            <Card
              key={map.id}
              className="hover:border-primary/50 transition-colors cursor-pointer group"
              onClick={() => router.push(`/store-maps/editor?id=${map.id}`)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <Map className="h-5 w-5 text-muted-foreground" />
                    <CardTitle className="text-base">{map.name}</CardTitle>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          router.push(`/store-maps/editor?id=${map.id}`);
                        }}
                      >
                        <Edit className="h-4 w-4 mr-2" />
                        Open Editor
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="text-red-600"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteId(map.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {map.store_name && (
                  <p className="text-sm text-muted-foreground">
                    Store: {map.store_name}
                  </p>
                )}
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="secondary">
                    <Layers className="h-3 w-3 mr-1" />
                    {map.floor_count ?? 0} floors
                  </Badge>
                  <Badge variant="secondary">
                    {map.area_count ?? 0} areas
                  </Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Dialog */}
      <CreateMapDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onSuccess={handleMapCreated}
      />

      {/* Delete Confirmation */}
      <AlertDialog
        open={deleteId !== null}
        onOpenChange={(open) => !open && setDeleteId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Store Map</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this map? This will remove all
              floors, areas, and coordinates. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => {
                if (deleteId) {
                  deleteMap.mutate(deleteId);
                  setDeleteId(null);
                }
              }}
            >
              {deleteMap.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
