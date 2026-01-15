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
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
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
  ImageIcon,
  Search,
  CheckCircle,
  XCircle,
  Layers,
  ArrowUpCircle,
  ArrowDownCircle,
  Clock,
  Database,
} from "lucide-react";
import type { CutoutImage } from "@/types";
import Image from "next/image";

const PAGE_SIZE = 50;

export default function CutoutsPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [searchUPC, setSearchUPC] = useState("");
  const [filterEmbedding, setFilterEmbedding] = useState<string>("all");
  const [filterMatched, setFilterMatched] = useState<string>("all");

  // Sync dialog state
  const [isSyncNewDialogOpen, setIsSyncNewDialogOpen] = useState(false);
  const [isBackfillDialogOpen, setIsBackfillDialogOpen] = useState(false);
  const [syncLimit, setSyncLimit] = useState("1000");
  const [backfillStartPage, setBackfillStartPage] = useState("1");

  // Fetch sync state
  const { data: syncState, isLoading: syncStateLoading } = useQuery({
    queryKey: ["cutout-sync-state"],
    queryFn: () => apiClient.getSyncState(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch cutout stats
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["cutout-stats"],
    queryFn: () => apiClient.getCutoutStats(),
  });

  // Fetch cutouts with filters
  const {
    data: cutoutsData,
    isLoading: cutoutsLoading,
    isFetching,
  } = useQuery({
    queryKey: ["cutouts", page, filterEmbedding, filterMatched, searchUPC],
    queryFn: () =>
      apiClient.getCutouts({
        page,
        limit: PAGE_SIZE,
        has_embedding:
          filterEmbedding === "all"
            ? undefined
            : filterEmbedding === "with",
        is_matched:
          filterMatched === "all" ? undefined : filterMatched === "matched",
        predicted_upc: searchUPC || undefined,
      }),
  });

  // Sync New mutation
  const syncNewMutation = useMutation({
    mutationFn: (params: { max_items?: number; page_size?: number }) =>
      apiClient.syncNewCutouts(params),
    onSuccess: (data) => {
      if (data.synced_count > 0) {
        toast.success(
          `Synced ${data.synced_count} new cutouts (ID ${data.lowest_external_id} - ${data.highest_external_id})`
        );
      } else {
        toast.info("No new cutouts to sync - already up to date");
      }
      queryClient.invalidateQueries({ queryKey: ["cutouts"] });
      queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
      queryClient.invalidateQueries({ queryKey: ["cutout-sync-state"] });
      setIsSyncNewDialogOpen(false);
    },
    onError: (error) => {
      toast.error(`Sync failed: ${error.message}`);
    },
  });

  // Backfill mutation
  const backfillMutation = useMutation({
    mutationFn: (params: { max_items?: number; page_size?: number; start_page?: number }) =>
      apiClient.backfillCutouts(params),
    onSuccess: (data) => {
      if (data.synced_count > 0) {
        toast.success(
          `Backfilled ${data.synced_count} cutouts (ID ${data.lowest_external_id} - ${data.highest_external_id}). Last page: ${data.last_page}`
        );
        // Auto-update start page for continuation
        if (data.last_page) {
          setBackfillStartPage(String(data.last_page + 1));
        }
      } else {
        toast.info(`No new cutouts found. Last page checked: ${data.last_page}. Try a higher start page.`);
      }
      queryClient.invalidateQueries({ queryKey: ["cutouts"] });
      queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
      queryClient.invalidateQueries({ queryKey: ["cutout-sync-state"] });
      setIsBackfillDialogOpen(false);
    },
    onError: (error) => {
      toast.error(`Backfill failed: ${error.message}`);
    },
  });

  const handleSyncNew = () => {
    const limit = parseInt(syncLimit) || 1000;
    syncNewMutation.mutate({ max_items: limit, page_size: 100 });
  };

  const handleBackfill = () => {
    const limit = parseInt(syncLimit) || 1000;
    const startPage = parseInt(backfillStartPage) || 1;
    backfillMutation.mutate({ max_items: limit, page_size: 100, start_page: startPage });
  };

  const totalPages = cutoutsData
    ? Math.ceil(cutoutsData.total / PAGE_SIZE)
    : 0;

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return "Never";
    return new Date(dateStr).toLocaleString();
  };

  const isSyncing = syncNewMutation.isPending || backfillMutation.isPending;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Cutouts</h1>
          <p className="text-muted-foreground">
            Real shelf images from BuyBuddy for matching
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["cutouts"] });
              queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
              queryClient.invalidateQueries({ queryKey: ["cutout-sync-state"] });
            }}
            disabled={isFetching}
          >
            <RefreshCw
              className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
          <Button
            variant="default"
            onClick={() => setIsSyncNewDialogOpen(true)}
            disabled={isSyncing}
          >
            {syncNewMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Syncing...
              </>
            ) : (
              <>
                <ArrowUpCircle className="h-4 w-4 mr-2" />
                Sync New
              </>
            )}
          </Button>
          <Button
            variant="outline"
            onClick={() => setIsBackfillDialogOpen(true)}
            disabled={isSyncing}
          >
            {backfillMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Backfilling...
              </>
            ) : (
              <>
                <ArrowDownCircle className="h-4 w-4 mr-2" />
                Backfill
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Sync State Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Database className="h-4 w-4" />
            Sync Status
          </CardTitle>
          <CardDescription>
            BuyBuddy cutout synchronization progress
          </CardDescription>
        </CardHeader>
        <CardContent>
          {syncStateLoading ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading sync state...
            </div>
          ) : syncState ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Synced Range</p>
                <p className="text-sm font-mono">
                  {syncState.min_synced_external_id?.toLocaleString() || "N/A"} - {syncState.max_synced_external_id?.toLocaleString() || "N/A"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Total Synced</p>
                <p className="text-sm font-bold">{syncState.total_synced?.toLocaleString() || 0}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">BuyBuddy Max ID</p>
                <p className="text-sm font-mono">{syncState.buybuddy_max_id?.toLocaleString() || "Unknown"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Est. Remaining</p>
                <p className="text-sm">
                  {syncState.buybuddy_max_id && syncState.max_synced_external_id
                    ? (syncState.buybuddy_max_id - syncState.max_synced_external_id).toLocaleString()
                    : "Unknown"
                  } new
                </p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> Last Sync New
                </p>
                <p className="text-sm">{formatDate(syncState.last_sync_new_at)}</p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> Last Backfill
                </p>
                <p className="text-sm">{formatDate(syncState.last_backfill_at)}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No sync state available</p>
          )}
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total Cutouts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {statsLoading ? "-" : stats?.total.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              With Embedding
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {statsLoading ? "-" : stats?.with_embedding.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Without Embedding
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">
              {statsLoading ? "-" : stats?.without_embedding.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Matched
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {statsLoading ? "-" : stats?.matched.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Unmatched
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-600">
              {statsLoading ? "-" : stats?.unmatched.toLocaleString()}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Progress indicator for embedding coverage */}
      {stats && stats.total > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Embedding Coverage</CardTitle>
            <CardDescription>
              {stats.with_embedding} of {stats.total} cutouts have embeddings (
              {Math.round((stats.with_embedding / stats.total) * 100)}%)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Progress
              value={(stats.with_embedding / stats.total) * 100}
              className="h-2"
            />
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Cutouts List</CardTitle>
          <CardDescription>
            Browse and filter cutout images synced from BuyBuddy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filter Row */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by UPC..."
                  value={searchUPC}
                  onChange={(e) => {
                    setSearchUPC(e.target.value);
                    setPage(1);
                  }}
                  className="pl-10"
                />
              </div>
            </div>
            <Select
              value={filterEmbedding}
              onValueChange={(v) => {
                setFilterEmbedding(v);
                setPage(1);
              }}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Embedding status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Embeddings</SelectItem>
                <SelectItem value="with">With Embedding</SelectItem>
                <SelectItem value="without">Without Embedding</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={filterMatched}
              onValueChange={(v) => {
                setFilterMatched(v);
                setPage(1);
              }}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Match status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="matched">Matched</SelectItem>
                <SelectItem value="unmatched">Unmatched</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Table */}
          {cutoutsLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : cutoutsData?.items.length === 0 ? (
            <div className="text-center py-12">
              <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No cutouts found</p>
              <p className="text-sm text-muted-foreground mt-1">
                Try adjusting filters or sync from BuyBuddy
              </p>
            </div>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[80px]">Image</TableHead>
                    <TableHead>Predicted UPC</TableHead>
                    <TableHead>External ID</TableHead>
                    <TableHead>Embedding</TableHead>
                    <TableHead>Match Status</TableHead>
                    <TableHead>Synced</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {cutoutsData?.items.map((cutout: CutoutImage) => (
                    <TableRow key={cutout.id}>
                      <TableCell>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="relative w-16 h-16 rounded overflow-hidden bg-muted cursor-pointer">
                              <Image
                                src={cutout.image_url}
                                alt={cutout.predicted_upc || "Cutout"}
                                fill
                                className="object-cover"
                                unoptimized
                              />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent
                            side="right"
                            className="p-0 bg-transparent border-0"
                            sideOffset={8}
                          >
                            <img
                              src={cutout.image_url}
                              alt={cutout.predicted_upc || "Cutout"}
                              className="w-48 h-48 object-contain rounded-lg shadow-lg bg-white"
                            />
                          </TooltipContent>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <code className="text-sm bg-muted px-2 py-1 rounded">
                          {cutout.predicted_upc || "-"}
                        </code>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        #{cutout.external_id}
                      </TableCell>
                      <TableCell>
                        {cutout.has_embedding ? (
                          <Badge variant="default" className="bg-green-600">
                            <Layers className="h-3 w-3 mr-1" />
                            Has Embedding
                          </Badge>
                        ) : (
                          <Badge variant="secondary">
                            <XCircle className="h-3 w-3 mr-1" />
                            No Embedding
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        {cutout.matched_product_id ? (
                          <Badge variant="default" className="bg-blue-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Matched
                          </Badge>
                        ) : (
                          <Badge variant="outline">
                            Unmatched
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {new Date(cutout.created_at).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-4">
                  <p className="text-sm text-muted-foreground">
                    Page {page} of {totalPages} ({cutoutsData?.total} total)
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Sync New Dialog */}
      <Dialog open={isSyncNewDialogOpen} onOpenChange={setIsSyncNewDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ArrowUpCircle className="h-5 w-5" />
              Sync New Cutouts
            </DialogTitle>
            <DialogDescription>
              Fetch new cutouts from BuyBuddy (ID &gt; {syncState?.max_synced_external_id?.toLocaleString() || "0"})
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Maximum cutouts to sync</Label>
              <Input
                type="number"
                value={syncLimit}
                onChange={(e) => setSyncLimit(e.target.value)}
                placeholder="1000"
                min="100"
                max="100000"
                step="100"
              />
              <p className="text-xs text-muted-foreground">
                Will sync up to {parseInt(syncLimit) || 0} new cutouts.
                {syncState?.buybuddy_max_id && syncState?.max_synced_external_id && (
                  <span className="ml-1">
                    (~{(syncState.buybuddy_max_id - syncState.max_synced_external_id).toLocaleString()} available)
                  </span>
                )}
              </p>
            </div>
            {parseInt(syncLimit) > 10000 && (
              <p className="text-sm text-orange-600 bg-orange-50 p-3 rounded-lg">
                Warning: Syncing more than 10,000 cutouts may take a while.
              </p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSyncNewDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSyncNew} disabled={syncNewMutation.isPending}>
              {syncNewMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Syncing...
                </>
              ) : (
                <>
                  <ArrowUpCircle className="h-4 w-4 mr-2" />
                  Sync New
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Backfill Dialog */}
      <Dialog open={isBackfillDialogOpen} onOpenChange={setIsBackfillDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ArrowDownCircle className="h-5 w-5" />
              Backfill Old Cutouts
            </DialogTitle>
            <DialogDescription>
              Fetch historical cutouts from BuyBuddy (ID &lt; {syncState?.min_synced_external_id?.toLocaleString() || "0"})
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Maximum cutouts to backfill</Label>
              <Input
                type="number"
                value={syncLimit}
                onChange={(e) => setSyncLimit(e.target.value)}
                placeholder="1000"
                min="100"
                max="100000"
                step="100"
              />
              <p className="text-xs text-muted-foreground">
                Will backfill up to {parseInt(syncLimit) || 0} historical cutouts.
              </p>
            </div>
            <div className="space-y-2">
              <Label>Start from page</Label>
              <Input
                type="number"
                value={backfillStartPage}
                onChange={(e) => setBackfillStartPage(e.target.value)}
                placeholder="1"
                min="1"
                step="1"
              />
              {syncState?.last_backfill_page && syncState.last_backfill_page > 1 && (
                <p className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                  Last backfill ended at page {syncState.last_backfill_page}.
                  <button
                    type="button"
                    className="ml-1 underline font-medium"
                    onClick={() => setBackfillStartPage(String(syncState.last_backfill_page! + 1))}
                  >
                    Continue from page {syncState.last_backfill_page + 1}
                  </button>
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                Start from this page (100 items/page). Use higher values to fill gaps in the middle.
              </p>
            </div>
            {syncState?.backfill_completed && (
              <p className="text-sm text-green-600 bg-green-50 p-3 rounded-lg">
                Backfill already completed - all historical data has been synced.
              </p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsBackfillDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleBackfill} disabled={backfillMutation.isPending}>
              {backfillMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Backfilling...
                </>
              ) : (
                <>
                  <ArrowDownCircle className="h-4 w-4 mr-2" />
                  Start Backfill
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
