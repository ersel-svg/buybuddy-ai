"use client";

import { useState, useEffect } from "react";
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
import type { CutoutImage, Job } from "@/types";
import Image from "next/image";
import { Checkbox } from "@/components/ui/checkbox";

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
  const [selectedMerchantIds, setSelectedMerchantIds] = useState<number[]>([]);
  const [insertedAt, setInsertedAt] = useState("");
  const [updatedAt, setUpdatedAt] = useState("");

  // Active sync job tracking
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<Job | null>(null);

  // Poll active job status
  useEffect(() => {
    if (!activeJobId) return;

    const pollInterval = setInterval(async () => {
      try {
        const job = await apiClient.getJob(activeJobId);
        setActiveJob(job);

        if (job.status === "completed") {
          const result = job.result as { synced_count?: number; message?: string } | undefined;
          toast.success(result?.message || `Synced ${result?.synced_count || 0} cutouts`);
          setActiveJobId(null);
          setActiveJob(null);
          queryClient.invalidateQueries({ queryKey: ["cutouts"] });
          queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
          queryClient.invalidateQueries({ queryKey: ["cutout-sync-state"] });
        } else if (job.status === "failed") {
          toast.error(`Sync failed: ${job.error || "Unknown error"}`);
          setActiveJobId(null);
          setActiveJob(null);
        } else if (job.status === "cancelled") {
          toast.info("Sync job was cancelled");
          setActiveJobId(null);
          setActiveJob(null);
        }
      } catch {
        console.error("Failed to poll job status");
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [activeJobId, queryClient]);

  // Fetch merchants from BuyBuddy API
  const { data: merchants, isLoading: merchantsLoading } = useQuery({
    queryKey: ["merchants"],
    queryFn: () => apiClient.getMerchants(),
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

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

  // Sync New mutation - creates background job
  const syncNewMutation = useMutation({
    mutationFn: (params: { merchant_ids: number[]; max_items?: number; page_size?: number; inserted_at?: string; updated_at?: string }) =>
      apiClient.syncNewCutouts(params),
    onSuccess: (data) => {
      toast.info(`Sync job started. Tracking progress...`);
      setActiveJobId(data.job_id);
      setIsSyncNewDialogOpen(false);
    },
    onError: (error) => {
      toast.error(`Failed to start sync: ${error.message}`);
    },
  });

  // Backfill mutation - creates background job
  const backfillMutation = useMutation({
    mutationFn: (params: { merchant_ids: number[]; max_items?: number; page_size?: number; start_page?: number; inserted_at?: string; updated_at?: string }) =>
      apiClient.backfillCutouts(params),
    onSuccess: (data) => {
      toast.info(`Backfill job started. Tracking progress...`);
      setActiveJobId(data.job_id);
      setIsBackfillDialogOpen(false);
    },
    onError: (error) => {
      toast.error(`Failed to start backfill: ${error.message}`);
    },
  });

  const handleSyncNew = () => {
    if (selectedMerchantIds.length === 0) {
      toast.error("Please select at least one merchant");
      return;
    }
    const limit = parseInt(syncLimit) || 1000;
    syncNewMutation.mutate({
      merchant_ids: selectedMerchantIds,
      max_items: limit,
      page_size: 100,
      inserted_at: insertedAt || undefined,
      updated_at: updatedAt || undefined,
    });
  };

  const handleBackfill = () => {
    if (selectedMerchantIds.length === 0) {
      toast.error("Please select at least one merchant");
      return;
    }
    const limit = parseInt(syncLimit) || 1000;
    const startPage = parseInt(backfillStartPage) || 1;
    backfillMutation.mutate({
      merchant_ids: selectedMerchantIds,
      max_items: limit,
      page_size: 100,
      start_page: startPage,
      inserted_at: insertedAt || undefined,
      updated_at: updatedAt || undefined,
    });
  };

  const toggleMerchant = (merchantId: number) => {
    setSelectedMerchantIds((prev) =>
      prev.includes(merchantId)
        ? prev.filter((id) => id !== merchantId)
        : [...prev, merchantId]
    );
  };

  const totalPages = cutoutsData
    ? Math.ceil(cutoutsData.total / PAGE_SIZE)
    : 0;

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return "Never";
    return new Date(dateStr).toLocaleString();
  };

  const isSyncing = syncNewMutation.isPending || backfillMutation.isPending || !!activeJobId;

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
            {syncNewMutation.isPending || (activeJob && activeJob.config?.mode === "sync_new") ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                {activeJob ? `${activeJob.progress}%` : "Starting..."}
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
            {backfillMutation.isPending || (activeJob && activeJob.config?.mode === "backfill") ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                {activeJob ? `${activeJob.progress}%` : "Starting..."}
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

          {/* Active Job Progress */}
          {activeJob && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                  <span className="text-sm font-medium">
                    {activeJob.config?.mode === "sync_new" ? "Syncing New" : "Backfilling"}
                  </span>
                </div>
                <span className="text-sm text-muted-foreground">{activeJob.progress}%</span>
              </div>
              <Progress value={activeJob.progress} className="h-2" />
              {activeJob.current_step && (
                <p className="text-xs text-muted-foreground mt-1">{activeJob.current_step}</p>
              )}
            </div>
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
                    <TableHead>Merchant</TableHead>
                    <TableHead>Predicted UPC</TableHead>
                    <TableHead>Annotated UPC</TableHead>
                    <TableHead>Position</TableHead>
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
                        <span className="text-sm font-medium">
                          {cutout.merchant || "-"}
                        </span>
                      </TableCell>
                      <TableCell>
                        <code className="text-sm bg-muted px-2 py-1 rounded">
                          {cutout.predicted_upc || "-"}
                        </code>
                      </TableCell>
                      <TableCell>
                        {cutout.annotated_upc ? (
                          <code className="text-sm bg-green-100 text-green-800 px-2 py-1 rounded">
                            {cutout.annotated_upc}
                          </code>
                        ) : (
                          <span className="text-muted-foreground text-sm">-</span>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {cutout.row_index && cutout.column_index
                          ? `R${cutout.row_index} C${cutout.column_index}`
                          : "-"}
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
              <Label>Select Merchants (Required)</Label>
              <div className="grid grid-cols-2 gap-2 p-3 border rounded-lg">
                {merchantsLoading ? (
                  <div className="col-span-2 flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading merchants...
                  </div>
                ) : merchants && merchants.length > 0 ? (
                  merchants.map((merchant) => (
                    <div key={merchant.id} className="flex items-center space-x-2">
                      <Checkbox
                        id={`sync-merchant-${merchant.id}`}
                        checked={selectedMerchantIds.includes(merchant.id)}
                        onCheckedChange={() => toggleMerchant(merchant.id)}
                      />
                      <label
                        htmlFor={`sync-merchant-${merchant.id}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        {merchant.name}
                      </label>
                    </div>
                  ))
                ) : (
                  <p className="col-span-2 text-sm text-muted-foreground">No merchants available</p>
                )}
              </div>
              {selectedMerchantIds.length === 0 && !merchantsLoading && (
                <p className="text-xs text-red-500">Please select at least one merchant</p>
              )}
            </div>
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
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Inserted After (optional)</Label>
                <Input
                  type="date"
                  value={insertedAt}
                  onChange={(e) => setInsertedAt(e.target.value)}
                  placeholder="YYYY-MM-DD"
                />
              </div>
              <div className="space-y-2">
                <Label>Updated After (optional)</Label>
                <Input
                  type="date"
                  value={updatedAt}
                  onChange={(e) => setUpdatedAt(e.target.value)}
                  placeholder="YYYY-MM-DD"
                />
              </div>
            </div>
            {(insertedAt || updatedAt) && (
              <p className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                Filtering by date: {insertedAt && `inserted_at ≥ ${insertedAt}`} {insertedAt && updatedAt && " and "} {updatedAt && `updated_at ≥ ${updatedAt}`}
              </p>
            )}
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
            <Button onClick={handleSyncNew} disabled={syncNewMutation.isPending || selectedMerchantIds.length === 0}>
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
              <Label>Select Merchants (Required)</Label>
              <div className="grid grid-cols-2 gap-2 p-3 border rounded-lg">
                {merchantsLoading ? (
                  <div className="col-span-2 flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading merchants...
                  </div>
                ) : merchants && merchants.length > 0 ? (
                  merchants.map((merchant) => (
                    <div key={merchant.id} className="flex items-center space-x-2">
                      <Checkbox
                        id={`backfill-merchant-${merchant.id}`}
                        checked={selectedMerchantIds.includes(merchant.id)}
                        onCheckedChange={() => toggleMerchant(merchant.id)}
                      />
                      <label
                        htmlFor={`backfill-merchant-${merchant.id}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        {merchant.name}
                      </label>
                    </div>
                  ))
                ) : (
                  <p className="col-span-2 text-sm text-muted-foreground">No merchants available</p>
                )}
              </div>
              {selectedMerchantIds.length === 0 && !merchantsLoading && (
                <p className="text-xs text-red-500">Please select at least one merchant</p>
              )}
            </div>
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
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Inserted After (optional)</Label>
                <Input
                  type="date"
                  value={insertedAt}
                  onChange={(e) => setInsertedAt(e.target.value)}
                  placeholder="YYYY-MM-DD"
                />
              </div>
              <div className="space-y-2">
                <Label>Updated After (optional)</Label>
                <Input
                  type="date"
                  value={updatedAt}
                  onChange={(e) => setUpdatedAt(e.target.value)}
                  placeholder="YYYY-MM-DD"
                />
              </div>
            </div>
            {(insertedAt || updatedAt) && (
              <p className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                Filtering by date: {insertedAt && `inserted_at ≥ ${insertedAt}`} {insertedAt && updatedAt && " and "} {updatedAt && `updated_at ≥ ${updatedAt}`}
              </p>
            )}
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
            <Button onClick={handleBackfill} disabled={backfillMutation.isPending || selectedMerchantIds.length === 0}>
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
