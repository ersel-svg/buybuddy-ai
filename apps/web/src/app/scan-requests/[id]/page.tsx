"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import Link from "next/link";
import Image from "next/image";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  ArrowLeft,
  Loader2,
  Package,
  Clock,
  CheckCircle2,
  XCircle,
  PlayCircle,
  Mail,
  User,
  FileText,
  ImageIcon,
  ExternalLink,
} from "lucide-react";
import type { ScanRequestStatus } from "@/types";

const statusConfig: Record<
  ScanRequestStatus,
  { label: string; color: string; icon: React.ReactNode }
> = {
  pending: {
    label: "Pending",
    color: "bg-yellow-100 text-yellow-800",
    icon: <Clock className="h-4 w-4" />,
  },
  in_progress: {
    label: "In Progress",
    color: "bg-blue-100 text-blue-800",
    icon: <PlayCircle className="h-4 w-4" />,
  },
  completed: {
    label: "Completed",
    color: "bg-green-100 text-green-800",
    icon: <CheckCircle2 className="h-4 w-4" />,
  },
  cancelled: {
    label: "Cancelled",
    color: "bg-gray-100 text-gray-800",
    icon: <XCircle className="h-4 w-4" />,
  },
};

export default function ScanRequestDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const id = params.id as string;

  // Fetch scan request
  const { data: request, isLoading } = useQuery({
    queryKey: ["scan-request", id],
    queryFn: () => apiClient.getScanRequest(id),
    enabled: !!id,
  });

  // Update status mutation
  const updateStatusMutation = useMutation({
    mutationFn: (status: string) => apiClient.updateScanRequest(id, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      queryClient.invalidateQueries({ queryKey: ["scan-request", id] });
      queryClient.invalidateQueries({ queryKey: ["scan-requests"] });
    },
    onError: () => {
      toast.error("Failed to update status");
    },
  });

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: () => apiClient.deleteScanRequest(id),
    onSuccess: () => {
      toast.success("Request cancelled");
      router.push("/scan-requests");
    },
    onError: () => {
      toast.error("Failed to cancel request");
    },
  });

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      weekday: "long",
      month: "long",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (!request) {
    return (
      <div className="text-center py-12">
        <Package className="h-12 w-12 mx-auto text-gray-300 mb-4" />
        <h2 className="text-xl font-semibold mb-2">Request not found</h2>
        <p className="text-gray-500 mb-4">
          The scan request you&apos;re looking for doesn&apos;t exist.
        </p>
        <Link href="/scan-requests">
          <Button>Back to Scan Requests</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              Scan Request
              <Badge
                className={`${statusConfig[request.status].color} flex items-center gap-1`}
              >
                {statusConfig[request.status].icon}
                {statusConfig[request.status].label}
              </Badge>
            </h1>
            <p className="text-gray-500">
              Created {formatDate(request.created_at)}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {request.status === "pending" && (
            <Button
              onClick={() => updateStatusMutation.mutate("in_progress")}
              disabled={updateStatusMutation.isPending}
            >
              {updateStatusMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <PlayCircle className="h-4 w-4 mr-2" />
              )}
              Start Processing
            </Button>
          )}
          {request.status === "in_progress" && (
            <Button
              onClick={() => updateStatusMutation.mutate("completed")}
              disabled={updateStatusMutation.isPending}
            >
              {updateStatusMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <CheckCircle2 className="h-4 w-4 mr-2" />
              )}
              Mark Completed
            </Button>
          )}
          {(request.status === "pending" || request.status === "in_progress") && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive">
                  <XCircle className="h-4 w-4 mr-2" />
                  Cancel Request
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Cancel Scan Request</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to cancel this scan request? This action
                    cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Keep Request</AlertDialogCancel>
                  <AlertDialogAction
                    className="bg-red-600 hover:bg-red-700"
                    onClick={() => cancelMutation.mutate()}
                  >
                    {cancelMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      "Cancel Request"
                    )}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Product Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              Product Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-500">Barcode</label>
              <p className="text-lg font-mono">{request.barcode}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Product Name</label>
              <p className="text-lg">{request.product_name || "-"}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Brand</label>
              <p className="text-lg">{request.brand_name || "-"}</p>
            </div>
            {request.completed_by_product_id && (
              <div className="pt-4 border-t">
                <Link href={`/products/${request.completed_by_product_id}`}>
                  <Button variant="outline" className="w-full">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View Created Product
                  </Button>
                </Link>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Requester Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Requester Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center">
                <User className="h-5 w-5 text-indigo-600" />
              </div>
              <div>
                <p className="font-medium">{request.requester_name}</p>
                <a
                  href={`mailto:${request.requester_email}`}
                  className="text-sm text-indigo-600 hover:underline flex items-center gap-1"
                >
                  <Mail className="h-3 w-3" />
                  {request.requester_email}
                </a>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notes */}
        {request.notes && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Notes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 whitespace-pre-wrap">{request.notes}</p>
            </CardContent>
          </Card>
        )}

        {/* Reference Images */}
        <Card className={request.notes ? "" : "md:col-span-2"}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ImageIcon className="h-5 w-5" />
              Reference Images
              {request.reference_images?.length > 0 && (
                <Badge variant="secondary">{request.reference_images.length}</Badge>
              )}
            </CardTitle>
            <CardDescription>
              Images provided as reference for the product scan
            </CardDescription>
          </CardHeader>
          <CardContent>
            {request.reference_images?.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {request.reference_images.map((imageUrl, index) => (
                  <a
                    key={index}
                    href={imageUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="relative aspect-square rounded-lg overflow-hidden border hover:border-indigo-500 transition-colors group"
                  >
                    <Image
                      src={imageUrl}
                      alt={`Reference image ${index + 1}`}
                      fill
                      className="object-cover"
                      sizes="(max-width: 768px) 50vw, 25vw"
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                      <ExternalLink className="h-6 w-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </a>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <ImageIcon className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                <p>No reference images provided</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
              </div>
              <div>
                <p className="font-medium">Request Created</p>
                <p className="text-sm text-gray-500">{formatDate(request.created_at)}</p>
              </div>
            </div>
            {request.status !== "pending" && request.status !== "cancelled" && (
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                  <PlayCircle className="h-4 w-4 text-blue-600" />
                </div>
                <div>
                  <p className="font-medium">Processing Started</p>
                  <p className="text-sm text-gray-500">{formatDate(request.updated_at)}</p>
                </div>
              </div>
            )}
            {request.status === "completed" && request.completed_at && (
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                </div>
                <div>
                  <p className="font-medium">Request Completed</p>
                  <p className="text-sm text-gray-500">{formatDate(request.completed_at)}</p>
                </div>
              </div>
            )}
            {request.status === "cancelled" && (
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center shrink-0">
                  <XCircle className="h-4 w-4 text-red-600" />
                </div>
                <div>
                  <p className="font-medium">Request Cancelled</p>
                  <p className="text-sm text-gray-500">{formatDate(request.updated_at)}</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
