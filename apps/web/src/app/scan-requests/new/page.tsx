"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ArrowLeft, Loader2, Upload, X, AlertTriangle } from "lucide-react";
import type { ScanRequest } from "@/types";

export default function NewScanRequestPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Form state - initialize barcode from URL if present
  const [barcode, setBarcode] = useState(searchParams.get("barcode") || "");
  const [productName, setProductName] = useState("");
  const [brandName, setBrandName] = useState("");
  const [notes, setNotes] = useState("");
  const [requesterName, setRequesterName] = useState("");
  const [requesterEmail, setRequesterEmail] = useState("");

  // Image upload state
  const [images, setImages] = useState<{ file: File; preview: string; path?: string }[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // Duplicate check state
  const [duplicateWarning, setDuplicateWarning] = useState<ScanRequest[] | null>(null);
  const [isCheckingDuplicate, setIsCheckingDuplicate] = useState(false);

  // Check for duplicate when barcode changes
  const checkDuplicate = useCallback(async (barcodeValue: string) => {
    if (!barcodeValue || barcodeValue.length < 3) {
      setDuplicateWarning(null);
      return;
    }

    setIsCheckingDuplicate(true);
    try {
      const result = await apiClient.checkDuplicateScanRequest(barcodeValue);
      if (result.has_duplicate) {
        setDuplicateWarning(result.existing_requests);
      } else {
        setDuplicateWarning(null);
      }
    } catch {
      // Ignore errors during duplicate check
    } finally {
      setIsCheckingDuplicate(false);
    }
  }, []);

  // Check duplicate on initial load if barcode is provided via URL
  useEffect(() => {
    const initialBarcode = searchParams.get("barcode");
    if (initialBarcode) {
      checkDuplicate(initialBarcode);
    }
  }, [searchParams, checkDuplicate]);

  // Handle barcode input with debounced duplicate check
  const handleBarcodeChange = (value: string) => {
    setBarcode(value);
    // Debounce the duplicate check
    const timeoutId = setTimeout(() => checkDuplicate(value), 500);
    return () => clearTimeout(timeoutId);
  };

  // Handle image upload
  const handleImageUpload = async (files: FileList | null) => {
    if (!files) return;

    const maxImages = 5;
    const maxSize = 5 * 1024 * 1024; // 5MB

    const newFiles = Array.from(files).slice(0, maxImages - images.length);

    for (const file of newFiles) {
      if (file.size > maxSize) {
        toast.error(`${file.name} is too large. Maximum size is 5MB.`);
        continue;
      }

      if (!file.type.startsWith("image/")) {
        toast.error(`${file.name} is not an image.`);
        continue;
      }

      // Create preview
      const preview = URL.createObjectURL(file);
      setImages((prev) => [...prev, { file, preview }]);

      // Upload to server
      setIsUploading(true);
      try {
        const result = await apiClient.uploadScanRequestImage(file);
        setImages((prev) =>
          prev.map((img) =>
            img.file === file ? { ...img, path: result.path } : img
          )
        );
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Upload failed";
        toast.error(`Failed to upload ${file.name}: ${errorMessage}`);
        setImages((prev) => prev.filter((img) => img.file !== file));
        URL.revokeObjectURL(preview);
      } finally {
        setIsUploading(false);
      }
    }
  };

  // Remove image
  const removeImage = (index: number) => {
    setImages((prev) => {
      const newImages = [...prev];
      URL.revokeObjectURL(newImages[index].preview);
      newImages.splice(index, 1);
      return newImages;
    });
  };

  // Create mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createScanRequest({
        barcode,
        product_name: productName || undefined,
        brand_name: brandName || undefined,
        notes: notes || undefined,
        requester_name: requesterName,
        requester_email: requesterEmail,
        reference_images: images.filter((img) => img.path).map((img) => img.path!),
      });
    },
    onSuccess: () => {
      toast.success("Scan request created successfully!");
      router.push("/scan-requests");
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : "Failed to create scan request");
    },
  });

  // Form validation
  const isValid =
    barcode.trim() !== "" &&
    requesterName.trim() !== "" &&
    requesterEmail.trim() !== "" &&
    requesterEmail.includes("@");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isValid) {
      createMutation.mutate();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/scan-requests">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">New Scan Request</h1>
          <p className="text-gray-500">
            Request a product scan when a product is not in the system
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid gap-6 md:grid-cols-2">
          {/* Product Information */}
          <Card>
            <CardHeader>
              <CardTitle>Product Information</CardTitle>
              <CardDescription>
                Enter the product details you want scanned
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Barcode */}
              <div className="space-y-2">
                <Label htmlFor="barcode">Barcode *</Label>
                <div className="relative">
                  <Input
                    id="barcode"
                    value={barcode}
                    onChange={(e) => handleBarcodeChange(e.target.value)}
                    placeholder="Enter product barcode"
                    required
                  />
                  {isCheckingDuplicate && (
                    <Loader2 className="absolute right-3 top-2.5 h-4 w-4 animate-spin text-gray-400" />
                  )}
                </div>
              </div>

              {/* Duplicate Warning */}
              {duplicateWarning && duplicateWarning.length > 0 && (
                <div className="flex items-start gap-3 p-4 rounded-lg border border-yellow-200 bg-yellow-50 text-yellow-800">
                  <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Duplicate Request</p>
                    <p className="text-sm mt-1">
                      There {duplicateWarning.length === 1 ? "is" : "are"} already{" "}
                      {duplicateWarning.length} pending request
                      {duplicateWarning.length > 1 ? "s" : ""} for this barcode.
                      You can still create a new request if needed.
                    </p>
                  </div>
                </div>
              )}

              {/* Product Name */}
              <div className="space-y-2">
                <Label htmlFor="productName">Product Name (Optional)</Label>
                <Input
                  id="productName"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  placeholder="e.g., Cola 330ml"
                />
              </div>

              {/* Brand Name */}
              <div className="space-y-2">
                <Label htmlFor="brandName">Brand Name (Optional)</Label>
                <Input
                  id="brandName"
                  value={brandName}
                  onChange={(e) => setBrandName(e.target.value)}
                  placeholder="e.g., Coca-Cola"
                />
              </div>

              {/* Notes */}
              <div className="space-y-2">
                <Label htmlFor="notes">Notes (Optional)</Label>
                <Textarea
                  id="notes"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Any additional information about the product..."
                  rows={3}
                />
              </div>
            </CardContent>
          </Card>

          {/* Requester Information & Images */}
          <div className="space-y-6">
            {/* Requester Information */}
            <Card>
              <CardHeader>
                <CardTitle>Your Information</CardTitle>
                <CardDescription>
                  So we can notify you when the product is ready
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="requesterName">Your Name *</Label>
                  <Input
                    id="requesterName"
                    value={requesterName}
                    onChange={(e) => setRequesterName(e.target.value)}
                    placeholder="Enter your name"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="requesterEmail">Your Email *</Label>
                  <Input
                    id="requesterEmail"
                    type="email"
                    value={requesterEmail}
                    onChange={(e) => setRequesterEmail(e.target.value)}
                    placeholder="Enter your email"
                    required
                  />
                </div>
              </CardContent>
            </Card>

            {/* Reference Images */}
            <Card>
              <CardHeader>
                <CardTitle>Reference Images (Optional)</CardTitle>
                <CardDescription>
                  Upload up to 5 images of the product (max 5MB each)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Image Grid */}
                <div className="grid grid-cols-3 gap-4 mb-4">
                  {images.map((img, index) => (
                    <div key={index} className="relative group">
                      <img
                        src={img.preview}
                        alt={`Reference ${index + 1}`}
                        className="w-full h-24 object-cover rounded-lg border"
                      />
                      {!img.path && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
                          <Loader2 className="h-6 w-6 animate-spin text-white" />
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => removeImage(index)}
                        className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>

                {/* Upload Button */}
                {images.length < 5 && (
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-8 h-8 mb-2 text-gray-400" />
                      <p className="text-sm text-gray-500">
                        Click to upload or drag and drop
                      </p>
                      <p className="text-xs text-gray-400">
                        PNG, JPG, WebP (max 5MB)
                      </p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      multiple
                      onChange={(e) => handleImageUpload(e.target.files)}
                      disabled={isUploading}
                    />
                  </label>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end gap-4 mt-6">
          <Link href="/scan-requests">
            <Button variant="outline">Cancel</Button>
          </Link>
          <Button
            type="submit"
            disabled={!isValid || createMutation.isPending || isUploading}
          >
            {createMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              "Create Request"
            )}
          </Button>
        </div>
      </form>
    </div>
  );
}
