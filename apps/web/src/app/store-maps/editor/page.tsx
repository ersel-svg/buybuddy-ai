"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Loader2 } from "lucide-react";

import { StoreMapEditor } from "@/components/store-maps/editor/store-map-editor";

function EditorContent() {
  const searchParams = useSearchParams();
  const mapId = searchParams.get("id");

  if (!mapId || isNaN(parseInt(mapId))) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center space-y-2">
          <p className="text-lg font-medium">No map selected</p>
          <p className="text-sm text-muted-foreground">
            Please select a map from the store maps list.
          </p>
        </div>
      </div>
    );
  }

  return <StoreMapEditor mapId={parseInt(mapId)} />;
}

export default function StoreMapEditorPage() {
  return (
    <Suspense
      fallback={
        <div className="h-full flex items-center justify-center bg-background">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <EditorContent />
    </Suspense>
  );
}
