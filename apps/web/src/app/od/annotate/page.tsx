"use client";

import { Card, CardContent } from "@/components/ui/card";
import { PenTool, Construction } from "lucide-react";

export default function ODAnnotatePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Annotate</h1>
        <p className="text-muted-foreground">
          Draw bounding boxes and annotate images with AI assistance
        </p>
      </div>

      <Card className="py-12">
        <CardContent className="text-center">
          <div className="flex justify-center gap-2 mb-4">
            <PenTool className="h-12 w-12 text-muted-foreground" />
            <Construction className="h-12 w-12 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium">Coming Soon</h3>
          <p className="text-muted-foreground mt-1">
            Annotation editor will be implemented in Phase 5
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
