"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Brain, Construction } from "lucide-react";

export default function ODTrainingPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Training</h1>
        <p className="text-muted-foreground">
          Train detection models with RF-DETR, RT-DETR, or YOLO-NAS
        </p>
      </div>

      <Card className="py-12">
        <CardContent className="text-center">
          <div className="flex justify-center gap-2 mb-4">
            <Brain className="h-12 w-12 text-muted-foreground" />
            <Construction className="h-12 w-12 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium">Coming Soon</h3>
          <p className="text-muted-foreground mt-1">
            Training pipeline will be implemented in Phase 8
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
