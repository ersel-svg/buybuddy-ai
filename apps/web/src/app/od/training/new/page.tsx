"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Brain, Construction } from "lucide-react";

export default function ODNewTrainingPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">New Training Run</h1>
        <p className="text-muted-foreground">
          Configure and start a new detection model training
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
            Training wizard will be implemented in Phase 8
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
