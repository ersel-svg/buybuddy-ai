"use client";

import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, Settings2 } from "lucide-react";
import { useState } from "react";
import type { DataLoadingConfig, PreloadConfig, DataLoaderConfig } from "@/types";

interface DataLoadingConfigPanelProps {
  value?: DataLoadingConfig;
  onChange: (config: DataLoadingConfig) => void;
  showDataLoader?: boolean;
  defaultOpen?: boolean;
}

export function DataLoadingConfigPanel({
  value,
  onChange,
  showDataLoader = true,
  defaultOpen = false,
}: DataLoadingConfigPanelProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const preload = value?.preload || {};
  const dataloader = value?.dataloader || {};

  const updatePreload = (updates: Partial<PreloadConfig>) => {
    onChange({
      ...value,
      preload: { ...preload, ...updates },
    });
  };

  const updateDataLoader = (updates: Partial<DataLoaderConfig>) => {
    onChange({
      ...value,
      dataloader: { ...dataloader, ...updates },
    });
  };

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="flex items-center justify-between w-full p-3 bg-muted/50 rounded-lg hover:bg-muted transition-colors">
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4" />
          <span className="font-medium">Data Loading (Advanced)</span>
        </div>
        <ChevronDown
          className={`h-4 w-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </CollapsibleTrigger>

      <CollapsibleContent className="pt-4 space-y-6">
        {/* Preload Section */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-muted-foreground">
            Image Preloading
          </h4>

          <div className="grid grid-cols-2 gap-4">
            {/* Batched Preload Toggle */}
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <Label htmlFor="batched" className="font-medium">
                  Batched Preload
                </Label>
                <p className="text-xs text-muted-foreground mt-1">
                  Memory-efficient loading with gc.collect()
                </p>
              </div>
              <Switch
                id="batched"
                checked={preload.batched ?? false}
                onCheckedChange={(v) => updatePreload({ batched: v })}
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Batch Size</Label>
                <span className="text-sm text-muted-foreground">
                  {preload.batch_size ?? 500}
                </span>
              </div>
              <Slider
                value={[preload.batch_size ?? 500]}
                onValueChange={([v]) => updatePreload({ batch_size: v })}
                min={100}
                max={2000}
                step={100}
                disabled={!preload.batched}
              />
              <p className="text-xs text-muted-foreground">
                Images per batch (100-2000)
              </p>
            </div>

            {/* Max Workers */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Max Workers</Label>
                <span className="text-sm text-muted-foreground">
                  {preload.max_workers ?? 16}
                </span>
              </div>
              <Slider
                value={[preload.max_workers ?? 16]}
                onValueChange={([v]) => updatePreload({ max_workers: v })}
                min={1}
                max={64}
                step={1}
              />
              <p className="text-xs text-muted-foreground">
                Parallel download threads
              </p>
            </div>

            {/* HTTP Timeout */}
            <div className="space-y-2">
              <Label>HTTP Timeout (seconds)</Label>
              <Input
                type="number"
                value={preload.http_timeout ?? 30}
                onChange={(e) =>
                  updatePreload({ http_timeout: parseInt(e.target.value) || 30 })
                }
                min={5}
                max={120}
              />
            </div>
          </div>
        </div>

        {/* DataLoader Section */}
        {showDataLoader && (
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-muted-foreground">
              DataLoader Settings
            </h4>

            <div className="grid grid-cols-3 gap-4">
              {/* Num Workers */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Workers</Label>
                  <span className="text-sm text-muted-foreground">
                    {dataloader.num_workers ?? 4}
                  </span>
                </div>
                <Slider
                  value={[dataloader.num_workers ?? 4]}
                  onValueChange={([v]) => updateDataLoader({ num_workers: v })}
                  min={0}
                  max={16}
                  step={1}
                />
              </div>

              {/* Pin Memory */}
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <Label htmlFor="pin_memory">Pin Memory</Label>
                <Switch
                  id="pin_memory"
                  checked={dataloader.pin_memory ?? true}
                  onCheckedChange={(v) => updateDataLoader({ pin_memory: v })}
                />
              </div>

              {/* Prefetch Factor */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Prefetch</Label>
                  <span className="text-sm text-muted-foreground">
                    {dataloader.prefetch_factor ?? 2}
                  </span>
                </div>
                <Slider
                  value={[dataloader.prefetch_factor ?? 2]}
                  onValueChange={([v]) =>
                    updateDataLoader({ prefetch_factor: v })
                  }
                  min={1}
                  max={8}
                  step={1}
                />
              </div>
            </div>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
