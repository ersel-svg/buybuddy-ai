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
import { InfoTooltip } from "@/components/ui/info-tooltip";
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
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-muted-foreground">
              Image Preloading
            </h4>
            <InfoTooltip content="Download and cache images before training starts. This speeds up training by eliminating download time during the training loop." />
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Batched Preload Toggle */}
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <div className="flex items-center gap-2">
                  <Label htmlFor="batched" className="font-medium">
                    Batched Preload
                  </Label>
                  <InfoTooltip content="Load images in batches and run garbage collection between batches. Use this for large datasets (10K+ images) to prevent out-of-memory errors." />
                </div>
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
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Label>Batch Size</Label>
                  <InfoTooltip content="Number of images to load per batch. Larger batches are faster but use more memory. Start with 500 and increase if you have plenty of RAM." />
                </div>
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
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Label>Max Workers</Label>
                  <InfoTooltip content="Number of parallel threads for downloading images. More workers = faster downloads, but too many can overwhelm your network or server. Recommended: 16-32." />
                </div>
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
              <div className="flex items-center gap-2">
                <Label>HTTP Timeout (seconds)</Label>
                <InfoTooltip content="Maximum time to wait for each image download. If a download takes longer than this, it will be retried. Increase this if you have slow internet or large images." />
              </div>
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
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-medium text-muted-foreground">
                DataLoader Settings
              </h4>
              <InfoTooltip content="PyTorch DataLoader configuration for feeding batches to the GPU during training. These settings affect training speed and memory usage." />
            </div>

            <div className="grid grid-cols-3 gap-4">
              {/* Num Workers */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Label>Workers</Label>
                    <InfoTooltip content="Number of CPU processes loading batches in parallel during training. 0 = main process only (slower), 4-8 = recommended. More workers can speed up training if CPU is the bottleneck." />
                  </div>
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
                <div className="flex items-center gap-2">
                  <Label htmlFor="pin_memory">Pin Memory</Label>
                  <InfoTooltip content="Pin memory in RAM for faster GPU transfer. Should almost always be enabled for GPU training. Only disable if you run into memory issues." />
                </div>
                <Switch
                  id="pin_memory"
                  checked={dataloader.pin_memory ?? true}
                  onCheckedChange={(v) => updateDataLoader({ pin_memory: v })}
                />
              </div>

              {/* Prefetch Factor */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Label>Prefetch</Label>
                    <InfoTooltip content="Number of batches each worker loads ahead of time. Higher values keep the GPU fed with data, but use more memory. 2 is usually optimal." />
                  </div>
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
