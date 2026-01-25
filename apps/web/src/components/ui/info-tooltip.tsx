"use client";

import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface InfoTooltipProps {
  content: string;
  side?: "top" | "right" | "bottom" | "left";
  className?: string;
}

/**
 * InfoTooltip - A reusable info icon with tooltip for training pages
 *
 * Usage:
 * <InfoTooltip content="This is helpful information for users" />
 */
export function InfoTooltip({ content, side = "top", className }: InfoTooltipProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Info
          className={`h-4 w-4 text-muted-foreground hover:text-foreground cursor-help inline-block ${className || ""}`}
        />
      </TooltipTrigger>
      <TooltipContent side={side} className="max-w-xs">
        <p className="text-sm">{content}</p>
      </TooltipContent>
    </Tooltip>
  );
}
