/**
 * SmartRecommendationCard Component
 *
 * Displays AI-generated recommendations based on dataset analysis.
 */

"use client";

import { Lightbulb, Sparkles, Check, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { SmartDefaults } from "../types/wizard";
import { getRecommendationSummary } from "../hooks/useSmartDefaults";

interface SmartRecommendationCardProps {
  smartDefaults: SmartDefaults | null;
  onApply: () => void;
  isApplied: boolean;
  className?: string;
}

export function SmartRecommendationCard({
  smartDefaults,
  onApply,
  isApplied,
  className,
}: SmartRecommendationCardProps) {
  if (!smartDefaults) return null;

  const summary = getRecommendationSummary(smartDefaults);

  return (
    <Card className={cn("border-primary/20 bg-primary/5", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          Smart Recommendations
          <Badge variant="secondary" className="ml-auto">
            AI-Generated
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Reasoning */}
        <div className="space-y-2">
          <p className="text-sm font-medium">Based on your dataset:</p>
          <ul className="space-y-1">
            {smartDefaults.reasoning.map((reason, index) => (
              <li
                key={index}
                className="text-sm text-muted-foreground flex items-start gap-2"
              >
                <Lightbulb className="h-3 w-3 mt-1 flex-shrink-0 text-yellow-500" />
                {reason}
              </li>
            ))}
          </ul>
        </div>

        {/* Summary */}
        <div className="space-y-2">
          <p className="text-sm font-medium">Recommended settings:</p>
          <div className="flex flex-wrap gap-2">
            {summary.map((item, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {item}
              </Badge>
            ))}
          </div>
        </div>

        {/* Apply button */}
        <Button
          type="button"
          variant={isApplied ? "secondary" : "default"}
          size="sm"
          onClick={onApply}
          disabled={isApplied}
          className="w-full gap-2"
        >
          {isApplied ? (
            <>
              <Check className="h-4 w-4" />
              Applied
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Apply Recommendations
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

/**
 * Inline recommendation prompt
 */
interface InlineRecommendationProps {
  recommendation: string;
  onApply: () => void;
  className?: string;
}

export function InlineRecommendation({
  recommendation,
  onApply,
  className,
}: InlineRecommendationProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between p-3 rounded-lg bg-primary/5 border border-primary/20",
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-yellow-500" />
        <span className="text-sm">{recommendation}</span>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onApply}
        className="gap-1"
      >
        Apply
        <ChevronRight className="h-3 w-3" />
      </Button>
    </div>
  );
}

/**
 * Recommendation list item
 */
interface RecommendationItemProps {
  title: string;
  description: string;
  currentValue?: string;
  recommendedValue: string;
  onApply: () => void;
  isApplied?: boolean;
}

export function RecommendationItem({
  title,
  description,
  currentValue,
  recommendedValue,
  onApply,
  isApplied,
}: RecommendationItemProps) {
  return (
    <div className="flex items-center justify-between py-2 border-b last:border-b-0">
      <div className="flex-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
        {currentValue && currentValue !== recommendedValue && (
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-muted-foreground line-through">
              {currentValue}
            </span>
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-primary font-medium">
              {recommendedValue}
            </span>
          </div>
        )}
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onApply}
        disabled={isApplied}
        className="gap-1"
      >
        {isApplied ? (
          <Check className="h-3 w-3" />
        ) : (
          <>
            Apply
            <ChevronRight className="h-3 w-3" />
          </>
        )}
      </Button>
    </div>
  );
}
