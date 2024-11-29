"use client";

import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { Clock } from "lucide-react";

export default function Progress({
  progress,
  total,
  startTime,
  batch,
  batchSize,
}: {
  progress: number;
  total: number;
  startTime: number;
  batch: number;
  batchSize: number;
}) {
  const elapsedTime = Date.now() - startTime;
  const hours = Math.floor(elapsedTime / (1000 * 60 * 60));
  const minutes = Math.floor((elapsedTime % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((elapsedTime % (1000 * 60)) / 1000);

  const progressPercentage = Math.min(
    ((progress * batchSize + batch) / (total * batchSize)) * 100,
    100,
  );

  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>PROGRESS</div>
        <div className="flex items-center gap-2">
          <Clock className="size-3.5" />
          {`${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`}
        </div>
      </div>
      <div className="mb-2 flex items-end justify-between">
        <div className="text-4xl font-medium">
          {progressPercentage.toFixed(2)}%
        </div>
        <div className="text-lg text-muted-foreground">
          {progress}/{total} Epochs
          {/* {batchProgress} */}
        </div>
      </div>
      <ProgressBar progress={progressPercentage} total={100} />
    </Card>
  );
}
export function ProgressBar({
  progress,
  total,
}: {
  progress: number;
  total: number;
}) {
  const scaledProgress = Math.min(Math.floor((progress / total) * 50), 50);

  return (
    <div className="relative z-0 flex h-6 w-full justify-between">
      <div
        className="absolute z-10 h-8 w-5 -translate-y-1 animate-pulse bg-primary blur-lg"
        style={{ left: `calc(${(scaledProgress - 1) * 2}% - 0.5rem)` }}
      />
      {Array.from({ length: 50 }).map((_, i) => (
        <div
          key={i}
          className={cn(
            `h-full w-0.5 rounded-full`,
            i < scaledProgress - 1
              ? "bg-foreground"
              : i === scaledProgress - 1 || (progress >= 100 && i === 49)
                ? "animate-pulse bg-primary"
                : "bg-muted",
          )}
        ></div>
      ))}
    </div>
  );
}
