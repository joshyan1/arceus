"use client";

import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { CircleDollarSign, Clock } from "lucide-react";

export default function Earnings({
  earnings,
  startTime,
}: {
  earnings: number;
  startTime: number;
}) {
  const elapsedTime = Date.now() - startTime;
  const minutelyEarnings = earnings / (elapsedTime / (1000 * 60));

  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>EARNINGS</div>
        <div className="flex items-center gap-2">
          <CircleDollarSign className="size-3.5" />
          BY COMPUTE
        </div>
      </div>
      <div className="flex items-end justify-between">
        <div className="text-4xl font-medium">
          ${minutelyEarnings.toFixed(2)}
        </div>
        <div className="text-lg text-muted-foreground">
          ${minutelyEarnings.toFixed(2)}/Min
        </div>
      </div>
    </Card>
  );
}

export function ProgressBar({ progress }: { progress: number }) {
  const scaledProgress = Math.floor(progress / 2);

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
              : i === scaledProgress - 1
                ? "animate-pulse bg-primary"
                : "bg-muted",
          )}
        ></div>
      ))}
    </div>
  );
}
