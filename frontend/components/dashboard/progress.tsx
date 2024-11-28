"use client";

import { cn } from "@/lib/utils";
import { useState } from "react";

export default function Progress({ progress }: { progress: number }) {
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
