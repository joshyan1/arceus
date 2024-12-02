"use client";

import React, { Children, useEffect, useRef, useState } from "react";
import { Card } from "../../ui/card";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { InputPlus, PosEncoding } from "./symbols";

const stages = [
  ["Token Generation", "mb-2"],
  ["Softmax", "mb-2"],
  ["Linear Projection", "mb-8"],
  ["Add & Norm", "mb-2"],
  ["Feed Forward", "mb-8"],
  ["Dropout", "mb-2"],
  ["Add & Norm", "mb-2"],
  ["Multi-Head Attention", "mb-12"],
  ["InputSymbols", ""],
  ["Input Embedding", "mt-8"],
];

export default function TransformerVisualization() {
  const containerRef = useRef<HTMLDivElement>(null);

  const [animationStage, setAnimationStage] = useState(0);
  const [animationKey, setAnimationKey] = useState(-1);

  useEffect(() => {
    setAnimationKey(0);
    const interval = setInterval(() => {
      setAnimationStage((prev) => {
        if (prev === 9) {
          setTimeout(() => {
            setAnimationStage(0);
            setAnimationKey((prev) => prev + 1);
          }, 1500);
          return prev + 1;
        }
        return Math.min(prev + 1, 11);
      });
    }, 150);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card
      ref={containerRef}
      className="relative z-0 row-span-2 flex items-center justify-center font-supply text-sm uppercase"
    >
      {animationKey >= 0 && (
        <div
          key={animationKey}
          className="absolute top-0 h-[250%] w-[250%] overflow-visible opacity-25"
        >
          <div className="transformer-pulse dotted-pattern absolute h-full w-full bg-primary" />
        </div>
      )}
      {/* {animationStage} */}
      <div className="flex flex-col items-center">
        <div className="relative mb-6 flex flex-col items-center">
          <div className="absolute -bottom-4 -z-20 h-full w-px bg-border" />
          {animationKey >= 0 && (
            <div
              key={animationKey}
              className="h-animated absolute -bottom-4 -z-10 w-px bg-primary"
            />
          )}

          {stages.map(([stage, className], index) => {
            if (stage === "InputSymbols") {
              return (
                <InputSymbols
                  key={index}
                  activePlus={animationStage >= 9 - index}
                  active={
                    animationStage === 9 - index ||
                    animationStage === 10 - index
                  }
                />
              );
            }
            return (
              <Stage
                key={index}
                active={
                  animationStage === 9 - index || animationStage === 10 - index
                }
                className={className}
              >
                {stage}
              </Stage>
            );
          })}
        </div>
        <div className="text-muted-foreground">Inputs</div>
      </div>
    </Card>
  );
}

function InputSymbols({
  active,
  activePlus,
}: {
  active: boolean;
  activePlus: boolean;
}) {
  return (
    <div className="relative flex h-4 w-4 items-center transition-all duration-300">
      <InputPlus active={activePlus} />
      <div
        className={cn(
          "absolute left-6 w-24 leading-tight",
          active ? "text-primary" : "text-muted-foreground",
        )}
      >
        Positional Encoding
      </div>
      <div className="absolute right-4 flex w-12 items-center">
        <PosEncoding active={active} />
        <div
          className={cn(
            "h-px w-6",
            active ? "bg-primary" : "bg-muted-foreground",
          )}
        />
      </div>
    </div>
  );
}

function Stage({
  children,
  className,
  active = false,
}: {
  children: React.ReactNode;
  className?: string;
  active?: boolean;
}) {
  return (
    <div
      className={cn(
        "z-0 flex w-32 items-center justify-center rounded-lg border bg-nested-card p-2 text-center text-muted-foreground shadow-lg shadow-muted/25 transition-all duration-300",
        active ? "border-primary text-foreground shadow-lg" : "",
        className,
      )}
    >
      {children}
    </div>
  );
}
