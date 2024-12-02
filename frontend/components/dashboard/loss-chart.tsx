"use client";

import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { EpochStats, TrainingData } from "@/lib/types";

const chartConfig = {
  training: {
    label: "Training Loss",
    color: "hsl(var(--primary))",
  },
  validation: {
    label: "Validation Loss",
    color: "hsl(var(--muted-foreground))",
  },
} satisfies ChartConfig;

export default function LossChart({
  epochStats,
  trainingData,
}: {
  epochStats: EpochStats[];
  trainingData: TrainingData[];
}) {
  const recentEpochStats = epochStats.slice(-100);
  const recentTrainingData = trainingData.slice(-1000);

  const containerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(200);

  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setHeight(containerRef.current.clientHeight);
      }
    };

    updateHeight();
    window.addEventListener("resize", updateHeight);
    return () => window.removeEventListener("resize", updateHeight);
  }, []);

  const chartData = recentTrainingData.map((train) => {
    const matchingEpoch = recentEpochStats.find(
      (e) => e.epoch === Math.floor(train.epoch),
    );
    return {
      step: train.epoch.toFixed(2),
      training: train.train_loss,
      validation: matchingEpoch?.val_loss ?? null,
    };
  });

  // Find the last valid validation point
  const lastValidationIndex =
    chartData
      .map((point, index) => ({
        hasValidation: point.validation !== null,
        index,
      }))
      .reverse()
      .find((item) => item.hasValidation)?.index ?? -1;

  return (
    <div ref={containerRef} className="h-full">
      <ChartContainer config={chartConfig} height={height}>
        <LineChart
          data={chartData}
          margin={{
            left: 4,
            right: 12,
            top: 12,
            bottom: 4,
          }}
        >
          <defs>
            <filter id="glow" x="-100%" y="-100%" width="400%" height="400%">
              <feGaussianBlur stdDeviation="6" result="blur1" />
              <feGaussianBlur stdDeviation="10" result="blur2" />
              <feMerge>
                <feMergeNode in="blur2" />
                <feMergeNode in="blur1" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <CartesianGrid vertical={false} />
          <ChartTooltip
            cursor={true}
            animationDuration={100}
            content={
              <ChartTooltipContent
                decimalPlaces={3}
                className="w-44"
                hideLabel
              />
            }
          />
          <Line
            dataKey="validation"
            type="linear"
            opacity={0.5}
            stroke="hsl(var(--muted-foreground))"
            strokeWidth={1.5}
            dot={(props) => {
              const isLastValidation = props.index === lastValidationIndex;
              return isLastValidation ? (
                <>
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={4}
                    fill="hsl(var(--muted-foreground))"
                    opacity={0.5}
                  />
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={8}
                    fill="hsl(var(--muted-foreground))"
                    opacity={0.5}
                    filter="url(#glow)"
                    className="dot-pulse"
                  />
                </>
              ) : (
                <></>
              );
            }}
            isAnimationActive={false}
          />
          <Line
            dataKey="training"
            type="linear"
            stroke="hsl(var(--primary))"
            strokeWidth={1.5}
            dot={(props) => {
              const isLast = props.index === chartData.length - 1;
              return isLast ? (
                <>
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={4}
                    fill="hsl(var(--primary))"
                  />
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={8}
                    fill="hsl(var(--primary))"
                    filter="url(#glow)"
                    className="dot-pulse"
                  />
                </>
              ) : (
                <></>
              );
            }}
            isAnimationActive={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}
