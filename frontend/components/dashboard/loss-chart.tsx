"use client";

import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

const generateRandomData = () => {
  const data = [];
  let trainLoss = 2.5;
  let valLoss = 2.8;

  // Generate 9 epochs worth of data (45 points)
  for (let epoch = 0; epoch < 10; epoch++) {
    for (let step = 0; step < 5; step++) {
      if (epoch < 9) {
        // Gradually decrease losses with some noise
        trainLoss = Math.max(
          0.2,
          trainLoss * 0.95 + (Math.random() * 0.2 - 0.1),
        );
        // Validation loss follows training loss but is slightly higher
        valLoss = Math.max(0.3, trainLoss + 0.2 + (Math.random() * 0.3 - 0.15));

        data.push({
          step: `${epoch}.${step}`,
          training: Number(trainLoss.toFixed(3)),
          validation: Number(valLoss.toFixed(3)),
        });
      } else {
        // Push null for the last 1 epochs to create empty space
        data.push({
          step: `${epoch}.${step}`,
          training: null,
          validation: null,
        });
      }
    }
  }
  return data;
};

const chartData = generateRandomData();

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

export default function LossChart() {
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

  return (
    <div ref={containerRef} className="h-full">
      <ChartContainer config={chartConfig} height={height}>
        <LineChart
          data={chartData}
          margin={{
            left: 4,
            right: 4,
            top: 12,
            bottom: 4,
          }}
        >
          <CartesianGrid vertical={false} />
          <ChartTooltip
            cursor={true}
            animationDuration={100}
            content={<ChartTooltipContent cursor className="w-48" hideLabel />}
          />
          <Line
            dataKey="validation"
            type="linear"
            opacity={0.5}
            stroke="hsl(var(--muted-foreground))"
            strokeWidth={1.5}
            dot={(props) => {
              const isLast = props.index === 44; // Last point of actual data (9 epochs * 5 points - 1)
              return isLast ? (
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
                    filter="blur(8px)"
                    className="animate-pulse"
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
              const isLast = props.index === 44;
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
                    filter="blur(8px)"
                    className="animate-pulse"
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
