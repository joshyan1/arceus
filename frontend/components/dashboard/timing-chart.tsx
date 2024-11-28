"use client";

import { useEffect, useRef, useState } from "react";
import { Area, CartesianGrid, AreaChart } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

const generateTimingData = () => {
  const data = [];
  const baseValues = {
    forward: 0.0027,
    backward: 0.0031,
    params: 0.0011,
    comm: 0.0012,
    prep: 0.0002,
  };

  // Generate 9 epochs worth of data (45 points)
  for (let epoch = 0; epoch < 10; epoch++) {
    for (let step = 0; step < 5; step++) {
      if (epoch < 9) {
        // Add some random variation (±15%)
        const variation = () => 1 + (Math.random() * 0.3 - 0.15);

        data.push({
          step: `${epoch}.${step}`,
          prep: Number((baseValues.prep * variation()).toFixed(4)),
          comm: Number((baseValues.comm * variation()).toFixed(4)),
          params: Number((baseValues.params * variation()).toFixed(4)),
          backward: Number((baseValues.backward * variation()).toFixed(4)),
          forward: Number((baseValues.forward * variation()).toFixed(4)),
        });
      } else {
        // Empty space for last epoch
        data.push({
          step: `${epoch}.${step}`,
          prep: null,
          comm: null,
          params: null,
          backward: null,
          forward: null,
        });
      }
    }
  }
  return data;
};

const chartData = generateTimingData();

const chartConfig = {
  forward: {
    label: "Forward Pass",
    color: "hsl(142, 76%, 36%)", // Green
  },
  backward: {
    label: "Backward Pass",
    color: "hsl(200, 95%, 14%)", // Dark blue
  },
  params: {
    label: "Parameter Update",
    color: "hsl(271, 91%, 65%)", // Purple
  },
  comm: {
    label: "Communication",
    color: "hsl(349, 89%, 60%)", // Red
  },
  prep: {
    label: "Data Preparation",
    color: "hsl(32, 95%, 44%)", // Orange
  },
} satisfies ChartConfig;

export default function TimingChart() {
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
        <AreaChart
          data={chartData}
          margin={{ left: 4, right: 4, top: 12, bottom: 4 }}
        >
          <CartesianGrid vertical={false} />
          <ChartTooltip
            cursor={true}
            animationDuration={100}
            content={
              <ChartTooltipContent
                hideLabel
                className="w-52"
                decimalPlaces={4}
              />
            }
          />
          {Object.entries(chartConfig)
            .reverse()
            .map(([key, config]) => (
              <Area
                key={key}
                dataKey={key}
                type="linear"
                stackId="1"
                stroke={config.color}
                fill={config.color}
                isAnimationActive={false}
              />
            ))}
        </AreaChart>
      </ChartContainer>
    </div>
  );
}
