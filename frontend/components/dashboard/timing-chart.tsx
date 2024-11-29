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
    forward: 2.763,
    backward: 3.162,
    params: 1.183,
    comm: 1.236,
    prep: 0.299,
  };

  // Generate 9 epochs worth of data (45 points)
  for (let epoch = 0; epoch < 10; epoch++) {
    for (let step = 0; step < 5; step++) {
      if (epoch < 10) {
        // Add some random variation (±15%)
        const variation = () => 1 + (Math.random() * 0.3 - 0.15);

        data.push({
          step: `${epoch}.${step}`,
          prep: Number((baseValues.prep * variation()).toFixed(3)),
          comm: Number((baseValues.comm * variation()).toFixed(3)),
          params: Number((baseValues.params * variation()).toFixed(3)),
          backward: Number((baseValues.backward * variation()).toFixed(3)),
          forward: Number((baseValues.forward * variation()).toFixed(3)),
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
    color: "hsl(20.5 90.2% 78.2%)",
  },
  backward: {
    label: "Backward Pass",
    color: "hsl(20.5 90.2% 68.2%)",
  },
  params: {
    label: "Parameter Update",
    color: "hsl(20.5 90.2% 48.2%)",
  },
  comm: {
    label: "Communication",
    color: "hsl(20.5 90.2% 38.2%)",
  },
  prep: {
    label: "Data Preparation",
    color: "hsl(20.5 90.2% 18.2%)",
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
                decimalPlaces={3}
                suffix="ms"
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
