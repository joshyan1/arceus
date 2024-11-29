"use client";

import { useEffect, useRef, useState } from "react";
import { Area, CartesianGrid, AreaChart } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { TimingData } from "@/lib/types";

const chartConfig = {
  avg_forward: {
    label: "Forward Pass",
    color: "hsl(20.5 20.2% 58.2%)",
  },
  avg_backward: {
    label: "Backward Pass",
    color: "hsl(20.5 50.2% 58.2%)",
  },
  avg_update: {
    label: "Parameter Update",
    color: "hsl(20.5 90.2% 48.2%)",
  },
  avg_comm: {
    label: "Communication",
    color: "hsl(20.5 90.2% 38.2%)",
  },
  avg_prep: {
    label: "Data Preparation",
    color: "hsl(20.5 90.2% 18.2%)",
  },
} satisfies ChartConfig;

export default function TimingChart({
  timingData,
}: {
  timingData: TimingData[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(200);

  const processedData = timingData.slice(-100).map((data) => ({
    ...data,
    avg_forward: data.avg_forward * 1000,
    avg_backward: data.avg_backward * 1000,
    avg_update: data.avg_update * 1000,
    avg_comm: data.avg_comm * 1000,
    avg_prep: data.avg_prep * 1000,
  }));

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
          data={processedData}
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
