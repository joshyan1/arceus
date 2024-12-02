"use client";

import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
} from "recharts";

import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

const formatTimestamp = (timestamp: Date) => {
  return timestamp.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

const chartConfig = {
  performance: {
    label: "Performance",
    color: "hsl(var(--primary))",
  },
} satisfies ChartConfig;

export default function PerformanceChart({
  totalCompute,
}: {
  totalCompute: number;
}) {
  const [history, setHistory] = useState<{ value: number; timestamp: Date }[]>(
    [],
  );

  useEffect(() => {
    setHistory((prev) => {
      const now = new Date();
      const lastValue = prev.length > 0 ? prev[prev.length - 1].value : 0;
      return [...prev, { value: lastValue + totalCompute, timestamp: now }];
    });
  }, [totalCompute]);

  const chartData = (() => {
    return history.map((point, i) => ({
      hour: `${i + 1}h`,
      performance: point.value,
      timestamp: point.timestamp,
    }));
  })();

  return (
    <div className="h-36">
      <ChartContainer config={chartConfig} height={144}>
        <LineChart
          accessibilityLayer
          data={chartData}
          margin={{
            left: 4,
            right: 12,
            top: 12,
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
            labelFormatter={(label, payload) => {
              const timestamp = payload[0].payload.timestamp;
              return formatTimestamp(timestamp);
            }}
            content={
              <ChartTooltipContent
                decimalPlaces={2}
                className="w-40"
                indicator="dot"
              />
            }
          />
          <Line
            dataKey="performance"
            type="linear"
            fill="var(--color-performance)"
            fillOpacity={0.4}
            stroke="var(--color-performance)"
            isAnimationActive={false}
            dot={(props) => {
              const isLast = props.index === history.length - 1;
              return isLast ? (
                <>
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={4}
                    fill="var(--color-performance)"
                  />
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={8}
                    fill="var(--color-performance)"
                    filter="url(#glow)"
                    className="dot-pulse"
                  />
                </>
              ) : (
                <></>
              );
            }}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}
