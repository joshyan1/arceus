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
      const newHistory = [
        ...prev,
        { value: totalCompute, timestamp: now },
      ].slice(-8); // Keep last 8 points
      return newHistory;
    });
  }, [totalCompute]);

  const chartData = (() => {
    const data = [];
    const now = new Date();

    // Fill with actual history
    history.forEach((point, i) => {
      data.push({
        hour: `${i + 1}h`,
        performance: point.value,
        timestamp: point.timestamp,
      });
    });

    // Pad with nulls if needed
    const remaining = 10 - history.length;
    for (let i = 0; i < remaining; i++) {
      const timestamp = new Date(now.getTime() + i * 60 * 60 * 1000);
      data.push({
        hour: `${history.length + i + 1}h`,
        performance: null,
        timestamp,
      });
    }

    return data;
  })();

  return (
    <div className="h-36">
      <ChartContainer config={chartConfig} height={144}>
        <LineChart
          accessibilityLayer
          data={chartData}
          margin={{
            left: 4,
            right: 4,
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
              const isLast = props.index === 7;
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
