"use client";

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

const generateRandomData = () => {
  const data = [];
  let currentValue = 5;
  const now = new Date();

  for (let i = 0; i < 10; i++) {
    const timestamp = new Date(now.getTime() - (9 - i) * 60 * 60 * 1000);

    if (i < 8) {
      const remainingPoints = 7 - i;
      const neededIncrease =
        remainingPoints > 0 ? (45 - currentValue) / remainingPoints : 0;
      currentValue += neededIncrease + (Math.random() * 2 - 1);
      currentValue = Math.min(currentValue, 45);

      data.push({
        hour: `${i + 1}h`,
        desktop: Math.round(currentValue),
        timestamp,
      });
    } else {
      data.push({
        hour: `${i + 1}h`,
        desktop: null,
        timestamp,
      });
    }
  }
  return data;
};

const chartData = generateRandomData();

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "hsl(var(--primary))",
  },
} satisfies ChartConfig;

export default function PerformanceChart() {
  return (
    <ChartContainer config={chartConfig}>
      <LineChart
        accessibilityLayer
        data={chartData}
        margin={{
          left: 4,
          right: 4,
          top: 12,
        }}
      >
        <CartesianGrid vertical={false} />
        <ChartTooltip
          cursor={false}
          animationDuration={100}
          labelFormatter={(label, payload) => {
            const timestamp = payload[0].payload.timestamp;
            return formatTimestamp(timestamp);
          }}
          content={<ChartTooltipContent indicator="dot" />}
        />

        <Line
          dataKey="desktop"
          type="linear"
          fill="var(--color-desktop)"
          fillOpacity={0.4}
          stroke="var(--color-desktop)"
          isAnimationActive={false}
          dot={(props) => {
            const isLast = props.index === 7;
            return isLast ? (
              <>
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={4}
                  fill="var(--color-desktop)"
                />
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={8}
                  fill="var(--color-desktop)"
                  filter="blur(8px)"
                />
              </>
            ) : (
              <></>
            );
          }}
        />
      </LineChart>
    </ChartContainer>
  );
}
