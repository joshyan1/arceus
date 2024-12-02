import { Activity, Layers } from "lucide-react";
import { Card } from "../ui/card";
import TimingChart from "./timing-chart";
import { TimingData } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function Timing({
  timingData,
  epoch,
  className,
}: {
  timingData: TimingData[];
  epoch: number;
  className?: string;
}) {
  const lastBatch =
    timingData.length > 0 ? timingData[timingData.length - 1].batch_idx : 0;

  return (
    <Card className={cn("flex flex-col p-4", className)}>
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>TIMING</div>
        <div className="flex items-center gap-2">
          <Activity className="size-3.5" />
          EPOCH {epoch} BATCH {lastBatch}
        </div>
      </div>
      <TimingChart timingData={timingData} />
    </Card>
  );
}
