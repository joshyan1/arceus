import { GaugeCircle, Layers } from "lucide-react";
import { Card } from "../ui/card";
import TimingChart from "./timing-chart";
import { TimingData } from "@/lib/types";

export default function Timing({ timingData }: { timingData: TimingData[] }) {
  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>TIMING</div>
        <div className="flex items-center gap-2">
          <GaugeCircle className="size-3.5" />
          BATCH 140
        </div>
      </div>
      <TimingChart timingData={timingData} />
    </Card>
  );
}
