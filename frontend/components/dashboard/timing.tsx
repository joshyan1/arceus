import { GaugeCircle, Layers } from "lucide-react";
import { Card } from "../ui/card";
import TimingChart from "./timing-chart";

export default function Timing() {
  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>TIMING</div>
        <div className="flex items-center gap-2">
          <GaugeCircle className="size-3.5" />
          0.01s
        </div>
      </div>
      {/* <TimingChart /> */}
    </Card>
  );
}
