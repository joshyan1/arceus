import { Card } from "@/components/ui/card";
import { Ruler } from "lucide-react";
import PerformanceChart from "./performance-chart";

export default function Compute() {
  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>COMPUTE</div>
        <div className="flex items-center gap-2">
          <Ruler className="size-3.5" />
          TFLOPS
        </div>
      </div>
      <div className="mb-1 flex items-end justify-between">
        <div className="text-4xl font-medium">10.88</div>
        <div className="text-lg text-muted-foreground">0.32 H100s</div>
      </div>
      <PerformanceChart />
    </Card>
  );
}
