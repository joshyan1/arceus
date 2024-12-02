import { Card } from "@/components/ui/card";
import { Ruler } from "lucide-react";
import PerformanceChart from "./performance-chart";
import { useState, useEffect } from "react";

export default function Compute({ totalCompute }: { totalCompute: number }) {
  const [cumulativeCompute, setCumulativeCompute] = useState(0);

  useEffect(() => {
    setCumulativeCompute((prev) => prev + totalCompute);
  }, [totalCompute]);

  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>COMPUTE</div>
        <div className="flex items-center gap-2">
          <Ruler className="size-3.5" />
          TFLOP TOTAL
        </div>
      </div>
      <div className="mb-1 flex items-end justify-between">
        <div className="text-4xl font-medium">
          {cumulativeCompute.toFixed(2)}
        </div>
        <div className="text-lg text-muted-foreground">
          Current: {totalCompute.toFixed(2)}
        </div>
      </div>
      <PerformanceChart totalCompute={totalCompute} />
    </Card>
  );
}
