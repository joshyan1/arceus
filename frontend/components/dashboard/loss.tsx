import { Check } from "lucide-react";
import { Card } from "../ui/card";
import LossChart from "./loss-chart";

export default function Loss() {
  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>LOSS</div>
        <div className="flex items-center gap-2">
          <Check className="size-3.5" />
          ACCURACY: 50%
        </div>
      </div>
      <LossChart />
    </Card>
  );
}
