import { Check } from "lucide-react";
import { Card } from "../ui/card";
import LossChart from "./loss-chart";
import { EpochStats, TrainingData } from "@/lib/types";

export default function Loss({
  epochStats,
  trainingData,
}: {
  epochStats: EpochStats[];
  trainingData: TrainingData[];
}) {
  const trainingAccuracy = trainingData.map((epoch) => epoch.train_acc);
  const latestAccuracy =
    trainingAccuracy.length > 0
      ? trainingAccuracy[trainingAccuracy.length - 1].toFixed(2)
      : "0.00";

  return (
    <Card className="flex flex-col p-4">
      <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
        <div>LOSS</div>
        <div className="flex items-center gap-2">
          <Check className="size-3.5" />
          ACCURACY: {Number(latestAccuracy) * 100}%
        </div>
      </div>
      <LossChart epochStats={epochStats} trainingData={trainingData} />
    </Card>
  );
}
