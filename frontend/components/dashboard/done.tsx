import { cn } from "@/lib/utils";
import { Card } from "../ui/card";
import { AIModel } from "../models/columns";
import getModelImage from "@/lib/model-image";
import Image from "next/image";
import {
  Check,
  CircleDollarSign,
  Loader2,
  MonitorCheck,
  Receipt,
  Users,
} from "lucide-react";
import { Button } from "../ui/button";

export default function DoneTraining({ model }: { model: AIModel }) {
  const modelImage = getModelImage(model.type);

  return (
    <Card className="group flex w-full max-w-xl flex-col items-start">
      <div className="w-full p-4 pb-0">
        <div className="relative z-0 flex aspect-[2/1] w-full items-center justify-center rounded-md border">
          {modelImage && (
            <>
              <Image
                src={modelImage}
                alt="AI Illustration"
                fill
                className="absolute rounded-sm object-cover duration-300"
              />
              <Image
                src={modelImage}
                alt="AI Illustration"
                fill
                className="absolute -z-10 object-cover blur-2xl duration-300"
              />
            </>
          )}
        </div>
      </div>
      <div className="relative z-0 flex w-full grow flex-col gap-4 overflow-y-auto p-4">
        <div className="z-10 flex w-full items-center justify-between">
          <div className="text-xl font-medium">{model.name}</div>
          <div className="rounded-md border px-3 py-1 font-supply text-sm uppercase text-muted-foreground">
            {model.type === "neuralnetwork" ? "neural network" : model.type}
          </div>
        </div>
      </div>
      <div className="flex w-full items-center justify-between border-t p-4">
        <div className="flex gap-4 font-supply uppercase text-muted-foreground">
          <div className="flex items-center gap-2">
            <Receipt className="size-4 text-primary" />
            <div className="text-sm">
              ${model.projectedEarnings.toFixed(2)} earned
            </div>
          </div>
        </div>
        <Button disabled variant="secondary">
          <Check className="size-4" />
          Done Training
        </Button>
      </div>
    </Card>
  );
}
