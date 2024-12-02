import { cn } from "@/lib/utils";
import { Card } from "../ui/card";
import { AIModel } from "../models/columns";
import getModelImage from "@/lib/model-image";
import Image from "next/image";
import {
  BrainCircuit,
  CircleDollarSign,
  CircleGauge,
  Cpu,
  Layers,
  Loader2,
  Monitor,
  Users,
  Zap,
} from "lucide-react";
import { Button } from "../ui/button";

export default function WaitingForTraining({
  model,
  startTraining,
}: {
  model: AIModel;
  startTraining: () => void;
}) {
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

        <div className="flex flex-col gap-2">
          <Card className="flex items-center justify-between gap-2 rounded-lg bg-nested-card p-2 font-supply text-sm transition-all">
            <div>RAJAN AGARWAL</div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Cpu className="size-3.5" />
              M3
            </div>
          </Card>
          <Card className="flex items-center justify-between gap-2 rounded-lg bg-nested-card p-2 font-supply text-sm transition-all">
            <div>SIMERUS MAHESH</div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Cpu className="size-3.5" />
              M3
            </div>
          </Card>
          <Card className="flex items-center justify-between gap-2 rounded-lg bg-nested-card p-2 font-supply text-sm transition-all">
            <div>BOSHUA YAN</div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Cpu className="size-3.5" />
              M3
            </div>
          </Card>
          <Card className="flex items-center justify-between gap-2 rounded-lg bg-nested-card p-2 font-supply text-sm transition-all">
            <div>ISHAAN DEY</div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Cpu className="size-3.5" />
              M1
            </div>
          </Card>
        </div>
      </div>

      <div className="flex w-full items-center justify-between border-t p-4">
        <div className="flex gap-4 font-supply uppercase text-muted-foreground">
          <div className="flex items-center gap-2">
            <Users className="size-4 text-primary" />
            <div className="text-sm">
              {parseInt(model.spots.split("/")[0]) + 1}/
              {model.spots.split("/")[1]} users
            </div>
          </div>
          <div className="flex items-center gap-2">
            <CircleDollarSign className="size-4 text-primary" />
            <div className="text-sm">
              ${model.projectedEarnings.toFixed(2)} (projected)
            </div>
          </div>
        </div>
        {/* <Button disabled variant="secondary">
          <Loader2 className="size-4 animate-spin" />
          Waiting for host...
        </Button> */}
        <Button variant="secondary" onClick={startTraining}>
          Start Training
        </Button>
      </div>
    </Card>
  );
}
