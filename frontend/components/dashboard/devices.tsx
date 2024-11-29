"use client";

import { Card } from "@/components/ui/card";
import { CircleGauge, Cpu, Laptop, Layers, Monitor, Zap } from "lucide-react";
import { useAppContext } from "../providers/context";

const you = {
  name: "CURSOR REJECT",
  cpu: "M1",
  tflops: 1.2,
  task: [3],
  usage: 0.5,
  battery: 1,
};

const devices = [
  {
    name: "THE RAGE",
    cpu: "M3 MAX",
    tflops: 3.8,
    task: [1],
    usage: 0.7,
    battery: 1,
  },
  {
    name: "PLAYSTATION 5",
    cpu: "M2",
    tflops: 2.9,
    task: [2],
    usage: 0.3,
    battery: 0.5,
  },
];

export default function Devices() {
  return (
    <Card className="relative z-0 flex flex-1 overflow-hidden">
      <div className="absolute left-0 top-0 h-4 w-full bg-gradient-to-b from-card via-card/75 to-transparent" />
      <div className="absolute bottom-0 left-0 h-4 w-full bg-gradient-to-t from-card via-card/75 to-transparent" />
      <div className="flex flex-1 flex-col overflow-y-auto p-4">
        <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
          <div>DEVICES</div>
          <div className="flex items-center gap-2">
            <Laptop className="size-3.5" />
            {devices.length + 1}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <DeviceCard device={{ ...you, name: "YOUR DEVICE" }} />

          <div className="my-2 h-px w-full bg-muted" />

          {devices.map((device) => (
            <DeviceCard key={device.name} device={device} />
          ))}
        </div>
      </div>
    </Card>
  );
}

function DeviceCard({ device }: { device: (typeof devices)[number] }) {
  const { setHoveredLayers } = useAppContext();

  return (
    <Card
      key={device.name}
      className="flex select-none flex-col rounded-lg bg-nested-card p-2 pr-3 font-supply text-sm"
      onMouseEnter={() => setHoveredLayers(device.task)}
      onMouseLeave={() => setHoveredLayers([])}
    >
      <div className="flex w-full items-center gap-2">
        <div>{device.name}</div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Cpu className="size-3.5" />
          {device.cpu}
        </div>
      </div>
      <div className="grid grid-cols-2">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Layers className="size-3.5 text-primary" />L{device.task.join(",")}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <CircleGauge className="size-3.5 text-primary" />
          {device.tflops} TFLOPS
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Monitor className="size-3.5 text-primary" />
          {device.usage * 100}%
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Zap className="size-3.5 text-primary" />
          {device.battery * 100}%
        </div>
      </div>
    </Card>
  );
}
