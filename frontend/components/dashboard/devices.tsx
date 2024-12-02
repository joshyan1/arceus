"use client";

import { Card } from "@/components/ui/card";
import { CircleGauge, Cpu, Laptop, Layers, Monitor, Zap } from "lucide-react";
import { useAppContext } from "../providers/context";
import { cn } from "@/lib/utils";
// import { you, devices } from "@/lib/devices";
import { Device } from "@/lib/types";

export default function Devices({ deviceData }: { deviceData: Device[] }) {
  return (
    <Card className="relative z-0 flex flex-1 overflow-hidden">
      <div className="absolute left-0 top-0 h-4 w-full bg-gradient-to-b from-card via-card/75 to-transparent" />
      <div className="absolute bottom-0 left-0 h-4 w-full bg-gradient-to-t from-card via-card/75 to-transparent" />
      <div className="flex flex-1 flex-col overflow-y-auto p-4">
        <div className="mb-4 flex justify-between font-supply text-sm text-muted-foreground">
          <div>DEVICES</div>
          <div className="flex items-center gap-2">
            <Laptop className="size-3.5" />
            {deviceData.length}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          {/* <DeviceCard device={{ ...you, name: "YOUR DEVICE" }} /> */}

          {/* <div className="my-2 h-px w-full bg-muted" /> */}

          {deviceData.map((device) => (
            <DeviceCard key={device.device_id} device={device} />
          ))}
        </div>
      </div>
    </Card>
  );
}

const demoData = [
  {
    device_id: 1,
    chip: "M1",
    name: "ISHAAN DEY",
  },
  {
    device_id: 2,
    chip: "M2 Pro",
    name: "USER 1",
  },
  {
    device_id: 3,
    chip: "M3 Max",
    name: "USER 2",
  },
];

function DeviceCard({
  device,
}: {
  device: {
    device_id: number;
    total_teraflops: number;
    chip: string;
    device_layers: {
      [key: string]: number[];
    };
  };
}) {
  const { hoveredDeviceId, setHoveredDeviceId } = useAppContext();

  return (
    <Card
      className={cn(
        "flex select-none flex-col rounded-lg bg-nested-card p-2 pr-3 font-supply text-sm transition-all",
        hoveredDeviceId !== null &&
          hoveredDeviceId !== device.device_id &&
          "opacity-50",
      )}
      onMouseEnter={() => setHoveredDeviceId(device.device_id)}
      onMouseLeave={() => setHoveredDeviceId(null)}
    >
      <div className="flex w-full items-center gap-2">
        <div>
          {demoData.find((d) => d.device_id === device.device_id)?.name}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Cpu className="size-3.5" />
          {demoData.find((d) => d.device_id === device.device_id)?.chip}
        </div>
      </div>
      <div className="grid grid-cols-2">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Layers className="size-3.5 text-primary" />L
          {device.device_layers[1].join(",")}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <CircleGauge className="size-3.5 text-primary" />
          {(device.total_teraflops * 2000).toFixed(
            2,
          )} TFLOPS
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Monitor className="size-3.5 text-primary" />
          50%
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Zap className="size-3.5 text-primary" />
          75%
        </div>
      </div>
    </Card>
  );
}
