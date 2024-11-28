import { Card } from "@/components/ui/card";
import { CircleGauge, Cpu, Laptop, Monitor, Zap } from "lucide-react";

const you = {
  name: "CURSOR REJECT",
  cpu: "M1",
  tflops: 1.2,
  task: "Layer 3",
  usage: 0.5,
};

const devices = [
  {
    name: "THE RAGE",
    cpu: "M3 MAX",
    tflops: 3.8,
    task: "Layer 1",
    usage: 0.7,
  },
  {
    name: "SPACEX EMPLOYEE",
    cpu: "M2",
    tflops: 2.9,
    task: "Layer 2",
    usage: 0.3,
  },
  {
    name: "6 FOOT 3 JUNGKOOK",
    cpu: "M3",
    tflops: 3.2,
    task: "Layer 4",
    usage: 0.4,
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
            {devices.length}
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
  return (
    <Card
      key={device.name}
      className="flex flex-col gap-2 rounded-lg bg-muted/25 p-2 pr-3"
    >
      <div className="flex items-center justify-between">
        <div className="font-supply text-sm">
          <div>{device.name}</div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Cpu className="size-3.5 text-primary" />
            {device.cpu}
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <CircleGauge className="size-3.5 text-primary" />
            {device.tflops} TFLOPS
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Monitor className="size-3.5 text-primary" />
            {device.usage * 100}%
          </div>
        </div>
        <div className="text-lg font-medium">{device.task}</div>
      </div>
    </Card>
  );
}
