import Nav from "@/components/dashboard/nav";
import Progress from "@/components/dashboard/progress";
import Devices from "@/components/dashboard/devices";
import Compute from "@/components/dashboard/compute";
import Loss from "@/components/dashboard/loss";
import Timing from "@/components/dashboard/timing";
import ModelVisualization from "@/components/dashboard/model-visualization";
import Earnings from "@/components/dashboard/earnings";

export default function Home() {
  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav />
      <div className="flex w-full grow gap-4 overflow-hidden bg-muted/25 p-4">
        <div className="flex w-96 flex-col gap-4">
          <Progress progress={75} />
          <Earnings />
          <Compute />
          <Devices />
        </div>
        <div className="grid grow grid-cols-2 grid-rows-2 gap-4">
          <Loss />
          <Timing />
          <ModelVisualization />
        </div>
      </div>
    </div>
  );
}
