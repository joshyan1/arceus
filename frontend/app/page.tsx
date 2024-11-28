import Nav from "@/components/dashboard/nav";
import PerformanceChart from "@/components/dashboard/performance-chart";
import Progress from "@/components/dashboard/progress";
import { Card } from "@/components/ui/card";
import { Clock, Laptop } from "lucide-react";

export default function Home() {
  return (
    <div className="flex h-full w-full flex-col">
      <Nav />
      <div className="flex w-full grow gap-4 bg-muted/25 p-4">
        <div className="flex w-96 flex-col gap-4">
          <Card className="flex flex-col p-4">
            <div className="font-supply mb-4 flex justify-between text-sm text-muted-foreground">
              <div>COMPUTE</div>
              <div className="flex items-center gap-2">
                <Laptop className="size-3.5" />3 DEVICES
              </div>
            </div>
            <div className="mb-1 flex items-end justify-between">
              <div className="text-4xl font-medium">11.5 TFLOPS</div>
              <div className="text-lg text-muted-foreground">0.34 H100s</div>
            </div>
            <div className="h-36">
              <PerformanceChart />
            </div>
          </Card>
          <Card className="flex flex-col p-4">
            <div className="font-supply mb-4 flex justify-between text-sm text-muted-foreground">
              <div>PROGRESS</div>
              <div className="flex items-center gap-2">
                <Clock className="size-3.5" />
                2:15:32
              </div>
            </div>
            <div className="mb-1 flex items-end justify-between">
              <div className="text-4xl font-medium">75%</div>
              <div className="text-lg text-muted-foreground">75/100 Epochs</div>
            </div>
            <Progress progress={75} />
          </Card>
          <Card className="grow p-4">card2</Card>
        </div>
        <Card className="grow p-4">card3</Card>
      </div>
    </div>
  );
}
