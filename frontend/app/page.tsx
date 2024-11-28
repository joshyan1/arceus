import Nav from "@/components/dashboard/nav";
import Progress from "@/components/dashboard/progress";
import { Card } from "@/components/ui/card";
import Devices from "@/components/dashboard/devices";
import Compute from "@/components/dashboard/compute";

export default function Home() {
  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav />
      <div className="flex w-full grow gap-4 overflow-hidden bg-muted/25 p-4">
        <div className="flex w-96 flex-col gap-4">
          <Compute />
          <Progress progress={75} />
          <Devices />
        </div>
        <Card className="grow p-4">card3</Card>
      </div>
    </div>
  );
}
