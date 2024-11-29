"use client";
import Nav from "@/components/dashboard/nav";
import Progress from "@/components/dashboard/progress";
import Devices from "@/components/dashboard/devices";
import Compute from "@/components/dashboard/compute";
import Loss from "@/components/dashboard/loss";
import Timing from "@/components/dashboard/timing";
import ModelVisualization from "@/components/dashboard/model-visualization";
import Earnings from "@/components/dashboard/earnings";

import { socket } from "@/lib/socket";
import { useState } from "react";
import { useEffect } from "react";

export default function Home() {
  const [timingData, setTimingData] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    socket.connect();

    return () => {
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    function onConnectEvent() {
      setIsConnected(true);
    }

    function onDisconnectEvent() {
      setIsConnected(false);
    }

    function onTimingEvent(value: any) {
      setTimingData((prev) => [...prev, value]);
    }

    socket.on("connect", onConnectEvent);
    socket.on("disconnect", onDisconnectEvent);
    socket.on("timing_stats", onTimingEvent);

    return () => {
      socket.off("connect", onConnectEvent);
      socket.off("disconnect", onDisconnectEvent);
      socket.off("timing_stats", onTimingEvent);
    };
  }, [timingData]);

  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav isConnected={isConnected} />
      <div>{JSON.stringify(timingData)}</div>
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
