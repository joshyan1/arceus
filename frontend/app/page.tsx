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

  function startTraining() {
    const jobId = "1";
    fetch(`http://127.0.0.1:4000/api/network/train/${jobId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        epochs: 10,
        learning_rate: 0.1,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message === "Training started") {
          console.log("Training started");
        } else {
          console.error("Training failed to start:", data.error);
        }
      })
      .catch((error) => {
        console.error("Error starting training:", error);
      });
  }

  useEffect(() => {
    function onConnectEvent() {
      setIsConnected(true);
      socket.emit("join", { room: "training_room" });
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
  }, []);

  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav isConnected={isConnected} />
      <div>{JSON.stringify(timingData)}</div>
      <button onClick={startTraining}>Start Training</button>
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
