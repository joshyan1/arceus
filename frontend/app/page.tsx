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
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { EpochStats, TimingData, TrainingData } from "@/lib/types";

export default function Home() {
  const [timingData, setTimingData] = useState<TimingData[]>([]);
  const [epochStats, setEpochStats] = useState<EpochStats[]>([]);
  const [trainingData, setTrainingData] = useState<TrainingData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [startTime, setStartTime] = useState(0);

  useEffect(() => {
    socket.connect();

    return () => {
      socket.disconnect();
    };
  }, []);

  async function startTraining() {
    const jobId = "1";
    try {
      const response = await fetch(
        `http://127.0.0.1:4000/api/network/train/${jobId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            epochs: 10,
            learning_rate: 0.1,
          }),
        },
      );
      const data = await response.json();
      if (data.message === "Training started") {
        console.log("Training started");
        setIsTraining(true);
        setStartTime(Date.now());
      } else {
        console.error("Training failed to start:", data.error);
      }
    } catch (error) {
      console.error("Error starting training:", error);
    }
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
      console.log(value);
      if (!isTraining) {
        setIsTraining(true);
      }
    }

    function onEpochStatsEvent(value: any) {
      setEpochStats((prev) => [...prev, value]);
      console.log(value);
      if (!isTraining) {
        setIsTraining(true);
      }
    }

    function onTrainingDataEvent(value: any) {
      setTrainingData((prev) => [...prev, value]);
      console.log(value);
      if (!isTraining) {
        setIsTraining(true);
      }
    }

    socket.on("connect", onConnectEvent);
    socket.on("disconnect", onDisconnectEvent);
    socket.on("timing_stats", onTimingEvent);
    socket.on("epoch_stats", onEpochStatsEvent);
    socket.on("training_data", onTrainingDataEvent);

    return () => {
      socket.off("connect", onConnectEvent);
      socket.off("disconnect", onDisconnectEvent);
      socket.off("timing_stats", onTimingEvent);
      socket.off("epoch_stats", onEpochStatsEvent);
      socket.off("training_data", onTrainingDataEvent);
    };
  }, []);

  const epoch =
    epochStats.length > 0 ? epochStats[epochStats.length - 1].epoch : 0;
  const totalEpochs =
    epochStats.length > 0 ? epochStats[epochStats.length - 1].epochs : 0;

  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav isConnected={isConnected} />
      <div>TIMING: {timingData.length}</div>
      <div>EPOCH: {epochStats.length}</div>
      <div>TRAINING: {trainingData.length}</div>
      {isTraining ? (
        <>
          <div className="flex w-full grow gap-4 overflow-hidden bg-muted/25 p-4">
            <div className="flex w-96 flex-col gap-4">
              <Progress
                progress={epoch}
                total={totalEpochs}
                startTime={startTime}
              />
              <Earnings />
              <Compute />
              <Devices />
            </div>
            <div className="grid grow grid-cols-2 grid-rows-2 gap-4">
              <Loss epochStats={epochStats} trainingData={trainingData} />
              <Timing timingData={timingData} epoch={epoch} />
              <ModelVisualization />
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="grid w-full grow grid-cols-2 gap-4 overflow-hidden bg-muted/25 p-4">
            <Card className="flex flex-col items-start gap-4 p-4">
              <div>Training</div>
              <Button variant="secondary" onClick={startTraining}>
                Start Training
              </Button>
            </Card>
            <Card className="flex flex-col items-start gap-4 p-4">Devices</Card>
          </div>
        </>
      )}
    </div>
  );
}
