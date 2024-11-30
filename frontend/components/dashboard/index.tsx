"use client";

import Nav from "./nav";
import Progress from "./progress";
import Devices from "./devices";
import Compute from "./compute";
import Loss from "./loss";
import Timing from "./timing";
import ModelVisualization from "./model-visualization";
import Earnings from "./earnings";

import { socket } from "@/lib/socket";
import { useState } from "react";
import { useEffect } from "react";
import { EpochStats, TimingData, TrainingData } from "@/lib/types";
import { AIModel } from "../models/columns";
import { cn } from "@/lib/utils";
import WaitingForTraining from "./waiting";

export default function Dashboard({ model }: { model: AIModel }) {
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
      console.log("timing_data", value);
      if (!isTraining) {
        setIsTraining(true);
      }
    }

    function onEpochStatsEvent(value: any) {
      setEpochStats((prev) => [...prev, value]);
      if (!isTraining) {
        setIsTraining(true);
      }
    }

    function onTrainingDataEvent(value: any) {
      setTrainingData((prev) => [...prev, value]);
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
  const batch =
    timingData.length > 0 ? timingData[timingData.length - 1].batch_idx : 0;
  const batchSize = 256;

  const progressPercentage = Math.min(
    ((epoch * batchSize + batch) / (totalEpochs * batchSize)) * 100,
    100,
  );

  const totalCompute =
    timingData.length > 0
      ? timingData[timingData.length - 1].device_data.reduce(
          (acc, curr) => acc + curr.total_teraflops,
          0,
        ) * 2000
      : 0;
  const deviceData =
    timingData.length > 0 ? timingData[timingData.length - 1].device_data : [];

  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav isConnected={isConnected} modelName={model.name} />
      <div
        className={cn(
          "flex w-full grow gap-4 overflow-hidden bg-muted/25 p-4",
          !isTraining && "items-center justify-center",
        )}
      >
        {isTraining ? (
          <>
            <div className="flex w-96 flex-col gap-4">
              <Progress
                epoch={epoch}
                totalEpochs={totalEpochs}
                startTime={startTime}
                progressPercentage={progressPercentage}
              />
              {/* <Earnings /> */}
              <Compute totalCompute={totalCompute} />
              <Devices deviceData={deviceData} />
            </div>
            <div className="grid grow grid-cols-2 grid-rows-2 gap-4">
              <Loss epochStats={epochStats} trainingData={trainingData} />
              <Timing timingData={timingData} epoch={epoch} />
              <ModelVisualization
                pause={progressPercentage >= 100}
                deviceData={deviceData}
              />
            </div>
          </>
        ) : (
          <WaitingForTraining model={model} />
          // <div>Waiting for training to start...</div>
        )}
      </div>
    </div>
  );
}
