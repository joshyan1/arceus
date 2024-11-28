"use client";

import { Card } from "../ui/card";
import { useRef, useState, useEffect } from "react";

export default function ModelVisualization() {
  const dimensions = [30, 20];
  const connections = generateConnections(dimensions);
  const containerRef = useRef<HTMLDivElement>(null);
  const [layerSpacing, setLayerSpacing] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const totalWidth = containerRef.current.clientWidth;
    const numGaps = dimensions.length - 1;
    // Account for the width of each layer (w-10 = 2.5rem = 40px) and divide remaining space
    const spacing = (totalWidth - dimensions.length * 40) / (numGaps + 1);
    setLayerSpacing(spacing);
  }, [dimensions.length]);

  return (
    <Card
      ref={containerRef}
      className="relative col-span-2 flex items-center justify-evenly font-supply"
    >
      {dimensions.map((dimension, i) => (
        <Layer
          key={i}
          layer={i + 1}
          dimension={dimension}
          spacing={layerSpacing}
          connections={
            i < connections.length
              ? { data: connections[i], nextDimension: dimensions[i + 1] }
              : undefined
          }
        />
      ))}
    </Card>
  );
}

function Layer({
  layer,
  dimension,
  spacing,
  connections,
}: {
  layer: number;
  dimension: number;
  spacing: number;
  connections?: { data: Connection[]; nextDimension: number };
}) {
  return (
    <div className="flex h-5/6 w-10 flex-col justify-between rounded-lg border bg-muted/20 shadow-lg shadow-muted/50">
      <div className="h-full py-4">
        <div className="relative flex h-full w-full justify-center">
          {connections &&
            connections.data.map(([from, to], index) => (
              <>
                <div
                  key={index}
                  style={{
                    top: `${(from / dimension) * 100}%`,
                  }}
                  className="absolute size-1.5 rounded-full bg-foreground"
                />
              </>
            ))}
        </div>
      </div>
      <div className="flex flex-col items-center border-t bg-muted/40 py-2">
        <div>L{layer}</div>
        <div className="text-xs text-muted-foreground">{dimension}</div>
      </div>
    </div>
  );
}

type Connection = [number, number];

function generateConnections(dimensions: number[]): Connection[][] {
  const connections: Connection[][] = [];

  for (let layer = 0; layer < dimensions.length - 1; layer++) {
    const fromDim = dimensions[layer];
    const toDim = dimensions[layer + 1];
    const maxConnections = Math.min(fromDim, toDim);

    // Random number of connections between 0 and maxConnections
    const numConnections = Math.floor(Math.random() * (maxConnections + 1));

    const layerConnections: Connection[] = [];
    for (let i = 0; i < numConnections; i++) {
      const from = Math.floor(Math.random() * fromDim);
      const to = Math.floor(Math.random() * toDim);
      layerConnections.push([from, to]);
    }

    connections.push(layerConnections);
  }

  return connections;
}
