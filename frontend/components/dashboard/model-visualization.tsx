"use client";

import { Card } from "../ui/card";
import { useRef, useState, useLayoutEffect, Fragment } from "react";

export default function ModelVisualization() {
  const dimensions = [30, 20, 20];
  const connections = generateConnections(dimensions);
  const containerRef = useRef<HTMLDivElement>(null);
  const [layerSpacing, setLayerSpacing] = useState(0);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const totalWidth = containerRef.current.clientWidth;
    const numGaps = dimensions.length - 1;
    const spacing =
      (totalWidth - dimensions.length * 40) / (dimensions.length + 1);

    console.log("totalWidth", totalWidth);
    console.log("numGaps", numGaps);
    console.log("spacing", spacing);

    setLayerSpacing(spacing);
  }, [dimensions.length]);

  return (
    <Card
      ref={containerRef}
      className="relative col-span-2 flex items-center justify-evenly font-supply"
      style={{ minHeight: "200px" }}
    >
      {layerSpacing > 0 &&
        dimensions.map((dimension, i) => (
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
              <Fragment key={index}>
                <div
                  style={{
                    top: `${(from / dimension) * 100}%`,
                  }}
                  className="absolute size-1.5 rounded-full bg-foreground"
                />
                <div
                  style={{
                    top: `${(to / connections.nextDimension) * 100}%`,
                    transform: `translateX(${spacing + 40}px)`,
                  }}
                  className="absolute size-1.5 rounded-full bg-foreground"
                />
              </Fragment>
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
