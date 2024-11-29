"use client";

import { Card } from "../ui/card";
import { useRef, useState, useLayoutEffect, Fragment } from "react";

export default function ModelVisualization() {
  const dimensions = [30, 30, 30, 30];
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
      className="relative z-0 col-span-2 flex items-center justify-evenly font-supply"
      style={{ minHeight: "200px" }}
    >
      {/* <div className="absolute -z-10 h-full w-[200%] overflow-visible">
        <div
          style={{
            left: `${((layerSpacing + 40) * (dimensions.length - 1)) / 2}px`,
          }}
          className="dotted-pattern absolute h-full w-full bg-primary"
        ></div>
      </div> */}

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
  if (connections) console.log("connections", connections);

  const lineContainerRef = useRef<HTMLDivElement>(null);
  const [lineContainerHeight, setLineContainerHeight] = useState(0);

  useLayoutEffect(() => {
    if (!lineContainerRef.current) return;
    setLineContainerHeight(lineContainerRef.current.clientHeight);
  }, [lineContainerRef.current]);

  return (
    <div className="bg-nested-card flex h-5/6 w-10 flex-col justify-between rounded-lg border shadow-lg shadow-muted/50">
      <div className="h-full py-4">
        <div
          className="relative flex h-full w-full justify-center"
          ref={lineContainerRef}
        >
          {connections &&
            connections.data.map(([from, to], index) => {
              const top = {
                from: from / (dimension - 1),
                to: to / (connections.nextDimension - 1),
              };

              const verticalDistance =
                (top.to - top.from) * lineContainerHeight;
              const angle = Math.atan(verticalDistance / (spacing + 40));
              const lineLength = Math.sqrt(
                (spacing + 40) ** 2 + verticalDistance ** 2,
              );

              return (
                <Fragment key={index}>
                  <div
                    style={{
                      top: `${top.from * 100}%`,
                    }}
                    className="nn-start-node absolute z-10 size-1.5 rounded-full bg-foreground"
                  />
                  <div
                    style={{
                      top: `${top.from * 100}%`,
                      width: `${lineLength}px`,
                      transformOrigin: "0 0",
                      transform: `rotate(${angle}rad) translateY(2px)`,
                    }}
                    className="nn-line-pulse absolute left-5 h-px bg-foreground"
                  />
                  <div
                    style={{
                      top: `${top.to * 100}%`,
                      transform: `translateX(${spacing + 40}px)`,
                    }}
                    className="nn-end-node absolute z-10 size-1.5 rounded-full bg-foreground"
                  />
                </Fragment>
              );
            })}
        </div>
      </div>
      <div className="flex flex-col items-center border-t bg-muted/40 py-1 text-sm">
        <div>L{layer}</div>
        <div className="text-muted-foreground">{dimension}</div>
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
