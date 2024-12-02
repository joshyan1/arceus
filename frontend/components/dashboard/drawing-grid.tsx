import { useEffect, useRef, useState } from "react";
import { Button } from "../ui/button";
import { RotateCcw, RotateCw } from "lucide-react";

interface DrawingGridProps {
  onDrawingComplete?: (pixels: number[][]) => void;
}

export default function DrawingGrid({ onDrawingComplete }: DrawingGridProps) {
  const [isDrawing, setIsDrawing] = useState(false);
  const [grid, setGrid] = useState<number[][]>(
    Array(28)
      .fill(0)
      .map(() => Array(28).fill(0)),
  );
  const gridRef = useRef<HTMLDivElement>(null);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  const drawPoint = (x: number, y: number) => {
    if (x >= 0 && x < 28 && y >= 0 && y < 28) {
      setGrid((prev) => {
        const newGrid = [...prev];
        // Draw a 3x3 square around the point for thickness
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const newX = x + dx;
            const newY = y + dy;
            if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
              newGrid[newY][newX] = 1;
            }
          }
        }
        return newGrid;
      });
    }
  };

  const interpolatePoints = (
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ) => {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    while (true) {
      drawPoint(x0, y0);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }
  };

  const handleDraw = (e: MouseEvent | TouchEvent) => {
    if (!isDrawing || !gridRef.current || isRunning || result) return;

    const rect = gridRef.current.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

    const x = Math.floor(((clientX - rect.left) / rect.width) * 28);
    const y = Math.floor(((clientY - rect.top) / rect.height) * 28);

    if (lastPos.current) {
      interpolatePoints(lastPos.current.x, lastPos.current.y, x, y);
    } else {
      drawPoint(x, y);
    }
    lastPos.current = { x, y };
  };

  const clearGrid = () => {
    setGrid(
      Array(28)
        .fill(0)
        .map(() => Array(28).fill(0)),
    );
    lastPos.current = null;
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => handleDraw(e);
    const handleTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      handleDraw(e);
    };

    if (isDrawing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("touchmove", handleTouchMove, { passive: false });
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchmove", handleTouchMove);
    };
  }, [isDrawing]);

  const handleStartDrawing = () => {
    if (isRunning || result) return;
    setIsDrawing(true);
    lastPos.current = null;
  };

  const handleStopDrawing = () => {
    setIsDrawing(false);
    lastPos.current = null;
  };

  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    if (isRunning) {
      setTimeout(() => {
        setIsRunning(false);
        setResult("8");
      }, 1000);
    }
  }, [isRunning]);

  return (
    <div className="flex w-full flex-col items-center gap-2">
      <div
        ref={gridRef}
        className={`grid aspect-square w-full max-w-[280px] touch-none grid-cols-[repeat(28,1fr)] overflow-hidden rounded-md border bg-background ${
          isRunning || result ? "cursor-not-allowed opacity-80" : ""
        }`}
        onMouseDown={handleStartDrawing}
        onMouseUp={handleStopDrawing}
        onMouseLeave={handleStopDrawing}
        onTouchStart={handleStartDrawing}
        onTouchEnd={handleStopDrawing}
      >
        {grid.map((row, y) =>
          row.map((cell, x) => (
            <div
              key={`${x}-${y}`}
              className={`aspect-square border border-border ${
                cell ? "bg-primary" : ""
              }`}
            />
          )),
        )}
      </div>
      <div className="flex w-[280px] gap-2">
        {result ? (
          <div className="inner-shadow flex h-9 grow items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium">
            <div className="text-center font-medium">
              Result: <span className="font-supply">{result}</span>
            </div>
          </div>
        ) : (
          <Button
            variant="secondary"
            size="icon"
            className="grow"
            onClick={() => setIsRunning(!isRunning)}
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Run Model"}
          </Button>
        )}
        <Button
          variant="outline"
          size="icon"
          onClick={() => {
            setIsRunning(false);
            setResult(null);
            clearGrid();
          }}
        >
          <RotateCw className="!size-3.5" />
        </Button>
      </div>
    </div>
  );
}
