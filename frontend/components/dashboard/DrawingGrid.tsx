import { useEffect, useRef, useState } from "react";

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

  const handleDraw = (e: MouseEvent | TouchEvent) => {
    if (!isDrawing || !gridRef.current) return;

    const rect = gridRef.current.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

    const x = Math.floor(((clientX - rect.left) / rect.width) * 28);
    const y = Math.floor(((clientY - rect.top) / rect.height) * 28);

    if (x >= 0 && x < 28 && y >= 0 && y < 28) {
      setGrid((prev) => {
        const newGrid = [...prev];
        newGrid[y][x] = 1;
        // Also fill adjacent pixels for smoother drawing
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

  const clearGrid = () => {
    setGrid(
      Array(28)
        .fill(0)
        .map(() => Array(28).fill(0)),
    );
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

  return (
    <div className="flex w-full flex-col items-center gap-4">
      <div
        ref={gridRef}
        className="grid aspect-square w-full max-w-[280px] touch-none grid-cols-[repeat(28,1fr)] overflow-hidden rounded-md border bg-background"
        onMouseDown={() => setIsDrawing(true)}
        onMouseUp={() => setIsDrawing(false)}
        onMouseLeave={() => setIsDrawing(false)}
        onTouchStart={() => setIsDrawing(true)}
        onTouchEnd={() => setIsDrawing(false)}
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
      <button
        onClick={clearGrid}
        className="rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80"
      >
        Clear Drawing
      </button>
    </div>
  );
}
