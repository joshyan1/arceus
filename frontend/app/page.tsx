"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Image, { StaticImageData } from "next/image";

// UI Components
import Nav from "@/components/dashboard/nav";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  CircleDollarSign,
  Grid2X2Plus,
  Plus,
  Search,
  Users,
} from "lucide-react";

// Table Components
import { ColumnFiltersState } from "@tanstack/react-table";
import { DataTable } from "@/components/models/data-table";
import { columns } from "@/components/models/columns";
import { getData } from "@/components/models/data";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import getModelImage from "@/lib/model-image";
import Link from "next/link";

export default function Home() {
  const searchParams = useSearchParams();
  const data = getData();
  const [isFocused, setIsFocused] = useState(false);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);
  const [currentImage, setCurrentImage] = useState<StaticImageData | null>(
    null,
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (hoveredModelId) {
          window.location.href = `/model/${hoveredModelId}`;
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [hoveredModelId]);

  useEffect(() => {
    if (hoveredModel) {
      const randomImage = getModelImage(hoveredModel.type);
      setCurrentImage(randomImage);
    } else {
      setCurrentImage(null);
    }
  }, [hoveredModelId]);

  const handleSearch = (value: string) => {
    setColumnFilters([
      {
        id: "name",
        value: value,
      },
    ]);
  };

  const hoveredModel = data.find((model) => model.id === hoveredModelId);

  console.log("hoveredModel", hoveredModelId);

  return (
    <div className="flex h-full max-h-screen w-full flex-col">
      <Nav />

      <div className="grid w-full grow grid-cols-2 gap-4 overflow-hidden bg-muted/25 p-4">
        <Card className="flex flex-col items-start gap-4 p-4">
          <div className="mb-4 flex w-full items-center justify-between">
            <div className="font-supply text-sm text-muted-foreground">
              MARKETPLACE
            </div>
            <div className="flex items-center gap-2">
              <Search
                className={`size-4 ${isFocused ? "text-primary" : "text-muted-foreground"}`}
              />
              <Input
                inputSize="sm"
                placeholder="Search models..."
                className="w-48"
                value={(columnFilters[0]?.value as string) ?? ""}
                onChange={(e) => handleSearch(e.target.value)}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
              />
              <Button size="sm">
                <Plus className="size-4" />
                Post Training Job
              </Button>
            </div>
          </div>
          <DataTable
            columns={columns}
            data={data}
            columnFilters={columnFilters}
            setColumnFilters={setColumnFilters}
            onHoverChange={setHoveredModelId}
          />
        </Card>
        <Card
          className={cn(
            "group flex flex-col items-start transition-all",
            hoveredModelId ? "opacity-100" : "opacity-0",
          )}
        >
          <div className="w-full p-4 pb-0">
            <div className="relative z-0 flex aspect-[2/1] w-full items-center justify-center rounded-md border">
              {currentImage && (
                <>
                  <Image
                    src={currentImage}
                    alt="AI Illustration"
                    fill
                    className="absolute rounded-sm object-cover saturate-0 duration-300 group-hover:saturate-100"
                  />
                  <Image
                    src={currentImage}
                    alt="AI Illustration"
                    fill
                    className="absolute -z-10 object-cover opacity-0 blur-2xl duration-300 group-hover:opacity-100"
                  />
                </>
              )}
            </div>
          </div>
          <div className="relative z-0 flex w-full grow overflow-hidden">
            <div className="absolute bottom-0 left-0 z-20 h-8 w-full bg-gradient-to-t from-background to-transparent" />
            <div className="flex w-full flex-col gap-4 overflow-y-auto p-4 pb-8">
              <div className="z-10 flex w-full items-center justify-between">
                <div className="text-xl font-medium">{hoveredModel?.name}</div>
                <div className="rounded-md border px-3 py-1 font-supply text-sm uppercase text-muted-foreground">
                  {hoveredModel?.type === "neuralnetwork"
                    ? "neural network"
                    : hoveredModel?.type}
                </div>
              </div>
              <div className="flex w-full flex-col gap-1">
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[30%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[36%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[19%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[25%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[22%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[42%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="flex w-full justify-between">
                  <div className="h-5 w-[32%] animate-pulse rounded-md bg-muted" />
                  <div className="h-5 w-24 animate-pulse rounded-md bg-muted" />
                </div>
              </div>
            </div>
          </div>
          <div className="flex w-full items-center justify-between border-t p-4">
            <div className="flex gap-4 font-supply uppercase text-muted-foreground">
              <div className="flex items-center gap-2">
                <Users className="size-4 text-primary" />
                <div className="text-sm">{hoveredModel?.spots} users</div>
              </div>
              <div className="flex items-center gap-2">
                <CircleDollarSign className="size-4 text-primary" />
                <div className="text-sm">
                  ${hoveredModel?.projectedEarnings.toFixed(2)} (projected)
                </div>
              </div>
            </div>
            <Link href={`/model/${hoveredModel?.id}`}>
              <Button variant="secondary">
                Join Training Run
                <div className="text-muted-foreground">⌘K</div>
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
