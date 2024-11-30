"use client";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import Image from "next/image";

// UI Components
import Nav from "@/components/dashboard/nav";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";

// Table Components
import { ColumnFiltersState } from "@tanstack/react-table";
import { DataTable } from "@/components/models/data-table";
import { columns } from "@/components/models/columns";
import { getData } from "@/components/models/data";
import { AIModel } from "@/components/models/columns";

// Assets
import Transformer from "@/assets/images/transformer.png";
import NeuralNetwork from "@/assets/images/nn.png";

export default function Home() {
  const searchParams = useSearchParams();
  const data = getData();
  const [isFocused, setIsFocused] = useState(false);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);

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
        <Card className="group flex flex-col items-start gap-4 p-4">
          <div className="relative flex aspect-[2/1] w-full items-center justify-center overflow-hidden rounded-md border">
            {hoveredModel && (
              <Image
                src={
                  hoveredModel.type === "transformer"
                    ? Transformer
                    : NeuralNetwork
                }
                alt="AI Illustration"
                fill
                className="absolute object-cover saturate-0 duration-300 group-hover:saturate-100"
              />
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
