"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Image, { StaticImageData } from "next/image";

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

// Assets
import Transformer1 from "@/assets/images/transformer1.png";
import Transformer2 from "@/assets/images/transformer2.png";
import Transformer3 from "@/assets/images/transformer3.png";
import NeuralNetwork1 from "@/assets/images/nn1.png";
import NeuralNetwork2 from "@/assets/images/nn2.png";
import NeuralNetwork3 from "@/assets/images/nn3.png";

const transformerImages = [Transformer1, Transformer2, Transformer3];
const neuralNetworkImages = [NeuralNetwork1, NeuralNetwork2, NeuralNetwork3];

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
    if (hoveredModel) {
      const images =
        hoveredModel.type === "transformer"
          ? transformerImages
          : neuralNetworkImages;
      const randomImage = images[Math.floor(Math.random() * images.length)];
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
            {currentImage && (
              <Image
                src={currentImage}
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
