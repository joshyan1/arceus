"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Image, { StaticImageData } from "next/image";

// UI Components
import Nav from "@/components/dashboard/nav";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { CircleDollarSign, Grid2X2Plus, Search, Users } from "lucide-react";

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
              <div className="text-sm text-muted-foreground">
                Elit in adipisicing nulla duis eiusmod Lorem eiusmod tempor
                reprehenderit esse enim eu anim consequat Lorem. Dolor eiusmod
                veniam commodo culpa aliqua voluptate anim veniam reprehenderit
                commodo. Id excepteur esse proident sit reprehenderit veniam
                aliquip adipisicing. Veniam aliqua ex sint ea incididunt minim.
                Dolor laborum mollit pariatur ipsum consectetur labore officia.
                Lorem non officia aute eu. Mollit labore pariatur ea dolor.
                <br />
                <br />
                Dolor occaecat tempor enim fugiat. Irure officia dolore elit non
                occaecat voluptate anim irure proident nostrud nulla ex laborum
                excepteur cupidatat. Consectetur irure dolore quis Lorem amet
                aliqua est aute ipsum sit. Nisi non sint occaecat reprehenderit
                in.
                <br />
                <br />
                Duis ea minim exercitation est adipisicing cupidatat ipsum ut
                velit anim proident. Minim officia nisi eu do duis non ipsum
                eiusmod amet ullamco commodo minim. Magna cillum non proident
                Lorem anim veniam consectetur elit nisi. Enim occaecat non
                adipisicing occaecat in.
                <br />
                <br />
                Est laboris pariatur nisi ea in Lorem eu consequat irure
                consequat enim dolor officia. Consequat enim consequat proident
                dolor ad occaecat. Cillum aute eiusmod Lorem mollit veniam ex
                sit dolore nostrud in. Eu tempor sint excepteur ex ipsum enim
                voluptate.
                <br />
                <br />
                Sit minim ullamco labore qui excepteur aliquip veniam. Anim enim
                ea ipsum mollit consequat sint. Proident nostrud nisi dolore
                sunt Lorem pariatur pariatur ut adipisicing fugiat occaecat.
                Ipsum mollit exercitation ad ad.
                <br />
                <br />
                Nulla eu deserunt adipisicing adipisicing quis. Duis irure Lorem
                pariatur incididunt voluptate aliqua anim cillum eiusmod eiusmod
                proident id. Laboris minim ex aliquip proident excepteur non
                velit ex quis aliquip dolor ut.
                <br />
                <br />
                Aliquip proident pariatur pariatur mollit quis aute ad cupidatat
                voluptate nostrud cillum do labore minim. Aliqua consectetur
                proident enim. Consectetur adipisicing dolor mollit elit fugiat
                enim sit excepteur nostrud ea aliqua eiusmod occaecat. Ut
                eiusmod dolore veniam.
                <br />
                <br />
                Ut mollit qui esse do quis commodo nisi elit quis culpa ut in
                pariatur. Non quis in fugiat id Lorem quis ad. Commodo quis est
                ipsum ea velit ad velit exercitation aute dolor quis incididunt
                irure velit. Esse reprehenderit sit qui. Est in voluptate
                excepteur dolor anim aliquip sint sint irure ex duis ullamco
                culpa aute. Sit adipisicing ipsum officia in enim do fugiat
                exercitation laborum ut esse amet sunt. Id sunt dolor quis magna
                ex cillum consequat do. Dolor deserunt sunt consequat non
                officia amet Lorem eu enim ad eu.
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
