"use client";

import { useSearchParams, useRouter } from "next/navigation";
import Nav from "@/components/dashboard/nav";
import { Card } from "@/components/ui/card";
import { Search, Box, Users, CircleDollarSign, Network } from "lucide-react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getFilteredRowModel,
  ColumnFiltersState,
} from "@tanstack/react-table";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { useState } from "react";

type AIModel = {
  id: string;
  name: string;
  type: "transformer" | "neuralnetwork";
  spots: string;
  projectedEarnings: number;
};

export const models: AIModel[] = [
  {
    id: "gpt-1",
    name: "GPT-Mini",
    type: "transformer",
    spots: "50/50",
    projectedEarnings: 3.5,
  },
  {
    id: "nn-1",
    name: "ImageNet-S",
    type: "neuralnetwork",
    spots: "2/4",
    projectedEarnings: 7.2,
  },
];

export const columns: ColumnDef<AIModel>[] = [
  {
    accessorKey: "name",
    header: () => (
      <div className="flex items-center gap-1.5">
        <Box className="size-3.5" />
        Model Name
      </div>
    ),
    cell: ({ row }) => {
      const spots = row.getValue("spots") as string;
      const [current, total] = spots.split("/").map(Number);
      return (
        <span className={current === total ? "text-muted-foreground" : ""}>
          {row.getValue("name")}
        </span>
      );
    },
  },
  {
    accessorKey: "spots",
    header: () => (
      <div className="flex items-center gap-1.5">
        <Users className="size-3.5" />
        Spots
      </div>
    ),
    cell: ({ row }) => {
      const spots = row.getValue("spots") as string;
      const [current, total] = spots.split("/").map(Number);
      return (
        <span className={current === total ? "text-muted-foreground" : ""}>
          {spots}
        </span>
      );
    },
  },
  {
    accessorKey: "projectedEarnings",
    header: () => (
      <div className="flex items-center gap-1.5">
        <CircleDollarSign className="size-3.5" />
        Projected
      </div>
    ),
    cell: ({ row }) => {
      const spots = row.getValue("spots") as string;
      const [current, total] = spots.split("/").map(Number);
      const earnings = row.getValue("projectedEarnings") as number;
      return (
        <span className={current === total ? "text-muted-foreground" : ""}>
          ${earnings.toFixed(2)}
        </span>
      );
    },
  },
  {
    accessorKey: "type",
    header: () => (
      <div className="flex items-center gap-1.5">
        <Network className="size-3.5" />
        Architecture
      </div>
    ),
    cell: ({ row }) => {
      const type = row.getValue("type") as string;
      const spots = row.getValue("spots") as string;
      const [current, total] = spots.split("/").map(Number);
      return (
        <span className={current === total ? "text-muted-foreground" : ""}>
          {type === "transformer" ? "Transformer" : "Neural Network"}
        </span>
      );
    },
  },
];

function getData(): AIModel[] {
  return [
    {
      id: "gpt-1",
      name: "GPT-Mini",
      type: "transformer",
      spots: "50/50",
      projectedEarnings: 3.5,
    },
    {
      id: "nn-1",
      name: "ImageNet-S",
      type: "neuralnetwork",
      spots: "2/4",
      projectedEarnings: 7.2,
    },
    {
      id: "t-1",
      name: "BERT-Lite",
      type: "transformer",
      spots: "12/20",
      projectedEarnings: 4.8,
    },
    {
      id: "nn-2",
      name: "ResNet-Mini",
      type: "neuralnetwork",
      spots: "10/10",
      projectedEarnings: 5.9,
    },
    {
      id: "t-2",
      name: "T5-Small",
      type: "transformer",
      spots: "8/15",
      projectedEarnings: 6.3,
    },
    {
      id: "nn-3",
      name: "VGG-Compact",
      type: "neuralnetwork",
      spots: "8/8",
      projectedEarnings: 4.2,
    },
    {
      id: "t-3",
      name: "RoBERTa-Tiny",
      type: "transformer",
      spots: "15/30",
      projectedEarnings: 3.8,
    },
    {
      id: "nn-4",
      name: "DenseNet-S",
      type: "neuralnetwork",
      spots: "12/12",
      projectedEarnings: 5.1,
    },
    {
      id: "t-4",
      name: "XLM-Mini",
      type: "transformer",
      spots: "9/25",
      projectedEarnings: 7.5,
    },
    {
      id: "nn-5",
      name: "EfficientNet-Lite",
      type: "neuralnetwork",
      spots: "6/6",
      projectedEarnings: 8.2,
    },
    {
      id: "t-5",
      name: "ALBERT-Compact",
      type: "transformer",
      spots: "7/14",
      projectedEarnings: 6.7,
    },
    {
      id: "nn-6",
      name: "MobileNet-S",
      type: "neuralnetwork",
      spots: "3/5",
      projectedEarnings: 4.5,
    },
    {
      id: "t-6",
      name: "DistilBERT-S",
      type: "transformer",
      spots: "11/22",
      projectedEarnings: 5.4,
    },
    {
      id: "nn-7",
      name: "Inception-Mini",
      type: "neuralnetwork",
      spots: "5/7",
      projectedEarnings: 6.8,
    },
    {
      id: "t-7",
      name: "ELECTRA-Tiny",
      type: "transformer",
      spots: "6/18",
      projectedEarnings: 4.9,
    },
    {
      id: "nn-8",
      name: "SqueezeNet-S",
      type: "neuralnetwork",
      spots: "2/3",
      projectedEarnings: 3.2,
    },
    {
      id: "t-8",
      name: "Longformer-Mini",
      type: "transformer",
      spots: "13/26",
      projectedEarnings: 7.8,
    },
    {
      id: "nn-9",
      name: "ShuffleNet-S",
      type: "neuralnetwork",
      spots: "4/8",
      projectedEarnings: 5.6,
    },
    {
      id: "t-9",
      name: "DeBERTa-Lite",
      type: "transformer",
      spots: "10/20",
      projectedEarnings: 8.9,
    },
    {
      id: "nn-10",
      name: "Xception-Mini",
      type: "neuralnetwork",
      spots: "7/10",
      projectedEarnings: 6.1,
    },
  ];
}

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  columnFilters: ColumnFiltersState;
  setColumnFilters: React.Dispatch<React.SetStateAction<ColumnFiltersState>>;
}

export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const data = getData();
  const search = searchParams.get("q") ?? "";
  const [isFocused, setIsFocused] = useState(false);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  const handleSearch = (value: string) => {
    setColumnFilters([
      {
        id: "name",
        value: value,
      },
    ]);
  };

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
          />
        </Card>
        <Card className="flex flex-col items-start gap-4 p-4">Devices</Card>
      </div>
    </div>
  );
}

export function DataTable<TData, TValue>({
  columns,
  data,
  columnFilters,
  setColumnFilters,
}: DataTableProps<TData, TValue>) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    onColumnFiltersChange: setColumnFilters,
    getFilteredRowModel: getFilteredRowModel(),
    state: {
      columnFilters,
    },
  });

  return (
    <div className="w-full rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="hover:bg-background">
              {headerGroup.headers.map((header) => {
                return (
                  <TableHead key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                  </TableHead>
                );
              })}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => {
              const spots = row.getValue("spots") as string;
              const [current, total] = spots.split("/").map(Number);
              const isFullCapacity = current === total;

              return (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                  className={
                    isFullCapacity ? "hover:bg-background" : "hover:bg-muted/50"
                  }
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center">
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
