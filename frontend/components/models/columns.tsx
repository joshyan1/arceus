import { ColumnDef } from "@tanstack/react-table";
import { Box, Users, CircleDollarSign, Network } from "lucide-react";

export type AIModel = {
  id: string;
  name: string;
  type: "transformer" | "neuralnetwork";
  spots: string;
  projectedEarnings: number;
};

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
