import { useState } from "react";
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

interface WithId {
  id: string;
}

interface DataTableProps<TData extends WithId, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  columnFilters: ColumnFiltersState;
  setColumnFilters: React.Dispatch<React.SetStateAction<ColumnFiltersState>>;
  onHoverChange?: (id: string | null) => void;
}

export function DataTable<TData extends WithId, TValue>({
  columns,
  data,
  columnFilters,
  setColumnFilters,
  onHoverChange,
}: DataTableProps<TData, TValue>) {
  const [hoveredRowId, setHoveredRowId] = useState<string | null>(null);

  const handleHover = (id: string | null, isFullCapacity: boolean) => {
    if (!isFullCapacity && id !== null) {
      setHoveredRowId(id);
      onHoverChange?.(id);
    }
  };

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
                  className={`${
                    isFullCapacity ? "hover:bg-background" : "hover:bg-muted/50"
                  } ${hoveredRowId === row.original.id ? "bg-muted/50" : ""}`}
                  onMouseEnter={() =>
                    handleHover(row.original.id, isFullCapacity)
                  }
                  onClick={() => handleHover(row.original.id, isFullCapacity)}
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
