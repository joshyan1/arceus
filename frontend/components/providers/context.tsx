"use client";

import { createContext, useContext, useState } from "react";

type ContextType = {
  hoveredLayers: number[];
  setHoveredLayers: (layers: number[]) => void;
};

const AppContext = createContext<ContextType | undefined>(undefined);

export default function AppContextProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [hoveredLayers, setHoveredLayers] = useState<number[]>([]);

  return (
    <AppContext.Provider value={{ hoveredLayers, setHoveredLayers }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within a AppContextProvider");
  }
  return context;
}
