"use client";

import { createContext, useContext, useState } from "react";

type ContextType = {
  hoveredDeviceId: number | null;
  setHoveredDeviceId: (id: number | null) => void;
};

const AppContext = createContext<ContextType | undefined>(undefined);

export default function AppContextProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [hoveredDeviceId, setHoveredDeviceId] = useState<number | null>(null);

  return (
    <AppContext.Provider value={{ hoveredDeviceId, setHoveredDeviceId }}>
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
