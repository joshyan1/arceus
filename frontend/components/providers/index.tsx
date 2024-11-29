"use client";

import AppContextProvider from "./context";
import ThemeProvider from "./theme";
import { initSocket } from "@/lib/socket";
import { useEffect } from "react";

export default function Providers({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    initSocket();
  }, []);

  return (
    <AppContextProvider>
      <ThemeProvider
        attribute="class"
        forcedTheme="dark"
        disableTransitionOnChange
      >
        {children}
      </ThemeProvider>
    </AppContextProvider>
  );
}
