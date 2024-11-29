"use client";

import AppContextProvider from "./context";
import ThemeProvider from "./theme";
import { useEffect } from "react";

export default function Providers({ children }: { children: React.ReactNode }) {
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
