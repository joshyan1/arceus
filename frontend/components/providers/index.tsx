"use client";

import AppContextProvider from "./context";
import ThemeProvider from "./theme";
import { Toaster } from "@/components/ui/sonner";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AppContextProvider>
      <ThemeProvider
        attribute="class"
        forcedTheme="dark"
        disableTransitionOnChange
      >
        {children}
        <Toaster />
      </ThemeProvider>
    </AppContextProvider>
  );
}
