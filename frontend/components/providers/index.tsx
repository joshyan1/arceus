import AppContextProvider from "./context";
import ThemeProvider from "./theme";

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
