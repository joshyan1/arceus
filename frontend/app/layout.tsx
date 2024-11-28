import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { ThemeProvider } from "@/components/providers/theme";

const montreal = localFont({
  src: [
    {
      path: "./fonts/montreal-book.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "./fonts/montreal-medium.woff2",
      weight: "500",
      style: "normal",
    },
    {
      path: "./fonts/montreal-bold.woff2",
      weight: "700",
      style: "normal",
    },
  ],
  display: "swap",
  variable: "--font-montreal",
});

const editorial = localFont({
  src: [
    {
      path: "./fonts/editorial.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "./fonts/editorial-italic.woff2",
      weight: "400",
      style: "italic",
    },
  ],
  display: "swap",
  variable: "--font-editorial",
});

const supply = localFont({
  src: "./fonts/supply.woff2",
  display: "swap",
  variable: "--font-supply",
});

export const metadata: Metadata = {
  title: "Arceus",
  description:
    "A decentralized compute network that transforms idle computers into a global marketplace, letting anyone earn by sharing computing power or access affordable compute on demand.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${montreal.variable} ${editorial.variable} ${supply.variable}`}
      >
        <ThemeProvider
          attribute="class"
          forcedTheme="dark"
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
