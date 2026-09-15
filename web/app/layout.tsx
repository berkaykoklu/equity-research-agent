import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

const DESCRIPTION =
  "Research notes in which every sentence is traceable to the filing passage it came from, and every figure is read from the company's own filed data rather than written by a model. The checker is ordinary code, not a model judging a model.";

export const metadata: Metadata = {
  metadataBase: new URL("https://equity.berkaykoklu.com"),
  title: "Research notes that cannot cite what does not exist",
  description: DESCRIPTION,
  openGraph: { title: "Research notes that cannot cite what does not exist", description: DESCRIPTION, type: "website", locale: "en" },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" />
      </head>
      <body>{children}</body>
    </html>
  );
}
