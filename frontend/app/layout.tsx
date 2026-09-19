import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "CV Match — AI-Powered CV–Job Compatibility Analysis",
  description:
    "Drop in your CV and job description. Get a scored breakdown with clear feedback on what to fix before you apply.",
};

// Explicit (Next's default matches this, but responsive layout depends on
// it, so it must never be accidentally dropped).
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
