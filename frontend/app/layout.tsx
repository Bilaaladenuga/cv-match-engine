import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Career Match — CV–Job Compatibility Analysis",
  description:
    "An explainable ML system that analyzes how well your CV matches a job description.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
