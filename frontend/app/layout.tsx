import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "CV Match — AI-Powered CV–Job Compatibility Analysis",
  description:
    "Drop in your CV and job description. Get a scored breakdown with clear feedback on what to fix before you apply.",
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
