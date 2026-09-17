import type { Metadata } from "next";

import "./globals.css";
import { SiteHeader } from "@/components/SiteHeader";

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
      <body className="antialiased">
        <SiteHeader />
        {children}
        <footer className="mt-auto border-t border-gray-200/40 bg-white/40 backdrop-blur-sm">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5 text-xs text-gray-400">
            <span>Career Match — Explainable CV–Job compatibility</span>
            <span className="hidden sm:inline">
              Privacy-first · No accounts · Your data stays in your browser
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
