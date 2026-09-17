"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { FileCheck2 } from "lucide-react";

const links = [
  { href: "/analyze", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/history", label: "History" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 glass-strong border-b border-white/30">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3.5">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-700 via-primary-600 to-accent-600 shadow-glow-blue transition-shadow duration-300 group-hover:shadow-glow-blue-lg">
            <FileCheck2 className="h-4 w-4 text-white" />
          </div>
          <span className="font-heading text-lg font-bold tracking-tight text-gray-900">
            Career
            <span className="gradient-text-blue"> Match</span>
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {links.map((l) => {
            const active =
              pathname === l.href || pathname.startsWith(l.href + "/");
            return (
              <Link
                key={l.href}
                href={l.href}
                className={clsx(
                  "relative rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200",
                  active
                    ? "bg-primary-700/8 text-primary-700"
                    : "text-gray-500 hover:bg-gray-100/60 hover:text-gray-900"
                )}
              >
                {l.label}
                {active ? (
                  <span className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-gradient-to-r from-primary-600 to-accent-500" />
                ) : null}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
