"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const links = [
  { href: "/analyze", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/history", label: "History" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-xl font-semibold text-gray-900">
          Career Match
        </Link>
        <nav className="flex gap-5 text-sm">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={clsx(
                "transition-colors",
                pathname === l.href || pathname.startsWith(l.href + "/")
                  ? "font-medium text-gray-900"
                  : "text-gray-500 hover:text-gray-900"
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
