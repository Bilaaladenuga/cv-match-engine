"use client";

/**
 * MobileNav — hamburger trigger + slide-in panel for small screens.
 *
 * The desktop nav is a centered pill; on phones there isn't room for links
 * plus a CTA, and hiding the CTA outright left phones without a visible
 * path into the analyzer. This panel restores it: big tappable links, the
 * primary CTA, and product facts, styled with the same brutalist tokens
 * (paper pill, carbon cards, mint highlight).
 *
 * Behaviour: locks body scroll while open, closes on Escape / backdrop tap /
 * route change, and the hamburger is a real <button> with proper ARIA.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { FileText, Menu, X } from "lucide-react";

const LINKS = [
  { href: "/analyze", label: "Analyze" },
  { href: "/history", label: "History" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function MobileNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // Close automatically whenever the route changes.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  // Scroll lock + Escape to close while the panel is open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      {/* Trigger — replaces the desktop pill entirely below md. */}
      <div className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-3 md:hidden">
        <Link
          href="/"
          className="rounded-pill bg-paper px-4 py-2 font-display text-lg uppercase tracking-tight text-carbon"
        >
          CV Match
        </Link>
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="Open navigation menu"
          aria-expanded={open}
          aria-controls="mobile-nav-panel"
          className="rounded-pill bg-paper p-2.5 text-carbon shadow-sm active:bg-mist"
        >
          <Menu className="h-6 w-6" />
        </button>
      </div>

      {/* Slide-in panel + backdrop */}
      <div
        id="mobile-nav-panel"
        className={`fixed inset-0 z-[60] md:hidden ${open ? "" : "pointer-events-none"}`}
        aria-hidden={!open}
      >
        {/* Backdrop */}
        <button
          type="button"
          aria-label="Close navigation menu"
          onClick={() => setOpen(false)}
          className={`absolute inset-0 h-full w-full bg-carbon/40 transition-opacity duration-200 ${
            open ? "opacity-100" : "opacity-0"
          }`}
          tabIndex={open ? 0 : -1}
        />

        {/* Panel */}
        <div
          className={`absolute top-0 right-0 h-full w-[78%] max-w-xs bg-paper shadow-2xl transition-transform duration-300 ease-out ${
            open ? "translate-x-0" : "translate-x-full"
          }`}
          role="dialog"
          aria-modal="true"
          aria-label="Navigation"
        >
          <div className="flex items-center justify-between border-b border-ash px-5 py-4">
            <span className="font-display text-lg uppercase tracking-tight text-carbon">
              Menu
            </span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close menu"
              className="rounded-full p-2 text-carbon active:bg-mist"
            >
              <X className="h-6 w-6" />
            </button>
          </div>

          <nav className="flex flex-col px-5 py-6">
            {LINKS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`flex items-center justify-between border-b border-ash/60 py-4 font-display text-2xl uppercase tracking-tight ${
                  pathname === href ? "text-carbon" : "text-slate"
                }`}
              >
                {label}
                <FileText className="h-4 w-4 text-smoke" />
              </Link>
            ))}
          </nav>

          <div className="px-5">
            <Link href="/analyze" className="btn-primary w-full justify-center text-base">
              Start Analyzing
            </Link>
            <p className="mt-4 text-caption text-smoke">
              Free, no sign-up. Your CV never leaves your browser session.
            </p>
          </div>

          <div className="absolute bottom-6 left-5 right-5 rounded-card bg-carbon p-4 text-paper">
            <p className="font-mono text-[11px] uppercase tracking-wide text-mint">
              Explainable ML
            </p>
            <p className="mt-1 text-caption text-ash">
              Scores come from a real model — matched, partial, and missing
              skills are shown with evidence.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
