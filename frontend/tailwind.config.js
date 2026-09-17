/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        heading: ['"Space Grotesk"', "system-ui", "sans-serif"],
        body: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        primary: {
          DEFAULT: "#1E3A5F",
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1E3A5F",
          800: "#1E3A5F",
          900: "#0F172A",
        },
        accent: {
          DEFAULT: "#059669",
          50: "#ECFDF5",
          100: "#D1FAE5",
          200: "#A7F3D0",
          300: "#6EE7B7",
          400: "#34D399",
          500: "#10B981",
          600: "#059669",
          700: "#047857",
          800: "#065F46",
          900: "#064E3B",
        },
        surface: {
          DEFAULT: "#F8FAFC",
          50: "#FFFFFF",
          100: "#F8FAFC",
          200: "#F1F5F9",
          300: "#E2E8F0",
        },
      },
      boxShadow: {
        glass: "0 8px 32px rgba(30, 58, 95, 0.08)",
        "card-hover":
          "0 20px 40px rgba(30, 58, 95, 0.12), 0 8px 16px rgba(30, 58, 95, 0.06)",
        "glow-blue": "0 0 20px rgba(37, 99, 235, 0.15)",
        "glow-green": "0 0 20px rgba(5, 150, 105, 0.15)",
        "glow-blue-lg": "0 0 40px rgba(37, 99, 235, 0.2)",
        "inner-glow": "inset 0 1px 2px rgba(37, 99, 235, 0.06)",
        premium:
          "0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.06), 0 12px 28px rgba(15, 23, 42, 0.04)",
      },
      backgroundImage: {
        "gradient-premium":
          "linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #059669 100%)",
        "gradient-card":
          "linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%)",
        "gradient-hero":
          "linear-gradient(160deg, #0F172A 0%, #1E3A5F 40%, #2563EB 100%)",
        "gradient-score":
          "linear-gradient(90deg, #2563EB 0%, #059669 100%)",
        "gradient-surface":
          "linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%)",
        "dot-pattern":
          "radial-gradient(circle, #CBD5E1 1px, transparent 1px)",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out forwards",
        "slide-up": "slideUp 0.5s ease-out forwards",
        "slide-up-delayed": "slideUp 0.5s ease-out 0.15s forwards",
        shimmer: "shimmer 2s infinite linear",
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
        "score-fill": "scoreFill 1s ease-out forwards",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        scoreFill: {
          "0%": { strokeDashoffset: "var(--circumference)" },
          "100%": { strokeDashoffset: "var(--target-offset)" },
        },
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      spacing: {
        18: "4.5rem",
        22: "5.5rem",
      },
    },
  },
  plugins: [],
};
