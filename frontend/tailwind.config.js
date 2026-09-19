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
        display: ['"Anton"', '"Bebas Neue"', '"Barlow Condensed"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        body: ['"Inter"', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        carbon: '#000000',
        paper: '#ffffff',
        canvas: '#e5e5e5',
        mist: '#f3f3f3',
        ash: '#c6c6c6',
        smoke: '#979797',
        slate: '#444444',
        graphite: '#2f2f2f',
        mint: '#d1ffca',
        voltage: '#fff100',
      },
      borderRadius: {
        'card': '24px',
        'card-lg': '32px',
        'card-xl': '64px',
        'pill': '48px',
        'tag': '64px',
      },
      spacing: {
        'unit': '8px',
        'section': '80px',
      },
      fontSize: {
        'caption': ['12px', { lineHeight: '1.6', letterSpacing: '-0.03em' }],
        'body-sm': ['14px', { lineHeight: '1.3', letterSpacing: '-0.011em' }],
        'body': ['16px', { lineHeight: '1.25' }],
        'sub': ['18px', { lineHeight: '1.33' }],
        'sub-lg': ['20px', { lineHeight: '1.2' }],
        // Display sizes scale fluidly with the viewport (clamp) so the
        // brutalist headlines never overflow phones or tablets.
        'heading-sm': ['clamp(24px, 5.5vw, 28px)', { lineHeight: '1.3', letterSpacing: '-0.03em' }],
        'heading': ['clamp(30px, 6.5vw, 40px)', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
        'heading-lg': ['clamp(34px, 8vw, 48px)', { lineHeight: '0.9', letterSpacing: '-0.03em' }],
        'display': ['clamp(42px, 10.5vw, 80px)', { lineHeight: '0.9', letterSpacing: '-0.03em' }],
        'display-xl': ['clamp(52px, 13vw, 130px)', { lineHeight: '0.9', letterSpacing: '-0.03em' }],
      },
      maxWidth: {
        // Container used by every page (max-w-page). 72rem = 1152px.
        'page': '72rem',
      },
    },
  },
  plugins: [],
};
