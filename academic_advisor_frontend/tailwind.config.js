/** @type {import('tailwindcss').Config} */
export default {
  important: true,
  theme: {
    extend: {
      colors: {
        brand: {
          red: '#B41E2A',
          red2: '#8F1721',
          navy: '#0B1330',
          ink: '#1E2235',
          line: '#E6E8EE',
          cream: '#FAFAFD',
        },
      },
      boxShadow: { soft: '0 1px 2px rgba(0,0,0,.06), 0 10px 30px rgba(0,0,0,.06)' },
      borderRadius: { xl2: '1rem' },
    },
  },
  plugins: [],
};
