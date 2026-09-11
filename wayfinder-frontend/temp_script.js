tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        page: '#040D3F',
                        surface: '#141C4E',
                        card: '#1D2552',
                        'card-hover': '#243066',
                        primary: '#FFFFFF',
                        secondary: '#B9BBC7',
                        tertiary: '#6A7178',
                        'neutral-4': '#DEE2E6',
                        'neutral-6': '#ADB5BD',
                        positive: '#7FE089',
                        negative: '#F97D7D',
                        'info-cyan': '#0D8091',
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['IBM Plex Mono', 'JetBrains Mono', 'monospace'],
                    },
                    fontSize: {
                        'hero': ['48px', { lineHeight: '1.05', fontWeight: '700' }],
                        'display': ['32px', { lineHeight: '1.2', fontWeight: '700' }],
                        'h1': ['22px', { lineHeight: '1.25', fontWeight: '700' }],
                        'h2': ['16px', { lineHeight: '1.3', fontWeight: '600' }],
                        'card-title': ['16px', { lineHeight: '1.3', fontWeight: '600' }],
                        'body': ['14px', { lineHeight: '1.4', fontWeight: '500' }],
                        'body-subtle': ['13px', { lineHeight: '1.4', fontWeight: '500' }],
                        'caption': ['12px', { lineHeight: '1.4', fontWeight: '400' }],
                        'micro': ['11px', { lineHeight: '1.3', fontWeight: '600' }],
                        'mono-num': ['14px', { lineHeight: '1.4', fontWeight: '500' }],
                    },
                    spacing: {
                        'xs': '4px',
                        'sm': '8px',
                        'md': '12px',
                        'lg': '16px',
                        'xl': '20px',
                        '2xl': '24px',
                        '3xl': '32px',
                    },
                    borderRadius: {
                        'xs': '6px',
                        'sm': '8px',
                        'md': '12px',
                        'lg': '16px',
                        'full': '9999px',
                    },
                    boxShadow: {
                        'none': '0 0 #0000',
                        'tile': '0 2px 4px 0 rgba(0, 0, 0, 0.10)',
                        'card-big': '0 0 60px 0 rgba(0, 4, 26, 0.60)',
                        'modal': '0 8px 32px 0 rgba(0, 0, 0, 0.30)',
                    },
                    keyframes: {
                        scan: {
                            '0%, 100%': { transform: 'translateY(0)' },
                            '50%': { transform: 'translateY(240px)' },
                        }
                    },
                    animation: {
                        scan: 'scan 2.5s ease-in-out infinite',
                    }
                }
            }
        }