# Relax Documentation

This directory contains the VitePress documentation site for Relax.

## Features

- 📚 Bilingual documentation (English & Chinese)
- 🎨 Beautiful VitePress theme with custom branding
- 🔍 Full-text search support
- 🖼️ **Image zoom functionality** - Click any image to view it in full size
- 📊 Mermaid diagram support
- 🌓 Dark mode support

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

## Structure

```
docs/
├── .vitepress/
│   ├── config.mts          # VitePress configuration
│   └── theme/              # Custom theme
├── public/                 # Static assets
├── en/                     # English documentation
│   ├── guide/              # English guides
│   │   ├── introduction.md
│   │   ├── installation.md
│   │   ├── quick-start.md
│   │   └── ...
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   └── index.md            # English homepage
├── zh/                     # Chinese documentation
│   ├── guide/              # Chinese guides
│   ├── api/                # API documentation
│   ├── examples/           # Example documentation
│   └── index.md            # Chinese homepage
├── draft/                  # Draft documentation
│   ├── design.md
│   ├── metrics_service_usage.md
│   └── ...
└── index.md                # Root homepage (defaults to Chinese)
```

## Adding New Pages

1. Create a new markdown file in the appropriate directory
2. Add the page to the sidebar in `.vitepress/config.mts`
3. Add translations if needed

## Image Zoom Feature

All images in the documentation support click-to-zoom functionality powered by `medium-zoom`.

**Usage:**

- Hover over any image to see the zoom cursor
- Click to enlarge the image
- Click again or press ESC to close

**Documentation:**

- [Quick Start Guide](QUICK_START_IMAGE_ZOOM.md)
- [Feature Details](IMAGE_ZOOM_FEATURE.md)
- [Implementation Summary](IMAGE_ZOOM_IMPLEMENTATION.md)
- [Testing Guide](TESTING_IMAGE_ZOOM.md)

## Deployment

The documentation can be deployed to:

- GitHub Pages
- Vercel
- Netlify
- Any static hosting service

See [VitePress deployment guide](https://vitepress.dev/guide/deploy) for details.
