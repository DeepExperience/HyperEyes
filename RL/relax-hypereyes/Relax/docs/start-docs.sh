#!/bin/bash
# Quick start script for Relax documentation

set -e

echo "🚀 Starting Relax Documentation..."

# Navigate to docs directory
cd "$(dirname "$0")/../docs"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start dev server
echo "🌐 Starting development server..."
echo "📖 Documentation will be available at http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm run docs:dev
