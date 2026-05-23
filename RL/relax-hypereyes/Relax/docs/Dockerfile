# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM artifactory.devops.xiaohongshu.com/media/node:20-alpine AS builder

WORKDIR /app

# Switch to Aliyun mirror and install git + Python (required for OpenAPI generation)
RUN sed -i 's|https://dl-cdn.alpinelinux.org|https://mirrors.aliyun.com|g' /etc/apk/repositories \
    && apk add --no-cache git python3 py3-pip \
    && pip3 install --no-cache-dir --break-system-packages fastapi pydantic

# Copy dependency manifests first for layer caching
COPY package.json ./

RUN npm install

# Copy source files needed for the build; include .git for git-log access
COPY .git/ ./.git/
COPY docs/ ./docs/

# Copy project source needed by scripts/tools/generate_openapi.py
# (heavy deps like Ray/PyTorch are shimmed; only fastapi + pydantic are needed)
COPY scripts/tools/generate_openapi.py ./scripts/tools/generate_openapi.py
COPY relax/ ./relax/

# Build VitePress site and apply the WAF chunk-name fix
RUN npm run docs:build

# ── Stage 2: serve ────────────────────────────────────────────────────────────
FROM artifactory.devops.xiaohongshu.com/media/nginx:1.27-alpine AS runner

# Copy built site into the /Relax/ sub-path to match VitePress base: '/Relax/'
RUN mkdir -p /usr/share/nginx/html/Relax && rm -rf /usr/share/nginx/html/index.html
COPY --from=builder /app/docs/.vitepress/dist /usr/share/nginx/html/Relax

# Nginx config: serve under /Relax/ prefix with clean-URL and asset-cache support
RUN printf 'server {\n\
    listen       80;\n\
    server_name  _;\n\
\n\
    # Redirect bare root to the docs base path\n\
    location = / {\n\
        return 301 /Relax/;\n\
    }\n\
\n\
    location /Relax/ {\n\
        root  /usr/share/nginx/html;\n\
        index index.html;\n\
\n\
        # VitePress cleanUrls:false — try exact file, then .html, then directory index\n\
        try_files $uri $uri.html $uri/index.html =404;\n\
    }\n\
\n\
    # Cache immutable hashed assets aggressively\n\
    location ~* /Relax/assets/.*\\.(js|css|woff2?|ttf|svg|png|jpg|ico)$ {\n\
        root  /usr/share/nginx/html;\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
\n\
    error_page 404 /Relax/404.html;\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
