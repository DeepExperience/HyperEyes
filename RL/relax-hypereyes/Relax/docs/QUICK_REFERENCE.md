# Relax Documentation Quick Reference

## 🚀 快速命令

### 开发

```bash
# 启动文档开发服务器
./docs/start-docs.sh
# 或
make docs-dev
# 或
cd docs && npm run docs:dev
```

### 构建

```bash
# 构建生产版本
make docs-build
# 或
cd docs && npm run docs:build
```

### 预览

```bash
# 预览生产版本
make docs-preview
# 或
cd docs && npm run docs:preview
```

### 部署

```bash
# GitHub Pages
./docs/deploy-docs.sh github

# Vercel
./docs/deploy-docs.sh vercel

# Netlify
./docs/deploy-docs.sh netlify
```

## 📁 文件位置

| 内容           | 位置                         |
| -------------- | ---------------------------- |
| VitePress 配置 | `docs/.vitepress/config.mts` |
| 根首页（中文） | `docs/index.md`              |
| 英文首页       | `docs/en/index.md`           |
| 中文首页       | `docs/zh/index.md`           |
| 英文指南       | `docs/en/guide/*.md`         |
| 中文指南       | `docs/zh/guide/*.md`         |
| 英文 API 文档  | `docs/en/api/*.md`           |
| 中文 API 文档  | `docs/zh/api/*.md`           |
| 英文示例文档   | `docs/en/examples/*.md`      |
| 中文示例文档   | `docs/zh/examples/*.md`      |
| 草稿文档       | `docs/draft/*.md`            |
| 自定义主题     | `docs/.vitepress/theme/`     |
| 静态资源       | `docs/public/`               |

## 🎨 主题定制

### 修改颜色

编辑 `docs/.vitepress/theme/custom.css`:

```css
:root {
  --vp-c-brand-1: #ff2442;  /* 主色调 */
}
```

### 修改 Logo

替换 `docs/public/logo.jpg`

### 修改导航

编辑 `docs/.vitepress/config.mts` 中的 `nav` 配置

### 修改侧边栏

编辑 `docs/.vitepress/config.mts` 中的 `sidebar` 配置

## 📝 添加新页面

### 1. 创建文件

```bash
# 英文页面
touch docs/en/guide/new-page.md

# 中文页面
touch docs/zh/guide/new-page.md
```

### 2. 添加到侧边栏

编辑 `docs/.vitepress/config.mts`:

```typescript
sidebar: {
  '/guide/': [
    {
      text: 'Section',
      items: [
        { text: 'New Page', link: '/guide/new-page' }
      ]
    }
  ]
}
```

## 🌐 URL 结构

| 页面     | 英文 URL               | 中文 URL                  |
| -------- | ---------------------- | ------------------------- |
| 首页     | `/`                    | `/zh/`                    |
| 介绍     | `/guide/introduction`  | `/zh/guide/introduction`  |
| 安装     | `/guide/installation`  | `/zh/guide/installation`  |
| 快速上手 | `/guide/quick-start`   | `/zh/guide/quick-start`   |
| 配置     | `/guide/configuration` | `/zh/guide/configuration` |
| 架构     | `/guide/architecture`  | `/zh/guide/architecture`  |

## 🔧 常用 Markdown 语法

### 代码块

\`\`\`python
def hello():
print("Hello, Relax!")
\`\`\`

### 提示框

```markdown
::: tip 提示
这是提示内容
:::

::: warning 警告
这是警告内容
:::

::: danger 危险
这是危险警告
:::
```

### 链接

```markdown
[文本](./path/to/page.md)
[外部链接](https://example.com)
```

### 图片

```markdown
![Alt text](/path/to/image.jpg)
```

## 🐛 故障排除

### 端口被占用

```bash
npm run docs:dev -- --port 5174
```

### 清除缓存

```bash
rm -rf docs/.vitepress/cache
rm -rf docs/node_modules
cd docs && npm install
```

### 构建失败

```bash
cd docs
npm run docs:build -- --debug
```

## 📊 项目结构

```
docs/
├── .vitepress/
│   ├── config.mts          # 主配置文件
│   ├── theme/              # 自定义主题
│   │   ├── index.ts
│   │   └── custom.css
│   └── cache/              # 构建缓存（git ignore）
├── public/                 # 静态资源
│   └── logo.jpg
├── guide/                  # 英文指南
├── zh/                     # 中文内容
│   ├── index.md
│   ├── guide/
│   ├── api/
│   └── examples/
├── api/                    # API 文档
├── examples/               # 示例文档
├── index.md                # 英文首页
├── package.json            # NPM 配置
└── README.md               # 说明文档
```

## 🔗 有用链接

- [VitePress 文档](https://vitepress.dev/)
- [Markdown 指南](https://vitepress.dev/guide/markdown)
- [主题配置](https://vitepress.dev/reference/default-theme-config)
- [部署指南](https://vitepress.dev/guide/deploy)
- [完整文档指南](./DOCS_GUIDE.md)
