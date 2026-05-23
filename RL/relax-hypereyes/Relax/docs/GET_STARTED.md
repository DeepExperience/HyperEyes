# 🚀 快速开始 Relax 文档

## 一分钟启动文档站点

```bash
# 1. 安装依赖
cd docs
npm install

# 2. 启动开发服务器
npm run docs:dev

# 3. 打开浏览器访问
# http://localhost:5173
```

就这么简单！🎉

## 或者使用快捷命令

```bash
# 使用脚本
./docs/start-docs.sh

# 使用 Makefile
make docs-dev
```

## 构建生产版本

```bash
# 构建
make docs-build

# 预览
make docs-preview
```

## 部署到生产环境

```bash
# GitHub Pages（推荐）
./docs/deploy-docs.sh github

# Vercel
./docs/deploy-docs.sh vercel

# Netlify
./docs/deploy-docs.sh netlify
```

## 文档特性

✨ **双语支持** - 中文和英文
✨ **搜索功能** - 快速查找内容
✨ **暗色模式** - 自动切换
✨ **响应式** - 支持移动端
✨ **代码高亮** - 多语言支持

## 需要帮助？

- 📖 [完整文档指南](../DOCS_GUIDE.md)
- 📋 [快速参考](./QUICK_REFERENCE.md)
- 📊 [部署总结](./DEPLOYMENT_SUMMARY.md)
- ✅ [完成报告](../DOCUMENTATION_COMPLETE.md)

## 文档结构

```
docs/
├── en/             # 英文文档
│   ├── guide/      # 英文指南
│   ├── api/        # API 文档
│   └── examples/   # 示例文档
├── zh/             # 中文文档
│   ├── guide/      # 中文指南
│   ├── api/        # API 文档
│   └── examples/   # 示例文档
├── draft/          # 草稿文档
└── public/         # 静态资源
```

## 添加新页面

1. 创建 Markdown 文件
2. 在 `.vitepress/config.mts` 添加到侧边栏
3. 保存并查看效果（热重载）

就是这么简单！开始探索吧！🚀
