<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const props = defineProps<{
  specUrl: string
}>()

const container = ref<HTMLElement | null>(null)

async function renderSwagger() {
  if (!container.value) return

  // Dynamic import to avoid SSR issues
  const SwaggerUIBundle = (await import('swagger-ui-dist/swagger-ui-bundle.js')).default

  // Clear any previous render
  container.value.innerHTML = ''

  SwaggerUIBundle({
    url: props.specUrl,
    domNode: container.value,
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIBundle.SwaggerUIStandalonePreset,
    ],
    layout: 'BaseLayout',
    deepLinking: true,
    defaultModelsExpandDepth: 1,
    defaultModelExpandDepth: 1,
    docExpansion: 'list',
    filter: false,
    showExtensions: true,
    showCommonExtensions: true,
    tryItOutEnabled: false,
  })
}

onMounted(() => {
  renderSwagger()
})

watch(() => props.specUrl, () => {
  renderSwagger()
})
</script>

<template>
  <div class="swagger-ui-wrapper">
    <div ref="container" class="swagger-container" />
  </div>
</template>

<style>
@import 'swagger-ui-dist/swagger-ui.css';

/* ═══════════════════════════════════════════════════════════════════════
   Shared overrides (both light & dark)
   ═══════════════════════════════════════════════════════════════════════ */

.swagger-ui-wrapper {
  margin: 1rem 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
}

.swagger-container .swagger-ui {
  font-family: var(--vp-font-family-base);
}

.swagger-container .swagger-ui .topbar {
  display: none;
}

.swagger-container .swagger-ui .info {
  margin: 20px 0;
}

/* Hide "Try it out" button — this is offline docs */
.swagger-container .swagger-ui .try-out {
  display: none;
}

/* Hide server selector — services are not running */
.swagger-container .swagger-ui .scheme-container {
  display: none;
}

/* ═══════════════════════════════════════════════════════════════════════
   Dark mode — comprehensive overrides
   
   Swagger UI assumes a white background and uses hard-coded dark colors
   everywhere.  We must force *all* text / border / background tokens to
   use VitePress CSS custom properties so that the component adapts to
   the dark theme.
   ═══════════════════════════════════════════════════════════════════════ */

/* -- Global background & base text color -------------------------------- */
.dark .swagger-container .swagger-ui {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .wrapper {
  background: var(--vp-c-bg);
}

/* -- Info header (title / description / version badge) ------------------ */
.dark .swagger-container .swagger-ui .info .title,
.dark .swagger-container .swagger-ui .info .title small,
.dark .swagger-container .swagger-ui .info .title small pre,
.dark .swagger-container .swagger-ui .info hgroup.main a {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .info .description,
.dark .swagger-container .swagger-ui .info .description p,
.dark .swagger-container .swagger-ui .info .description div,
.dark .swagger-container .swagger-ui .info .description li,
.dark .swagger-container .swagger-ui .info .description code,
.dark .swagger-container .swagger-ui .info .base-url,
.dark .swagger-container .swagger-ui .info a {
  color: var(--vp-c-text-2);
}

/* -- Tag groups --------------------------------------------------------- */
.dark .swagger-container .swagger-ui .opblock-tag {
  color: var(--vp-c-text-1);
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .opblock-tag:hover {
  background: var(--vp-c-bg-soft);
}

.dark .swagger-container .swagger-ui .opblock-tag small {
  color: var(--vp-c-text-2);
}

/* -- Operation blocks (GET / POST / …) ---------------------------------- */
.dark .swagger-container .swagger-ui .opblock {
  border-color: var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}

.dark .swagger-container .swagger-ui .opblock .opblock-summary {
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .opblock .opblock-summary-description {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .opblock .opblock-summary-operation-id,
.dark .swagger-container .swagger-ui .opblock .opblock-summary-path,
.dark .swagger-container .swagger-ui .opblock .opblock-summary-path a {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .opblock .opblock-section-header {
  background: var(--vp-c-bg-alt);
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .opblock .opblock-section-header h4,
.dark .swagger-container .swagger-ui .opblock .opblock-section-header label {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .opblock-body pre.microlight {
  background: var(--vp-c-bg-alt) !important;
  color: var(--vp-c-text-1) !important;
  border-color: var(--vp-c-divider);
}

/* -- GET blocks --------------------------------------------------------- */
.dark .swagger-container .swagger-ui .opblock.opblock-get {
  background: rgba(97, 175, 254, 0.08);
  border-color: rgba(97, 175, 254, 0.3);
}

.dark .swagger-container .swagger-ui .opblock.opblock-get .opblock-summary {
  border-color: rgba(97, 175, 254, 0.3);
}

/* -- POST blocks -------------------------------------------------------- */
.dark .swagger-container .swagger-ui .opblock.opblock-post {
  background: rgba(73, 204, 144, 0.08);
  border-color: rgba(73, 204, 144, 0.3);
}

.dark .swagger-container .swagger-ui .opblock.opblock-post .opblock-summary {
  border-color: rgba(73, 204, 144, 0.3);
}

/* -- PUT blocks --------------------------------------------------------- */
.dark .swagger-container .swagger-ui .opblock.opblock-put {
  background: rgba(252, 161, 48, 0.08);
  border-color: rgba(252, 161, 48, 0.3);
}

/* -- DELETE blocks ------------------------------------------------------ */
.dark .swagger-container .swagger-ui .opblock.opblock-delete {
  background: rgba(249, 62, 62, 0.08);
  border-color: rgba(249, 62, 62, 0.3);
}

/* -- Operation description / body text ---------------------------------- */
.dark .swagger-container .swagger-ui .opblock-description-wrapper,
.dark .swagger-container .swagger-ui .opblock-description-wrapper p,
.dark .swagger-container .swagger-ui .opblock-external-docs-wrapper,
.dark .swagger-container .swagger-ui .opblock-external-docs-wrapper p,
.dark .swagger-container .swagger-ui .opblock-body .opblock-description {
  color: var(--vp-c-text-2);
}

/* -- Parameter table ---------------------------------------------------- */
.dark .swagger-container .swagger-ui table thead tr th,
.dark .swagger-container .swagger-ui table thead tr td {
  color: var(--vp-c-text-1);
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui table tbody tr td {
  color: var(--vp-c-text-1);
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .parameter__name {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .parameter__name.required::after {
  color: #f93e3e;
}

.dark .swagger-container .swagger-ui .parameter__type {
  color: var(--vp-c-text-3);
}

.dark .swagger-container .swagger-ui .parameter__in {
  color: var(--vp-c-text-3);
}

.dark .swagger-container .swagger-ui .parameter__deprecated {
  color: #f93e3e;
}

/* -- Parameter input fields --------------------------------------------- */
.dark .swagger-container .swagger-ui input[type="text"],
.dark .swagger-container .swagger-ui input[type="password"],
.dark .swagger-container .swagger-ui input[type="search"],
.dark .swagger-container .swagger-ui input[type="email"],
.dark .swagger-container .swagger-ui input[type="file"],
.dark .swagger-container .swagger-ui textarea,
.dark .swagger-container .swagger-ui select {
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-divider);
}

/* -- Response section --------------------------------------------------- */
.dark .swagger-container .swagger-ui .responses-inner h4,
.dark .swagger-container .swagger-ui .responses-inner h5,
.dark .swagger-container .swagger-ui .responses-inner label {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .response-col_status {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .response-col_description,
.dark .swagger-container .swagger-ui .response-col_description p {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .response-col_links {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .response-col_links a {
  color: var(--vp-c-brand-1);
}

.dark .swagger-container .swagger-ui .responses-table {
  border-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .response {
  color: var(--vp-c-text-1);
}

/* -- Markdown rendered in descriptions ---------------------------------- */
.dark .swagger-container .swagger-ui .markdown p,
.dark .swagger-container .swagger-ui .markdown li,
.dark .swagger-container .swagger-ui .markdown h1,
.dark .swagger-container .swagger-ui .markdown h2,
.dark .swagger-container .swagger-ui .markdown h3,
.dark .swagger-container .swagger-ui .markdown h4,
.dark .swagger-container .swagger-ui .markdown h5,
.dark .swagger-container .swagger-ui .markdown pre,
.dark .swagger-container .swagger-ui .markdown code,
.dark .swagger-container .swagger-ui .renderedMarkdown p,
.dark .swagger-container .swagger-ui .renderedMarkdown li,
.dark .swagger-container .swagger-ui .renderedMarkdown code {
  color: var(--vp-c-text-2);
}

/* -- Code / JSON blocks ------------------------------------------------- */
.dark .swagger-container .swagger-ui .highlight-code,
.dark .swagger-container .swagger-ui .highlight-code .microlight {
  background: var(--vp-c-bg-alt) !important;
  color: var(--vp-c-text-1) !important;
}

.dark .swagger-container .swagger-ui .example.microlight,
.dark .swagger-container .swagger-ui pre.example.microlight {
  background: var(--vp-c-bg-alt) !important;
  color: var(--vp-c-text-1) !important;
  border-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .microlight code {
  color: var(--vp-c-text-1) !important;
}

/* -- Schema / Model section --------------------------------------------- */
.dark .swagger-container .swagger-ui section.models {
  border-color: var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.dark .swagger-container .swagger-ui section.models h4,
.dark .swagger-container .swagger-ui section.models h4 span {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui section.models.is-open h4 {
  border-bottom-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .model-container {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-divider);
}

.dark .swagger-container .swagger-ui .model-box {
  background: var(--vp-c-bg-soft);
}

.dark .swagger-container .swagger-ui .model {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .model-title {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .model .property,
.dark .swagger-container .swagger-ui .model .property.primitive {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .prop-type {
  color: var(--vp-c-brand-1);
}

.dark .swagger-container .swagger-ui .prop-format {
  color: var(--vp-c-text-3);
}

/* -- Expand / collapse arrows (SVG fill) -------------------------------- */
.dark .swagger-container .swagger-ui .expand-operation svg,
.dark .swagger-container .swagger-ui .model-toggle::after {
  fill: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .arrow,
.dark .swagger-container .swagger-ui svg.arrow {
  fill: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui button {
  color: var(--vp-c-text-1);
}

/* -- Tabs (e.g. "Example Value" / "Schema") ----------------------------- */
.dark .swagger-container .swagger-ui .tab li {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .tab li.active {
  color: var(--vp-c-text-1);
}

.dark .swagger-container .swagger-ui .tab li button.tablinks {
  color: var(--vp-c-text-2);
}

.dark .swagger-container .swagger-ui .tab li.active button.tablinks {
  color: var(--vp-c-text-1);
}

/* -- Content-type dropdown selectors ------------------------------------ */
.dark .swagger-container .swagger-ui .opblock-body select,
.dark .swagger-container .swagger-ui .scheme-container select {
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-divider);
}

/* -- Copy-to-clipboard button ------------------------------------------- */
.dark .swagger-container .swagger-ui .copy-to-clipboard {
  background: var(--vp-c-bg-alt);
}

.dark .swagger-container .swagger-ui .copy-to-clipboard button {
  color: var(--vp-c-text-2);
}

/* -- Authorization / lock icon ------------------------------------------ */
.dark .swagger-container .swagger-ui .authorization__btn svg {
  fill: var(--vp-c-text-2);
}

/* -- Loading indicator -------------------------------------------------- */
.dark .swagger-container .swagger-ui .loading-container .loading::after {
  color: var(--vp-c-text-2);
}

/* -- Blanket rule: catch anything we missed ----------------------------- */
.dark .swagger-container .swagger-ui p,
.dark .swagger-container .swagger-ui span,
.dark .swagger-container .swagger-ui li,
.dark .swagger-container .swagger-ui label,
.dark .swagger-container .swagger-ui small,
.dark .swagger-container .swagger-ui h1,
.dark .swagger-container .swagger-ui h2,
.dark .swagger-container .swagger-ui h3,
.dark .swagger-container .swagger-ui h4,
.dark .swagger-container .swagger-ui h5,
.dark .swagger-container .swagger-ui h6 {
  color: var(--vp-c-text-1);
}

/* Slightly subdued for secondary text */
.dark .swagger-container .swagger-ui .opblock-description-wrapper p,
.dark .swagger-container .swagger-ui .opblock-description-wrapper span,
.dark .swagger-container .swagger-ui .info .description p,
.dark .swagger-container .swagger-ui .info .description span {
  color: var(--vp-c-text-2);
}
</style>
