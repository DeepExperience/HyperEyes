<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'

const { lang } = useData()
const route = useRoute()
const isZh = computed(() => lang.value === 'zh-CN')

const isHome = ref(false)

function checkIsHome() {
  const path = route.path
  const p = path.replace(/index\.html$/, '').replace(/\/+$/, '')
  isHome.value = p === '' || p.endsWith('/en') || p.endsWith('/zh')
}

onMounted(() => checkIsHome())
watch(() => route.path, () => checkIsHome())

const langPrefix = computed(() => (isZh.value ? '/zh' : '/en'))

interface ShowcaseItem {
  icon: string
  label: string
  labelZh: string
  desc: string
  descZh: string
  tags: string[]
  link: string
  category: 'model' | 'algorithm' | 'task' | 'feature'
}

const items: ShowcaseItem[] = [
  // ── Models ──
  {
    icon: '◇',
    label: 'Qwen3',
    labelZh: 'Qwen3',
    desc: 'Dense & MoE text models (4B, 30B-A3B). Math reasoning, code generation, and multi-turn dialogue.',
    descZh: 'Dense 与 MoE 文本模型 (4B, 30B-A3B)，支持数学推理、代码生成与多轮对话。',
    tags: ['Text', 'MoE', 'GRPO'],
    link: '/guide/quick-start',
    category: 'model',
  },
  {
    icon: '◇',
    label: 'Qwen3-VL',
    labelZh: 'Qwen3-VL',
    desc: 'Vision-Language models (4B, 30B-A3B). Image understanding, visual QA, and multimodal reasoning.',
    descZh: '视觉语言模型 (4B, 30B-A3B)，支持图像理解、视觉问答与多模态推理。',
    tags: ['Vision', 'MoE'],
    link: '/guide/quick-start',
    category: 'model',
  },
  {
    icon: '◇',
    label: 'Qwen3-Omni',
    labelZh: 'Qwen3-Omni',
    desc: 'Omni-modal model (30B-A3B). Joint image, text, and audio understanding in a unified framework.',
    descZh: '全模态模型 (30B-A3B)，在统一框架下联合理解图像、文本与音频。',
    tags: ['Omni', 'Audio', 'Vision'],
    link: '/guide/quick-start',
    category: 'model',
  },
  {
    icon: '◇',
    label: 'Qwen3.5',
    labelZh: 'Qwen3.5',
    desc: 'Latest multimodal model (9B, 30B-A3B). Enhanced visual reasoning and instruction following.',
    descZh: '最新多模态模型 (9B, 30B-A3B)，增强视觉推理与指令跟随能力。',
    tags: ['Vision', 'MoE'],
    link: '/guide/quick-start',
    category: 'model',
  },
  // ── Algorithms ──
  {
    icon: '△',
    label: 'GRPO',
    labelZh: 'GRPO',
    desc: 'Group Relative Policy Optimization. Reference-free advantage estimation with group-level normalization.',
    descZh: 'Group Relative Policy Optimization，基于组级归一化的无参考优势估计。',
    tags: ['On-Policy', 'Group Sampling'],
    link: '/guide/customize-training',
    category: 'algorithm',
  },
  {
    icon: '△',
    label: 'GSPO',
    labelZh: 'GSPO',
    desc: 'Group Sample Policy Optimization. Sample-aware variant with improved credit assignment.',
    descZh: 'Group Sample Policy Optimization，改进 credit assignment 的样本感知变体。',
    tags: ['On-Policy', 'Sample-Aware'],
    link: '/guide/customize-training',
    category: 'algorithm',
  },
  {
    icon: '△',
    label: 'SAPO',
    labelZh: 'SAPO',
    desc: 'Sample-Aware Policy Optimization. Fine-grained per-sample advantage weighting for stable training.',
    descZh: 'Sample-Aware Policy Optimization，细粒度逐样本优势加权，训练更稳定。',
    tags: ['On-Policy', 'Stable'],
    link: '/guide/customize-training',
    category: 'algorithm',
  },
  {
    icon: '△',
    label: 'On-Policy Distillation',
    labelZh: '在线蒸馏',
    desc: 'Teacher-student knowledge distillation with KL penalty. Train compact models from larger teachers.',
    descZh: '基于 KL 惩罚的师生知识蒸馏，从大模型蒸馏出高效紧凑模型。',
    tags: ['KL Penalty', 'Teacher-Student'],
    link: '/examples/on-policy-distillation',
    category: 'algorithm',
  },
  // ── Tasks ──
  {
    icon: '○',
    label: 'DAPO Math',
    labelZh: 'DAPO 数学',
    desc: 'Math reasoning on dapo-math-17k. Rule-based symbolic verification with GRPO training.',
    descZh: '基于 dapo-math-17k 的数学推理，规则符号验证配合 GRPO 训练。',
    tags: ['Text', 'Qwen3-4B'],
    link: '/guide/quick-start',
    category: 'task',
  },
  {
    icon: '○',
    label: 'Open-R1 VL',
    labelZh: 'Open-R1 视觉语言',
    desc: 'Vision-language reasoning on multimodal-open-r1-8k. Image + text understanding with symbolic math verification.',
    descZh: '基于 multimodal-open-r1-8k 的视觉语言推理，图像+文本理解配合符号数学验证。',
    tags: ['Vision', 'Qwen3-VL-4B'],
    link: '/guide/quick-start',
    category: 'task',
  },
  {
    icon: '○',
    label: 'AVQA Omni',
    labelZh: 'AVQA 全模态',
    desc: 'Omni-modal QA on AVQA-R1-6K. Joint image + audio understanding with Qwen3-Omni.',
    descZh: '基于 AVQA-R1-6K 的全模态问答，Qwen3-Omni 联合理解图像+音频。',
    tags: ['Omni', 'Qwen3-Omni'],
    link: '/guide/quick-start',
    category: 'task',
  },
  {
    icon: '○',
    label: 'DeepEyes',
    labelZh: 'DeepEyes',
    desc: 'Agentic multi-turn VL task with tool use. The model interacts with visual tools (zoom, rotate) to solve tasks.',
    descZh: 'Agentic 多轮视觉语言任务，模型通过视觉工具（缩放、旋转）交互式解决任务。',
    tags: ['Agentic', 'Multi-Turn', 'Tool Use'],
    link: '/examples/deepeyes',
    category: 'task',
  },
  {
    icon: '○',
    label: 'R2E-Gym',
    labelZh: 'R2E-Gym',
    desc: 'Code generation and repository-level editing environment. Coming soon.',
    descZh: '代码生成与仓库级编辑环境，敬请期待。',
    tags: ['Code', 'Coming Soon'],
    link: '',
    category: 'task',
  },
  {
    icon: '○',
    label: 'NextQA',
    labelZh: 'NextQA',
    desc: 'Video question answering benchmark for temporal and causal reasoning with Qwen3-Omni.',
    descZh: '视频问答基准，基于 Qwen3-Omni 进行时序与因果推理。',
    tags: ['Video', 'Qwen3-Omni'],
    link: '/guide/quick-start',
    category: 'task',
  },
  // ── Features ──
  {
    icon: '▣',
    label: 'Fully Async Training',
    labelZh: '全异步训练',
    desc: 'Decouple training, inference, and forward computation on independent GPU clusters. Eliminate GPU idle time with TransferQueue streaming.',
    descZh: '将训练、推理、前向计算解耦到独立 GPU 集群，通过 TransferQueue 流式传输消除 GPU 空闲。',
    tags: ['TransferQueue', 'StreamingDataLoader'],
    link: '/guide/fully-async-training',
    category: 'feature',
  },
  {
    icon: '▣',
    label: 'Lossless R3',
    labelZh: '无损 R3',
    desc: 'Routing Replay for lossless MoE training. Async D-to-H copy and NCCL broadcast for zero-overhead expert routing.',
    descZh: 'Routing Replay 实现 MoE 无损训练，异步 D-to-H 拷贝与 NCCL 广播实现零开销专家路由。',
    tags: ['MoE', 'Routing Replay'],
    link: '/guide/configuration',
    category: 'feature',
  },
  {
    icon: '▣',
    label: 'Omni-Modal',
    labelZh: '全模态',
    desc: 'Native support for text, image, video, and audio RL training in a unified framework. Few systems in the industry can match this.',
    descZh: '在统一框架内原生支持文本、图像、视频与音频 RL 训练，业界少有的全模态能力。',
    tags: ['Text', 'Vision', 'Audio'],
    link: '/guide/introduction',
    category: 'feature',
  },
  {
    icon: '▣',
    label: 'DCS Weight Sync',
    labelZh: 'DCS 权重同步',
    desc: 'Distributed Checkpoint Service with NCCL broadcast. Control/data plane separation for pipelined weight updates overlapping training.',
    descZh: '基于 NCCL 广播的分布式 Checkpoint 服务，控制/数据平面分离，权重更新与训练重叠。',
    tags: ['NCCL', 'Checkpoint'],
    link: '/guide/distributed-checkpoint',
    category: 'feature',
  },
  {
    icon: '▣',
    label: 'Elastic Scaling',
    labelZh: '弹性扩缩容',
    desc: 'Dynamically scale Rollout engines via HTTP REST API during training. Same-cluster and cross-cluster federation modes.',
    descZh: '训练中通过 HTTP REST API 动态扩缩 Rollout 引擎，支持同集群与跨集群联邦模式。',
    tags: ['REST API', 'Non-Blocking'],
    link: '/guide/elastic-rollout',
    category: 'feature',
  },
  {
    icon: '▣',
    label: 'Health Monitor',
    labelZh: '健康监控',
    desc: 'Two-tier auto-recovery with heartbeat monitoring. In-place restart for transient failures, global restart for persistent ones.',
    descZh: '基于心跳的两级自动恢复机制，瞬态故障就地重启，持久故障全局重启。',
    tags: ['Heartbeat', 'Auto-Recovery'],
    link: '/guide/health-check-manager',
    category: 'feature',
  },
]

type Category = ShowcaseItem['category']

const categories: { key: Category; label: string; labelZh: string }[] = [
  { key: 'feature', label: 'Features', labelZh: '特性' },
  { key: 'task', label: 'Tasks', labelZh: '任务' },
  { key: 'model', label: 'Models', labelZh: '模型' },
  { key: 'algorithm', label: 'Algorithms', labelZh: '算法' },
]

const activeCategory = ref<Category>('feature')

const filteredItems = computed(() =>
  items.filter((item) => item.category === activeCategory.value)
)

function getCategoryLabel(cat: { label: string; labelZh: string }): string {
  return isZh.value ? cat.labelZh : cat.label
}

function getItemLink(item: ShowcaseItem): string {
  return withBase(langPrefix.value + item.link)
}

const sectionTitle = computed(() =>
  isZh.value ? '生态全景' : 'Ecosystem at a Glance'
)

const categoryColors: Record<Category, string> = {
  feature: '#e8384a',
  task: '#bda2ff',
  model: '#38e8c4',
  algorithm: '#ff69b4',
}
</script>

<template>
  <section v-if="isHome" class="ecosystem-section">
    <h2 class="ecosystem-title">{{ sectionTitle }}</h2>

    <!-- Category tabs -->
    <div class="ecosystem-tabs">
      <button
        v-for="cat in categories"
        :key="cat.key"
        class="ecosystem-tab"
        :class="{ 'ecosystem-tab--active': activeCategory === cat.key }"
        :style="
          activeCategory === cat.key
            ? { '--tab-color': categoryColors[cat.key] }
            : {}
        "
        @click="activeCategory = cat.key"
      >
        {{ getCategoryLabel(cat) }}
      </button>
    </div>

    <!-- Cards grid -->
    <div class="ecosystem-grid">
      <component
        :is="item.link ? 'a' : 'div'"
        v-for="item in filteredItems"
        :key="item.label"
        :href="item.link ? getItemLink(item) : undefined"
        class="ecosystem-card"
        :class="{ 'ecosystem-card--wip': !item.link }"
        :style="{ '--card-color': categoryColors[item.category] }"
      >
        <div class="card-header">
          <span class="card-icon">{{ item.icon }}</span>
          <h3 class="card-label">{{ isZh ? item.labelZh : item.label }}</h3>
          <span v-if="!item.link" class="card-badge">Coming Soon</span>
        </div>
        <p class="card-desc">{{ isZh ? item.descZh : item.desc }}</p>
        <div class="card-tags">
          <span v-for="tag in item.tags" :key="tag" class="card-tag">
            {{ tag }}
          </span>
        </div>
        <span v-if="item.link" class="card-arrow">→</span>
      </component>
    </div>

    <!-- Scrolling marquee below -->
    <div class="marquee-track">
      <div class="marquee-content">
        <template v-for="copy in 2" :key="copy">
          <span
            v-for="(item, i) in items"
            :key="copy + '-' + i"
            class="marquee-chip"
            :style="{ '--chip-color': categoryColors[item.category] }"
            :aria-hidden="copy === 2 ? 'true' : undefined"
          >
            {{ item.icon }} {{ isZh ? item.labelZh : item.label }}
          </span>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.ecosystem-section {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 80px 24px 96px;
}

.ecosystem-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: #f0edf1;
  text-align: center;
  margin-bottom: 48px;
  letter-spacing: -0.02em;
}

@media (min-width: 640px) {
  .ecosystem-title {
    font-size: 2.4rem;
  }
}

/* ── Tabs ── */
.ecosystem-tabs {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 36px;
  flex-wrap: wrap;
}

.ecosystem-tab {
  padding: 10px 28px;
  border-radius: 8px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  color: #acaaae;
  background: rgba(37, 37, 42, 0.5);
  border: 1px solid rgba(72, 71, 75, 0.2);
  cursor: pointer;
  transition: all 0.25s ease;
}

.ecosystem-tab:hover {
  color: #f0edf1;
  background: rgba(37, 37, 42, 0.8);
}

.ecosystem-tab--active {
  color: #f0edf1;
  background: rgba(37, 37, 42, 0.9);
  border-color: var(--tab-color);
  box-shadow: 0 0 16px color-mix(in srgb, var(--tab-color) 20%, transparent);
}

/* ── Cards Grid ── */
.ecosystem-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  margin-bottom: 48px;
}

@media (min-width: 640px) {
  .ecosystem-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 960px) {
  .ecosystem-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.ecosystem-card {
  position: relative;
  display: flex;
  flex-direction: column;
  padding: 24px;
  border-radius: 12px;
  background: rgba(19, 19, 22, 0.7);
  border: 1px solid rgba(72, 71, 75, 0.15);
  text-decoration: none;
  transition: all 0.25s ease;
  overflow: hidden;
}

.ecosystem-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--card-color);
  opacity: 0;
  transition: opacity 0.25s ease;
}

.ecosystem-card:hover {
  border-color: color-mix(in srgb, var(--card-color) 40%, transparent);
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.ecosystem-card:hover::before {
  opacity: 1;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.card-icon {
  font-size: 1.1rem;
  color: var(--card-color);
  line-height: 1;
}

.card-label {
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: #f0edf1;
  margin: 0;
}

.card-desc {
  font-family: "Manrope", sans-serif;
  font-size: 0.85rem;
  color: #acaaae;
  line-height: 1.6;
  margin: 0 0 16px;
  flex: 1;
}

.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.card-tag {
  padding: 3px 10px;
  border-radius: 12px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: color-mix(in srgb, var(--card-color) 80%, #f0edf1);
  background: color-mix(in srgb, var(--card-color) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--card-color) 20%, transparent);
}

.card-arrow {
  position: absolute;
  top: 24px;
  right: 24px;
  font-size: 1rem;
  color: #48474b;
  transition: all 0.25s ease;
}

.ecosystem-card:hover .card-arrow {
  color: var(--card-color);
  transform: translateX(3px);
}

/* ── WIP Card ── */
.ecosystem-card--wip {
  opacity: 0.55;
  cursor: default;
}

.ecosystem-card--wip:hover {
  transform: none;
  box-shadow: none;
  border-color: rgba(72, 71, 75, 0.15);
}

.ecosystem-card--wip:hover::before {
  opacity: 0;
}

.card-badge {
  margin-left: auto;
  padding: 2px 10px;
  border-radius: 10px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #e8c838;
  background: rgba(232, 200, 56, 0.12);
  border: 1px solid rgba(232, 200, 56, 0.25);
}

/* ── Marquee ── */
.marquee-track {
  overflow: hidden;
  -webkit-mask-image: linear-gradient(
    to right,
    transparent 0%,
    black 6%,
    black 94%,
    transparent 100%
  );
  mask-image: linear-gradient(
    to right,
    transparent 0%,
    black 6%,
    black 94%,
    transparent 100%
  );
}

.marquee-content {
  display: flex;
  gap: 10px;
  width: max-content;
  animation: marquee-scroll 50s linear infinite;
}

.marquee-content:hover {
  animation-play-state: paused;
}

@keyframes marquee-scroll {
  0% {
    transform: translateX(0);
  }
  100% {
    transform: translateX(-50%);
  }
}

.marquee-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  border-radius: 16px;
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.75rem;
  font-weight: 600;
  white-space: nowrap;
  color: color-mix(in srgb, var(--chip-color) 70%, #f0edf1);
  background: color-mix(in srgb, var(--chip-color) 6%, transparent);
  border: 1px solid color-mix(in srgb, var(--chip-color) 15%, transparent);
}
</style>
