<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const mounted = ref(false)
const showTitle = ref(false)
const showSubtitle = ref(false)

let timers: number[] = []

function sleep(ms: number) {
  return new Promise<void>(resolve => {
    const id = window.setTimeout(resolve, ms)
    timers.push(id)
  })
}

onMounted(async () => {
  mounted.value = true

  // Staggered entrance animation
  await sleep(100)
  showTitle.value = true

  await sleep(400)
  showSubtitle.value = true
})

onUnmounted(() => {
  timers.forEach(id => clearTimeout(id))
  timers = []
})
</script>

<template>
  <div class="hero-title-section" v-if="mounted">
    <!-- Ambient glow blob -->
    <div class="hero-glow-blob" />

    <!-- Large typographic RELAX title -->
    <transition name="hero-fade">
      <h1 v-if="showTitle" class="hero-title">RELAX</h1>
    </transition>

    <!-- Subtitle -->
    <transition name="hero-fade">
      <p v-if="showSubtitle" class="hero-subtitle">
        Towards <span class="hero-emphasis">Async</span>,
        <span class="hero-emphasis">Omni-Modal</span>
        RL at <span class="hero-emphasis">Scale</span>, Just <span class="hero-brand">Relax</span>.
      </p>
    </transition>

  </div>
</template>

<style scoped>
.hero-title-section {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  position: relative;
  padding-top: 32px;
  padding-bottom: 24px;
}

/* Ambient glow blob behind title */
.hero-glow-blob {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 800px;
  height: 800px;
  background: rgba(232, 56, 74, 0.15);
  border-radius: 50%;
  filter: blur(140px);
  pointer-events: none;
  z-index: 0;
}

/* Large typographic title — matching relax-web hero */
.hero-title {
  position: relative;
  z-index: 1;
  margin: 0;
  padding: 0;
  font-family: "Space Grotesk", sans-serif;
  font-size: 6rem;
  font-weight: 700;
  letter-spacing: -4px;
  line-height: 1;
  color: #e8384a;
  text-shadow: 0 0 20px rgba(232, 56, 74, 0.4);
  margin-bottom: 32px;
}

@media (min-width: 640px) {
  .hero-title {
    font-size: 8rem;
    letter-spacing: -6px;
  }
}

@media (min-width: 960px) {
  .hero-title {
    font-size: 10rem;
    letter-spacing: -8px;
  }
}

/* Subtitle */
.hero-subtitle {
  position: relative;
  z-index: 1;
  font-family: "Manrope", sans-serif;
  font-size: 1.125rem;
  line-height: 1.6;
  color: #acaaae;
  max-width: 600px;
  margin: 0 auto 20px;
}

@media (min-width: 640px) {
  .hero-subtitle {
    font-size: 1.25rem;
  }
}

@media (min-width: 960px) {
  .hero-subtitle {
    font-size: 1.375rem;
  }
}

.hero-emphasis {
  color: #f0edf1;
  font-weight: 600;
}

.hero-brand {
  color: #e8384a;
  font-weight: 700;
  text-shadow: 0 0 12px rgba(232, 56, 74, 0.35);
}

/* Entrance transitions */
.hero-fade-enter-active {
  transition: opacity 0.8s ease, transform 0.8s ease;
}

.hero-fade-enter-from {
  opacity: 0;
  transform: translateY(20px);
}
</style>
