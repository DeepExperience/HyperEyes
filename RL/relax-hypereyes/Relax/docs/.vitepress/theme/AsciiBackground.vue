<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'

const canvasRef = ref<HTMLCanvasElement | null>(null)
const isHome = ref(false)
const route = useRoute()

let animationId: number | null = null

// --- Contour-line / Plasma wave configuration ---
const FONT_SIZE = 14
const WAVE_SPEED = 0.0005 // Ultra slow, barely perceptible drift
const NOISE_SCALE = 0.015 // Controls contour ring size

const BAND_COUNT = 14 // Number of contour bands
const BLANK_RATIO = 0.0 // 0% blank — continuous, no gaps between bands

// Characters for different contour levels (valley → peak)
const CHARS = ['&', '%', '@', '#', '0', 'R', 'E', 'L', 'A', 'X']


// --- Wave displacement (cloth-like ripple from vue2) ---
const WAVE_DISPLACEMENT_SPEED = 0.0125
const WAVE_AMP = 28 // Pixel amplitude of wave displacement

// --- Mouse interaction physics ---
const MOUSE_RADIUS = 130
const REPULSION_FORCE = 9
const SPRING_STRENGTH = 0.08
const FRICTION = 0.80

interface Particle {
  baseX: number
  baseY: number
  x: number
  y: number
  vx: number
  vy: number
  col: number
  row: number
  char: string
}

function checkIsHome() {
  const path = route.path
  const p = path.replace(/index\.html$/, '').replace(/\/+$/, '')
  isHome.value = p === '' || p.endsWith('/en') || p.endsWith('/zh')
}

onMounted(() => {
  checkIsHome()

  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  let particles: Particle[] = []
  let cols = 0
  let rows = 0
  let time = 0
  let waveTime = 0 // Separate clock for wave displacement (faster)

  // Mouse state
  const mouse = { x: -1000, y: -1000, isActive: false }

  function onMouseMove(e: MouseEvent) {
    mouse.x = e.clientX
    mouse.y = e.clientY
    mouse.isActive = true
  }

  function onMouseOut() {
    mouse.isActive = false
  }

  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseout', onMouseOut)

  function initGrid() {
    canvas!.width = window.innerWidth
    canvas!.height = window.innerHeight
    cols = Math.ceil(canvas!.width / FONT_SIZE)
    rows = Math.ceil(canvas!.height / FONT_SIZE)
    ctx!.font = `bold ${FONT_SIZE}px monospace`
    ctx!.textBaseline = 'top'

    particles = []

    // Build a char map with random n×m blocks (n,m ∈ [2,9]) sharing the same char
    const charMap: string[][] = Array.from({ length: rows }, () => new Array(cols).fill(''))
    for (let y = 0; y < rows;) {
      const bh = 2 + Math.floor(Math.random() * 8) // block height 2–9
      for (let x = 0; x < cols;) {
        const bw = 2 + Math.floor(Math.random() * 8) // block width 2–9
        const ch = CHARS[Math.floor(Math.random() * CHARS.length)]
        for (let dy = 0; dy < bh && y + dy < rows; dy++) {
          for (let dx = 0; dx < bw && x + dx < cols; dx++) {
            charMap[y + dy][x + dx] = ch
          }
        }
        x += bw
      }
      y += bh
    }

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        particles.push({
          baseX: x * FONT_SIZE,
          baseY: y * FONT_SIZE,
          x: x * FONT_SIZE,
          y: y * FONT_SIZE,
          vx: 0,
          vy: 0,
          col: x,
          row: y,
          char: charMap[y][x],
        })
      }
    }
  }

  function onResize() {
    initGrid()
    buildDarkGradient()
  }

  window.addEventListener('resize', onResize)
  initGrid()

  // Smoothstep for silky easing
  function smoothstep(x: number): number {
    const t = Math.max(0, Math.min(1, x))
    return t * t * (3 - 2 * t)
  }

  // Reusable gradient object (recreated on resize)
  let darkGradient: CanvasGradient | null = null

  function buildDarkGradient() {
    // Vertical gradient from top (transparent) to bottom (opaque)
    darkGradient = ctx!.createLinearGradient(0, 0, 0, canvas!.height)
  }

  buildDarkGradient()

  function animate() {
    // Clear canvas — transparent so body bg (#141414) shows through
    ctx!.clearRect(0, 0, canvas!.width, canvas!.height)

    time += WAVE_SPEED
    waveTime += WAVE_DISPLACEMENT_SPEED

    // --------------------------------------------------
    // 0. Read scroll progress from CSS variable (set by FeatureShowcase)
    // --------------------------------------------------
    const progressStr = getComputedStyle(document.documentElement)
      .getPropertyValue('--showcase-scroll-progress')
    const rawProgress = parseFloat(progressStr) || 0
    const sp = smoothstep(rawProgress)

    const W = canvas!.width
    const H = canvas!.height

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i]

      // --------------------------------------------------
      // 1a. Wave displacement — multi-directional cloth ripple
      // --------------------------------------------------
      const nx = p.baseX / W // 0..1
      const ny = p.baseY / H // 0..1
      const px = nx * 6.28
      const py = ny * 6.28

      const w1 = Math.sin((-0.866 * px - 0.500 * py) * 1.0 - waveTime * 0.9 + 0.0) * 1.00
      const w2 = Math.sin((-0.342 * px + 0.940 * py) * 1.6 - waveTime * 1.15 + 0.8) * 0.55
      const w3 = Math.sin((-0.707 * px - 0.707 * py) * 4.5 - waveTime * 2.1 + 1.5) * 0.18
      const w4 = Math.sin((-0.866 * px + 0.500 * py) * 8.0 - waveTime * 3.25 + 5.0) * 0.07

      const waveSum = w1 + w2 + w3 + w4

      const waveDx = (-0.866 * w1 - 0.342 * w2 - 0.707 * w3 - 0.866 * w4) * WAVE_AMP
      const waveDy = (-0.500 * w1 + 0.940 * w2 - 0.707 * w3 + 0.500 * w4) * WAVE_AMP

      const anchorX = p.baseX + waveDx
      const anchorY = p.baseY + waveDy

      // --------------------------------------------------
      // 1b. Contour-line / Plasma wave texture (based on original grid position)
      // --------------------------------------------------
      const n1 = Math.sin(p.col * NOISE_SCALE + time)
      const n2 = Math.cos(p.row * NOISE_SCALE - time * 0.8)
      const n3 = Math.sin((p.col + p.row) * NOISE_SCALE * 0.7 + time * 1.2)

      // Normalize to 0.0 ~ 1.0
      const noise = (n1 + n2 + n3 + 3) / 6

      // Contour-line algorithm
      const bandedValue = noise * BAND_COUNT
      const fractionalPart = bandedValue - Math.floor(bandedValue)

      // Blank gap between contour lines
      if (fractionalPart < BLANK_RATIO) {
        continue
      }

      // --------------------------------------------------
      // Edge-to-center brightness gradient within each band
      // --------------------------------------------------
      const bandPos = (fractionalPart - BLANK_RATIO) / (1.0 - BLANK_RATIO)
      // Sine curve instead of triangle wave — much softer edge falloff
      const edgeBrightness = Math.sin(bandPos * Math.PI)

      // NOTE(wuhuan): 渐变或者不渐变调这里
      // Single smoothstep — gentler contrast between bright and dark bands
      const bandGlow = edgeBrightness * edgeBrightness * (3.0 - 2.0 * edgeBrightness)

      // --------------------------------------------------
      // 1c. Wave slope → light shading (cloth folds catch light)
      // --------------------------------------------------
      const slope = Math.max(-1, Math.min(1, waveSum / 1.8))
      const waveLight = 0.5 + slope * 0.42
      const light = 0.3 * waveLight + 0.7 * waveLight * waveLight // mild power curve

      // --------------------------------------------------
      // 2. Mouse interaction physics (repulsion & spring-back)
      // --------------------------------------------------
      if (mouse.isActive) {
        const dx = p.x - mouse.x
        const dy = p.y - mouse.y
        const distSq = dx * dx + dy * dy

        if (distSq < MOUSE_RADIUS * MOUSE_RADIUS) {
          const distance = Math.sqrt(distSq)
          const force = (MOUSE_RADIUS - distance) / MOUSE_RADIUS
          p.vx += (dx / distance) * force * REPULSION_FORCE
          p.vy += (dy / distance) * force * REPULSION_FORCE
        }
      }

      // Spring physics: pull particle back to wave-displaced anchor
      p.vx += (anchorX - p.x) * SPRING_STRENGTH
      p.vy += (anchorY - p.y) * SPRING_STRENGTH

      p.vx *= FRICTION
      p.vy *= FRICTION

      p.x += p.vx
      p.y += p.vy

      // --------------------------------------------------
      // 3. Draw character — combine band glow + wave light for shading
      // --------------------------------------------------
      // Mix contour band glow (50%) with wave slope light (50%)
      const combined = bandGlow * 0.5 + light * 0.5

      const r = Math.floor(110 + combined * 105) // edge 110 → center 215
      const g = Math.floor(15 + combined * 35)   // 15 → 50
      const b = Math.floor(20 + combined * 40)   // 20 → 60

      const baseAlpha = 0.12 + combined * 0.30 // edge 0.12 → center 0.42
      const a = baseAlpha * (1 - sp * 0.55)

      if (a < 0.015) continue

      ctx!.fillStyle = `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`
      ctx!.fillText(p.char, p.x, p.y)
    }

    // --------------------------------------------------
    // 4. Draw dark overlay gradient directly on canvas
    //    Bottom-to-top sweep: dark at bottom, transparent at top.
    //    As scroll progresses, the dark region rises and intensifies.
    // --------------------------------------------------
    if (sp > 0.01) {
      const maxAlpha = 0.45
      const alpha = sp * maxAlpha

      // How far up the dark region covers (0% = none, 100% = full canvas)
      const coverage = sp   // 0..1

      if (darkGradient) {
        // Rebuild gradient stops each frame (cheap, just stop values change)
        darkGradient = ctx!.createLinearGradient(0, 0, 0, canvas!.height)

        // Feather edge: the top portion stays transparent,
        // then smoothly fades into the dark region
        const feather = (1 - coverage) * 0.4 // feather band shrinks as coverage grows
        const clearEnd = Math.max(0, 1 - coverage - feather) // where fully transparent ends
        const darkStart = Math.max(0, 1 - coverage)          // where full dark begins

        darkGradient.addColorStop(0, 'rgba(14, 14, 17, 0)')
        darkGradient.addColorStop(Math.min(clearEnd, 0.99), 'rgba(14, 14, 17, 0)')
        darkGradient.addColorStop(Math.min(darkStart + 0.001, 1), `rgba(14, 14, 17, ${(alpha * 0.35).toFixed(3)})`)
        darkGradient.addColorStop(1, `rgba(14, 14, 17, ${(alpha * 0.8).toFixed(3)})`)

        ctx!.fillStyle = darkGradient
        ctx!.fillRect(0, 0, canvas!.width, canvas!.height)
      }
    }

    animationId = requestAnimationFrame(animate)
  }

  animate()

  onUnmounted(() => {
    if (animationId) cancelAnimationFrame(animationId)
    window.removeEventListener('resize', onResize)
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mouseout', onMouseOut)
  })
})

watch(
  () => route.path,
  () => {
    checkIsHome()
  },
)
</script>

<template>
  <canvas v-show="isHome" ref="canvasRef" class="ascii-bg-canvas" />
</template>

<style scoped>
.ascii-bg-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
  pointer-events: auto;
  cursor: crosshair;
}
</style>
