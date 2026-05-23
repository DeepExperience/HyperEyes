<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'

const preRef = ref<HTMLPreElement | null>(null)
const isHome = ref(false)
const route = useRoute()

let timer: number | null = null

function checkIsHome() {
  const path = route.path
  const p = path.replace(/index\.html$/, '').replace(/\/+$/, '')
  isHome.value = p === '' || p.endsWith('/en') || p.endsWith('/zh')
}

// --- ASCII firework engine (pure text grid) ---

interface Spark {
  x: number
  y: number
  vx: number
  vy: number
  life: number
  char: string
  color: string
}

interface Rocket {
  x: number
  y: number
  vy: number
  targetY: number
  fuse: number
}

const COLS = 120
const ROWS = 32

const BURST_CHARS = ['.', '*', '#', '@', 'o', '+', '~', '^', ':', ';', '!', '=', 'x', '`', "'"]
const TRAIL_CHARS = ['|', ':', '.']
const ROCKET_CHAR = '^'


const COLORS = [
  '#e8384a', '#d4323e', '#ef6471',  // xiaohongshu red (brand)
  '#ffd700', '#ffb347', '#ffe066',  // gold
  '#00ffcc', '#33ffdd', '#66ffee',  // cyan
  '#bda2ff', '#cc44ff', '#dd66ff',  // purple / lavender
  '#ffffff', '#cccccc',             // white / silver
]

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

onMounted(() => {
  checkIsHome()

  const pre = preRef.value
  if (!pre) return

  // The grid holds { char, color, alpha } per cell
  type Cell = { char: string; color: string; alpha: number }
  const grid: Cell[][] = []
  for (let r = 0; r < ROWS; r++) {
    grid[r] = []
    for (let c = 0; c < COLS; c++) {
      grid[r][c] = { char: ' ', color: '#333', alpha: 0 }
    }
  }

  const sparks: Spark[] = []
  const rockets: Rocket[] = []
  let frame = 0

  function launchRocket() {
    const x = 10 + Math.floor(Math.random() * (COLS - 20))
    rockets.push({
      x,
      y: ROWS - 1,
      vy: -(1.2 + Math.random() * 0.8),
      targetY: 3 + Math.floor(Math.random() * (ROWS * 0.4)),
      fuse: 0,
    })
  }

  function explode(rx: number, ry: number) {
    const color = randomFrom(COLORS)
    const count = 20 + Math.floor(Math.random() * 30)
    const style = Math.random()

    for (let i = 0; i < count; i++) {
      let angle: number, speed: number

      if (style < 0.4) {
        // circular
        angle = (Math.PI * 2 * i) / count
        speed = 1.5 + Math.random() * 1.5
      } else if (style < 0.7) {
        // random scatter
        angle = Math.random() * Math.PI * 2
        speed = 0.5 + Math.random() * 2.5
      } else {
        // starburst with double ring
        angle = (Math.PI * 2 * i) / count
        speed = i % 2 === 0 ? 1 + Math.random() : 2.5 + Math.random()
      }

      // ASCII coords: x moves faster visually because chars are taller than wide
      const vx = Math.cos(angle) * speed * 1.8
      const vy = Math.sin(angle) * speed

      sparks.push({
        x: rx,
        y: ry,
        vx,
        vy,
        life: 12 + Math.floor(Math.random() * 16),
        char: randomFrom(BURST_CHARS),
        color: Math.random() < 0.3 ? randomFrom(COLORS) : color,
      })
    }
  }

  function step() {
    // Fade the grid
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = grid[r][c]
        cell.alpha *= 0.55
        if (cell.alpha < 0.05) {
          cell.char = ' '
          cell.alpha = 0
        }
      }
    }

    // Auto-launch
    if (frame % 18 === 0 || (frame % 12 === 0 && Math.random() < 0.3)) {
      launchRocket()
    }

    // Update rockets
    for (let i = rockets.length - 1; i >= 0; i--) {
      const r = rockets[i]
      r.y += r.vy
      r.fuse++

      const iy = Math.round(r.y)
      const ix = Math.round(r.x)

      // Draw rocket trail
      if (iy >= 0 && iy < ROWS && ix >= 0 && ix < COLS) {
        grid[iy][ix] = { char: ROCKET_CHAR, color: '#ffcc00', alpha: 1 }
      }
      // Trail behind
      const ty = iy + 1
      if (ty >= 0 && ty < ROWS && ix >= 0 && ix < COLS) {
        grid[ty][ix] = { char: randomFrom(TRAIL_CHARS), color: '#ff8844', alpha: 0.6 }
      }
      const ty2 = iy + 2
      if (ty2 >= 0 && ty2 < ROWS && ix >= 0 && ix < COLS) {
        grid[ty2][ix] = { char: '.', color: '#664422', alpha: 0.3 }
      }

      if (r.y <= r.targetY) {
        explode(r.x, r.y)
        rockets.splice(i, 1)
      }
    }

    // Update sparks
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i]
      s.x += s.vx
      s.y += s.vy
      s.vy += 0.07 // gravity
      s.vx *= 0.96
      s.life--

      const iy = Math.round(s.y)
      const ix = Math.round(s.x)

      if (iy >= 0 && iy < ROWS && ix >= 0 && ix < COLS && s.life > 0) {
        const alpha = Math.min(1, s.life / 8)
        // Char changes as spark ages
        let ch = s.char
        if (s.life < 4) ch = randomFrom(['.', '·', "'", '`'])
        else if (s.life < 8 && Math.random() < 0.3) ch = randomFrom(['*', '+', '.'])

        grid[iy][ix] = { char: ch, color: s.color, alpha }
      }

      if (s.life <= 0) {
        sparks.splice(i, 1)
      }
    }

    // Render grid to <pre> as colored HTML
    let html = ''
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = grid[r][c]
        if (cell.char === ' ' || cell.alpha < 0.05) {
          html += ' '
        } else {
          const opacity = Math.max(0.15, Math.min(1, cell.alpha))
          html += `<span style="color:${cell.color};opacity:${opacity.toFixed(2)}">${cell.char}</span>`
        }
      }
      html += '\n'
    }

    pre!.innerHTML = html
    frame++
  }

  // Run at ~15 fps for a nice retro feel
  timer = window.setInterval(step, 66)

  onUnmounted(() => {
    if (timer) clearInterval(timer)
  })
})

watch(() => route.path, () => {
  checkIsHome()
})
</script>

<template>
  <div v-show="isHome" class="ascii-fireworks-wrapper">
    <div class="relax-bg-text" aria-hidden="true">R E L A X</div>
    <pre ref="preRef" class="ascii-fireworks-pre"></pre>
  </div>
</template>

<style scoped>
.ascii-fireworks-wrapper {
  position: relative;
  width: 100%;
  overflow: hidden;
  z-index: 1;
  background: #0e0e11;
  border-top: 1px solid rgba(232, 56, 74, 0.08);
}

.relax-bg-text {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
  font-weight: 900;
  font-size: 18vw;
  letter-spacing: 0.15em;
  color: var(--vp-c-brand-1, #e8384a);
  opacity: 0.8;
  user-select: none;
  pointer-events: none;
  line-height: 1;
  white-space: nowrap;
}

.ascii-fireworks-pre {
  position: relative;
  display: block;
  margin: 0 auto;
  padding: 16px 0;
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.15;
  text-align: center;
  color: #222;
  background: transparent;
  white-space: pre;
  overflow: hidden;
  user-select: none;
  pointer-events: none;
}
</style>
