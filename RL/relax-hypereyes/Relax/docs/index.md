---
layout: page
title: Relax
---

<script setup>
import { onMounted } from 'vue'
import { useData } from 'vitepress'

onMounted(() => {
  const { site } = useData()
  const base = site.value.base || '/'
  window.location.replace(base + 'en/')
})
</script>

<div style="display:flex;align-items:center;justify-content:center;min-height:50vh;color:#888;">
  Redirecting to English...
</div>
