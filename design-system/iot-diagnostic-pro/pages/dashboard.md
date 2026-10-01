# Dashboard Page Design Overrides (Futuristic Luxury Tech)

> **Inherits from:** `design-system/iot-diagnostic-pro/MASTER.md`
> **Tone:** Futuristic, Professional, Trendy, Luxurious (Tương lai - Chuyên nghiệp - Thời thượng - Sang trọng)
> **Aesthetic:** Cyber-HUD Glassmorphism 2.0 + Deep Obsidian Palette + Luminous Neon Telemetry

## Overrides & Enhancements

### 1. Color Palette (Cyber Luxury)
- **Canvas Base:** `#080C15` with subtle multi-source radial aura:
  - Top-left: `radial-gradient(ellipse at 15% 10%, rgba(0, 240, 255, 0.07) 0%, transparent 60%)`
  - Top-right: `radial-gradient(ellipse at 85% 15%, rgba(99, 102, 241, 0.06) 0%, transparent 60%)`
- **Surfaces (Glassmorphism 2.0):**
  - Card background: `rgba(15, 23, 42, 0.72)`
  - Backdrop filter: `blur(20px)`
  - Micro-borders: `1px solid rgba(255, 255, 255, 0.08)`
  - Outer glow: `box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08)`
- **Luminous Neon Accents:**
  - Electric Cyan: `#00F0FF` (Live status, telemetry values, primary CTA)
  - Emerald Cyber: `#10B981` (Normal state, optimal IAQ)
  - Cyber Coral: `#F43F5E` (Ventilation issue, critical alert)
  - Amber Gold: `#F59E0B` (Thermal warning, load alert)
  - Royal Indigo: `#6366F1` (AI Neural engine, Spark cluster)

### 2. Typography
- **Headings & Badges:** `Space Grotesk`, sans-serif (Geometric, high-tech, luxury)
- **Body & Captions:** `DM Sans` / `Fira Sans`, sans-serif (Ultra-clean legibility)
- **Telemetry & Numbers:** `JetBrains Mono` / `Fira Code`, monospace (Crisp sensor data)

### 3. Interactive Components
- **Primary CTA Button:**
  - Gradient: `linear-gradient(135deg, #00F0FF 0%, #0284C7 50%, #6366F1 100%)`
  - Glow: `box-shadow: 0 0 20px rgba(0, 240, 255, 0.35)`
  - Active: Scale `0.98`, high-contrast text
- **Telemetry HUD Cards:** Bento-grid layout with icon aura, delta indicators, and status badges.
- **Pulsing Radar Indicator:** CSS `@keyframes radar-pulse` for real-time live sensor streaming.
