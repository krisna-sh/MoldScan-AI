"""
╔══════════════════════════════════════════════════════════════╗
║   MOLD DETECTION WEBSITE — Flask Backend                     ║
║   Run: python app.py                                         ║
║   Open: http://localhost:5000                                ║
╚══════════════════════════════════════════════════════════════╝

Install requirements first:
    pip install flask tensorflow pillow numpy
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import tensorflow as tf
import io, base64
import sys

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\DELL\Downloads\ResNet50V2_BiLSTM_FINAL\ResNet50V2_BiLSTM_FINAL.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['Stachybotrys', 'aspergillus', 'cladosporium', 'penicillium']

# ── MOLD INFO DATABASE ────────────────────────────────────────────────
MOLD_INFO = {
    'Stachybotrys': {
        'common_name': 'Black Mold (Stachybotrys)',
        'danger_level': 'HIGH',
        'danger_color': '#ff4444',
        'icon': '☠️',
        'description': (
            'Stachybotrys chartarum, commonly known as "black mold" or "toxic black mold", '
            'is one of the most dangerous household molds. It thrives in areas with high '
            'moisture and produces mycotoxins that can cause serious health problems including '
            'respiratory issues, chronic fatigue, headaches, and in severe cases neurological damage. '
            'It typically appears as dark greenish-black slimy patches and has a musty odor.'
        ),
        'health_risks': ['Respiratory problems', 'Chronic fatigue', 'Headaches & migraines',
                         'Skin irritation', 'Neurological damage (long-term exposure)'],
        'removal_steps': [
            '🚨 Evacuate children, elderly, and pets from the area immediately.',
            '🥽 Wear full protective gear: N95 mask, goggles, gloves, and disposable coveralls.',
            '🔒 Seal off the affected area with plastic sheeting to prevent spore spread.',
            '💧 Lightly mist the mold with water to prevent airborne spores during removal.',
            '🧴 Apply a commercial mold killer (bleach solution: 1 cup bleach per gallon of water) and let sit 15 minutes.',
            '🪛 Scrub thoroughly with a stiff brush, then wipe clean with disposable rags.',
            '🗑️ Seal all contaminated materials in heavy-duty plastic bags before disposal.',
            '💨 Use a HEPA air purifier to clean the air after removal.',
            '⚠️ For large infestations (>10 sq ft), contact a professional mold remediation service.',
        ],
        'prevention': 'Fix all water leaks immediately. Keep indoor humidity below 50%. Use dehumidifiers in basements and bathrooms.',
    },
    'aspergillus': {
        'common_name': 'Aspergillus Mold',
        'danger_level': 'MODERATE',
        'danger_color': '#ff8c00',
        'icon': '⚠️',
        'description': (
            'Aspergillus is one of the most common molds found in homes and buildings. '
            'While many species are harmless to healthy individuals, it can cause serious '
            'lung infections (aspergillosis) in people with weakened immune systems, asthma, '
            'or other respiratory conditions. It appears in various colors including green, '
            'yellow, brown, and black, and commonly grows on walls, food, and HVAC systems.'
        ),
        'health_risks': ['Allergic reactions', 'Asthma attacks', 'Lung infections (aspergillosis)',
                         'Sinus infections', 'Serious risk for immunocompromised individuals'],
        'removal_steps': [
            '😷 Wear an N95 respirator, rubber gloves, and safety goggles.',
            '🪟 Open windows for ventilation before starting.',
            '🧴 Mix 1 cup of white vinegar or a commercial antifungal cleaner in a spray bottle.',
            '💦 Spray the affected surface and let soak for 1 hour.',
            '🖌️ Scrub with a stiff brush until fully removed.',
            '🌀 Wipe area dry completely — moisture encourages regrowth.',
            '🔄 Apply a mold-resistant paint or sealant after cleaning.',
            '🗑️ Dispose of all cleaning materials in sealed bags.',
        ],
        'prevention': 'Ensure proper ventilation in bathrooms and kitchens. Clean and replace HVAC filters regularly. Address water leaks promptly.',
    },
    'cladosporium': {
        'common_name': 'Cladosporium Mold',
        'danger_level': 'LOW-MODERATE',
        'danger_color': '#ffd700',
        'icon': '⚡',
        'description': (
            'Cladosporium is an extremely common mold found both indoors and outdoors. '
            'It typically appears as olive-green, brown, or black patches and has a suede-like '
            'or powdery texture. While generally less toxic than other molds, it is one of the '
            'leading causes of mold-related allergies and can trigger asthma attacks. '
            'It commonly grows on fabrics, wood, and damp surfaces.'
        ),
        'health_risks': ['Nasal congestion & sneezing', 'Eye irritation', 'Skin rashes',
                         'Asthma and allergy flare-ups', 'Respiratory discomfort'],
        'removal_steps': [
            '🧤 Put on rubber gloves and an N95 mask before starting.',
            '🧹 For small areas, scrub with undiluted white vinegar using a stiff brush.',
            '🧴 For porous surfaces (wood, fabric), use a borax solution (1 cup borax per gallon of water).',
            '⏱️ Let the solution sit for 10-15 minutes before scrubbing.',
            '🌀 Wipe clean and dry the area thoroughly.',
            '☀️ Expose treated areas to sunlight if possible — UV light kills mold spores.',
            '♻️ For fabrics: wash with hot water and add 1 cup of white vinegar to the cycle.',
        ],
        'prevention': 'Reduce indoor humidity with dehumidifiers. Ensure carpets and fabrics stay dry. Improve ventilation in closets and storage areas.',
    },
    'penicillium': {
        'common_name': 'Penicillium Mold',
        'danger_level': 'LOW-MODERATE',
        'danger_color': '#4caf50',
        'icon': '🔵',
        'description': (
            'Penicillium is the blue-green fuzzy mold most people recognize from spoiled food. '
            'Indoors, it spreads rapidly and can colonize walls, carpets, insulation, and HVAC '
            'ducts. While famous for producing the antibiotic penicillin, many Penicillium species '
            'produce mycotoxins harmful to humans. It is a major cause of allergic reactions and '
            'can worsen asthma. It has a characteristic musty, earthy smell.'
        ),
        'health_risks': ['Allergic rhinitis', 'Chronic sinus infections', 'Asthma aggravation',
                         'Hypersensitivity pneumonitis', 'Penicillin-like reactions in sensitive individuals'],
        'removal_steps': [
            '🥽 Wear gloves, safety glasses, and an N95 mask.',
            '🚿 For non-porous surfaces: spray with hydrogen peroxide (3%) and leave for 10 minutes.',
            '🧽 Scrub the area with a brush, then wipe clean with a damp cloth.',
            '🧴 Follow up with a tea tree oil solution (1 tsp per cup of water) as a natural fungicide.',
            '💧 Dry the area completely using fans or a dehumidifier.',
            '🔍 Check nearby areas — Penicillium spreads quickly through HVAC systems.',
            '🔄 Replace any heavily contaminated porous materials (carpet, drywall, insulation).',
        ],
        'prevention': 'Store food properly and discard spoiled items quickly. Keep HVAC and duct systems clean. Address water intrusion within 24-48 hours to prevent colonization.',
    },
}

# ── LOAD MODEL ────────────────────────────────────────────────────────
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model loaded!")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    model = None

# ══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════
HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MoldScan AI — Mold Detection System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0d0f;
  --surface: #111518;
  --surface2: #181d21;
  --border: #1e2428;
  --accent: #00ff87;
  --accent2: #00c9ff;
  --text: #e8eaed;
  --muted: #6b7680;
  --danger: #ff4444;
  --warn: #ff8c00;
  --ok: #00cc6a;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── NOISE TEXTURE OVERLAY ── */
body::before {
  content:'';
  position:fixed; inset:0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events:none; z-index:0; opacity:0.4;
}

/* ── HERO ── */
.hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  overflow: hidden;
}

.hero-bg {
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,255,135,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,201,255,0.06) 0%, transparent 50%),
    radial-gradient(ellipse 40% 50% at 10% 60%, rgba(0,255,135,0.04) 0%, transparent 50%);
  animation: bgPulse 8s ease-in-out infinite alternate;
}

@keyframes bgPulse {
  from { opacity: 0.6; }
  to   { opacity: 1; }
}

/* ── GRID LINES ── */
.grid-lines {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(0,255,135,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,135,0.03) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
}

.hero-content {
  position: relative;
  z-index: 1;
  text-align: center;
  max-width: 800px;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(0,255,135,0.08);
  border: 1px solid rgba(0,255,135,0.2);
  color: var(--accent);
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 6px 16px;
  border-radius: 100px;
  margin-bottom: 32px;
  animation: fadeDown 0.6s ease both;
}

.badge::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
  animation: blink 1.4s ease infinite;
}

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

h1 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(4rem, 12vw, 9rem);
  line-height: 0.9;
  letter-spacing: 0.02em;
  background: linear-gradient(135deg, #ffffff 30%, var(--accent) 70%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: fadeDown 0.7s 0.1s ease both;
}

.subtitle {
  font-size: 1.1rem;
  color: var(--muted);
  margin-top: 20px;
  line-height: 1.7;
  font-weight: 300;
  max-width: 520px;
  margin-inline: auto;
  animation: fadeDown 0.7s 0.2s ease both;
}

.hero-stats {
  display: flex;
  gap: 40px;
  justify-content: center;
  margin-top: 48px;
  animation: fadeDown 0.7s 0.3s ease both;
}

.stat {
  text-align: center;
}
.stat-num {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.2rem;
  color: var(--accent);
  line-height: 1;
}
.stat-label {
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 4px;
}

.scroll-hint {
  position: absolute;
  bottom: 32px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  animation: fadeDown 1s 0.8s ease both;
}
.scroll-arrow {
  width: 24px; height: 24px;
  border-right: 2px solid var(--muted);
  border-bottom: 2px solid var(--muted);
  transform: rotate(45deg);
  animation: bounce 1.5s ease infinite;
}
@keyframes bounce {
  0%,100%{ transform: rotate(45deg) translateY(0); }
  50%    { transform: rotate(45deg) translateY(5px); }
}

/* ── SCANNER SECTION ── */
.scanner-section {
  position: relative;
  z-index: 1;
  padding: 80px 20px;
  max-width: 900px;
  margin: 0 auto;
}

.section-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--accent);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 12px;
}

.section-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(2.5rem, 6vw, 4.5rem);
  line-height: 1;
  margin-bottom: 16px;
}

.section-desc {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.7;
  max-width: 500px;
  margin-bottom: 40px;
}

/* ── UPLOAD ZONE ── */
.upload-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 40px 80px rgba(0,0,0,0.4);
}

.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 16px;
  margin: 24px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.upload-zone::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 50%, rgba(0,255,135,0.05), transparent 70%);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s;
}

.upload-zone:hover::before,
.upload-zone.drag-over::before { opacity: 1; }

.upload-zone:hover,
.upload-zone.drag-over {
  border-color: var(--accent);
  transform: translateY(-2px);
}

.upload-icon {
  font-size: 3.5rem;
  margin-bottom: 16px;
  display: block;
  filter: grayscale(0.3);
}

.upload-title {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 8px;
}

.upload-sub {
  font-size: 0.85rem;
  color: var(--muted);
}

.upload-sub label {
  color: var(--accent);
  cursor: pointer;
  text-decoration: underline;
  text-decoration-style: dotted;
}

#fileInput { display: none; }

/* preview */
.preview-container {
  display: none;
  padding: 24px;
  gap: 24px;
  align-items: flex-start;
}

.preview-img-wrap {
  position: relative;
  flex-shrink: 0;
}

.preview-img {
  width: 220px;
  height: 220px;
  object-fit: cover;
  border-radius: 12px;
  border: 1px solid var(--border);
  display: block;
}

.preview-overlay {
  position: absolute; inset: 0;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(0,255,135,0.15), transparent);
  pointer-events: none;
}

.preview-info {
  flex: 1;
}

.preview-name {
  font-family: 'DM Mono', monospace;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 8px;
  word-break: break-all;
}

.preview-size {
  font-size: 0.8rem;
  color: var(--muted);
}

.change-btn {
  margin-top: 16px;
  background: none;
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 0.8rem;
  cursor: pointer;
  font-family: 'DM Sans', sans-serif;
  transition: all 0.2s;
}
.change-btn:hover { border-color: var(--accent); color: var(--accent); }

/* analyze button */
.btn-analyze {
  display: block;
  width: calc(100% - 48px);
  margin: 0 24px 24px;
  padding: 18px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #000;
  border: none;
  border-radius: 12px;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.4rem;
  letter-spacing: 0.08em;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 8px 24px rgba(0,255,135,0.25);
}

.btn-analyze:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 12px 32px rgba(0,255,135,0.35);
}

.btn-analyze:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn-analyze .btn-shimmer {
  position: absolute;
  top: 0; left: -100%;
  width: 60%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  animation: shimmer 2.5s ease infinite;
}

@keyframes shimmer {
  0%  { left: -100%; }
  100%{ left: 200%; }
}

/* ── LOADER ── */
.loader {
  display: none;
  padding: 48px 24px;
  text-align: center;
}

.scan-animation {
  width: 120px; height: 120px;
  margin: 0 auto 24px;
  position: relative;
}

.scan-ring {
  position: absolute; inset: 0;
  border-radius: 50%;
  border: 2px solid transparent;
}
.scan-ring:nth-child(1) {
  border-top-color: var(--accent);
  animation: spin 1s linear infinite;
}
.scan-ring:nth-child(2) {
  inset: 12px;
  border-right-color: var(--accent2);
  animation: spin 1.4s linear infinite reverse;
}
.scan-ring:nth-child(3) {
  inset: 24px;
  border-bottom-color: var(--accent);
  animation: spin 0.8s linear infinite;
}

.scan-dot {
  position: absolute; inset: 44px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 20px var(--accent);
  animation: pulse 1s ease infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.2)} }

.loader-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.85rem;
  color: var(--accent);
  letter-spacing: 0.1em;
}

.loader-steps {
  margin-top: 16px;
  font-size: 0.8rem;
  color: var(--muted);
}

/* ── RESULTS ── */
.results { display: none; padding: 24px; }

.result-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}

.result-icon {
  font-size: 2.8rem;
  flex-shrink: 0;
}

.result-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  line-height: 1;
}

.danger-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 4px;
}

/* confidence bars */
.confidence-section { margin-bottom: 28px; }

.conf-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 14px;
}

.conf-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.conf-name {
  width: 160px;
  font-size: 0.85rem;
  flex-shrink: 0;
}

.conf-bar-bg {
  flex: 1;
  height: 8px;
  background: var(--border);
  border-radius: 100px;
  overflow: hidden;
}

.conf-bar-fill {
  height: 100%;
  border-radius: 100px;
  background: var(--accent);
  width: 0%;
  transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
}

.conf-bar-fill.predicted {
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  box-shadow: 0 0 8px rgba(0,255,135,0.4);
}

.conf-pct {
  width: 52px;
  text-align: right;
  font-family: 'DM Mono', monospace;
  font-size: 0.85rem;
  font-weight: 500;
}

/* info sections */
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

@media(max-width:600px){ .info-grid{ grid-template-columns:1fr; } }

.info-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
}

.info-card-title {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 12px;
}

.desc-text {
  font-size: 0.88rem;
  line-height: 1.7;
  color: #c8cdd2;
  grid-column: 1 / -1;
}

.risk-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.risk-list li {
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

.risk-list li::before {
  content: '•';
  color: var(--danger);
  font-size: 1.2em;
  flex-shrink: 0;
}

/* removal steps */
.steps-section { margin-bottom: 24px; }

.step-item {
  display: flex;
  gap: 14px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.88rem;
  line-height: 1.6;
  opacity: 0;
  transform: translateY(12px);
  animation: stepIn 0.4s ease forwards;
}

.step-item:last-child { border-bottom: none; }

.step-num {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  color: var(--accent);
  background: rgba(0,255,135,0.08);
  border: 1px solid rgba(0,255,135,0.2);
  width: 26px; height: 26px;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}

@keyframes stepIn {
  to { opacity:1; transform:translateY(0); }
}

.prevention-box {
  background: rgba(0,255,135,0.05);
  border: 1px solid rgba(0,255,135,0.15);
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 0.85rem;
  line-height: 1.6;
  color: #b8f0d0;
}

.prevention-box strong {
  color: var(--accent);
  display: block;
  margin-bottom: 4px;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-family: 'DM Mono', monospace;
}

.scan-again-btn {
  display: block;
  width: 100%;
  padding: 14px;
  background: none;
  border: 1px solid var(--border);
  color: var(--muted);
  border-radius: 10px;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.9rem;
  cursor: pointer;
  margin-top: 8px;
  transition: all 0.2s;
}

.scan-again-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

/* ── HOW IT WORKS ── */
.how-section {
  position: relative; z-index:1;
  padding: 80px 20px;
  max-width: 900px;
  margin: 0 auto;
  border-top: 1px solid var(--border);
}

.steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
  margin-top: 48px;
}

.how-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 24px;
  position: relative;
  overflow: hidden;
  transition: transform 0.3s, border-color 0.3s;
}

.how-card:hover {
  transform: translateY(-4px);
  border-color: rgba(0,255,135,0.2);
}

.how-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.4s ease;
}

.how-card:hover::before { transform: scaleX(1); }

.how-num {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.5rem;
  color: var(--border);
  line-height: 1;
  margin-bottom: 12px;
}

.how-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 8px;
}

.how-desc {
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.6;
}

/* ── FOOTER ── */
footer {
  border-top: 1px solid var(--border);
  padding: 32px 20px;
  text-align: center;
  color: var(--muted);
  font-size: 0.8rem;
  position: relative; z-index:1;
}

/* ── ANIMATIONS ── */
@keyframes fadeDown {
  from { opacity:0; transform: translateY(-20px); }
  to   { opacity:1; transform: translateY(0); }
}

.fade-in {
  opacity: 0;
  transform: translateY(24px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}
.fade-in.visible {
  opacity: 1;
  transform: translateY(0);
}
</style>
</head>
<body>

<!-- ══ HERO ══════════════════════════════════════════════════════ -->
<section class="hero">
  <div class="hero-bg"></div>
  <div class="grid-lines"></div>

  <div class="hero-content">
    <div class="badge">AI-Powered Detection System</div>
    <h1>MOLD<br>SCAN</h1>
    <p class="subtitle">
      Upload a photo of any suspicious mold growth.
      Our deep learning model — trained on thousands of samples —
      identifies the mold species in seconds and tells you exactly how to deal with it.
    </p>

    <div class="hero-stats">
      <div class="stat">
        <div class="stat-num">98.7%</div>
        <div class="stat-label">Accuracy</div>
      </div>
      <div class="stat">
        <div class="stat-num">4</div>
        <div class="stat-label">Mold Species</div>
      </div>
      <div class="stat">
        <div class="stat-num">&lt;2s</div>
        <div class="stat-label">Detection Time</div>
      </div>
    </div>
  </div>

  <div class="scroll-hint">
    <span>Scan now</span>
    <div class="scroll-arrow"></div>
  </div>
</section>

<!-- ══ SCANNER ════════════════════════════════════════════════════ -->
<section class="scanner-section">
  <div class="fade-in">
    <p class="section-label">// Neural Scanner</p>
    <h2 class="section-title">IDENTIFY YOUR MOLD</h2>
    <p class="section-desc">
      Upload a clear, close-up photo of the mold.
      The AI will classify it and provide a complete remediation guide.
    </p>
  </div>

  <div class="upload-card fade-in">
    <!-- Upload zone -->
    <div class="upload-zone" id="dropZone">
      <span class="upload-icon">🔬</span>
      <p class="upload-title">Drop your image here</p>
      <p class="upload-sub">or <label for="fileInput">browse files</label></p>
      <p class="upload-sub" style="margin-top:8px;font-size:0.75rem;">JPG, PNG, WEBP · Max 10MB</p>
      <input type="file" id="fileInput" accept="image/*"/>
    </div>

    <!-- Preview -->
    <div class="preview-container" id="previewContainer">
      <div class="preview-img-wrap">
        <img class="preview-img" id="previewImg" src="" alt="Preview"/>
        <div class="preview-overlay"></div>
      </div>
      <div class="preview-info">
        <p class="preview-name" id="previewName"></p>
        <p class="preview-size" id="previewSize"></p>
        <button class="change-btn" onclick="resetUpload()">↩ Change image</button>
      </div>
    </div>

    <!-- Loader -->
    <div class="loader" id="loader">
      <div class="scan-animation">
        <div class="scan-ring"></div>
        <div class="scan-ring"></div>
        <div class="scan-ring"></div>
        <div class="scan-dot"></div>
      </div>
      <p class="loader-text">ANALYZING IMAGE…</p>
      <p class="loader-steps" id="loaderStep">Preprocessing image</p>
    </div>

    <!-- Results -->
    <div class="results" id="results"></div>

    <!-- Analyze button -->
    <button class="btn-analyze" id="analyzeBtn" disabled onclick="analyze()">
      <span class="btn-shimmer"></span>
      RUN MOLD ANALYSIS
    </button>
  </div>
</section>

<!-- ══ HOW IT WORKS ═══════════════════════════════════════════════ -->
<section class="how-section">
  <div class="fade-in">
    <p class="section-label">// The Process</p>
    <h2 class="section-title">HOW IT WORKS</h2>
  </div>
  <div class="steps-grid">
    <div class="how-card fade-in">
      <div class="how-num">01</div>
      <div class="how-title">Upload Photo</div>
      <div class="how-desc">Take a clear, close-up photo of the mold and upload it to the scanner.</div>
    </div>
    <div class="how-card fade-in">
      <div class="how-num">02</div>
      <div class="how-title">AI Analysis</div>
      <div class="how-desc">ResNet50V2 + BiLSTM neural network processes the image through 50+ layers.</div>
    </div>
    <div class="how-card fade-in">
      <div class="how-num">03</div>
      <div class="how-title">Classification</div>
      <div class="how-desc">The model identifies the mold species with confidence scores for all 4 types.</div>
    </div>
    <div class="how-card fade-in">
      <div class="how-num">04</div>
      <div class="how-title">Action Plan</div>
      <div class="how-desc">Get a complete remediation guide — danger level, health risks, and step-by-step removal.</div>
    </div>
  </div>
</section>

<footer>
  <p>MoldScan AI · Powered by ResNet50V2 + 1D-CNN + BiLSTM · 98.7% Accuracy</p>
  <p style="margin-top:6px;font-size:0.72rem;">For severe infestations, always consult a professional mold remediation service.</p>
</footer>

<script>
// ── FILE HANDLING ─────────────────────────────────────────────────────
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const analyzeBtn  = document.getElementById('analyzeBtn');
const previewCont = document.getElementById('previewContainer');
const previewImg  = document.getElementById('previewImg');
const previewName = document.getElementById('previewName');
const previewSize = document.getElementById('previewSize');
const loader      = document.getElementById('loader');
const results     = document.getElementById('results');
const loaderStep  = document.getElementById('loaderStep');

let currentFile = null;

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1048576).toFixed(1) + ' MB';
}

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatSize(file.size);
    dropZone.style.display = 'none';
    previewCont.style.display = 'flex';
    loader.style.display = 'none';
    results.style.display = 'none';
    analyzeBtn.disabled = false;
    analyzeBtn.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  currentFile = null;
  fileInput.value = '';
  dropZone.style.display = 'block';
  previewCont.style.display = 'none';
  loader.style.display = 'none';
  results.style.display = 'none';
  analyzeBtn.disabled = true;
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

// ── ANALYZE ───────────────────────────────────────────────────────────
async function analyze() {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  previewCont.style.display = 'none';
  loader.style.display = 'block';
  results.style.display = 'none';

  const steps = ['Preprocessing image…', 'Running neural network…', 'Classifying mold species…', 'Generating report…'];
  let si = 0;
  const stepInterval = setInterval(() => {
    loaderStep.textContent = steps[si++ % steps.length];
  }, 600);

  const formData = new FormData();
  formData.append('image', currentFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    clearInterval(stepInterval);
    loader.style.display = 'none';
    showResults(data);
  } catch (err) {
    clearInterval(stepInterval);
    loader.style.display = 'none';
    results.style.display = 'block';
    results.innerHTML = `<div style="padding:24px;text-align:center;color:#ff4444;">
      ❌ Error: ${err.message}<br><small style="color:#6b7680">Make sure the Flask server is running</small>
    </div>`;
    analyzeBtn.disabled = false;
    analyzeBtn.style.display = 'block';
  }
}

// ── RENDER RESULTS ────────────────────────────────────────────────────
function showResults(data) {
  if (data.error) {
    results.innerHTML = `<div style="padding:24px;text-align:center;color:#ff4444;">❌ ${data.error}</div>`;
    results.style.display = 'block';
    analyzeBtn.disabled = false;
    return;
  }

  const info       = data.info;
  const classNames = data.class_names;
  const probs      = data.probabilities;
  const predicted  = data.predicted_class;

  let confHTML = '';
  classNames.forEach((name, i) => {
    const pct   = (probs[i] * 100).toFixed(1);
    const isPred = name === predicted;
    confHTML += `
      <div class="conf-row">
        <span class="conf-name">${name}</span>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill ${isPred?'predicted':''}" data-width="${pct}" style="width:0%"></div>
        </div>
        <span class="conf-pct" style="color:${isPred?'var(--accent)':'inherit'}">${pct}%</span>
      </div>`;
  });

  let stepsHTML = '';
  info.removal_steps.forEach((step, i) => {
    stepsHTML += `
      <div class="step-item" style="animation-delay:${i*0.07}s">
        <div class="step-num">${String(i+1).padStart(2,'0')}</div>
        <div>${step}</div>
      </div>`;
  });

  let risksHTML = info.health_risks.map(r => `<li>${r}</li>`).join('');

  results.innerHTML = `
    <div class="result-header">
      <span class="result-icon">${info.icon}</span>
      <div>
        <div class="result-name">${info.common_name}</div>
        <span class="danger-badge" style="background:${info.danger_color}22;color:${info.danger_color};border:1px solid ${info.danger_color}44;">
          ${info.danger_level} DANGER
        </span>
      </div>
    </div>

    <div class="confidence-section">
      <p class="conf-label">// Detection confidence</p>
      ${confHTML}
    </div>

    <div class="info-grid">
      <div class="info-card" style="grid-column:1/-1">
        <p class="info-card-title">About this mold</p>
        <p class="desc-text">${info.description}</p>
      </div>
      <div class="info-card">
        <p class="info-card-title">⚠️ Health Risks</p>
        <ul class="risk-list">${risksHTML}</ul>
      </div>
      <div class="info-card">
        <p class="info-card-title">🛡️ Prevention</p>
        <p style="font-size:0.85rem;line-height:1.7;color:#c8cdd2">${info.prevention}</p>
      </div>
    </div>

    <div class="steps-section">
      <p class="conf-label" style="margin-bottom:16px">// Step-by-step removal guide</p>
      ${stepsHTML}
    </div>

    <div class="prevention-box">
      <strong>💡 Pro Prevention Tip</strong>
      ${info.prevention}
    </div>

    <button class="scan-again-btn" onclick="resetUpload(); analyzeBtn.style.display='block';">
      ↩ Scan another image
    </button>
  `;

  results.style.display = 'block';
  analyzeBtn.style.display = 'none';

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.conf-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 100);
}

// ── SCROLL ANIMATIONS ─────────────────────────────────────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.1 });
document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
</script>
</body>
</html>'''

# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Load and preprocess image
        img  = Image.open(io.BytesIO(file.read())).convert('RGB')
        img  = img.resize(IMG_SIZE)
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = np.expand_dims(arr, axis=0)    # (1, 224, 224, 3)

        if model is None:
            return jsonify({'error': 'Model not loaded. Check MODEL_PATH in app.py'})

        # Predict
        probs     = model.predict(arr, verbose=0)[0]
        pred_idx  = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]

        return jsonify({
            'predicted_class': pred_name,
            'confidence':      float(probs[pred_idx]) * 100,
            'probabilities':   [float(p) for p in probs],
            'class_names':     CLASS_NAMES,
            'info':            MOLD_INFO[pred_name]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  MoldScan AI  --  Starting server...")
    print("  Model  :", MODEL_PATH)
    print("  Open   : http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)