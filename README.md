<div align="center">

# ☦️ OKN Social Media Analytics Pipeline

**Automated social media intelligence for the Orthodox Korea Network**

[![Live Report](https://img.shields.io/badge/📊_Live_Report-okn--analytics.pages.dev-1a3a5c?style=for-the-badge)](https://okn-analytics.pages.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML Models](https://img.shields.io/badge/ML_Models-14-c4953a?style=for-the-badge)](https://github.com/CyberSystema/okn-analytics)
[![Platforms](https://img.shields.io/badge/Platforms-Instagram_•_TikTok-E4405F?style=for-the-badge)](https://github.com/CyberSystema/okn-analytics)

A Python-based pipeline that processes weekly CSV exports, runs **14 ML & neural network models** (including multilingual semantic AI for English, Korean & Greek), and generates a comprehensive HTML intelligence report — deployed automatically via GitHub Actions to Cloudflare Pages.

<br>

**No APIs. No OAuth. No rate limits. Just data in, insights out.**

</div>

<br>

## 🔄 How It Works

```
  ┌─────────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────────┐
  │  📥 Export   │────▶│  📤 Push  │────▶│  ⚙️ GitHub   │────▶│  🧠 14   │────▶│  📊 Report   │
  │  CSV files   │     │  to repo  │     │   Actions    │     │ ML Models│     │  on CF Pages │
  └─────────────┘     └──────────┘     └──────────────┘     └──────────┘     └──────────────┘
  Meta Business Suite    git push        Runs automatically    Semantic AI       okn-analytics
  TikTok Studio                          on every push         Neural Networks   .pages.dev
```

<br>

## ✨ Features

### 📋 Executive Summary
> Every report opens with a **plain-language weekly summary** — no data science required. Perfect for the whole team.

| Section | What it tells you |
|---|---|
| 🟢🟡🔴 **Health Pulse** | One sentence — "Things are great!" or "We need to adjust" |
| 📊 **Key Numbers** | Total reach, interactions, followers gained, week-over-week |
| ✅ **What's Working** | Best content types, best posting times, viral hits |
| ⚠️ **What Needs Attention** | Declining reach, content fatigue, low comments |
| 🎯 **This Week's Actions** | 3–4 specific things anyone on the team can do right now |

### 📈 Analysis Engine
- **Recency-weighted** — last 90 days get full weight, older data fades out
- **Per-platform** sections — Instagram and TikTok never blended (different methodologies)
- **Temporal patterns** — best hours and days to post in KST
- **Growth trajectory** — reach, engagement, follower trend detection
- **Anomaly detection** — viral hits and underperformers flagged automatically
- **Cross-platform comparison** — with methodology disclaimers

### 🧠 14 ML & Neural Network Models

<details>
<summary><strong>Core Models (1–10)</strong> — run on all data</summary>
<br>

| # | Model | Algorithm | What it does |
|:---:|---|---|---|
| 1 | **Feature Importance** | GradientBoosting | Ranks what drives engagement |
| 2 | **Engagement Predictor** | MLP Neural Network (32→16→8) | Predicts engagement for new posts |
| 3 | **Content Clusters** | KMeans | Groups posts: Top / Average / Low |
| 4 | **Anomaly Detection** | IsolationForest | Finds statistically unusual posts |
| 5 | **Caption NLP** | TF-IDF + stopwordsiso | Words correlated with engagement |
| 6 | **Engagement Drivers** | Statistical correlation | Caption length, hashtags, emoji, etc. |
| 7 | **Content Fatigue** | Weighted linear regression | Detects declining engagement per type |
| 8 | **Optimal Cadence** | Cadence analysis | Finds ideal posts/week |
| 9 | **Momentum Score** | Composite 0–100 | Forward-looking health metric |
| 10 | **Root Cause Analysis** | Feature attribution | Explains WHY posts went viral or flopped |

</details>

<details>
<summary><strong>Semantic AI Models (11–14)</strong> — powered by multilingual embeddings</summary>
<br>

> Uses `paraphrase-multilingual-MiniLM-L12-v2` — 384-dimensional vectors that understand **English, Korean, and Greek** simultaneously. Loaded once, shared across all 4 models.

| # | Model | What it does |
|:---:|---|---|
| 11 | **Topic Discovery** | Clusters posts by *meaning* — finds themes like "feast days", "youth camp", "liturgy" |
| 12 | **Similar Post Predictor** | Finds the 5 most similar past posts and predicts expected engagement |
| 13 | **Hashtag Cluster Strategy** | Groups hashtags into semantic themes, tracks which clusters drive engagement |
| 14 | **Semantic Features** | PCA-reduced caption embeddings fed into GradientBoosting as features |

These models gracefully skip if `sentence-transformers` is not installed.

</details>

### 🌐 Multilingual Intelligence

The pipeline natively handles three languages across all analysis:

| | English | Korean (한국어) | Greek (Ελληνικά) |
|---|:---:|:---:|:---:|
| Caption analysis | ✅ | ✅ | ✅ |
| Stop words (2,824) | ✅ | ✅ | ✅ with accent stripping |
| Semantic embeddings | ✅ | ✅ | ✅ |
| Topic discovery | ✅ | ✅ | ✅ |
| Hashtag clustering | ✅ | ✅ | ✅ |

### 🎨 Report Design
- Self-contained HTML with embedded charts and logos
- **OKN logo** in header with gold gradient ring
- **CyberSystema branding** — header badge, chart watermarks, built-by card, footer
- Per-platform color-coded sections
- All links open to [cybersystema.com](https://cybersystema.com)

<br>

## 📁 Repository Structure

```
okn-analytics/
│
├── 📂 data/
│   ├── 📸 instagram/            ← Meta Business Suite exports
│   │   ├── content.csv          ← Weekly content (renamed)
│   │   ├── Follows.csv
│   │   ├── Interactions.csv
│   │   ├── Link clicks.csv
│   │   ├── Reach.csv
│   │   ├── Views.csv
│   │   ├── Visits.csv
│   │   └── Audience.csv
│   └── 🎵 tiktok/              ← TikTok Studio exports
│       ├── Content.csv
│       ├── Overview.csv
│       ├── Viewers.csv
│       ├── FollowerHistory.csv
│       ├── FollowerActivity.csv
│       ├── FollowerGender.csv
│       └── FollowerTopTerritories.csv
│
├── 📂 scripts/
│   ├── main.py                  ← Pipeline orchestrator
│   ├── config.py                ← Configuration & branding
│   ├── ingest.py                ← Instagram data normalization
│   ├── ingest_tiktok.py         ← TikTok + Greek date parser
│   ├── ingest_account.py        ← Account daily metrics
│   ├── analyze.py               ← Core analysis (8 modules)
│   ├── report.py                ← HTML report + executive summary
│   └── 📂 models/
│       ├── ml_engine.py         ← 14 ML models
│       ├── timing.py            ← Posting time optimization
│       ├── scoring.py           ← Content scoring
│       └── forecast.py          ← Growth forecasting
│
├── 📂 history/                  ← Auto-managed (grows weekly)
├── 📂 reports/                  ← Generated output + logos
├── 📂 .github/workflows/       ← CI/CD pipeline
└── requirements.txt
```

<br>

## 📥 Data Export Guides

### 📸 Instagram (Meta Business Suite)

> Exports are in **PST (America/Los_Angeles)** — the pipeline converts to KST automatically.

1. Go to **Meta Business Suite** → Insights → Content
2. Click **Export Data** → select date range → CSV format
3. **Rename the file to `content.csv`** and place in `data/instagram/`
4. Also export account files: Follows, Reach, Views, Visits, Interactions, Audience

### 🎵 TikTok (TikTok Studio)

> Exports use **Greek month names** (e.g., "15 Μαρτίου") — parsed automatically.

1. Go to **TikTok Studio** → Analytics
2. Export all available sections as CSV
3. Place in `data/tiktok/`

<br>

## 🚀 Setup

```bash
# Clone
git clone https://github.com/CyberSystema/okn-analytics.git
cd okn-analytics

# Install dependencies
pip install -r requirements.txt

# Run
python scripts/main.py
```

### Dependencies

| Package | Purpose | Required? |
|---|---|:---:|
| pandas, numpy | Data processing | ✅ |
| scikit-learn | ML models (1–10) | ✅ |
| matplotlib, seaborn | Charts | ✅ |
| stopwordsiso | Multilingual stop words | ✅ |
| pyarrow | Parquet history storage | ✅ |
| sentence-transformers | Semantic AI models (11–14) | Optional |
| prophet | Time series forecasting | Optional |

### GitHub Actions Secrets

| Secret | Description |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Cloudflare Pages deploy token |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |

<br>

## ⚙️ GitHub Actions Pipeline

Runs automatically on every push to `data/`, weekly on Mondays, or manually.

```
☦️ Checkout  →  🐍 Python 3.11  →  🧠 Cache ML Model  →  📦 Install
                                          ↓
📋 Summary  ←  🚀 Deploy CF Pages  ←  📤 Commit  ←  🔬 Run Pipeline
```

| Step | What happens | Time |
|---|---|---|
| 🧠 Cache | HuggingFace model (118MB) cached between runs | ~3s (after first run) |
| 📦 Install | All pip packages from cache | ~30s |
| 🧠 Verify | Pre-downloads + tests embedding model | ~5s (cached) |
| 🔬 Pipeline | Ingest → Analyze → 14 ML models → Report | ~10s |
| 🚀 Deploy | Cloudflare Pages deployment | ~15s |

**First run:** ~5 minutes (downloads PyTorch + model). **Every run after:** ~2 minutes.

<br>

## 🔧 Configuration

Edit `scripts/config.py`:

| Setting | Default | Description |
|---|---|---|
| `TIMELINE["active_since"]` | December 2025 | When OKN social media became consistent |
| `TIMELINE["platform_start"]["tiktok"]` | January 6, 2026 | TikTok account creation |
| `TIMELINE["recency_weights"]` | 90/180 days | How much weight recent vs old data gets |
| `TIMEZONE` | Asia/Seoul | All times displayed in KST |
| `BRANDING` | OKN colors | Navy, Byzantine gold, deep red |

<br>

## 📅 Weekly Workflow

```
Monday:
  1. Export CSV from Meta Business Suite (last 60 days is enough)
  2. Rename content export → content.csv
  3. Export account files (Follows, Views, etc.)
  4. Export TikTok CSVs
  5. git add . && git commit -m "Week N data" && git push
  6. ☕ Wait ~2 minutes
  7. Report live at okn-analytics.pages.dev
```

> 💡 The pipeline **accumulates history** — each week's data merges with all previous weeks. Post metrics update to latest values. ML models improve as data grows.

<br>

## 📝 Technical Notes

| Topic | Detail |
|---|---|
| Instagram engagement rate | interactions ÷ reach |
| TikTok engagement rate | interactions ÷ views (different denominator) |
| Export timezone | PST (America/Los_Angeles) → auto-converted to KST |
| Greek stop words | Unicode accent stripping (NFKD normalization) for τής → της matching |
| Stop words total | 2,824 across English, Greek, Korean via `stopwordsiso` |
| Semantic model | `paraphrase-multilingual-MiniLM-L12-v2` (384d, 50+ languages) |
| Chart rendering | DejaVu Sans — Korean Hangul filtered for compatibility |
| History format | Parquet (CSV fallback if pyarrow unavailable) |
| ML weighting | All models use recency-weighted `sample_weight` |

<br>

---

<div align="center">

**Built by Nikolaos Pinatsis**

[![CyberSystema](https://img.shields.io/badge/CyberSystema-cybersystema.com-00d4ff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHJ4PSI0IiBmaWxsPSIjMWExYTJlIi8+PHRleHQgeD0iMyIgeT0iMTUiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSIjMDBkNGZmIj5DUzwvdGV4dD48L3N2Zz4=)](https://cybersystema.com)
[![GitHub](https://img.shields.io/badge/GitHub-CyberSystema-181717?style=for-the-badge&logo=github)](https://github.com/CyberSystema)

*Social media intelligence & data engineering*

</div>
