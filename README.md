# OKN Social Media Analytics Pipeline

**Automated social media intelligence for the Orthodox Korea Network**

A Python-based analytics pipeline that processes weekly CSV exports from Instagram and TikTok, runs ML/neural network models, and generates a comprehensive HTML intelligence report — deployed automatically via GitHub Actions to Cloudflare Pages.

🔗 **Live Report:** [okn-analytics.pages.dev](https://okn-analytics.pages.dev)

---

## How It Works

1. **Export** analytics data from Meta Business Suite (Instagram) and TikTok Studio (weekly)
2. **Push** the CSV files to the `data/` folder in this repo
3. **GitHub Actions** triggers automatically on push
4. **Pipeline** runs: Ingest → Analyze → ML Models → Generate Report
5. **Report** is deployed to Cloudflare Pages

No APIs. No OAuth. No rate limits. Just data in, insights out.

---

## Features

### Data Processing
- Unified schema normalizing Instagram and TikTok exports
- Greek timezone → KST conversion (Meta Business Suite exports in Europe/Athens)
- Greek date parsing for TikTok exports (handles 365-day year-wrapping)
- Historical data accumulation with metric freshness (keeps latest values on dedup)
- Per-platform account daily data with merge-on-append
- Demographics versioned to JSONL history for tracking audience changes

### Analysis Engine
- Recency-weighted analysis: last 90 days = weight 1.0, 90–180 days = 0.3, older = 0.1
- Platform overview with weighted KPIs and week-over-week trends
- Content performance ranking per platform (never mixes engagement methodologies)
- Temporal patterns: best hours and days to post (KST), weighted by recency
- Growth trajectory and reach/engagement trend detection
- Anomaly detection: identifies viral hits and underperformers
- Cross-platform comparison with methodology disclaimers

### ML & Neural Network Models (10 models)

| # | Model | Algorithm | Purpose |
|---|---|---|---|
| 1 | Feature Importance | GradientBoosting (100 trees, sample_weight) | Ranks engagement drivers |
| 2 | Engagement Predictor | MLP Neural Network (32→16→8) | Predicts engagement, finds over/underperformers |
| 3 | Content Clusters | KMeans | Groups posts into Top/Average/Low |
| 4 | Anomaly Detection | IsolationForest | Finds viral outliers and flops |
| 5 | Caption NLP | TF-IDF + stopwordsiso (EN/EL/KO) | Words/hashtags correlated with engagement |
| 6 | Engagement Drivers | Statistical correlation | Caption length, hashtags, emoji, multilingual, weekend |
| 7 | Content Fatigue | Weighted linear regression | Detects declining engagement per content type |
| 8 | Optimal Cadence | Cadence vs engagement analysis | Finds optimal posts/week |
| 9 | Momentum Score | Composite 0–100 metric | Forward-looking health score |
| 10 | Root Cause Analysis | Feature attribution (SHAP-like) | Explains WHY posts went viral or flopped |

### Report
- Self-contained HTML report with embedded charts and OKN logo
- Per-platform sections (Instagram, TikTok) — never blends data
- Interactive KPI cards, charts, ML insights, and recommendations
- Deployed to Cloudflare Pages on every push

---

## Repository Structure

```
okn-analytics/
├── data/
│   ├── instagram/         ← Meta Business Suite CSV exports
│   │   ├── content.csv    ← Renamed weekly content export
│   │   ├── Follows.csv
│   │   ├── Interactions.csv
│   │   ├── Link clicks.csv
│   │   ├── Reach.csv
│   │   ├── Views.csv
│   │   ├── Visits.csv
│   │   └── Audience.csv
│   └── tiktok/            ← TikTok Studio CSV exports
│       ├── Content.csv
│       ├── Overview.csv
│       ├── Viewers.csv
│       ├── FollowerHistory.csv
│       ├── FollowerActivity.csv
│       ├── FollowerGender.csv
│       └── FollowerTopTerritories.csv
├── scripts/
│   ├── main.py            ← Pipeline orchestrator
│   ├── config.py          ← Configuration, timeline, recency weights
│   ├── ingest.py          ← Data normalization (Instagram)
│   ├── ingest_tiktok.py   ← TikTok ingestion + Greek date parser
│   ├── ingest_account.py  ← Account-level data (daily metrics, demographics)
│   ├── analyze.py         ← Core analysis engine (8 modules)
│   ├── report.py          ← HTML report generation
│   └── models/
│       ├── ml_engine.py   ← ML & Neural Network models (10 models)
│       ├── timing.py      ← Optimal posting time model
│       ├── scoring.py     ← Content performance scoring
│       └── forecast.py    ← Growth forecasting (Prophet or statistical)
├── history/               ← Auto-managed historical data
│   ├── unified_history.parquet  ← All posts (accumulates over time)
│   ├── unified_history.csv      ← CSV fallback
│   ├── account_daily_instagram.csv
│   ├── account_daily_tiktok.csv
│   └── demographics_history_tiktok.jsonl
├── reports/
│   ├── weekly_report.html ← Generated report
│   ├── okn_logo.png       ← OKN logo (embedded in report header)
│   ├── index.html         ← Cloudflare Pages redirect
│   └── full_results.json  ← Machine-readable results
├── .github/
│   └── workflows/
│       └── analyze.yml    ← GitHub Actions workflow
└── requirements.txt
```

---

## Data Export Guides

### Instagram (Meta Business Suite)

Exports are in **Greek timezone** (Europe/Athens) — the pipeline converts to KST automatically.

1. Go to **Meta Business Suite** → Insights → Content
2. Click **Export Data** → select date range → CSV format
3. **Rename the file to `content.csv`** and place in `data/instagram/`
4. Also export account-level files:
   - Insights → Follows → Export → `Follows.csv`
   - Insights → Reach → Export → `Reach.csv`
   - Insights → Views → Export → `Views.csv`
   - Insights → Visits → Export → `Visits.csv`
   - Insights → Interactions → Export → `Interactions.csv`
   - Audience → Export → `Audience.csv`

### TikTok (TikTok Studio)

Exports use **Greek month names** (e.g., "15 Μαρτίου") — the pipeline parses these automatically.

1. Go to **TikTok Studio** → Analytics
2. Export all available sections as CSV
3. Place in `data/tiktok/`

---

## Setup

```bash
# Clone
git clone https://github.com/CyberSystema/okn-analytics.git
cd okn-analytics

# Install dependencies
pip install -r requirements.txt

# Run
python scripts/main.py
```

### Requirements

- Python 3.11+
- Key packages: pandas, numpy, scikit-learn, matplotlib, stopwordsiso, pyarrow
- Optional: Prophet (falls back to statistical forecasting if not installed)

### GitHub Actions Secrets

| Secret | Description |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Cloudflare Pages deploy token |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |

---

## Configuration

Edit `scripts/config.py` to customize:

### Timeline
- `TIMELINE["active_since"]` — When OKN social media became consistently active (default: December 2025)
- `TIMELINE["platform_start"]["tiktok"]` — TikTok account creation date (default: January 6, 2026)
- `TIMELINE["recency_weights"]` — How much weight recent vs old data gets

### Timezone
- All timestamps displayed in **KST (UTC+9)**
- Instagram exports assumed to be in **Europe/Athens** timezone
- "Now" always means the most recent date in the data, not today

### Branding
- Colors: Deep navy, Byzantine gold, deep red
- OKN logo embedded from `reports/okn_logo.png`
- Report title, footer text, font family

---

## Weekly Workflow

1. Export data from Meta Business Suite and TikTok Studio
2. Rename Instagram content export to `content.csv`
3. Drop all CSV files in the appropriate `data/` subfolder
4. `git add . && git commit -m "Week N data" && git push`
5. GitHub Actions runs automatically → report appears at okn-analytics.pages.dev

The pipeline accumulates history: each week's data merges with previous weeks. Post metrics are updated to their latest values. ML models get smarter as data grows.

---

## Technical Notes

- Instagram engagement rate = interactions ÷ reach
- TikTok engagement rate = interactions ÷ views (different denominator — noted in report)
- Caption NLP strips Unicode accents for proper Greek/Korean stop word matching
- Charts filter out Korean Hangul characters that can't render in DejaVu Sans
- All ML models use recency-weighted `sample_weight` so recent data matters more
- History uses Parquet format (falls back to CSV if pyarrow not installed)

---

## License

Internal tool for the Orthodox Korea Network (OKN) team.

Built by Nikolaos Pinatsis / [CyberSystema](https://github.com/CyberSystema)
