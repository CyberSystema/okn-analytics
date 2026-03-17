# OKN Social Media Analytics Pipeline

**Automated social media intelligence for the Orthodox Korea Network**

A Python-based analytics pipeline that processes weekly CSV/JSON exports from YouTube, Instagram, Facebook, and TikTok — generating insights, predictions, and visual reports via GitHub Actions.

## How It Works

1. **Export** analytics data from each platform dashboard (weekly)
2. **Push** the CSV/JSON files to the `data/` folder in this repo
3. **GitHub Actions** triggers automatically on push
4. **Reports** are generated and committed to `reports/`

No APIs. No OAuth. No rate limits. Just data in, insights out.

---

## Repository Structure

```
okn-analytics/
├── data/
│   ├── youtube/           ← Drop YouTube Studio CSV exports here
│   ├── instagram/         ← Drop Meta Business Suite CSV exports here
│   ├── facebook/          ← Drop Facebook Page Insights exports here
│   └── tiktok/            ← Drop TikTok Analytics exports here
├── scripts/
│   ├── main.py            ← Pipeline orchestrator
│   ├── config.py          ← Configuration & settings
│   ├── ingest.py          ← Data normalization layer
│   ├── analyze.py         ← Core analytics engine
│   ├── report.py          ← Report generation (HTML + charts)
│   └── models/
│       ├── timing.py      ← Optimal posting time model
│       ├── scoring.py     ← Content performance scoring
│       └── forecast.py    ← Growth forecasting
├── history/               ← Accumulated historical data (auto-managed)
├── reports/               ← Generated reports (auto-committed)
├── .github/
│   └── workflows/
│       └── analyze.yml    ← GitHub Actions workflow
└── requirements.txt
```

---

## Data Export Guides

### Instagram (Priority Platform)
1. Go to **Meta Business Suite** → Insights → Content
2. Click **Export Data** (top right)
3. Select date range (past 7 days) and format (CSV)
4. Download and place in `data/instagram/`
5. Also export **Audience** data separately

### YouTube
1. Go to **YouTube Studio** → Analytics
2. Click **Advanced Mode** → Export (top right) → CSV
3. Export **Content**, **Audience**, and **Revenue** tabs separately
4. Place all files in `data/youtube/`

### Facebook
1. Go to **Meta Business Suite** → Insights
2. Click **Export Data** → select metrics and date range
3. Download CSV and place in `data/facebook/`

### TikTok
1. Go to **TikTok Studio** → Analytics
2. Export available data (or manually fill the template in `data/tiktok/template.csv`)
3. Place in `data/tiktok/`

---

## Setup

```bash
# Clone the repo
git clone https://github.com/CyberSystema/okn-analytics.git
cd okn-analytics

# Install dependencies
pip install -r requirements.txt

# Run manually (optional — Actions does this automatically)
python scripts/main.py
```

---

## Reports

After each run, the pipeline generates:
- `reports/weekly_report.html` — Full interactive HTML report
- `reports/latest_summary.json` — Machine-readable summary
- `reports/charts/` — Individual chart images

---

## Configuration

Edit `scripts/config.py` to customize:
- Platform priorities and weights
- Analysis parameters
- Report branding (OKN colors, logo)
- Alert thresholds

---

## License

Internal tool for the Orthodox Korea Network (OKN) team.
Built by Nikos Pinatsis / CyberSystema.
