"""
OKN Analytics Pipeline — Configuration
=======================================
Central configuration for all pipeline components.
Edit this file to customize behavior, branding, and thresholds.
"""

from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
HISTORY_DIR = ROOT_DIR / "history"
REPORTS_DIR = ROOT_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Platform data directories
PLATFORM_DIRS = {
    "instagram": DATA_DIR / "instagram",
    "youtube": DATA_DIR / "youtube",
    "facebook": DATA_DIR / "facebook",
    "tiktok": DATA_DIR / "tiktok",
}

# ──────────────────────────────────────────────
# PLATFORM CONFIGURATION
# ──────────────────────────────────────────────
PLATFORMS = {
    "instagram": {
        "name": "Instagram",
        "priority": 1,           # Highest priority
        "weight": 0.35,          # Weight in composite scoring
        "color": "#E4405F",      # Platform brand color
        "icon": "📸",
    },
    "youtube": {
        "name": "YouTube",
        "priority": 2,
        "weight": 0.30,
        "color": "#FF0000",
        "icon": "▶️",
    },
    "facebook": {
        "name": "Facebook",
        "priority": 3,
        "weight": 0.20,
        "color": "#1877F2",
        "icon": "👤",
    },
    "tiktok": {
        "name": "TikTok",
        "priority": 4,
        "weight": 0.15,
        "color": "#000000",
        "icon": "🎵",
    },
}

# ──────────────────────────────────────────────
# UNIFIED DATA SCHEMA
# ──────────────────────────────────────────────
# All platform data gets normalized to this schema
UNIFIED_SCHEMA = {
    "post_id": str,           # Unique identifier
    "platform": str,          # instagram, youtube, facebook, tiktok
    "published_at": str,      # ISO datetime
    "content_type": str,      # reel, story, post, video, short, live, etc.
    "title": str,             # Post title or caption (truncated)
    "permalink": str,         # Direct link to the post (if available)
    "duration_sec": int,      # Content duration in seconds (0 if N/A)
    "reach": int,             # Total reach / impressions
    "views": int,             # Video views (0 for non-video)
    "likes": int,
    "comments": int,
    "shares": int,            # Shares / reposts / retweets
    "saves": int,             # Saves / bookmarks (0 if unavailable)
    "engagement_total": int,  # Sum of all interactions
    "engagement_rate": float, # engagement_total / reach
    "watch_time_sec": int,    # Total watch time in seconds (0 if N/A)
    "avg_watch_sec": float,   # Average watch time per view (0 if N/A)
    "followers_gained": int,  # Net followers gained from this post
    "link_clicks": int,       # External link clicks (0 if N/A)
}

# ──────────────────────────────────────────────
# CONTENT TYPE MAPPING
# ──────────────────────────────────────────────
# Normalize platform-specific content types to unified categories
CONTENT_TYPE_MAP = {
    # Instagram (real Meta Business Suite export values)
    "ig reel": "short_video",
    "ig carousel": "carousel",
    "ig image": "image",
    "ig story": "story",
    # Instagram (alternative/legacy formats)
    "reel": "short_video",
    "reels": "short_video",
    "story": "story",
    "stories": "story",
    "carousel": "carousel",
    "photo": "image",
    "image": "image",
    "igtv": "long_video",
    "ig_reel": "short_video",
    # YouTube
    "video": "long_video",
    "short": "short_video",
    "shorts": "short_video",
    "live": "live",
    "premiere": "long_video",
    # Facebook
    "link": "link_post",
    "status": "text_post",
    "photo_post": "image",
    "video_post": "long_video",
    "reel_fb": "short_video",
    # TikTok
    "tiktok_video": "short_video",
    # Fallback
    "post": "image",
    "other": "other",
}

# Unified content categories for analysis
CONTENT_CATEGORIES = [
    "short_video",   # Reels, Shorts, TikToks
    "long_video",    # YouTube videos, IGTV
    "image",         # Photos, single images
    "carousel",      # Multi-image posts
    "story",         # Stories (ephemeral)
    "live",          # Live streams
    "text_post",     # Text-only posts
    "link_post",     # Link shares
    "other",
]

# ──────────────────────────────────────────────
# ANALYSIS PARAMETERS
# ──────────────────────────────────────────────
ANALYSIS = {
    # Minimum posts needed per category for reliable analysis
    "min_posts_for_analysis": 5,

    # Viral threshold: post performs this many times above average
    "viral_multiplier": 2.5,

    # Underperformer threshold: post performs below this fraction of average
    "underperform_threshold": 0.3,

    # Number of top/bottom posts to highlight in reports
    "top_n_posts": 5,

    # Time buckets for posting time analysis (hours in KST)
    "time_buckets": {
        "early_morning": (5, 8),
        "morning": (8, 12),
        "afternoon": (12, 17),
        "evening": (17, 21),
        "night": (21, 24),
        "late_night": (0, 5),
    },

    # Growth rate alert thresholds
    "growth_alert_positive": 0.10,   # +10% week-over-week = great
    "growth_alert_negative": -0.05,  # -5% week-over-week = concern

    # Engagement rate benchmarks (cross-industry averages)
    # Methodology: interactions / reach — matching our pipeline calculation
    #
    # Sources (latest available as of early 2026):
    #   Hootsuite Q1 2025 (1M+ posts, published Feb 2026)
    #   Socialinsider 2026 Benchmarks Report (70M posts)
    #   Buffer State of Engagement 2026 (52M+ posts)
    #   Metricool 2024 YouTube Study (10M+ videos)
    #
    # Note: These are cross-industry averages across all account sizes.
    # Smaller community accounts (<5K followers) like OKN typically see
    # 2-5x higher engagement rates — which is normal and healthy.
    "engagement_benchmarks": {
        "instagram": 0.020,   # 2.0% — Hootsuite 2026
        "youtube": 0.039,     # 3.9% — Metricool 2024 / Buffer 2026
        "facebook": 0.014,    # 1.4% — Hootsuite 2026
        "tiktok": 0.025,      # 2.5% — Sprout Social / Hootsuite 2026
    },
}

# ──────────────────────────────────────────────
# FORECASTING PARAMETERS
# ──────────────────────────────────────────────
FORECAST = {
    "horizon_days": 30,          # How far ahead to forecast
    "min_history_weeks": 4,      # Minimum weeks of data needed
    "confidence_interval": 0.80, # 80% confidence interval
}

# ──────────────────────────────────────────────
# REPORT BRANDING
# ──────────────────────────────────────────────
BRANDING = {
    "org_name": "Orthodox Korea Network",
    "org_short": "OKN",
    "primary_color": "#1a3a5c",     # Deep navy (Orthodox aesthetic)
    "secondary_color": "#c4953a",   # Byzantine gold
    "accent_color": "#8b1a1a",      # Deep red
    "bg_color": "#f8f6f0",          # Warm parchment
    "text_color": "#2c2c2c",
    "font_family": "'Segoe UI', 'Noto Sans KR', sans-serif",
    "report_title": "OKN Social Media Intelligence Report",
    "footer_text": "Generated by OKN Analytics Pipeline • CyberSystema",
}

# ──────────────────────────────────────────────
# TIMEZONE
# ──────────────────────────────────────────────
TIMEZONE = "Asia/Seoul"  # KST — primary audience timezone


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for d in [DATA_DIR, HISTORY_DIR, REPORTS_DIR, CHARTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for platform_dir in PLATFORM_DIRS.values():
        platform_dir.mkdir(parents=True, exist_ok=True)
    # Create .gitkeep files
    for platform_dir in PLATFORM_DIRS.values():
        gitkeep = platform_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
