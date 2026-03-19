"""
OKN Analytics Pipeline — Report Generator v2
=============================================
Per-platform sections with ML/NN insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, logging, base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config import BRANDING, PLATFORMS, ANALYSIS, REPORTS_DIR, CHARTS_DIR, ensure_dirs

logger = logging.getLogger("okn.report")


def _safe(text, max_len=80):
    """Escape HTML-sensitive characters in text for safe embedding."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#fafafa","axes.edgecolor":"#cccccc","axes.grid":True,"grid.alpha":0.3,"font.family":"sans-serif","font.size":11})


class ReportGenerator:
    def __init__(self, df, analysis, scores=None, timing=None, forecast=None, account_data=None, ml_results=None):
        self.df = df
        self.analysis = analysis
        self.scores = scores or {}
        self.timing = timing or {}
        self.forecast = forecast or {}
        self.account_data = account_data or {"daily": pd.DataFrame(), "demographics": {}}
        self.ml_results = ml_results or {}
        self.charts = {}

    def generate(self) -> str:
        ensure_dirs()
        logger.info("📝 Generating report...")
        self._gen_all_charts()
        html = self._build_html()
        p = REPORTS_DIR / "weekly_report.html"
        p.write_text(html, encoding="utf-8")
        logger.info(f"📄 Report saved: {p}")
        (REPORTS_DIR / "latest_summary.json").write_text(json.dumps(self._summary(), indent=2, default=str), encoding="utf-8")
        return str(p)

    # ════════════════ CHARTS ════════════════

    def _gen_all_charts(self):
        logger.info("   📊 Generating charts...")
        for plat in self.df["platform"].unique():
            pdf = self.df[self.df["platform"] == plat]
            pc = PLATFORMS.get(plat, {}).get("color", "#666")
            pn = PLATFORMS.get(plat, {}).get("name", plat)
            self._ch_content(pdf, plat, pc, pn)
            self._ch_eng_pie(pdf, plat)
            self._ch_weekly(pdf, plat, pc, pn)
            self._ch_feature_imp(plat)
            self._ch_clusters(plat)
            self._ch_nlp(plat)
        self._ch_account_daily()
        self._ch_demographics()
        self._ch_cross()

    def _ch_content(self, pdf, p, color, pn):
        t = pdf.groupby("content_type").agg(r=("engagement_rate","mean"),n=("post_id","count")).sort_values("r",ascending=True)
        if t.empty: return
        fig,ax = plt.subplots(figsize=(9, max(3, len(t)*0.8)))
        bars = ax.barh([f"{ct} ({int(row['n'])})" for ct,row in t.iterrows()], t["r"]*100, color=color, alpha=0.85)
        for bar,val in zip(bars,t["r"]*100): ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",fontsize=10)
        ax.set_xlabel("Avg Engagement Rate (%)"); ax.set_title(f"Content Performance — {pn}"); plt.tight_layout()
        self.charts[f"{p}_content"] = self._b64(fig)

    def _ch_eng_pie(self, pdf, p):
        totals = {k:int(pdf[k].sum()) for k in ["likes","comments","shares","saves"] if k in pdf.columns and pdf[k].sum()>0}
        if not totals: return
        fig,ax = plt.subplots(figsize=(6,6))
        ax.pie(list(totals.values()), labels=[k.capitalize() for k in totals], colors=["#E4405F","#1877F2","#25D366","#FF6B35"][:len(totals)], autopct="%1.1f%%", startangle=90, pctdistance=0.8)
        ax.set_title(f"Engagement Mix — {PLATFORMS.get(p,{}).get('name',p)}"); plt.tight_layout()
        self.charts[f"{p}_eng_pie"] = self._b64(fig)

    def _ch_weekly(self, pdf, p, color, pn):
        pdf = pdf.copy(); pdf["published_at"] = pd.to_datetime(pdf["published_at"],errors="coerce"); pdf = pdf.dropna(subset=["published_at"])
        if pdf.empty: return
        w = pdf.set_index("published_at").resample("W").agg(reach=("reach","sum"),eng=("engagement_total","sum"))
        w = self._clip_90d(w)
        if len(w)<2: return
        fig,ax1 = plt.subplots(figsize=(10,4.5))
        # Use short "Mon DD" format for x labels
        weeks = [d.strftime("%b %d, %Y") for d in w.index]
        x = range(len(weeks))
        ax1.bar(x, w["reach"].values, color=color, alpha=0.3, label="Reach"); ax1.set_ylabel("Reach", color=color)
        ax2 = ax1.twinx(); ax2.plot(x, w["eng"].values, color=BRANDING["secondary_color"], marker="o", linewidth=2, label="Engagement"); ax2.set_ylabel("Engagement")
        ax1.set_xticks(x); ax1.set_xticklabels(weeks, rotation=45, ha="right", fontsize=9)
        ax1.set_title(f"Weekly Trends — {pn}")
        l1,la1=ax1.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax1.legend(l1+l2,la1+la2,loc="upper left")
        fig.subplots_adjust(bottom=0.2); plt.tight_layout()
        self.charts[f"{p}_weekly"] = self._b64(fig)

    def _ch_feature_imp(self, p):
        ml = self.ml_results.get(p,{}); feats = ml.get("feature_importance",{}).get("top_features",[])
        if not feats: return
        fig,ax = plt.subplots(figsize=(9, max(3,len(feats)*0.5)))
        ax.barh([f["feature"] for f in reversed(feats)],[f["pct"] for f in reversed(feats)],color=BRANDING["primary_color"],alpha=0.85)
        ax.set_xlabel("Importance (%)"); ax.set_title(f"What Drives Engagement — {PLATFORMS.get(p,{}).get('name',p)}"); plt.tight_layout()
        self.charts[f"{p}_feat_imp"] = self._b64(fig)

    def _ch_clusters(self, p):
        cl = self.ml_results.get(p,{}).get("content_clusters",{}).get("clusters",[])
        if not cl: return
        fig,ax = plt.subplots(figsize=(8,4))
        labels=[c["label"] for c in cl]; rates=[c["avg_engagement_rate"]*100 for c in cl]; sizes=[c["size"] for c in cl]
        colors=[BRANDING["secondary_color"],"#888",BRANDING["accent_color"]]
        bars=ax.bar(range(len(cl)),rates,color=colors[:len(cl)],alpha=0.85)
        ax.set_xticks(range(len(cl))); ax.set_xticklabels([f"{l}\n({s} posts)" for l,s in zip(labels,sizes)],fontsize=9)
        ax.set_ylabel("Avg Engagement Rate (%)"); ax.set_title(f"Content Clusters — {PLATFORMS.get(p,{}).get('name',p)}")
        for bar,val in zip(bars,rates): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{val:.1f}%",ha="center",fontsize=10)
        plt.tight_layout(); self.charts[f"{p}_clusters"] = self._b64(fig)

    def _ch_nlp(self, p):
        terms = self.ml_results.get(p,{}).get("caption_analysis",{}).get("top_engagement_terms",[])[:8]
        if not terms: return
        fig,ax = plt.subplots(figsize=(9, max(3,len(terms)*0.5)))
        import unicodedata
        def _chart_safe(text):
            """Keep only Latin, Greek, and common characters for chart labels."""
            safe = ''.join(c for c in text if ord(c) < 0x3000)
            return safe.strip() if safe.strip() else f"[{text[:8]}]"
        names=[_chart_safe(t["term"]) for t in reversed(terms)]; lifts=[t["engagement_lift"]*100 for t in reversed(terms)]
        cols=[BRANDING["secondary_color"] if l>0 else BRANDING["accent_color"] for l in lifts]
        ax.barh(names,lifts,color=cols,alpha=0.85); ax.axvline(x=0,color="#333",linewidth=0.8)
        ax.set_xlabel("Engagement Lift (%)"); ax.set_title(f"Caption Terms Impact — {PLATFORMS.get(p,{}).get('name',p)}"); plt.tight_layout()
        self.charts[f"{p}_nlp"] = self._b64(fig)

    def _ch_account_daily(self):
        plat_data = self.account_data.get("platforms",{})
        if not plat_data:
            daily = self.account_data.get("daily",pd.DataFrame())
            if not daily.empty: self._render_acct(daily,"instagram")
            return
        for plat,pdata in plat_data.items():
            d = pdata.get("daily",pd.DataFrame())
            if not d.empty: self._render_acct(d,plat)

    def _render_acct(self, daily, plat):
        pc = PLATFORMS.get(plat,{}).get("color","#666"); pn = PLATFORMS.get(plat,{}).get("name",plat)
        daily = self._clip_90d(daily)
        # Further trim: only show from first non-zero day to last non-zero day
        data_cols = [c for c in ["reach","views","follows","interactions","visits","follower_count"]
                     if c in daily.columns and daily[c].sum() > 0]
        if data_cols:
            has_data = daily[data_cols].ne(0).any(axis=1)
            if has_data.any():
                first_idx = has_data.idxmax()
                last_idx = has_data[::-1].idxmax()
                daily = daily.loc[first_idx:last_idx]
        dates = daily.index

        # Reach chart — only if reach has real data
        has_reach = "reach" in daily.columns and daily["reach"].sum() > 0
        has_views = "views" in daily.columns and daily["views"].sum() > 0

        if has_reach and has_views:
            fig,(a1,a2)=plt.subplots(2,1,figsize=(12,7),sharex=True)
            a1.fill_between(dates,daily["reach"],alpha=0.15,color=pc); a1.plot(dates,daily["reach"],color=pc,alpha=0.4,linewidth=0.8)
            if "reach_7d_avg" in daily.columns: a1.plot(dates,daily["reach_7d_avg"],color=pc,linewidth=2.5,label="7d avg")
            a1.set_ylabel("Reach"); a1.set_title(f"Daily Reach — {pn}"); a1.legend(loc="upper left")
            a2.fill_between(dates,daily["views"],alpha=0.15,color=BRANDING["secondary_color"]); a2.plot(dates,daily["views"],color=BRANDING["secondary_color"],alpha=0.4,linewidth=0.8)
            if "views_7d_avg" in daily.columns: a2.plot(dates,daily["views_7d_avg"],color=BRANDING["secondary_color"],linewidth=2.5,label="7d avg")
            a2.set_ylabel("Views"); a2.set_title(f"Daily Views — {pn}"); a2.legend(loc="upper left")
            plt.xticks(rotation=45); plt.tight_layout(); self.charts[f"{plat}_acct_rv"] = self._b64(fig)
        elif has_views:
            # Only views, no reach
            fig,ax=plt.subplots(figsize=(12,4))
            ax.fill_between(dates,daily["views"],alpha=0.15,color=BRANDING["secondary_color"]); ax.plot(dates,daily["views"],color=BRANDING["secondary_color"],alpha=0.4,linewidth=0.8)
            if "views_7d_avg" in daily.columns: ax.plot(dates,daily["views_7d_avg"],color=BRANDING["secondary_color"],linewidth=2.5,label="7d avg")
            ax.set_ylabel("Views"); ax.set_title(f"Daily Views — {pn}"); ax.legend(loc="upper left")
            plt.xticks(rotation=45); plt.tight_layout(); self.charts[f"{plat}_acct_rv"] = self._b64(fig)
        elif has_reach:
            fig,ax=plt.subplots(figsize=(12,4))
            ax.fill_between(dates,daily["reach"],alpha=0.15,color=pc); ax.plot(dates,daily["reach"],color=pc,alpha=0.4,linewidth=0.8)
            if "reach_7d_avg" in daily.columns: ax.plot(dates,daily["reach_7d_avg"],color=pc,linewidth=2.5,label="7d avg")
            ax.set_ylabel("Reach"); ax.set_title(f"Daily Reach — {pn}"); ax.legend(loc="upper left")
            plt.xticks(rotation=45); plt.tight_layout(); self.charts[f"{plat}_acct_rv"] = self._b64(fig)

        # Follows
        if "follows" in daily.columns and daily["follows"].sum()>0:
            fig,ax=plt.subplots(figsize=(12,4)); ax.bar(dates,daily["follows"],color=BRANDING["accent_color"],alpha=0.7,width=1.0)
            if "follows_7d_avg" in daily.columns: ax.plot(dates,daily["follows_7d_avg"],color=BRANDING["accent_color"],linewidth=2.5)
            ax.set_title(f"Daily New Followers — {pn}"); plt.xticks(rotation=45); plt.tight_layout()
            self.charts[f"{plat}_acct_fol"] = self._b64(fig)
        # Follower count
        if "follower_count" in daily.columns and daily["follower_count"].sum()>0:
            fig,ax=plt.subplots(figsize=(12,4)); ax.fill_between(dates,daily["follower_count"],alpha=0.2,color=pc)
            ax.plot(dates,daily["follower_count"],color=pc,linewidth=2.5); ax.set_title(f"Total Followers — {pn}"); plt.xticks(rotation=45); plt.tight_layout()
            self.charts[f"{plat}_fol_growth"] = self._b64(fig)

    def _ch_demographics(self):
        plat_data = self.account_data.get("platforms",{})
        if not plat_data:
            demo = self.account_data.get("demographics",{})
            if demo: self._render_demo(demo,"instagram")
            return
        for plat,pdata in plat_data.items():
            demo = pdata.get("demographics",{})
            if demo: self._render_demo(demo,plat)

    def _render_demo(self, demo, plat):
        pn = PLATFORMS.get(plat,{}).get("name",plat)
        countries = demo.get("countries",[])
        if countries:
            fig,ax=plt.subplots(figsize=(9,max(3,len(countries)*0.45)))
            ax.barh([c["country"] for c in reversed(countries)],[c["percentage"] for c in reversed(countries)],color=BRANDING["primary_color"],alpha=0.85)
            ax.set_xlabel("% of Audience"); ax.set_title(f"Countries — {pn}"); plt.tight_layout()
            self.charts[f"{plat}_demo_co"] = self._b64(fig)
        age_gender = demo.get("age_gender",[])
        if age_gender:
            fig,ax=plt.subplots(figsize=(9,max(3,len(age_gender)*0.7))); y=np.arange(len(age_gender)); h=0.35
            ax.barh(y-h/2,[a["men"] for a in age_gender],h,label="Men",color=BRANDING["primary_color"],alpha=0.85)
            ax.barh(y+h/2,[a["women"] for a in age_gender],h,label="Women",color=BRANDING["accent_color"],alpha=0.85)
            ax.set_yticks(y); ax.set_yticklabels([a["range"] for a in age_gender]); ax.set_xlabel("% of Audience"); ax.set_title(f"Age & Gender — {pn}"); ax.legend(); plt.tight_layout()
            self.charts[f"{plat}_demo_ag"] = self._b64(fig)
        gender = demo.get("gender",[])
        if gender and not age_gender:
            vals = [g["percentage"] for g in gender if g["percentage"]>0]; labs = [g["gender"] for g in gender if g["percentage"]>0]
            if vals:
                fig,ax=plt.subplots(figsize=(5,5)); ax.pie(vals,labels=labs,autopct="%1.1f%%",startangle=90,colors=[BRANDING["accent_color"],BRANDING["primary_color"],"#ccc"])
                ax.set_title(f"Gender — {pn}"); plt.tight_layout(); self.charts[f"{plat}_demo_ge"] = self._b64(fig)
        cities = demo.get("cities",[])
        if cities:
            fig,ax=plt.subplots(figsize=(9,max(3,len(cities)*0.45)))
            ax.barh([c["city"] for c in reversed(cities)],[c["percentage"] for c in reversed(cities)],color=BRANDING["secondary_color"],alpha=0.85)
            ax.set_xlabel("% of Audience"); ax.set_title(f"Cities — {pn}"); plt.tight_layout()
            self.charts[f"{plat}_demo_ci"] = self._b64(fig)

    def _ch_cross(self):
        platforms = self.df["platform"].unique()
        if len(platforms)<2: return
        ov = self.analysis.get("platform_overview",{})
        if not ov: return
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        pnames=[PLATFORMS.get(p,{}).get("name",p) for p in ov]; colors=[PLATFORMS.get(p,{}).get("color","#666") for p in ov]
        for ax,metric,title in [(axes[0],"total_reach","Total Reach"),(axes[1],"total_engagement","Total Engagement"),(axes[2],"total_posts","Total Posts")]:
            vals=[ov[p][metric] for p in ov]; ax.bar(pnames,vals,color=colors,alpha=0.85); ax.set_title(title)
            for i,v in enumerate(vals): ax.text(i,v,f"{v:,}",ha="center",va="bottom",fontsize=9)
        plt.suptitle("Cross-Platform Comparison",fontsize=14,fontweight="bold"); plt.tight_layout()
        self.charts["cross"] = self._b64(fig)

    @staticmethod
    def _b64(fig):
        # Watermark on every chart
        fig.text(0.99, 0.01, "cybersystema.com", fontsize=7, color="#cccccc",
                 ha="right", va="bottom", alpha=0.5, style="italic",
                 transform=fig.transFigure)
        buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _ci(self, key):
        return f'<img src="data:image/png;base64,{self.charts[key]}" class="chart-img">' if key in self.charts else ""

    def _clip_90d(self, dataframe):
        """Clip a time-indexed DataFrame to the last 90 days of actual data."""
        if dataframe.empty:
            return dataframe
        last_date = dataframe.index.max()
        cutoff = last_date - pd.Timedelta(days=90)
        return dataframe.loc[dataframe.index >= cutoff]

    def _get_followers_gained(self, platform, overview):
        """Get followers gained, falling back to account data if content-level is 0."""
        content_val = overview.get('total_followers_gained', 0)
        if content_val > 0:
            return content_val
        plat_data = self.account_data.get("platforms", {})
        acct = plat_data.get(platform, {})
        daily = acct.get("daily", pd.DataFrame())
        if not daily.empty and "follows" in daily.columns:
            return int(daily["follows"].sum())
        return content_val

    # ════════════════ HTML ════════════════

    def _build_html(self):
        now=datetime.now(); recs=self.analysis.get("recommendations",[])
        health=self.forecast.get("health",{}); meta=self.analysis.get("meta",{})

        # Load OKN logo (assets/ primary, reports/ fallback)
        logo_html = ""
        ASSETS_DIR = REPORTS_DIR.parent / "assets"
        for search_dir in [ASSETS_DIR, REPORTS_DIR]:
            for logo_name in ["okn_logo.png", "OKN_bg.png", "logo.png"]:
                logo_path = search_dir / logo_name
                if logo_path.exists():
                    import base64 as b64mod
                    with open(logo_path, "rb") as f:
                        logo_b64 = b64mod.b64encode(f.read()).decode("utf-8")
                    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" alt="OKN">'
                    break
            if logo_html:
                break

        # Load CyberSystema logo (assets/ primary, reports/ fallback)
        cs_logo_html = ""
        cs_logo_small = ""
        for search_dir in [ASSETS_DIR, REPORTS_DIR]:
            for cs_name in ["cybersystema_logo.png", "cybersystema-logo-2x.png"]:
                cs_path = search_dir / cs_name
                if cs_path.exists():
                    import base64 as b64mod
                    with open(cs_path, "rb") as f:
                        cs_b64 = b64mod.b64encode(f.read()).decode("utf-8")
                    cs_logo_html = f'<img src="data:image/png;base64,{cs_b64}" class="cs-logo" alt="CyberSystema">'
                    cs_logo_small = f'<img src="data:image/png;base64,{cs_b64}" class="cs-logo-sm" alt="CS">'
                    break
            if cs_logo_html:
                break

        platform_html = ""
        for plat in self.df["platform"].unique():
            platform_html += self._platform_section(plat)

        cross_html = self._cross_section()
        summary_html = self._build_summary()

        rec_html = ""
        for r in recs:
            icon={"high":"🔴","medium":"🟡","low":"🟢"}.get(r["priority"],"⚪")
            rec_html += f'<div class="rec-item rec-{r["priority"]}"><span class="rec-icon">{icon}</span><span class="rec-text"><strong>[{r.get("category","").title()}]</strong> {r["message"]}</span></div>'

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{BRANDING['report_title']}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:{BRANDING['font_family']};background:{BRANDING['bg_color']};color:{BRANDING['text_color']};line-height:1.6}}.container{{max-width:1100px;margin:0 auto;padding:24px}}
.header{{background:linear-gradient(160deg,{BRANDING['primary_color']} 0%,#1e4a6e 35%,#2a3a5c 60%,{BRANDING['accent_color']} 100%);color:white;padding:48px 40px 36px;border-radius:20px;margin-bottom:32px;text-align:center;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle at 30% 40%,rgba(196,149,58,0.08) 0%,transparent 50%);pointer-events:none}}
.header-logo-wrap{{display:inline-block;padding:6px;border-radius:50%;background:linear-gradient(135deg,rgba(196,149,58,0.6),rgba(255,255,255,0.2));margin-bottom:20px}}
.header-logo{{width:140px;height:140px;border-radius:50%;display:block;object-fit:cover;border:3px solid rgba(255,255,255,0.25)}}
.header h1{{font-size:28px;margin-bottom:6px;letter-spacing:0.5px;font-weight:700;text-shadow:0 2px 8px rgba(0,0,0,0.15)}}
.header .sub{{opacity:0.85;font-size:13px;margin-bottom:16px;letter-spacing:0.2px}}
.header-divider{{width:60px;height:2px;background:linear-gradient(90deg,transparent,{BRANDING['secondary_color']},transparent);margin:0 auto 16px;border-radius:1px}}
.health{{display:inline-block;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);padding:11px 26px;border-radius:30px;font-size:15px}}
.header-meta{{opacity:0.55;font-size:11px;margin-top:14px;letter-spacing:0.3px}}
.pblock{{margin-bottom:40px}}.phead{{background:white;padding:20px 28px;border-radius:12px 12px 0 0;border-bottom:3px solid;display:flex;align-items:center;gap:12px}}.phead h2{{font-size:22px;margin:0}}
.section{{background:white;border-radius:12px;padding:28px;margin-bottom:24px;border:1px solid #eee}}.section h2{{color:{BRANDING['primary_color']};font-size:20px;margin-bottom:20px;padding-bottom:12px;border-bottom:2px solid {BRANDING['secondary_color']}}}.section h3{{color:{BRANDING['primary_color']};font-size:16px;margin:20px 0 12px 0}}
.sub-s{{background:#fafafa;border-radius:10px;padding:20px;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}}.kpi{{background:#fafafa;padding:14px;border-radius:10px;text-align:center}}.kpi .v{{display:block;font-size:20px;font-weight:bold;color:{BRANDING['primary_color']}}}.kpi .l{{display:block;font-size:10px;color:#888;text-transform:uppercase}}
.chart-img{{width:100%;max-width:100%;height:auto;border-radius:8px;margin:12px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #eee}}th{{background:#f5f5f5;font-weight:600;color:{BRANDING['primary_color']}}}tr:hover{{background:#fafafa}}
a{{color:{BRANDING['primary_color']};text-decoration:none}}a:hover{{text-decoration:underline}}
.rec-item{{padding:14px 18px;border-radius:8px;margin-bottom:10px;display:flex;align-items:flex-start;gap:12px}}.rec-high{{background:#fff0f0;border-left:4px solid #e53e3e}}.rec-medium{{background:#fffbf0;border-left:4px solid #d69e2e}}.rec-low{{background:#f0fff0;border-left:4px solid #38a169}}.rec-icon{{font-size:18px;flex-shrink:0}}.rec-text{{font-size:14px}}
.ml{{display:inline-block;background:#e8f4f8;color:#1a5276;padding:3px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-left:6px}}
.dr{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0f0f0}}.dr .k{{font-weight:500}}.dr .val{{color:{BRANDING['primary_color']};font-weight:600}}
.no-data{{color:#999;font-style:italic;padding:20px;text-align:center}}.footer{{text-align:center;padding:24px;color:#999;font-size:12px}}
.cross{{background:linear-gradient(180deg,#f0f4f8,#fff);border:2px solid {BRANDING['primary_color']}20}}
.powered-badge{{display:inline-flex;align-items:center;gap:6px;margin-top:12px;padding:5px 14px;border-radius:20px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);font-size:10px;color:rgba(255,255,255,0.55);letter-spacing:0.5px}}
.powered-badge a{{color:rgba(255,255,255,0.75);text-decoration:none;display:inline-flex;align-items:center;gap:5px}}.powered-badge a:hover{{color:white;text-decoration:underline}}
.cs-logo-sm{{width:16px;height:16px;border-radius:3px;vertical-align:middle}}
.built-by{{text-align:center;padding:28px 20px;margin-bottom:8px;border-radius:12px;background:linear-gradient(135deg,#f8f6f0,#eef2f7);border:1px solid #e0e0e0}}
.cs-logo{{width:48px;height:48px;border-radius:10px;margin-bottom:8px}}
.built-by-label{{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}}.built-by-name{{font-size:16px;font-weight:600;color:{BRANDING['primary_color']}}}.built-by-name a{{color:{BRANDING['primary_color']};text-decoration:none}}.built-by-name a:hover{{text-decoration:underline}}.built-by-tagline{{font-size:12px;color:#888;margin-top:4px}}
.summary{{background:white;border-radius:16px;padding:32px;margin-bottom:32px;border:2px solid {BRANDING['secondary_color']}30}}
.summary h2{{color:{BRANDING['primary_color']};font-size:22px;margin-bottom:4px}}.summary-sub{{color:#888;font-size:13px;margin-bottom:20px}}
.summary-pulse{{font-size:16px;line-height:1.7;padding:16px 20px;border-radius:12px;margin-bottom:20px}}
.summary-pulse.good{{background:#f0faf0;border-left:4px solid #38a169}}.summary-pulse.ok{{background:#fffbf0;border-left:4px solid #d69e2e}}.summary-pulse.bad{{background:#fff5f5;border-left:4px solid #e53e3e}}
.summary-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:20px}}
.summary-card{{padding:18px;border-radius:12px;background:#fafbfc;border:1px solid #eee}}
.summary-card h3{{font-size:14px;color:{BRANDING['primary_color']};margin-bottom:10px}}.summary-card ul{{list-style:none;padding:0}}.summary-card li{{padding:6px 0;font-size:14px;color:#444;border-bottom:1px solid #f4f4f4}}.summary-card li:last-child{{border:none}}
.summary-actions{{background:linear-gradient(135deg,{BRANDING['primary_color']}08,{BRANDING['secondary_color']}10);padding:18px 20px;border-radius:12px;border:1px solid {BRANDING['primary_color']}15}}
.summary-actions h3{{font-size:14px;color:{BRANDING['primary_color']};margin-bottom:10px}}.summary-actions ol{{padding-left:20px;margin:0}}.summary-actions li{{padding:5px 0;font-size:14px;color:#333}}
</style></head><body><div class="container">
<div class="header">
{f'<div class="header-logo-wrap">{logo_html}</div>' if logo_html else ''}
<h1>{BRANDING['report_title']}</h1>
<div class="header-divider"></div>
<div class="sub">Generated: {now.strftime('%B %d, %Y at %H:%M')} KST by <a href="https://cybersystema.com" target="_blank" style="color:rgba(255,255,255,0.85);text-decoration:none;border-bottom:1px dotted rgba(255,255,255,0.4)">CyberSystema</a> &nbsp;•&nbsp; Data through: {meta.get('date_range',{}).get('latest','N/A')[:10]} &nbsp;•&nbsp; {meta.get('total_posts',0)} posts across {len(meta.get('platforms',[]))} platforms</div>
<div class="health">{health.get('emoji','📊')} Growth: <strong>{health.get('status','unknown').replace('_',' ').title()}</strong> — {health.get('message','Collecting data...')}</div>
<div class="header-meta">Active since December 2025 &nbsp;•&nbsp; TikTok since January 6, 2026 &nbsp;•&nbsp; All times in KST</div>
<div class="powered-badge">Powered by <a href="https://cybersystema.com" target="_blank">{cs_logo_small} CyberSystema</a></div>
</div>
{summary_html}
{platform_html}
{cross_html}
<div class="section"><h2>💡 Recommendations</h2>{rec_html if rec_html else '<p class="no-data">Need more data</p>'}</div>
<div class="built-by">{cs_logo_html}<div class="built-by-label">Analytics Infrastructure by</div><div class="built-by-name"><a href="https://cybersystema.com" target="_blank">CyberSystema</a></div><div class="built-by-tagline">Social media intelligence &amp; data engineering</div></div>
<div class="footer">{BRANDING['footer_text']}<br>Data: {str(meta.get('date_range',{}).get('earliest','N/A'))[:10]} → {str(meta.get('date_range',{}).get('latest','N/A'))[:10]}</div>
</div></body></html>"""

    # ──── EXECUTIVE SUMMARY ────

    def _build_summary(self):
        """Build a friendly, plain-language summary for all team members."""
        overview = self.analysis.get("platform_overview", {})
        content = self.analysis.get("content_performance", {})
        temporal = self.analysis.get("temporal", {})
        growth = self.analysis.get("growth", {})
        anomalies = self.analysis.get("anomalies", {})
        recs = self.analysis.get("recommendations", [])

        # ── 1. HEALTH PULSE ──
        # Determine overall mood from multiple signals
        health = self.forecast.get("health", {})
        trajectory = growth.get("trajectory", {})

        signals_good = 0
        signals_bad = 0
        for plat, ov in overview.items():
            vs = ov.get("vs_benchmark", 0)
            if vs > 0: signals_good += 1
            elif vs < -0.02: signals_bad += 1

        pg = growth.get("platform_growth", {})
        for plat, g in pg.items():
            if g.get("reach_trend") == "growing": signals_good += 1
            elif g.get("reach_trend") == "declining": signals_bad += 1

        # Check momentum scores
        for plat in overview:
            ms = self.ml_results.get(plat, {}).get("momentum_score", {})
            if ms.get("status") == "ok":
                score = ms.get("total_score", 50)
                if score >= 60: signals_good += 1
                elif score < 35: signals_bad += 1

        if signals_bad == 0 and signals_good >= 2:
            pulse_class = "good"
            pulse_text = "Things are looking great! Our content is connecting with the audience and engagement is healthy across platforms."
        elif signals_bad >= 2:
            pulse_class = "bad"
            pulse_text = "There are some areas that need our attention. Engagement or reach is dropping on one or more platforms — let's adjust our approach."
        else:
            pulse_class = "ok"
            pulse_text = "Our social media presence is steady. There's room to grow — the data shows a few opportunities we can take advantage of."

        # Add specific context
        viral_posts = anomalies.get("viral", [])
        if viral_posts:
            top_viral = viral_posts[0]
            pulse_text += f" We had a standout post that performed {top_viral['multiplier']}x above average!"

        pulse_html = f'<div class="summary-pulse {pulse_class}">{pulse_text}</div>'

        # ── 2. KEY NUMBERS ──
        numbers_html = ""
        total_reach = sum(ov.get("total_reach", 0) for ov in overview.values())
        total_engagement = sum(ov.get("total_engagement", 0) for ov in overview.values())
        total_followers = sum(ov.get("total_followers_gained", 0) for ov in overview.values())
        total_posts = sum(ov.get("total_posts", 0) for ov in overview.values())

        numbers = []
        numbers.append(f"📊 We've published <strong>{total_posts:,}</strong> posts in total")
        numbers.append(f"👀 Our content reached <strong>{total_reach:,}</strong> people")
        numbers.append(f"💬 We received <strong>{total_engagement:,}</strong> interactions (likes, comments, shares, saves)")

        if total_followers > 0:
            numbers.append(f"👥 We gained <strong>{total_followers:,}</strong> new followers across all platforms")

        # Per-platform highlights
        for plat, ov in overview.items():
            pname = PLATFORMS.get(plat, {}).get("name", plat)
            wow = ov.get("wow_reach_change")
            if wow is not None and wow > 0.1:
                numbers.append(f"📈 {pname} reach grew <strong>{wow:.0%}</strong> compared to last week")
            elif wow is not None and wow < -0.1:
                numbers.append(f"📉 {pname} reach dropped <strong>{abs(wow):.0%}</strong> compared to last week")

        numbers_html = "<ul>" + "".join(f"<li>{n}</li>" for n in numbers) + "</ul>"

        # ── 3. WHAT'S WORKING ──
        working = []
        platform_rankings = content.get("_platform_rankings", {})
        for plat, rankings in platform_rankings.items():
            if rankings:
                best = rankings[0]
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                ctype = best["type"].replace("_", " ").title()
                working.append(f"<strong>{ctype}</strong> posts on {pname} get the best engagement ({best['count']} posts analyzed)")

        best_hours = temporal.get("best_hours_overall", {})
        if best_hours:
            top_hour = list(best_hours.keys())[0]
            top_data = best_hours[top_hour]
            if top_data.get("post_count", 0) >= 3:
                working.append(f"Posting around <strong>{int(top_hour)}:00 KST</strong> gets the most engagement")

        best_days = temporal.get("best_days_overall", {})
        if best_days:
            top_day = list(best_days.keys())[0]
            top_day_data = best_days[top_day]
            if top_day_data.get("post_count", 0) >= 3:
                working.append(f"<strong>{top_day}</strong> is our best performing day")

        # Caption insights from ML
        for plat in overview:
            ml = self.ml_results.get(plat, {})
            drivers = ml.get("engagement_drivers", {})
            multi = drivers.get("multilingual", {})
            if multi.get("lift", 0) > 0.1:
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                working.append(f"Multilingual captions (Korean + English) boost engagement on {pname}")
            emoji = drivers.get("emoji", {})
            if emoji.get("lift", 0) > 0.1:
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                working.append(f"Using emojis increases engagement on {pname}")

        if viral_posts:
            for v in viral_posts[:2]:
                pname = PLATFORMS.get(v["platform"], {}).get("name", v["platform"])
                title = str(v.get("title", ""))[:50]
                if title:
                    working.append(f'Our post "{_safe(title)}" went viral on {pname} ({v["multiplier"]}x above average)')

        working_html = "<ul>" + "".join(f"<li>{w}</li>" for w in working[:6]) + "</ul>" if working else '<p style="color:#888">Not enough data yet — keep posting!</p>'

        # ── 4. WHAT NEEDS ATTENTION ──
        attention = []
        for plat, g in pg.items():
            pname = PLATFORMS.get(plat, {}).get("name", plat)
            if g.get("reach_trend") == "declining":
                attention.append(f"{pname} reach has been declining recently — try new content formats or increase posting frequency")

        # Content fatigue
        for plat in overview:
            ml = self.ml_results.get(plat, {})
            fatigue = ml.get("content_fatigue", {})
            fatigued = fatigue.get("fatigued_types", [])
            for ft in fatigued:
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                ctype = ft["content_type"].replace("_", " ").title()
                attention.append(f"Our audience may be getting tired of <strong>{ctype}</strong> posts on {pname} (engagement is dropping)")

        engagement_data = self.analysis.get("engagement", {})
        if engagement_data.get("conversation_ratio", 1) < 0.05:
            attention.append("We're getting lots of likes but very few comments — try asking questions or creating discussion-worthy content")

        for plat, ov in overview.items():
            if ov.get("vs_benchmark", 0) < -0.02:
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                attention.append(f"{pname} engagement is below the industry average — our content strategy on this platform needs a refresh")

        attention_html = "<ul>" + "".join(f"<li>{a}</li>" for a in attention[:5]) + "</ul>" if attention else '<p style="color:#888">Nothing urgent — keep up the good work!</p>'

        # ── 5. THIS WEEK'S ACTIONS ──
        actions = []

        # Best content type to create
        for plat, rankings in platform_rankings.items():
            if rankings:
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                best_type = rankings[0]["type"].replace("_", " ").title()
                actions.append(f"Create more <strong>{best_type}</strong> content on {pname} — it's your top performer")
                break

        # Best time to post
        if best_hours:
            top_hour = list(best_hours.keys())[0]
            actions.append(f"Schedule your most important post for <strong>{int(top_hour)}:00 KST</strong>")

        # Cadence recommendation
        for plat in overview:
            ml = self.ml_results.get(plat, {})
            cadence = ml.get("posting_cadence", {})
            if cadence.get("status") == "ok":
                opt = cadence.get("optimal_for_engagement", {}).get("posts_per_week", 0)
                current = cadence.get("current_cadence", 0)
                pname = PLATFORMS.get(plat, {}).get("name", plat)
                if opt > current + 1:
                    actions.append(f"Try posting <strong>{opt} times per week</strong> on {pname} (currently ~{current:.0f})")
                break

        # Content variety
        if attention:
            for plat in overview:
                ml = self.ml_results.get(plat, {})
                fatigue = ml.get("content_fatigue", {})
                growing = fatigue.get("growing_types", [])
                if growing:
                    pname = PLATFORMS.get(plat, {}).get("name", plat)
                    gtype = growing[0]["content_type"].replace("_", " ").title()
                    actions.append(f"<strong>{gtype}</strong> content is gaining momentum on {pname} — invest more in this format")
                    break

        if not actions:
            actions.append("Keep posting consistently — your data is still building and the models will improve each week")

        actions_html = "<ol>" + "".join(f"<li>{a}</li>" for a in actions[:4]) + "</ol>"

        # ── ASSEMBLE ──
        return f"""<div class="summary">
<h2>📋 Weekly Summary</h2>
<div class="summary-sub">Here's what you need to know — no data science required</div>
{pulse_html}
<div class="summary-grid">
<div class="summary-card"><h3>📊 Key Numbers</h3>{numbers_html}</div>
<div class="summary-card"><h3>✅ What's Working</h3>{working_html}</div>
<div class="summary-card"><h3>⚠️ What Needs Attention</h3>{attention_html}</div>
<div class="summary-card summary-actions"><h3>🎯 This Week's Actions</h3>{actions_html}</div>
</div>
</div>"""

    # ──── PER-PLATFORM SECTION ────

    def _platform_section(self, plat):
        pi = PLATFORMS.get(plat,{}); pn=pi.get("name",plat.title()); pc=pi.get("color","#666"); pico=pi.get("icon","📊")
        ov = self.analysis.get("platform_overview",{}).get(plat,{})
        pdf = self.df[self.df["platform"]==plat]
        ml = self.ml_results.get(plat,{})
        tp = [p for p in self.scores.get("top_posts",[]) if p["platform"]==plat]
        viral = [v for v in self.analysis.get("anomalies",{}).get("viral",[]) if v["platform"]==plat]

        # KPIs
        wow=ov.get("wow_reach_change"); ws=f"{wow:+.1%}" if wow is not None else "N/A"
        kpis = f"""<div class="kpis">
        <div class="kpi"><span class="v">{ov.get('total_posts',0)}</span><span class="l">Posts</span></div>
        <div class="kpi"><span class="v">{ov.get('total_reach',0):,}</span><span class="l">Total Reach</span></div>
        <div class="kpi"><span class="v">{ov.get('total_engagement',0):,}</span><span class="l">Engagement</span></div>
        <div class="kpi"><span class="v">{ov.get('avg_engagement_rate',0):.1%}</span><span class="l">Avg Eng Rate</span></div>
        <div class="kpi"><span class="v">{self._get_followers_gained(plat, ov):,}</span><span class="l">Followers Gained</span></div>
        <div class="kpi"><span class="v">{ws}</span><span class="l">WoW Reach</span></div></div>"""

        # Viral
        viral_html=""
        if viral:
            viral_html='<h3>🔥 Viral Content</h3><table><tr><th>Post</th><th>Engagement</th><th>vs Avg</th></tr>'
            for v in viral[:5]:
                t=str(v.get('title','') or '')[:60]; lk=v.get("permalink",""); cell=f'<a href="{lk}" target="_blank">{t}</a>' if lk else t
                viral_html+=f'<tr><td>{cell}</td><td>{v["engagement"]:,}</td><td><strong>{v["multiplier"]}x</strong></td></tr>'
            viral_html+="</table>"

        # Top posts
        top_html=""
        if tp:
            top_html='<h3>🏆 Top Scored Posts</h3><table><tr><th>#</th><th>Post</th><th>Score</th><th>Grade</th></tr>'
            for p2 in tp[:5]:
                t=str(p2.get('title','') or '')[:50]; lk=p2.get("permalink",""); cell=f'<a href="{lk}" target="_blank">{t}</a>' if lk else t
                top_html+=f'<tr><td>#{p2["rank"]}</td><td>{cell}</td><td><strong>{p2["score"]}</strong></td><td>{p2["grade"]}</td></tr>'
            top_html+="</table>"

        ml_html = self._ml_section(plat, ml, pn)
        acct_html = self._acct_section(plat)

        return f"""<div class="pblock">
<div class="phead" style="border-color:{pc}"><span style="font-size:28px">{pico}</span><h2>{pn}</h2>
<span style="color:#999;font-size:13px;margin-left:auto">{len(pdf)} posts</span></div>
<div class="section" style="border-radius:0 0 12px 12px;border-top:none">
{kpis}
<h3>🎬 Content Performance</h3>{self._ci(f'{plat}_content')}
<h3>💬 Engagement Mix</h3>{self._ci(f'{plat}_eng_pie')}
<h3>📈 Weekly Trends</h3>{self._ci(f'{plat}_weekly')}
{viral_html}{top_html}
</div>{ml_html}{acct_html}</div>"""

    # ──── ML SECTION ────

    def _ml_section(self, plat, ml, pn):
        if ml.get("status")!="ok": return ""
        parts=[]

        # Feature Importance
        fi=ml.get("feature_importance",{})
        if fi.get("status")=="ok":
            rows="".join(f'<div class="dr"><span class="k">{f["feature"]}</span><span class="val">{f["pct"]:.1f}%</span></div>' for f in fi.get("top_features",[])[:6])
            parts.append(f'<div class="sub-s"><h3>📊 What Drives Engagement <span class="ml">GradientBoosting</span></h3>{self._ci(f"{plat}_feat_imp")}{rows}</div>')

        # Neural Network
        nn=ml.get("engagement_prediction",{})
        if nn.get("status")=="trained":
            r2=nn["r2_score"]; cv=nn.get("cv_r2_score"); cv_t=f" · CV R²: {cv:.3f}" if cv else ""
            over_rows = ""
            for o in nn.get("overperformers",[])[:3]:
                link = o.get("permalink","")
                title = _safe(str(o.get('title','') or '')[:50])
                cell = f'<a href="{link}" target="_blank">{title}</a>' if link else title
                over_rows += f'<tr><td>{cell}</td><td>{o["actual"]:.1%}</td><td>{o["predicted"]:.1%}</td><td style="color:green">+{o["surplus"]:.1%}</td></tr>'
            under_rows = ""
            for u in nn.get("underperformers",[])[:3]:
                link = u.get("permalink","")
                title = _safe(str(u.get('title','') or '')[:50])
                cell = f'<a href="{link}" target="_blank">{title}</a>' if link else title
                under_rows += f'<tr><td>{cell}</td><td>{u["actual"]:.1%}</td><td>{u["predicted"]:.1%}</td><td style="color:red">{u["deficit"]:.1%}</td></tr>'
            parts.append(f"""<div class="sub-s"><h3>🧠 Neural Network Predictor <span class="ml">MLP (32→16→8)</span></h3>
<p>Model: <strong>R² = {r2:.3f}</strong>{cv_t} — {nn.get("interpretation","")}</p>
<p>Posts that <strong>outperformed</strong> the model's prediction:</p>
<table><tr><th>Post</th><th>Actual</th><th>Predicted</th><th>Surplus</th></tr>{over_rows}</table>
<p style="margin-top:16px">Posts that <strong>underperformed</strong> their predicted potential:</p>
<table><tr><th>Post</th><th>Actual</th><th>Predicted</th><th>Deficit</th></tr>{under_rows}</table></div>""")

        # Clusters
        cl=ml.get("content_clusters",{})
        if cl.get("status")=="ok":
            cl_html = ""
            for c in cl.get("clusters", []):
                label = c.get("label", "Cluster")
                cl_html += f'<div style="margin:8px 0;padding:12px;background:white;border-radius:8px"><strong>{label}</strong> — {c["size"]} posts, {c["avg_engagement_rate"]:.1%} avg engagement</div>'
            parts.append(f'<div class="sub-s"><h3>🎯 Content Clusters <span class="ml">KMeans</span></h3>{self._ci(f"{plat}_clusters")}{cl_html}</div>')

        # NLP
        nlp=ml.get("caption_analysis",{})
        if nlp.get("status")=="ok":
            terms=nlp.get("top_engagement_terms",[])[:6]
            t_html = ""
            for t in terms:
                color = "green" if t["engagement_lift"] > 0 else "red"
                t_html += f'<div class="dr"><span class="k">&quot;{_safe(t["term"])}&quot; ({t["posts_with_term"]} posts)</span><span class="val" style="color:{color}">{t["engagement_lift"]:+.0%} lift</span></div>'
            parts.append(f'<div class="sub-s"><h3>📝 Caption &amp; Hashtag Analysis <span class="ml">TF-IDF NLP</span></h3>{self._ci(f"{plat}_nlp")}{t_html}</div>')

        # Drivers
        drivers=ml.get("engagement_drivers",{})
        if drivers:
            d_html=""
            for k,d in drivers.items():
                if k=="caption_length": d_html+=f'<div class="dr"><span class="k">Caption Length</span><span class="val">{d["direction"]} (optimal: {d["optimal_range"]})</span></div>'
                elif k=="hashtags":
                    opt = d["optimal_count"]
                    opt_text = f"~{opt} per post" if opt > 0 else "minimal or none"
                    d_html+=f'<div class="dr"><span class="k">Hashtags</span><span class="val">{d["direction"]} (optimal: {opt_text})</span></div>'
                elif k=="multilingual": d_html+=f'<div class="dr"><span class="k">Multilingual</span><span class="val">{d["recommendation"]} ({d["lift"]:+.0%})</span></div>'
                elif k=="emoji": d_html+=f'<div class="dr"><span class="k">Emoji</span><span class="val">{"Helps" if d.get("lift",0)>0.05 else "No effect"} ({d.get("lift",0):+.0%})</span></div>'
                elif k=="weekend_effect":
                    we_label = f'{d["better"].title()} is better' if d["better"] != "no difference" else "No significant difference"
                    d_html+=f'<div class="dr"><span class="k">Weekend vs Weekday</span><span class="val">{we_label} (WE:{d["weekend_avg"]:.1%} vs WD:{d["weekday_avg"]:.1%})</span></div>'
            if d_html: parts.append(f'<div class="sub-s"><h3>🔬 Engagement Drivers <span class="ml">Statistical</span></h3>{d_html}</div>')

        # Content Fatigue
        fatigue = ml.get("content_fatigue", {})
        if fatigue.get("status") == "ok":
            fatigued = fatigue.get("fatigued_types", [])
            growing = fatigue.get("growing_types", [])
            f_html = ""
            for ft in fatigue.get("content_types", []):
                icon = "📉" if ft["trend"] == "declining" else "📈" if ft["trend"] == "growing" else "➡️"
                color = "red" if ft["trend"] == "declining" else "green" if ft["trend"] == "growing" else "#888"
                f_html += f'<div class="dr"><span class="k">{icon} {ft["content_type"]} ({ft["post_count"]} posts)</span><span class="val" style="color:{color}">{ft["change_pct"]:+.1f}% recent vs older</span></div>'
            alert = ""
            if fatigued:
                types = ", ".join(f["content_type"] for f in fatigued)
                alert = f'<p style="color:red;font-weight:500">⚠️ Audience fatigue detected for: {types}. Consider reducing frequency or refreshing the format.</p>'
            if growing:
                types = ", ".join(g["content_type"] for g in growing)
                alert += f'<p style="color:green;font-weight:500">🚀 Growing engagement for: {types}. Double down on these.</p>'
            parts.append(f'<div class="sub-s"><h3>🔄 Content Fatigue Detector <span class="ml">Trend Regression</span></h3>{alert}{f_html}</div>')

        # Posting Cadence
        cadence = ml.get("posting_cadence", {})
        if cadence.get("status") == "ok":
            opt_eng = cadence.get("optimal_for_engagement", {})
            opt_reach = cadence.get("optimal_for_reach", {})
            c_html = f'<div class="dr"><span class="k">Current pace</span><span class="val">~{cadence["current_cadence"]:.0f} posts/week</span></div>'
            c_html += f'<div class="dr"><span class="k">Optimal for engagement</span><span class="val">{opt_eng.get("posts_per_week",0)} posts/week ({opt_eng.get("avg_engagement_rate",0):.1%} avg)</span></div>'
            c_html += f'<div class="dr"><span class="k">Optimal for reach</span><span class="val">{opt_reach.get("posts_per_week",0)} posts/week ({opt_reach.get("avg_total_reach",0):,} avg reach)</span></div>'
            rec = cadence.get("recommendation", "")
            c_html += f'<p style="margin-top:8px"><strong>→</strong> {_safe(rec)}</p>'
            parts.append(f'<div class="sub-s"><h3>⏱️ Optimal Posting Cadence <span class="ml">Cadence Analysis</span></h3>{c_html}</div>')

        # Momentum Score
        momentum = ml.get("momentum_score", {})
        if momentum.get("status") == "ok":
            score = momentum["total_score"]
            verdict = momentum["verdict"]
            bd = momentum.get("breakdown", {})
            # Color based on score
            if score >= 60:
                score_color = "green"
            elif score >= 35:
                score_color = BRANDING["secondary_color"]
            else:
                score_color = "red"
            m_html = f'<div style="text-align:center;margin:16px 0"><span style="font-size:48px;font-weight:bold;color:{score_color}">{score:.0f}</span><span style="font-size:16px;color:#888">/100</span></div>'
            m_html += f'<p style="text-align:center;color:#666;margin-bottom:16px">{_safe(verdict)}</p>'
            m_html += f'<div class="dr"><span class="k">Engagement Trend</span><span class="val">{bd.get("engagement_trend",0):.0f}/25</span></div>'
            m_html += f'<div class="dr"><span class="k">Posting Consistency</span><span class="val">{bd.get("posting_consistency",0):.0f}/25</span></div>'
            m_html += f'<div class="dr"><span class="k">Reach Growth</span><span class="val">{bd.get("reach_growth",0):.0f}/25</span></div>'
            m_html += f'<div class="dr"><span class="k">Content Quality</span><span class="val">{bd.get("content_quality",0):.0f}/25</span></div>'
            parts.append(f'<div class="sub-s"><h3>🚀 Audience Momentum Score <span class="ml">Composite ML</span></h3>{m_html}</div>')

        # Root Cause Analysis
        rca = ml.get("root_cause", {})
        if rca.get("status") == "ok":
            rca_html = ""
            viral_exp = rca.get("viral_explanations", [])
            if viral_exp:
                rca_html += '<p><strong>Why these posts went viral:</strong></p>'
                for v in viral_exp[:2]:
                    title = _safe(str(v.get('title','') or '')[:50])
                    link = v.get("permalink", "")
                    cell = f'<a href="{link}" target="_blank">{title}</a>' if link else title
                    rca_html += f'<div style="margin:8px 0;padding:12px;background:white;border-radius:8px"><strong>{cell}</strong> (actual: {v["actual_rate"]:.1%}, predicted: {v["predicted_rate"]:.1%})<br>'
                    for w in v.get("why", [])[:3]:
                        arrow = "↑" if w["direction"] == "positive" else "↓"
                        color = "green" if w["direction"] == "positive" else "red"
                        rca_html += f'<span style="color:{color};margin-right:12px">{arrow} {w["feature"]}: {w["contribution_pct"]:+.1f}%</span>'
                    rca_html += '</div>'

            flop_exp = rca.get("flop_explanations", [])
            if flop_exp:
                rca_html += '<p style="margin-top:12px"><strong>Why these posts underperformed:</strong></p>'
                for v in flop_exp[:2]:
                    title = _safe(str(v.get('title','') or '')[:50])
                    link = v.get("permalink", "")
                    cell = f'<a href="{link}" target="_blank">{title}</a>' if link else title
                    rca_html += f'<div style="margin:8px 0;padding:12px;background:white;border-radius:8px"><strong>{cell}</strong> (actual: {v["actual_rate"]:.1%}, predicted: {v["predicted_rate"]:.1%})<br>'
                    for w in v.get("why", [])[:3]:
                        arrow = "↑" if w["direction"] == "positive" else "↓"
                        color = "green" if w["direction"] == "positive" else "red"
                        rca_html += f'<span style="color:{color};margin-right:12px">{arrow} {w["feature"]}: {w["contribution_pct"]:+.1f}%</span>'
                    rca_html += '</div>'

            if rca_html:
                parts.append(f'<div class="sub-s"><h3>🔍 Root Cause Analysis <span class="ml">Feature Attribution</span></h3>{rca_html}</div>')

        # Topic Discovery
        topics = ml.get("topic_discovery", {})
        if topics.get("status") == "ok":
            t_html = ""
            for t in topics.get("topics", [])[:6]:
                kw = ", ".join(t.get("keywords", [])[:4]) or "—"
                eng_pct = t["avg_engagement_rate"]
                bar_w = min(100, max(5, int(eng_pct * 500)))
                t_html += f'<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px"><span><strong>{_safe(t.get("representative_post","")[:45])}</strong> ({t["post_count"]} posts)</span><span style="color:{BRANDING["primary_color"]};font-weight:600">{eng_pct:.1%}</span></div>'
                t_html += f'<div style="background:#eee;border-radius:4px;height:8px"><div style="width:{bar_w}%;height:100%;background:{BRANDING["secondary_color"]};border-radius:4px"></div></div>'
                t_html += f'<div style="font-size:11px;color:#888;margin-top:2px">Keywords: {_safe(kw)}</div></div>'
            parts.append(f'<div class="sub-s"><h3>📑 Content Topic Discovery <span class="ml">Semantic Clustering</span></h3><p style="color:#666;font-size:13px;margin-bottom:12px">Posts grouped by meaning — works across English, Korean &amp; Greek</p>{t_html}</div>')

        # Similar Post Predictor
        similar = ml.get("similar_posts", {})
        if similar.get("status") == "ok":
            s_html = f'<div class="dr"><span class="k">Prediction accuracy (MAE)</span><span class="val">{similar["mean_absolute_error"]:.2%}</span></div>'
            s_html += f'<div class="dr"><span class="k">Median content similarity</span><span class="val">{similar["median_similarity"]:.0%}</span></div>'
            recent = similar.get("recent_predictions", [])
            if recent:
                s_html += '<p style="margin-top:14px;font-weight:600;font-size:13px">Recent posts — predicted vs actual:</p>'
                for rp in recent[:4]:
                    actual = rp["actual_rate"]
                    predicted = rp["predicted_rate"]
                    diff = actual - predicted
                    diff_color = "green" if diff > 0 else "red" if diff < -0.02 else "#888"
                    sim_titles = ", ".join(f'"{_safe(t[:25])}"' for t in rp.get("similar_to", [])[:2])
                    s_html += f'<div style="margin:8px 0;padding:10px;background:white;border-radius:8px;font-size:13px">'
                    s_html += f'<strong>{_safe(rp["title"][:45])}</strong><br>'
                    s_html += f'Predicted: {predicted:.1%} → Actual: {actual:.1%} <span style="color:{diff_color}">({diff:+.1%})</span><br>'
                    s_html += f'<span style="color:#999;font-size:11px">Similar to: {sim_titles} ({rp["similarity"]:.0%} match)</span></div>'
            parts.append(f'<div class="sub-s"><h3>🔮 Similar Post Predictor <span class="ml">Semantic Similarity</span></h3>{s_html}</div>')

        # Hashtag Clusters
        htclusters = ml.get("hashtag_clusters", {})
        if htclusters.get("status") == "ok":
            h_html = f'<div class="dr"><span class="k">Unique hashtags</span><span class="val">{htclusters["total_unique_hashtags"]}</span></div>'
            h_html += f'<div class="dr"><span class="k">Semantic clusters</span><span class="val">{htclusters["n_clusters"]}</span></div>'
            for cl in htclusters.get("clusters", [])[:5]:
                tags = " ".join(cl.get("top_hashtags", [])[:4])
                eng = cl["avg_engagement_rate"]
                h_html += f'<div style="margin:8px 0;padding:10px;background:white;border-radius:8px">'
                h_html += f'<div style="display:flex;justify-content:space-between;align-items:center"><span style="font-size:13px"><strong>{_safe(cl.get("label",""))}</strong> ({cl["total_posts"]} posts)</span><span style="color:{BRANDING["primary_color"]};font-weight:600">{eng:.1%}</span></div>'
                h_html += f'<div style="font-size:12px;color:#666;margin-top:4px">{_safe(tags)}</div></div>'
            # Top individual hashtags
            top_ht = htclusters.get("top_individual_hashtags", [])
            if top_ht:
                h_html += '<p style="margin-top:12px;font-weight:600;font-size:13px">Top performing hashtags:</p>'
                for ht in top_ht[:6]:
                    h_html += f'<div class="dr"><span class="k">{_safe(ht["tag"])} ({ht["post_count"]} posts)</span><span class="val">{ht["avg_engagement"]:.1%}</span></div>'
            parts.append(f'<div class="sub-s"><h3>#️⃣ Hashtag Strategy <span class="ml">Semantic Clustering</span></h3>{h_html}</div>')

        if not parts: return ""
        return f'<div class="section"><h2>🤖 ML & AI Insights — {pn}</h2>{"".join(parts)}</div>'

    # ──── ACCOUNT SECTION ────

    def _acct_section(self, plat):
        plat_data = self.account_data.get("platforms",{})
        acct = plat_data.get(plat,{}) if plat_data else {}
        if not acct and plat=="instagram":
            acct={"daily":self.account_data.get("daily",pd.DataFrame()),"demographics":self.account_data.get("demographics",{})}
        daily=acct.get("daily",pd.DataFrame()); demo=acct.get("demographics",{})
        if daily.empty and not demo: return ""
        pn=PLATFORMS.get(plat,{}).get("name",plat.title()); parts=[]
        if not daily.empty:
            days=len(self._clip_90d(daily)); kpi_items=""
            for col,lbl in [("reach","Reach"),("views","Views"),("follows","New Followers"),("interactions","Interactions"),("visits","Profile Visits"),("follower_count","Final Followers")]:
                if col in daily.columns and daily[col].sum()>0:
                    if col == "follower_count":
                        nonzero = daily[col][daily[col] > 0]
                        val = f'{int(nonzero.iloc[-1]):,}' if not nonzero.empty else '0'
                    else:
                        val = f'{int(daily[col].sum()):,}'
                    kpi_items+=f'<div class="kpi"><span class="v">{val}</span><span class="l">{lbl}</span></div>'
            parts.append(f'<h3>📈 Account Metrics (last 90 days)</h3><div class="kpis">{kpi_items}</div>{self._ci(f"{plat}_acct_rv")}{self._ci(f"{plat}_acct_fol")}{self._ci(f"{plat}_fol_growth")}')
        if demo:
            peak=demo.get("peak_hours",[])
            if peak:
                peak_parts = [str(p2.get("hour","?")) + ":00 (" + str(p2.get("avg_active",0)) + " active)" for p2 in peak]
                peak_t='<p><strong>🕐 Peak hours:</strong> ' + ", ".join(peak_parts) + '</p>'
            else:
                peak_t=""
            parts.append(f'<h3>👥 Audience</h3>{self._ci(f"{plat}_demo_co")}{self._ci(f"{plat}_demo_ag")}{self._ci(f"{plat}_demo_ge")}{self._ci(f"{plat}_demo_ci")}{peak_t}')
        if not parts: return ""
        return f'<div class="section"><h2>📊 Account Data — {pn}</h2>{"".join(parts)}</div>'

    # ──── CROSS-PLATFORM ────

    def _cross_section(self):
        if len(self.df["platform"].unique())<2: return ""
        ov=self.analysis.get("platform_overview",{})
        rows = ""
        for p, d in ov.items():
            pn = PLATFORMS.get(p, {}).get("name", p)
            pico = PLATFORMS.get(p, {}).get("icon", "")
            vs = d.get("vs_benchmark", 0)
            vs_color = "green" if vs > 0 else "red"
            rows += f'<tr><td>{pico} {pn}</td><td>{d.get("total_posts",0)}</td><td>{d.get("total_reach",0):,}</td><td>{d.get("total_engagement",0):,}</td><td>{d.get("avg_engagement_rate",0):.1%}</td><td>{d.get("benchmark_engagement",0):.1%}</td><td style="color:{vs_color}">{vs:+.1%}</td></tr>'
        return f"""<div class="section cross"><h2>🔄 Cross-Platform Comparison</h2>
<p style="color:#888;font-size:13px;margin-bottom:16px">⚠️ <strong>Note:</strong> Instagram rate = interactions ÷ reach. TikTok rate = interactions ÷ views. Different denominators — not directly comparable.</p>
{self._ci("cross")}
<table><tr><th>Platform</th><th>Posts</th><th>Reach</th><th>Engagement</th><th>Avg Rate</th><th>Benchmark</th><th>vs Bench</th></tr>{rows}</table></div>"""

    def _summary(self):
        return {"generated_at":datetime.now().isoformat(),"meta":self.analysis.get("meta",{}),"health":self.forecast.get("health",{}),"recommendations":self.analysis.get("recommendations",[])}


def generate_report(df, analysis, scores=None, timing=None, forecast=None, account_data=None, ml_results=None):
    gen = ReportGenerator(df, analysis, scores, timing, forecast, account_data, ml_results)
    return gen.generate()
