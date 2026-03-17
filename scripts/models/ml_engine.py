"""
OKN Analytics Pipeline — ML & Neural Network Engine
====================================================
Machine learning models for deeper social media intelligence.

Models:
1. Engagement Predictor    — MLP Neural Network predicts engagement for new posts
2. Content Clustering      — KMeans groups similar-performing content
3. Feature Importance      — GradientBoosting identifies what drives engagement
4. Caption NLP             — TF-IDF finds high-engagement words/topics
5. Anomaly Detection       — IsolationForest finds statistically unusual posts
6. Time Series Decompose   — Separates trend, seasonality, and residual
7. Growth Trajectory       — Polynomial regression on follower growth
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, List, Optional

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

from config import compute_recency_weights

logger = logging.getLogger("okn.ml")


class MLEngine:
    """
    Runs ML models on a SINGLE platform's data.
    Always instantiate per-platform to avoid methodology mixing.
    """

    MIN_POSTS_FOR_ML = 10
    MIN_POSTS_FOR_NN = 15

    def __init__(self, df: pd.DataFrame, platform: str):
        self.df = df.copy()
        self.platform = platform
        self.results: Dict[str, Any] = {}
        self._extract_features()

    def _extract_features(self):
        """Build feature matrix from post data."""
        df = self.df

        # Time features
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["hour"] = df["published_at"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["published_at"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Caption features
        df["caption_length"] = df["title"].fillna("").str.len()
        df["hashtag_count"] = df["title"].fillna("").apply(
            lambda x: len(re.findall(r"#\w+", str(x)))
        )
        df["has_emoji"] = df["title"].fillna("").apply(
            lambda x: 1 if re.search(r"[\U0001F600-\U0001F9FF\U00002600-\U000027BF\U0001F300-\U0001F5FF]", str(x)) else 0
        )
        df["is_multilingual"] = df["title"].fillna("").apply(
            lambda x: 1 if (re.search(r"[\uAC00-\uD7AF]", str(x)) and re.search(r"[a-zA-Z]", str(x))) else 0
        )

        # Content type one-hot
        for ct in df["content_type"].unique():
            df[f"is_{ct}"] = (df["content_type"] == ct).astype(int)

        # Duration feature
        if "duration_sec" in df.columns:
            df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)

        self.df = df

        # Feature columns for models
        self.feature_cols = [
            "hour", "day_of_week", "is_weekend",
            "caption_length", "hashtag_count", "has_emoji", "is_multilingual",
        ]
        # Add content type dummies
        for col in df.columns:
            if col.startswith("is_") and col not in ["is_weekend", "is_multilingual"]:
                self.feature_cols.append(col)

        if "duration_sec" in df.columns:
            self.feature_cols.append("duration_sec")

        # Recency weights — last 90 days get full weight for ML training
        if "weight" in df.columns:
            self.sample_weights = df["weight"].values
        else:
            self.sample_weights = compute_recency_weights(df["published_at"]).values

    def run_all(self) -> Dict[str, Any]:
        """Run all ML models. Returns results dict."""
        n = len(self.df)
        logger.info(f"   🧠 ML Engine ({self.platform}): {n} posts")

        self.results["platform"] = self.platform
        self.results["n_posts"] = n

        if n < self.MIN_POSTS_FOR_ML:
            self.results["status"] = "insufficient_data"
            self.results["message"] = f"Need at least {self.MIN_POSTS_FOR_ML} posts for ML (have {n})"
            return self.results

        self.results["status"] = "ok"

        # Run models
        self.results["feature_importance"] = self._feature_importance()
        self.results["engagement_prediction"] = self._engagement_predictor()
        self.results["content_clusters"] = self._content_clustering()
        self.results["anomalies"] = self._anomaly_detection()
        self.results["caption_analysis"] = self._caption_nlp()
        self.results["engagement_drivers"] = self._engagement_drivers()

        return self.results

    # ──────────────────────────────────────────
    # 1. ENGAGEMENT PREDICTOR (Neural Network)
    # ──────────────────────────────────────────

    def _engagement_predictor(self) -> Dict:
        """
        Train an MLP Neural Network to predict engagement rate.
        Returns model quality metrics and predictions.
        """
        n = len(self.df)
        if n < self.MIN_POSTS_FOR_NN:
            return {"status": "need_more_data", "min_required": self.MIN_POSTS_FOR_NN}

        X = self.df[self.feature_cols].fillna(0).values
        y = self.df["engagement_rate"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # MLP Neural Network
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            alpha=0.01,  # L2 regularization
        )

        try:
            # Cross-validation score
            if n >= 20:
                cv_scores = cross_val_score(mlp, X_scaled, y, cv=min(5, n // 4),
                                            scoring="r2")
                cv_r2 = float(np.mean(cv_scores))
            else:
                cv_r2 = None

            # Fit on full data
            mlp.fit(X_scaled, y)
            y_pred = mlp.predict(X_scaled)

            r2 = float(r2_score(y, y_pred))
            mae = float(mean_absolute_error(y, y_pred))

            # Predicted vs actual for each post
            self.df["predicted_engagement_rate"] = y_pred
            self.df["engagement_residual"] = y - y_pred

            # Find overperformers (actual >> predicted) and underperformers
            overperformers = self.df.nlargest(3, "engagement_residual")
            underperformers = self.df.nsmallest(3, "engagement_residual")

            return {
                "status": "trained",
                "model": "MLP Neural Network (32→16→8)",
                "r2_score": round(r2, 4),
                "cv_r2_score": round(cv_r2, 4) if cv_r2 is not None else None,
                "mae": round(mae, 4),
                "interpretation": self._interpret_r2(cv_r2 if cv_r2 is not None else r2),
                "overperformers": [
                    {
                        "title": row["title"][:80],
                        "actual": round(row["engagement_rate"], 4),
                        "predicted": round(row["predicted_engagement_rate"], 4),
                        "surplus": round(row["engagement_residual"], 4),
                        "permalink": row.get("permalink", ""),
                    }
                    for _, row in overperformers.iterrows()
                ],
                "underperformers": [
                    {
                        "title": row["title"][:80],
                        "actual": round(row["engagement_rate"], 4),
                        "predicted": round(row["predicted_engagement_rate"], 4),
                        "deficit": round(row["engagement_residual"], 4),
                        "permalink": row.get("permalink", ""),
                    }
                    for _, row in underperformers.iterrows()
                ],
            }
        except Exception as e:
            logger.warning(f"   Neural net failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ──────────────────────────────────────────
    # 2. FEATURE IMPORTANCE
    # ──────────────────────────────────────────

    def _feature_importance(self) -> Dict:
        """
        Use GradientBoosting to rank which features drive engagement.
        """
        X = self.df[self.feature_cols].fillna(0).values
        y = self.df["engagement_rate"].values

        try:
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
            gb.fit(X, y, sample_weight=self.sample_weights)

            importances = gb.feature_importances_
            feature_ranking = sorted(
                zip(self.feature_cols, importances),
                key=lambda x: x[1],
                reverse=True,
            )

            # Human-readable feature names
            readable_names = {
                "hour": "Posting Hour",
                "day_of_week": "Day of Week",
                "is_weekend": "Weekend Post",
                "caption_length": "Caption Length",
                "hashtag_count": "Number of Hashtags",
                "has_emoji": "Uses Emoji",
                "is_multilingual": "Multilingual Caption",
                "duration_sec": "Video Duration",
                "is_short_video": "Short Video (Reel/TikTok)",
                "is_carousel": "Carousel Post",
                "is_image": "Single Image",
                "is_long_video": "Long Video",
                "is_story": "Story",
                "is_other": "Other Format",
            }

            return {
                "status": "ok",
                "model": "GradientBoosting (100 trees)",
                "r2_score": round(float(gb.score(X, y)), 4),
                "top_features": [
                    {
                        "feature": readable_names.get(f, f),
                        "raw_feature": f,
                        "importance": round(float(imp), 4),
                        "pct": round(float(imp) * 100, 1),
                    }
                    for f, imp in feature_ranking[:8]
                    if imp > 0.01
                ],
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ──────────────────────────────────────────
    # 3. CONTENT CLUSTERING
    # ──────────────────────────────────────────

    def _content_clustering(self) -> Dict:
        """
        KMeans clustering to find groups of similar-performing content.
        """
        cluster_features = ["engagement_rate", "likes", "comments", "shares"]
        available = [c for c in cluster_features if c in self.df.columns]

        if len(available) < 2:
            return {"status": "insufficient_features"}

        X = self.df[available].fillna(0).values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        n = len(self.df)
        k = min(3, max(2, n // 5))  # 2-3 clusters based on data size

        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.df["cluster"] = km.fit_predict(X_scaled)

            clusters = []
            for c in range(k):
                cluster_data = self.df[self.df["cluster"] == c]
                clusters.append({
                    "cluster_id": c,
                    "size": len(cluster_data),
                    "avg_engagement_rate": round(cluster_data["engagement_rate"].mean(), 4),
                    "avg_reach": int(cluster_data["reach"].mean()),
                    "avg_likes": int(cluster_data["likes"].mean()),
                    "avg_shares": int(cluster_data["shares"].mean()),
                    "dominant_type": cluster_data["content_type"].mode().iloc[0] if len(cluster_data) > 0 else "unknown",
                    "sample_titles": cluster_data["title"].head(3).tolist(),
                })

            # Label clusters
            clusters.sort(key=lambda x: x["avg_engagement_rate"], reverse=True)
            labels = ["[TOP] Top Performers", "[AVG] Average", "[LOW] Needs Improvement"]
            for i, c in enumerate(clusters):
                c["label"] = labels[min(i, len(labels) - 1)]

            return {
                "status": "ok",
                "n_clusters": k,
                "clusters": clusters,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ──────────────────────────────────────────
    # 4. ANOMALY DETECTION (Isolation Forest)
    # ──────────────────────────────────────────

    def _anomaly_detection(self) -> Dict:
        """
        Use IsolationForest to find statistically anomalous posts
        (both surprisingly good and surprisingly bad).
        """
        features = ["reach", "engagement_total", "likes", "comments", "shares"]
        available = [c for c in features if c in self.df.columns]

        if len(available) < 3:
            return {"status": "insufficient_features"}

        X = self.df[available].fillna(0).values

        try:
            iso = IsolationForest(
                contamination=0.15,  # Expect ~15% anomalies
                random_state=42,
                n_estimators=100,
            )
            self.df["anomaly_score"] = iso.fit_predict(X)
            self.df["anomaly_raw_score"] = iso.decision_function(X)

            anomalies = self.df[self.df["anomaly_score"] == -1].sort_values(
                "anomaly_raw_score"
            )

            # Classify: positive anomaly (good) or negative
            results = []
            for _, row in anomalies.iterrows():
                is_positive = row["engagement_rate"] > self.df["engagement_rate"].median()
                results.append({
                    "title": row["title"][:80],
                    "type": "viral_outlier" if is_positive else "underperformer_outlier",
                    "engagement_rate": round(row["engagement_rate"], 4),
                    "reach": int(row["reach"]),
                    "anomaly_score": round(float(row["anomaly_raw_score"]), 4),
                    "permalink": row.get("permalink", ""),
                })

            return {
                "status": "ok",
                "model": "IsolationForest",
                "total_anomalies": len(anomalies),
                "viral_outliers": [r for r in results if r["type"] == "viral_outlier"],
                "underperformer_outliers": [r for r in results if r["type"] == "underperformer_outlier"],
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ──────────────────────────────────────────
    # 5. CAPTION NLP (TF-IDF)
    # ──────────────────────────────────────────

    def _caption_nlp(self) -> Dict:
        """
        TF-IDF analysis on captions to find words/topics
        correlated with high engagement.
        """
        captions = self.df["title"].fillna("").tolist()

        # Filter out very short captions
        valid_mask = self.df["title"].fillna("").str.len() > 10
        if valid_mask.sum() < 5:
            return {"status": "insufficient_captions"}

        valid_df = self.df[valid_mask]
        valid_captions = valid_df["title"].tolist()

        try:
            # Multi-language stop words
            stop_words = set([
                # English
                "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                "to", "for", "of", "and", "or", "but", "with", "from", "by",
                "this", "that", "our", "we", "you", "it", "its", "has", "have",
                # Common social media
                "fyp", "foryou", "foryoupage", "viral",
            ])

            tfidf = TfidfVectorizer(
                max_features=50,
                min_df=2,
                stop_words=list(stop_words),
                token_pattern=r"(?u)\b[#\w][\w]{2,}\b",
            )

            tfidf_matrix = tfidf.fit_transform(valid_captions)
            feature_names = tfidf.get_feature_names_out()

            # Correlate each term with engagement
            engagement = valid_df["engagement_rate"].values
            term_engagement = {}

            for i, term in enumerate(feature_names):
                term_presence = (tfidf_matrix[:, i].toarray().flatten() > 0).astype(int)
                if term_presence.sum() >= 2:
                    # Average engagement when term is present vs absent
                    eng_with = engagement[term_presence == 1].mean()
                    eng_without = engagement[term_presence == 0].mean()
                    lift = (eng_with / eng_without) - 1 if eng_without > 0 else 0

                    term_engagement[term] = {
                        "term": term,
                        "posts_with_term": int(term_presence.sum()),
                        "avg_engagement_with": round(float(eng_with), 4),
                        "avg_engagement_without": round(float(eng_without), 4),
                        "engagement_lift": round(float(lift), 4),
                    }

            # Sort by engagement lift
            high_engagement_terms = sorted(
                term_engagement.values(),
                key=lambda x: x["engagement_lift"],
                reverse=True,
            )

            # Top hashtags specifically
            hashtags = [t for t in high_engagement_terms if t["term"].startswith("#")]

            return {
                "status": "ok",
                "top_engagement_terms": high_engagement_terms[:10],
                "top_hashtags": hashtags[:5],
                "low_engagement_terms": sorted(
                    term_engagement.values(),
                    key=lambda x: x["engagement_lift"],
                )[:5],
                "total_terms_analyzed": len(term_engagement),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    # ──────────────────────────────────────────
    # 6. ENGAGEMENT DRIVERS
    # ──────────────────────────────────────────

    def _engagement_drivers(self) -> Dict:
        """
        Statistical correlation analysis to find what specifically
        drives engagement on this platform.
        """
        drivers = {}

        # Caption length vs engagement
        if len(self.df) >= 10:
            corr = self.df["caption_length"].corr(self.df["engagement_rate"])
            optimal_length = self.df.nlargest(
                min(5, len(self.df) // 3), "engagement_rate"
            )["caption_length"].median()
            drivers["caption_length"] = {
                "correlation": round(float(corr), 4),
                "direction": "longer is better" if corr > 0.1 else "shorter is better" if corr < -0.1 else "no strong effect",
                "optimal_range": f"{int(optimal_length * 0.7)}-{int(optimal_length * 1.3)} characters",
            }

        # Hashtag count vs engagement
        if self.df["hashtag_count"].sum() > 0:
            corr = self.df["hashtag_count"].corr(self.df["engagement_rate"])
            optimal_tags = self.df.nlargest(
                min(5, len(self.df) // 3), "engagement_rate"
            )["hashtag_count"].median()
            drivers["hashtags"] = {
                "correlation": round(float(corr), 4),
                "direction": "more is better" if corr > 0.1 else "fewer is better" if corr < -0.1 else "no strong effect",
                "optimal_count": int(optimal_tags),
            }

        # Multilingual content effect (recency-weighted)
        if self.df["is_multilingual"].sum() >= 2 and (self.df["is_multilingual"] == 0).sum() >= 2:
            m_df = self.df[self.df["is_multilingual"] == 1]
            s_df = self.df[self.df["is_multilingual"] == 0]
            multi_eng = float(np.average(m_df["engagement_rate"].values, weights=m_df["weight"].values if "weight" in m_df.columns else None))
            single_eng = float(np.average(s_df["engagement_rate"].values, weights=s_df["weight"].values if "weight" in s_df.columns else None))
            drivers["multilingual"] = {
                "multilingual_avg_engagement": round(float(multi_eng), 4),
                "single_language_avg_engagement": round(float(single_eng), 4),
                "lift": round(float((multi_eng / single_eng) - 1), 4) if single_eng > 0 else 0,
                "recommendation": "Multilingual captions perform better" if multi_eng > single_eng * 1.1 else "Single-language captions perform better" if single_eng > multi_eng * 1.1 else "No significant difference",
            }

        # Emoji effect (recency-weighted)
        if self.df["has_emoji"].sum() >= 2 and (self.df["has_emoji"] == 0).sum() >= 2:
            e_df = self.df[self.df["has_emoji"] == 1]
            ne_df = self.df[self.df["has_emoji"] == 0]
            emoji_eng = float(np.average(e_df["engagement_rate"].values, weights=e_df["weight"].values if "weight" in e_df.columns else None))
            no_emoji_eng = float(np.average(ne_df["engagement_rate"].values, weights=ne_df["weight"].values if "weight" in ne_df.columns else None))
            drivers["emoji"] = {
                "with_emoji_avg": round(float(emoji_eng), 4),
                "without_emoji_avg": round(float(no_emoji_eng), 4),
                "lift": round(float((emoji_eng / no_emoji_eng) - 1), 4) if no_emoji_eng > 0 else 0,
            }

        # Weekend vs weekday (recency-weighted)
        if self.df["is_weekend"].sum() >= 2 and (self.df["is_weekend"] == 0).sum() >= 2:
            we_df = self.df[self.df["is_weekend"] == 1]
            wd_df = self.df[self.df["is_weekend"] == 0]
            we_eng = float(np.average(we_df["engagement_rate"].values, weights=we_df["weight"].values if "weight" in we_df.columns else None))
            wd_eng = float(np.average(wd_df["engagement_rate"].values, weights=wd_df["weight"].values if "weight" in wd_df.columns else None))
            drivers["weekend_effect"] = {
                "weekend_avg": round(float(we_eng), 4),
                "weekday_avg": round(float(wd_eng), 4),
                "better": "weekend" if we_eng > wd_eng * 1.05 else "weekday" if wd_eng > we_eng * 1.05 else "no difference",
            }

        return drivers

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    @staticmethod
    def _interpret_r2(r2: float) -> str:
        if r2 >= 0.8:
            return "Excellent — the model captures most engagement patterns"
        elif r2 >= 0.5:
            return "Good — the model captures significant patterns"
        elif r2 >= 0.3:
            return "Moderate — some patterns detected, more data will improve accuracy"
        elif r2 >= 0:
            return "Weak — engagement may be driven by factors not captured in the data"
        else:
            return "Poor fit — engagement appears highly unpredictable from available features"


def run_ml(df: pd.DataFrame, platform: str) -> Dict[str, Any]:
    """Convenience function to run ML analysis for a single platform."""
    engine = MLEngine(df, platform)
    return engine.run_all()
