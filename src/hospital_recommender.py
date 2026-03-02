"""
hospital_recommender.py
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HospitalRecommender:
    def __init__(self):
        self.hospitals_df  = None
        self.tfidf         = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        self.hospital_vecs = None

    # ── Load & vectorise hospitals ───────────────────────────
    def load(self, path='data/processed/hospitals.csv'):
        self.hospitals_df = pd.read_csv(path)

        # Build combined text feature for TF-IDF
        self.hospitals_df['text_features'] = (
            self.hospitals_df['specialties'].fillna('') + ' ' +
            self.hospitals_df['facility_type'].fillna('') + ' ' +
            self.hospitals_df['tier'].fillna('') + ' ' +
            self.hospitals_df['ownership_type'].fillna('') + ' ' +
            self.hospitals_df['district'].fillna('')
        )

        self.hospital_vecs = self.tfidf.fit_transform(
            self.hospitals_df['text_features'])
        print(f"Loaded {len(self.hospitals_df)} hospitals")
        print(f"TF-IDF matrix shape: {self.hospital_vecs.shape}")
        return self

    # ── Stage 2: TF-IDF similarity ───────────────────────────
    def _tfidf_match(self, specialty: str, district: str = '') -> pd.DataFrame:
        query = f"{specialty} {district}"
        query_vec = self.tfidf.transform([query])
        scores = cosine_similarity(query_vec, self.hospital_vecs)[0]
        df = self.hospitals_df.copy()
        df['similarity_score'] = scores
        return df

    # ── Stage 3: Rule-Based Filter + Scoring ─────────────────
    def recommend(self,
                  specialty:    str,
                  budget_npr:   int,
                  district:     str  = '',
                  emergency:    bool = False,
                  top_n:        int  = 5) -> pd.DataFrame:
        """
        Parameters
        ----------
        specialty   : medical specialty needed (from disease classifier)
        budget_npr  : max OPD fee user can afford (NPR)
        district    : preferred district (empty = no preference)
        emergency   : True if urgent/emergency case
        top_n       : number of hospitals to return

        Returns
        -------
        DataFrame of top_n recommended hospitals with scores
        """
        df = self._tfidf_match(specialty, district)

        # ── HARD CONSTRAINTS ────────────────────────────────
        # 1. Specialty must match (hospital's specialties contains needed specialty)
        df['specialty_match'] = df['specialties'].apply(
            lambda s: specialty.lower() in str(s).lower()
        )

        # 2. Budget constraint
        df['budget_ok'] = df['opd_fee_npr'] <= budget_npr

        # 3. Emergency constraint (if user needs emergency, hospital must have it)
        if emergency:
            df['emergency_ok'] = df['emergency_available'] == True
        else:
            df['emergency_ok'] = True

        # Filter eligible hospitals
        eligible = df[
            df['specialty_match'] &
            df['budget_ok'] &
            df['emergency_ok']
        ].copy()

        # Fallback: if no exact specialty match, widen to General Medicine
        if len(eligible) == 0:
            eligible = df[df['budget_ok'] & df['emergency_ok']].copy()

        if len(eligible) == 0:
            eligible = df.copy()   # last resort — return all

        # ── PREFERENCE SCORING (weights sum to 100) ──────────
        eligible['score'] = 0.0

        # Similarity score (35%)
        eligible['score'] += eligible['similarity_score'] * 35

        # Rating (25%)
        eligible['score'] += (eligible['rating'] / 5.0) * 25

        # Budget efficiency: closer to budget = better (20%)
        eligible['budget_ratio'] = (
            budget_npr - eligible['opd_fee_npr']
        ).clip(lower=0) / max(budget_npr, 1)
        eligible['score'] += eligible['budget_ratio'] * 20

        # Location preference bonus (10%)
        if district:
            eligible['score'] += (
                eligible['district'].str.lower() == district.lower()
            ).astype(float) * 10

        # Emergency available bonus (5%)
        eligible['score'] += eligible['emergency_available'].astype(float) * 5

        # Ambulance bonus (5%)
        eligible['score'] += eligible['ambulance_service'].astype(float) * 5

        # Sort and return top N
        result = eligible.sort_values('score', ascending=False).head(top_n)

        output_cols = [
            'hospital_name', 'facility_type', 'district',
            'ownership_type', 'tier', 'specialties',
            'opd_fee_npr', 'emergency_available',
            'ambulance_service', 'bed_capacity',
            'rating', 'score', 'similarity_score',
        ]
        return result[output_cols].reset_index(drop=True)

    # ── Model comparison (for evaluation section) ────────────
    def evaluate_similarity_methods(self, specialty='General Medicine'):
        """Compare TF-IDF cosine vs Jaccard similarity."""
        import time

        query = specialty
        results = {}

        # TF-IDF cosine
        t0 = time.time()
        df_tfidf = self._tfidf_match(specialty)
        t1 = time.time()
        results['TF-IDF Cosine'] = {
            'top_match_score': df_tfidf['similarity_score'].max(),
            'mean_score':      df_tfidf['similarity_score'].mean(),
            'time_ms':         (t1 - t0) * 1000,
        }

        # Jaccard (keyword overlap)
        t0 = time.time()
        query_tokens = set(query.lower().split())
        def jaccard(text):
            doc_tokens = set(str(text).lower().split())
            inter = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            return len(inter) / len(union) if union else 0
        jaccard_scores = self.hospitals_df['text_features'].apply(jaccard)
        t1 = time.time()
        results['Jaccard'] = {
            'top_match_score': jaccard_scores.max(),
            'mean_score':      jaccard_scores.mean(),
            'time_ms':         (t1 - t0) * 1000,
        }

        return pd.DataFrame(results).T


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = HospitalRecommender().load()

    print("\n--- Sample Recommendation ---")
    results = rec.recommend(
        specialty='Cardiology',
        budget_npr=1500,
        district='Kathmandu',
        emergency=False,
        top_n=5
    )
    print(results[['hospital_name', 'district', 'tier',
                   'opd_fee_npr', 'rating', 'score']].to_string())

    print("\n--- Similarity Method Comparison ---")
    comp = rec.evaluate_similarity_methods('Cardiology')
    print(comp.to_string())
