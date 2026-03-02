"""
data_preparation.py
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# DISEASE → MEDICAL SPECIALTY MAPPING

DISEASE_SPECIALTY_MAP = {
    'Fungal infection':                        'Dermatology',
    'Allergy':                                 'General Medicine',
    'GERD':                                    'Gastroenterology',
    'Chronic cholestasis':                     'Gastroenterology',
    'Drug Reaction':                           'General Medicine',
    'Peptic ulcer diseae':                     'Gastroenterology',
    'AIDS':                                    'Infectious Disease',
    'Diabetes':                                'Endocrinology',
    'Gastroenteritis':                         'Gastroenterology',
    'Bronchial Asthma':                        'Pulmonology',
    'Hypertension':                            'Cardiology',
    'Migraine':                                'Neurology',
    'Cervical spondylosis':                    'Orthopedics',
    'Paralysis (brain hemorrhage)':            'Neurology',
    'Jaundice':                                'Gastroenterology',
    'Malaria':                                 'Infectious Disease',
    'Chicken pox':                             'Infectious Disease',
    'Dengue':                                  'Infectious Disease',
    'Typhoid':                                 'Infectious Disease',
    'hepatitis A':                             'Gastroenterology',
    'Hepatitis B':                             'Gastroenterology',
    'Hepatitis C':                             'Gastroenterology',
    'Hepatitis D':                             'Gastroenterology',
    'Hepatitis E':                             'Gastroenterology',
    'Alcoholic hepatitis':                     'Gastroenterology',
    'Tuberculosis':                            'Pulmonology',
    'Common Cold':                             'General Medicine',
    'Pneumonia':                               'Pulmonology',
    'Dimorphic hemmorhoids(piles)':            'General Surgery',
    'Heart attack':                            'Cardiology',
    'Varicose veins':                          'General Surgery',
    'Hypothyroidism':                          'Endocrinology',
    'Hyperthyroidism':                         'Endocrinology',
    'Hypoglycemia':                            'Endocrinology',
    'Osteoarthristis':                         'Orthopedics',
    'Arthritis':                               'Orthopedics',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurology',
    'Acne':                                    'Dermatology',
    'Urinary tract infection':                 'Urology',
    'Psoriasis':                               'Dermatology',
    'Impetigo':                                'Dermatology',
}

ALL_SPECIALTIES = sorted(set(DISEASE_SPECIALTY_MAP.values()))

# Specialties assigned per facility type
HOSPITAL_SPECIALTY_POOL   = ALL_SPECIALTIES          # hospitals get 3–6
CLINIC_SPECIALTY_POOL     = ['General Medicine', 'Infectious Disease',
                              'Dermatology', 'Gastroenterology']
DOCTOR_SPECIALTY_POOL     = ['General Medicine', 'Infectious Disease']


# NEPAL DISTRICTS  (used for location tagging)

KATHMANDU_VALLEY = ['Kathmandu', 'Lalitpur', 'Bhaktapur']
MAJOR_CITIES     = ['Pokhara', 'Biratnagar', 'Birgunj', 'Dharan',
                    'Butwal', 'Nepalgunj', 'Hetauda', 'Itahari']
OTHER_DISTRICTS  = ['Chitwan', 'Kaski', 'Morang', 'Rupandehi',
                    'Sunsari', 'Makwanpur', 'Dhanusha', 'Sarlahi',
                    'Banke', 'Bardiya', 'Dang', 'Palpa', 'Syangja',
                    'Gorkha', 'Lamjung', 'Tanahu', 'Baglung',
                    'Parbat', 'Myagdi', 'Mustang', 'Dolpa']


def infer_location(row):
    """Infer district from coordinates or city field."""
    city = str(row.get('city_raw', '')).strip().lower()
    lat  = row.get('latitude', 0)
    lon  = row.get('longitude', 0)

    if any(k in city for k in ['kathmandu', 'ktm']):  return 'Kathmandu'
    if 'lalitpur' in city or 'patan' in city:         return 'Lalitpur'
    if 'bhaktapur' in city:                           return 'Bhaktapur'
    if 'pokhara' in city:                             return 'Pokhara'
    if 'biratnagar' in city:                          return 'Biratnagar'
    if 'birgunj' in city:                             return 'Birgunj'
    if 'dharan' in city:                              return 'Dharan'
    if 'butwal' in city:                              return 'Butwal'
    if 'nepalgunj' in city:                           return 'Nepalgunj'
    if 'chitwan' in city:                             return 'Chitwan'

    # Approximate by lat/lon bands
    if 27.6 <= lat <= 27.8 and 85.2 <= lon <= 85.5:  return 'Kathmandu'
    if 27.6 <= lat <= 27.7 and 85.3 <= lon <= 85.4:  return 'Lalitpur'
    if 28.1 <= lat <= 28.3 and 83.9 <= lon <= 84.1:  return 'Pokhara'
    if 26.4 <= lat <= 26.6 and 87.2 <= lon <= 87.4:  return 'Biratnagar'

    # Random assignment from all districts weighted by population
    return random.choices(
        KATHMANDU_VALLEY + MAJOR_CITIES + OTHER_DISTRICTS,
        weights=[10, 8, 4] + [5]*len(MAJOR_CITIES) + [2]*len(OTHER_DISTRICTS)
    )[0]


def assign_tier(facility_type, operator):
    ft  = str(facility_type).lower()
    op  = str(operator).lower()
    if ft == 'hospital':
        if 'government' in op or 'public' in op:
            return random.choices(['Standard', 'Premium'], weights=[60, 40])[0]
        elif 'private' in op:
            return random.choices(['Standard', 'Premium'], weights=[30, 70])[0]
        else:
            return random.choices(['Budget', 'Standard'],  weights=[50, 50])[0]
    else:   # clinic / doctors
        if 'government' in op or 'public' in op:
            return random.choices(['Budget', 'Standard'],  weights=[70, 30])[0]
        return random.choices(['Budget', 'Standard'],      weights=[60, 40])[0]


def assign_opd_fee(tier, operator):
    op = str(operator).lower()
    if 'government' in op or 'public' in op:
        return random.randint(100, 400)
    if tier == 'Budget':   return random.randint(200,  600)
    if tier == 'Standard': return random.randint(500, 1500)
    return random.randint(1000, 3000)   # Premium


def assign_specialties(facility_type, tier):
    ft = str(facility_type).lower()
    if ft == 'hospital':
        n = {'Budget': 2, 'Standard': 4, 'Premium': 6}.get(tier, 3)
        return ', '.join(random.sample(HOSPITAL_SPECIALTY_POOL,
                                       min(n, len(HOSPITAL_SPECIALTY_POOL))))
    elif ft == 'clinic':
        n = random.randint(1, 3)
        return ', '.join(random.sample(CLINIC_SPECIALTY_POOL,
                                       min(n, len(CLINIC_SPECIALTY_POOL))))
    else:
        return 'General Medicine'


def assign_emergency(facility_type, tier):
    ft = str(facility_type).lower()
    if ft == 'hospital':
        prob = {'Budget': 0.4, 'Standard': 0.7, 'Premium': 0.95}.get(tier, 0.5)
    else:
        prob = 0.15
    return random.random() < prob


def assign_beds(facility_type, tier):
    ft = str(facility_type).lower()
    if ft != 'hospital': return 0
    return {
        'Budget':   random.randint(10,  50),
        'Standard': random.randint(50, 200),
        'Premium':  random.randint(100, 500),
    }.get(tier, random.randint(20, 100))


def assign_rating(tier, operator):
    op = str(operator).lower()
    if 'government' in op or 'public' in op:
        return round(random.uniform(2.5, 4.0), 1)
    return {
        'Budget':   round(random.uniform(2.5, 3.5), 1),
        'Standard': round(random.uniform(3.0, 4.2), 1),
        'Premium':  round(random.uniform(3.5, 5.0), 1),
    }.get(tier, round(random.uniform(3.0, 4.0), 1))


def clean_operator(val):
    if pd.isna(val): return 'Private'
    v = str(val).lower().strip()
    if 'government' in v or 'public' in v: return 'Government'
    if 'ngo' in v or 'community' in v or 'charitable' in v: return 'Community/NGO'
    return 'Private'


# MAIN PROCESSING
def prepare_hospitals():
    print("=" * 55)
    print("  HOSPITAL DATA PREPARATION")
    print("=" * 55)

    raw = pd.read_csv('data/raw/nepal_hxl.csv')
    print(f"Raw records      : {len(raw)}")

    # Rename columns
    raw = raw.rename(columns={
        'X':                       'longitude',
        'Y':                       'latitude',
        '#loc+amenity':            'facility_type',
        '#loc +name':              'hospital_name',
        '#meta +speciality':       'existing_speciality',
        '#meta +operator_type':    'operator_type',
        '#contact +phone':         'phone',
        '#capacity +beds':         'beds_existing',
        '#meta+emergency':         'emergency_existing',
        'addr_city':               'city_raw',
    })

    # Keep only medical facilities (no pharmacies)
    df = raw[raw['facility_type'].isin(
        ['hospital', 'clinic', 'doctors', 'health_post', 'healthpost']
    )].copy()

    # Keep named facilities only
    df = df[df['hospital_name'].notna()].copy()
    df['hospital_name'] = df['hospital_name'].str.strip()
    df = df.drop_duplicates(subset='hospital_name')
    print(f"After cleaning   : {len(df)} named facilities")

    # Clean operator
    df['ownership_type'] = df['operator_type'].apply(clean_operator)

    # ── Synthetic columns ──────────────────────────────────
    print("Adding synthetic columns...")
    df['district']          = df.apply(infer_location, axis=1)
    df['tier']              = df.apply(lambda r: assign_tier(r['facility_type'],
                                                              r['ownership_type']), axis=1)
    df['opd_fee_npr']       = df.apply(lambda r: assign_opd_fee(r['tier'],
                                                                  r['ownership_type']), axis=1)
    df['specialties']       = df.apply(lambda r: assign_specialties(r['facility_type'],
                                                                      r['tier']), axis=1)
    df['emergency_available'] = df.apply(lambda r: assign_emergency(r['facility_type'],
                                                                      r['tier']), axis=1)
    df['bed_capacity']      = df.apply(lambda r: assign_beds(r['facility_type'],
                                                              r['tier']), axis=1)
    df['rating']            = df.apply(lambda r: assign_rating(r['tier'],
                                                                r['ownership_type']), axis=1)
    df['scholarship_available'] = df['ownership_type'].apply(
        lambda x: random.random() < (0.8 if x == 'Government' else 0.2))
    df['parking_available'] = df['tier'].apply(
        lambda t: random.random() < {'Budget': 0.3, 'Standard': 0.6, 'Premium': 0.9}.get(t, 0.5))
    df['ambulance_service'] = df.apply(
        lambda r: random.random() < (0.9 if r['facility_type'] == 'hospital' else 0.2), axis=1)

    # Tier encoding for ML
    tier_enc = {'Budget': 0, 'Standard': 1, 'Premium': 2}
    df['tier_encoded']      = df['tier'].map(tier_enc)
    owner_enc = {'Government': 0, 'Community/NGO': 1, 'Private': 2}
    df['ownership_encoded'] = df['ownership_type'].map(owner_enc)

    # Final columns to keep
    final_cols = [
        'hospital_name', 'facility_type', 'district',
        'latitude', 'longitude',
        'ownership_type', 'ownership_encoded',
        'tier', 'tier_encoded',
        'specialties', 'opd_fee_npr',
        'emergency_available', 'bed_capacity',
        'rating', 'ambulance_service',
        'parking_available', 'phone',
    ]
    df = df[final_cols].reset_index(drop=True)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/hospitals.csv', index=False)

    print(f"\nEnriched dataset saved → data/processed/hospitals.csv")
    print(f"Total hospitals  : {len(df)}")
    print(f"\nFacility type breakdown:")
    print(df['facility_type'].value_counts().to_string())
    print(f"\nOwnership breakdown:")
    print(df['ownership_type'].value_counts().to_string())
    print(f"\nTier breakdown:")
    print(df['tier'].value_counts().to_string())
    print(f"\nDistrict sample (top 10):")
    print(df['district'].value_counts().head(10).to_string())
    print(f"\nFee range (NPR):")
    print(f"  Min : {df['opd_fee_npr'].min()}")
    print(f"  Max : {df['opd_fee_npr'].max()}")
    print(f"  Mean: {df['opd_fee_npr'].mean():.0f}")
    print(f"\nEmergency available: {df['emergency_available'].sum()} facilities")

    return df


def prepare_disease_data():
    print("\n" + "=" * 55)
    print("  DISEASE DATA PREPARATION")
    print("=" * 55)

    # Load all disease-related files
    disease_df = pd.read_csv('data/raw/dataset.csv')
    desc_df    = pd.read_csv('data/raw/symptom_Description.csv')
    sev_df     = pd.read_csv('data/raw/Symptom-severity.csv')
    prec_df    = pd.read_csv('data/raw/symptom_precaution.csv')

    # Clean disease names
    disease_df['Disease'] = disease_df['Disease'].str.strip()
    desc_df['Disease']    = desc_df['Disease'].str.strip()
    prec_df['Disease']    = prec_df['Disease'].str.strip()
    sev_df['Symptom']     = sev_df['Symptom'].str.strip()

    # Melt dataset: wide symptoms → long format
    symptom_cols = [c for c in disease_df.columns if 'Symptom' in c]
    melted = disease_df.melt(
        id_vars='Disease',
        value_vars=symptom_cols,
        value_name='symptom'
    ).dropna(subset=['symptom'])
    melted['symptom'] = melted['symptom'].str.strip()
    melted = melted[melted['symptom'] != ''][['Disease', 'symptom']].drop_duplicates()

    # Merge severity
    melted = melted.merge(sev_df, left_on='symptom', right_on='Symptom', how='left')
    melted['weight'] = melted['weight'].fillna(1)

    # Aggregate: for each disease, list of symptoms + severity score
    agg = melted.groupby('Disease').agg(
        symptoms=('symptom', lambda x: list(x)),
        avg_severity=('weight', 'mean')
    ).reset_index()

    # Merge description and precautions
    agg = agg.merge(desc_df, on='Disease', how='left')
    agg = agg.merge(prec_df, on='Disease', how='left')

    # Add specialty mapping
    agg['specialty'] = agg['Disease'].map(DISEASE_SPECIALTY_MAP)
    agg['specialty'] = agg['specialty'].fillna('General Medicine')

    # Urgency level based on avg severity
    def urgency(score):
        if score >= 4.5: return 'Emergency'
        if score >= 3.0: return 'Urgent'
        return 'Routine'

    agg['urgency_level'] = agg['avg_severity'].apply(urgency)

    agg.to_csv('data/processed/diseases.csv', index=False)
    print(f"Disease records  : {len(agg)}")
    print(f"Unique specialties: {agg['specialty'].nunique()}")
    print(f"\nSpecialty distribution:")
    print(agg['specialty'].value_counts().to_string())
    print(f"\nUrgency distribution:")
    print(agg['urgency_level'].value_counts().to_string())

    # Save symptom severity lookup
    sev_df.to_csv('data/processed/symptom_severity.csv', index=False)
    print(f"\nSaved → data/processed/diseases.csv")
    print(f"Saved → data/processed/symptom_severity.csv")

    return agg


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hospitals = prepare_hospitals()
    diseases  = prepare_disease_data()
    print("\n✓ Data preparation complete!")

