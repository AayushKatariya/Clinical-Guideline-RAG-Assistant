-- Structured layer of the clinical guideline corpus.
-- `guidelines.guideline_id` is the same identifier as `doc_id` in the text index,
-- so answers can cite both an SQL row and the passage it came from.

CREATE TABLE IF NOT EXISTS guidelines (
    guideline_id      TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    organization      TEXT,
    specialty         TEXT,
    country           TEXT,
    publication_year  INTEGER,
    last_updated      DATE,
    url               TEXT
);
COMMENT ON TABLE  guidelines IS 'One row per clinical practice guideline in the corpus.';
COMMENT ON COLUMN guidelines.guideline_id IS 'Stable id; equals doc_id in the retrieval index.';
COMMENT ON COLUMN guidelines.organization IS 'Issuing body, e.g. WHO, NICE, CDC, ADA, ACC/AHA.';
COMMENT ON COLUMN guidelines.specialty IS 'Clinical area, e.g. cardiology, infectious disease, endocrinology.';
COMMENT ON COLUMN guidelines.publication_year IS 'Year the current version was published.';

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id  TEXT PRIMARY KEY,
    guideline_id       TEXT NOT NULL REFERENCES guidelines(guideline_id) ON DELETE CASCADE,
    section            TEXT,
    condition          TEXT,
    population         TEXT,
    intervention       TEXT,
    recommendation_text TEXT NOT NULL,
    strength           TEXT,
    evidence_level     TEXT
);
COMMENT ON TABLE  recommendations IS 'Individual graded recommendations extracted from guidelines.';
COMMENT ON COLUMN recommendations.strength IS 'Recommendation strength as written by the guideline, e.g. strong, conditional, weak, Class I, Class IIa.';
COMMENT ON COLUMN recommendations.evidence_level IS 'Certainty/level of evidence, e.g. high, moderate, low, very low, A, B, C.';
COMMENT ON COLUMN recommendations.population IS 'Who the recommendation applies to, e.g. adults, children, pregnant women.';

CREATE TABLE IF NOT EXISTS drug_dosing (
    dosing_id         SERIAL PRIMARY KEY,
    guideline_id      TEXT NOT NULL REFERENCES guidelines(guideline_id) ON DELETE CASCADE,
    drug_name         TEXT NOT NULL,
    indication        TEXT,
    population        TEXT,
    route             TEXT,
    dose              TEXT,
    frequency         TEXT,
    duration          TEXT,
    max_daily_dose    TEXT,
    renal_adjustment  TEXT,
    notes             TEXT
);
COMMENT ON TABLE  drug_dosing IS 'Dosing regimens stated in guidelines. Dose fields are free text exactly as written (e.g. "500 mg").';
COMMENT ON COLUMN drug_dosing.drug_name IS 'Generic drug name, lower case.';
COMMENT ON COLUMN drug_dosing.population IS 'e.g. adults, children, neonates, pregnancy, renal impairment, older adults.';
COMMENT ON COLUMN drug_dosing.route IS 'e.g. oral, IV, IM, subcutaneous, inhaled.';

CREATE TABLE IF NOT EXISTS diagnostic_thresholds (
    threshold_id      SERIAL PRIMARY KEY,
    guideline_id      TEXT NOT NULL REFERENCES guidelines(guideline_id) ON DELETE CASCADE,
    condition         TEXT NOT NULL,
    test_name         TEXT NOT NULL,
    comparator        TEXT,
    value             NUMERIC,
    unit              TEXT,
    population        TEXT,
    interpretation    TEXT
);
COMMENT ON TABLE  diagnostic_thresholds IS 'Numeric diagnostic or treatment thresholds, e.g. HbA1c >= 6.5 % for diabetes.';
COMMENT ON COLUMN diagnostic_thresholds.comparator IS 'One of <, <=, =, >=, >.';

CREATE INDEX IF NOT EXISTS idx_guidelines_org ON guidelines (organization);
CREATE INDEX IF NOT EXISTS idx_guidelines_year ON guidelines (publication_year);
CREATE INDEX IF NOT EXISTS idx_reco_guideline ON recommendations (guideline_id);
CREATE INDEX IF NOT EXISTS idx_dosing_drug ON drug_dosing (lower(drug_name));
CREATE INDEX IF NOT EXISTS idx_thresholds_condition ON diagnostic_thresholds (lower(condition));
