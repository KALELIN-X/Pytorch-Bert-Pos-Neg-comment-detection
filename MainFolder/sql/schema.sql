-- Base documents table
CREATE TABLE IF NOT EXISTS documents (
  doc_id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,         -- 'jigsaw' | 'goemotions' | 'twitter_emotion'
  text   TEXT NOT NULL
);

-- Toxic labels (6)
CREATE TABLE IF NOT EXISTS toxic_labels (
  doc_id BIGINT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
  toxic BOOL, severe_toxic BOOL, obscene BOOL, threat BOOL, insult BOOL, identity_hate BOOL
);

-- Positive emotions subset (adjust columns if you include more)
CREATE TABLE IF NOT EXISTS emotion_labels (
  doc_id BIGINT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
  admiration BOOL, gratitude BOOL, joy BOOL, pride BOOL, love BOOL, optimism BOOL
);

-- Helpful views
CREATE OR REPLACE VIEW toxic_prevalence AS
SELECT SUM(toxic::int) toxic, SUM(severe_toxic::int) severe_toxic, SUM(obscene::int) obscene,
       SUM(threat::int) threat, SUM(insult::int) insult, SUM(identity_hate::int) identity_hate
FROM toxic_labels;

CREATE OR REPLACE VIEW emotion_prevalence AS
SELECT
  SUM(admiration::int) admiration, SUM(gratitude::int) gratitude, SUM(joy::int) joy,
  SUM(pride::int) pride, SUM(love::int) love, SUM(optimism::int) optimism
FROM emotion_labels;
