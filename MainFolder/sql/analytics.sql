-- Length distribution
SELECT source, width_bucket(length(text), 0, 1000, 20) AS len_bin, COUNT(*) AS n
FROM documents
GROUP BY 1,2
ORDER BY 1,2;

-- Toxic label co-occurrence
SELECT
  SUM((toxic::int)*(insult::int)) AS toxic_insult,
  SUM((toxic::int)*(obscene::int)) AS toxic_obscene,
  SUM((insult::int)*(obscene::int)) AS insult_obscene
FROM toxic_labels;

-- Emotion frequencies
SELECT 'admiration' AS emotion, SUM(admiration::int) AS cnt FROM emotion_labels
UNION ALL
SELECT 'gratitude', SUM(gratitude::int) FROM emotion_labels
UNION ALL
SELECT 'joy', SUM(joy::int) FROM emotion_labels
UNION ALL
SELECT 'pride', SUM(pride::int) FROM emotion_labels
UNION ALL
SELECT 'love', SUM(love::int) FROM emotion_labels
UNION ALL
SELECT 'optimism', SUM(optimism::int) FROM emotion_labels
ORDER BY cnt DESC;