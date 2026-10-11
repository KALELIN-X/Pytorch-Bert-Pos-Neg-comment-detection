import matplotlib.pyplot as plt
from .utils import fetch_dataframe

# Length distribution buckets by source
sql_len = """
SELECT source, len_bin, n FROM (
  SELECT source, width_bucket(length(text), 0, 1000, 20) AS len_bin, COUNT(*) AS n
  FROM documents GROUP BY 1,2
) t ORDER BY source, len_bin
"""

df = fetch_dataframe(sql_len)
for src, grp in df.groupby('source'):
    plt.figure()
    plt.title(f'Length distribution — {src}')
    plt.plot(grp['len_bin'], grp['n'])
    plt.xlabel('Length bin (0..1000, 20 buckets)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()