```
.
├── midterms
│   ├── dyad_metrics.csv
│   ├── emb_metrics.csv
│   ├── follow_graph_metrics.csv
│   ├── next_reply_metrics
│   │   ├── midterms_paired_sample.json.gz
│   │   ├── midterms_paired_sample_tweet_ids.json.gz
│   │   └── midterms_tweets_tox_p75_m25.pkl.gz
│   ├── polarization.csv
│   ├── prefix_metrics
│   │   ├── midterms.json.gz
│   │   └── midterms.pkl.gz
│   ├── reply_graph_metrics.csv
│   ├── subgraph_metrics.csv
│   ├── toxicity.csv
│   └── tree_metrics.csv
├── modeling
│   ├── next_reply
│   │   ├── datasets
│   │   │   ├── midterms_paired.pkl.gz
│   │   │   └── news_paired.pkl.gz
│   │   ├── feature_sets.json
│   │   ├── runs
│   │   │   ├── domain_transfer.json.gz
│   │   │   ├── midterms_paired_nested_cv.json.gz
│   │   │   └── news_paired_nested_cv.json.gz
│   │   └── runs_csvs
│   │       └── res_nested.csv
│   └── prefix
│       ├── datasets
│       │   ├── midterms_labels.pkl.gz
│       │   ├── midterms_p100.pkl.gz
│       │   ├── midterms_p10.pkl.gz
│       │   ├── midterms_p20.pkl.gz
│       │   ├── midterms_p30.pkl.gz
│       │   ├── midterms_p40.pkl.gz
│       │   ├── midterms_p50.pkl.gz
│       │   ├── midterms_p60.pkl.gz
│       │   ├── midterms_p70.pkl.gz
│       │   ├── midterms_p80.pkl.gz
│       │   ├── midterms_p90.pkl.gz
│       │   ├── news_labels.pkl.gz
│       │   ├── news_p100.pkl.gz
│       │   ├── news_p10.pkl.gz
│       │   ├── news_p20.pkl.gz
│       │   ├── news_p30.pkl.gz
│       │   ├── news_p40.pkl.gz
│       │   ├── news_p50.pkl.gz
│       │   ├── news_p60.pkl.gz
│       │   ├── news_p70.pkl.gz
│       │   ├── news_p80.pkl.gz
│       │   └── news_p90.pkl.gz
│       ├── feature_sets.json
│       ├── label_maps.pkl.gz
│       ├── runs
│       │   ├── domain_transfer.json.gz
│       │   ├── midterms_q50_nested_cv.json.gz
│       │   └── news_q50_nested_cv.json.gz
│       └── runs_csvs
│           └── res_q50_nested.csv
└── news
    ├── dyad_metrics.csv
    ├── emb_metrics.csv
    ├── follow_graph_metrics.csv
    ├── next_reply_metrics
    │   ├── news_paired_sample.json.gz
    │   ├── news_paired_sample_tweet_ids.json.gz
    │   └── news_tweets_tox_p75_m25.pkl.gz
    ├── polarization.csv
    ├── prefix_metrics
    │   ├── news.json.gz
    │   └── news.pkl.gz
    ├── reply_graph_metrics.csv
    ├── subgraph_metrics.csv
    ├── toxicity.csv
    └── tree_metrics.csv

15 directories, 61 files
```