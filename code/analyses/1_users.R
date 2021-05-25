
library(tidyverse)

# NB: Since the user ids are not consistent in the anonymized conversations, 
#   the data to compute the statistics below is not available in the publicly 
#   available dataset.

# NB: setwd to the project root directory

# load utils
source("code/analyses/utils.R", echo=F)

# settings
ds <- "news"
plt_csv_dir <- "data/plots_csvs/"

# read data
df <- read_csv(
  sprintf("data/%s/user_toxicity/user_metrics.csv", ds), 
  col_types = "cdddddd"
)


#
# (1) bucketing the users in log buckets
#

# distribution of number of tox/nontox tweets per user
df_user_bins <- rbind(
  df %>%
    filter(n_tox_tweets > 0) %>%
    mutate(bin = log10_bin(n_tox_tweets, n_breaks = 11)) %>%
    group_by(bin) %>%
    summarise(n_users = n()) %>%
    mutate(set="toxic"),
  df %>%
    filter(n_tweets > 0) %>%
    mutate(bin = log10_bin(n_tweets, n_breaks = 11)) %>%
    group_by(bin) %>%
    summarise(n_users = n()) %>%
    mutate(set="toxic + nontoxic")
)

write_csv(df_user_bins, sprintf("%s/ego_n_tweets_%s.csv", plt_csv_dir, ds))


#
# (2) bucketing the users in log buckets by num of TOXIC tweets
#
total_n_tweets <- sum(df$n_tweets)
total_n_tox_tweets <- sum(df$n_tox_tweets)

df_tox_user_bins <- df %>%
  filter(n_tox_tweets > 0) %>%
  mutate(n_tox_tweets_bin = log10_bin(n_tox_tweets, n_breaks = 11)) %>%
  group_by(n_tox_tweets_bin) %>%
  summarise(
    n_users = n(),
    n_tox_tweets_bin_f_all = sum(n_tox_tweets) / total_n_tox_tweets,
    n_tweets_avg = mean(n_tweets),
    n_tweets_avg_95_ci = gaussian_mean_95_ci(n_tweets),
    min_n_tox_tweets = min(n_tox_tweets),
    max_n_tox_tweets = max(n_tox_tweets)
  )

write_csv(df_tox_user_bins, sprintf("%s/ego_frac_tox_%s.csv", plt_csv_dir, ds))


#
# (3) bucketing users by the TOTAL number of tweets per user
#
total_n_tweets <- sum(df$n_tweets)
total_n_tox_tweets <- sum(df$n_tox_tweets)

df_user_bins <- df %>%
  # filter(n_tweets > 0) %>%
  mutate(n_tweets_bin = log10_bin(n_tweets, n_breaks = 11)) %>%
  group_by(n_tweets_bin) %>%
  summarise(
    n_users = n(),
    min_n_tweets = min(n_tweets),
    max_n_tweets = max(n_tweets),
    n_tox_tweets_bin_f_all = sum(n_tox_tweets) / total_n_tox_tweets,
    n_tox_tweets_avg = mean(n_tox_tweets),
    n_tox_tweets_avg_95_ci = gaussian_mean_95_ci(n_tox_tweets),
    f_tox_tweets_avg = mean(f_tox_tweets),
    f_tox_tweets_avg_95_ci = gaussian_mean_95_ci(f_tox_tweets)
  )

write_csv(df_user_bins, sprintf("%s/ego_rate_tox_%s.csv", plt_csv_dir, ds))

# END