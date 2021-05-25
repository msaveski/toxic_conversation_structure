
library(tidyverse)
library(texreg)


# NB: setwd to the project root directory

# load utils
source("code/analyses/utils.R", echo=F)

# settings
ds <- "news"
plt_csv_dir <- "data/plots_csvs/"

#
# load data
#
df_reply_trees <-
  read_csv(
    sprintf("data/%s/tree_metrics.csv", ds),
    col_types = cols(root_tweet_id = col_character())
  )

df_toxicity <-
  read_csv(
    sprintf("data/%s/toxicity.csv", ds),
    col_types = cols(root_tweet_id = col_character())
  )

# select only one column (makes the subseqent analysis easier)
df_toxicity <- df_toxicity %>% 
  rename(f_tox_tweets = f_tox_tweets_0_531) %>%
  select(root_tweet_id, f_tox_tweets)

df <- inner_join(df_toxicity, df_reply_trees, by="root_tweet_id")

rm(df_reply_trees, df_toxicity)


# general settings
n_bins= 30
min_n_per_bin = 250


#
# (1) toxicity ~ number of tweets & number of users 
#
df_n_tweets <- df %>%
  mutate(n_tweets_bin = log10_bin(n_tweets, n_breaks = n_bins)) %>%
  group_by(n_tweets_bin) %>%
  summarize(
    var = "Number of Tweets",
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

df_n_users <- df %>%
  mutate(n_users_bin = log10_bin(n_users, n_breaks = n_bins)) %>%
  group_by(n_users_bin) %>%
  summarize(
    var = "Number of Users",
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

df_n_tweets_users <- rbind(
  df_n_tweets %>% rename(x = n_tweets_bin), 
  df_n_users %>% rename(x = n_users_bin)
)

write_csv(df_n_tweets_users, sprintf("%s/rt_size_%s.csv", plt_csv_dir, ds))


#
# (2) toxicity ~ depth and width (max-breath at any level)
#

# depth
df_depth <- df %>%
  mutate(depth_bin = log10_bin(depth, n_breaks = n_bins)) %>%
  group_by(depth_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_depth, sprintf("%s/rt_depth_%s.csv", plt_csv_dir, ds))


# width
df_width <- df %>%
  mutate(width_bin = log10_bin(width, n_breaks = n_bins)) %>%
  group_by(width_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_width, sprintf("%s/rt_width_%s.csv", plt_csv_dir, ds))


#
# (3) toxicity ~ structural virality & (structural virality x n_tweets)
# 

# structural virality
df_sv <- df %>%
  mutate(sv_bin = log10_bin(wiener_index, n_breaks = n_bins)) %>%
  group_by(sv_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_sv, sprintf("%s/rt_wiener_%s.csv", plt_csv_dir, ds))


# structural virality x size
df_sv_x_size <- df %>%
  mutate(
    n_tweets_bin = log10_bin_factor(n_tweets, n_breaks = 7),
    sv_bin = log10_bin(wiener_index, n_breaks = n_bins)
  ) %>%
  group_by(n_tweets_bin, sv_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_sv_x_size, sprintf("%s/rt_wiener_size_%s.csv", plt_csv_dir, ds))


#
# correlations & linear models
#
res <- cor(df %>% select("n_tweets", "n_users", "depth", "width", "wiener_index"))
round(res, 2)

models <- list(
  lm(f_tox_tweets ~ log10(n_tweets), data=df),
  lm(f_tox_tweets ~ log10(wiener_index), data=df),
  lm(f_tox_tweets ~ log10(n_tweets) + log10(wiener_index), data=df)
)

screenreg(models)

# END