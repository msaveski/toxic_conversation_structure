
library(tidyverse)


# NB: setwd to the project root directory

# load utils
source("code/analyses/utils.R", echo=F)

# settings
ds <- "news"
plt_csv_dir <- "data/plots_csvs/"

#
# load data
#
df_follow_graphs <-
  read_csv(
    sprintf("data/%s/follow_graph_metrics.csv", ds),
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

df <- inner_join(df_toxicity, df_follow_graphs, by="root_tweet_id")

rm(df_follow_graphs, df_toxicity)


# general settings
n_bins <- 20
min_n_per_bin <- 250


#
# Density
#
df_density <- df %>%
  filter(n_nodes > 0, n_edges_ud > 0) %>% 
  mutate(
    density_bin = log10_bin(density_ud, n_breaks = n_bins)
  ) %>%
  group_by(density_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_density, sprintf("%s/fg_density_%s.csv", plt_csv_dir, ds))


#
# Number of Connected Components
#
df_n_cc <- df %>%
  mutate(n_CC_bin = log10_bin(n_CC_1, n_breaks = n_bins)) %>%
  group_by(n_CC_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_n_cc, sprintf("%s/fg_nCC_%s.csv", plt_csv_dir, ds))


#
# Modularity after louvain
#
mean(df$modularity_louvain == 0, na.rm = T)

df_modularity <- df %>%
  filter(n_edges_ud > 0) %>%
  mutate(modularity_bin = lin_bin(modularity_louvain, n_breaks = n_bins)) %>%
  group_by(modularity_bin) %>%
  summarize(
    n = n(),
    bin_tox_mean = mean(f_tox_tweets),
    bin_tox_ci = gaussian_mean_95_ci(f_tox_tweets)
  ) %>%
  filter(n > min_n_per_bin)

write_csv(df_modularity, sprintf("%s/fg_modularity_%s.csv", plt_csv_dir, ds))


# END