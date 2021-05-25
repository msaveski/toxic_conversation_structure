
library(tidyverse)


# NB: setwd to the project root directory

# load utils
source("code/analyses/utils.R", echo=F)

# settings
ds <- "midterms"
plt_csv_dir <- "data/plots_csvs/"

# load data
df <- read_csv(
  sprintf("data/%s/dyad_metrics.csv", ds), 
  col_types = "clddlddcd"
)


#
# TOXIC INTERACTIONS
#

# --- p(toxic reply | post)
df_tox <- df %>%
  group_by(parent_tox, child_tox) %>%
  summarise(n = n()) %>%
  na.omit

p_tox_reply <- df_tox %>%
  group_by(parent_tox) %>%
  mutate(n = n / sum(n)) %>%
  filter(child_tox == T)

print(p_tox_reply)

# --- p(reply = toxic | post, dyad_type)
df_dyad_type_tox <- df %>%
  filter(!is.na(dyad_type), !is.na(parent_tox), !is.na(child_tox)) %>%
  group_by(dyad_type, parent_tox, child_tox) %>%
  summarise(n = n())

df_dyad_type_tox <- df_dyad_type_tox %>%
  mutate(child_tox = ifelse(child_tox == T, 'child_tox', 'child_nontox')) %>%
  spread(child_tox, n) %>%
  mutate(
    n_child = child_tox + child_nontox,
    p_child_tox = child_tox / n_child,
    p_child_ci = binomial_95_ci(p_child_tox, n_child)
  ) %>%
  arrange(desc(p_child_tox))

df_dyad_type_tox <- df_dyad_type_tox %>%
  group_by() %>%
  mutate(
    dyad_type = factor(dyad_type, rev(c("O  O", "O<-O", "O==O", "O->O"))),
    parent_tox = case_when(parent_tox == T ~ "post = toxic", parent_tox == F ~ "post = nontoxic"),
    parent_tox = factor(parent_tox, c("post = toxic", "post = nontoxic"))
  )

write_csv(df_dyad_type_tox, sprintf("%s/dyad_edgetype_%s.csv", plt_csv_dir, ds))

#
# INFLUENCE GAP
# p(reply = tox | n_followers_ratio)
#

# bin by follower ratio
n_bins <- 16
min_bucket_n = 250

df_n_follows_bin <- df %>%
  select(parent_tox, child_tox, parent_n_followers, child_n_followers) %>%
  na.omit %>%
  mutate(
    pc_ratio = log10(parent_n_followers) - log10(child_n_followers),
    pc_bin = lin_bin(pc_ratio, n_breaks = n_bins)
  )

df_n_follows_bin_post <- df_n_follows_bin %>%
  group_by(parent_tox, child_tox, pc_bin) %>%
  summarise(n = n())

df_n_follows_bin_post <- df_n_follows_bin_post %>%
  group_by() %>%
  mutate(child_tox = ifelse(child_tox == T, 'child_tox', 'child_nontox')) %>%
  spread(child_tox, n) %>%
  mutate(
    n_child = child_tox + child_nontox,
    p_child_tox = child_tox / n_child,
    p_child_ci = binomial_95_ci(p_child_tox, n_child)
  ) %>%
  filter(n_child > min_bucket_n)

df_n_follows_bin_post <- df_n_follows_bin_post %>%
  group_by() %>%
  mutate(
    parent_tox = case_when(parent_tox == T ~ "post = toxic", parent_tox == F ~ "post = nontoxic"),
    parent_tox = factor(parent_tox, c("post = toxic", "post = nontoxic"))
  )

write_csv(df_n_follows_bin_post, sprintf("%s/dyad_inf_%s.csv", plt_csv_dir, ds))


#
# EMBEDDEDNESS
# p(reply = toxic | friends embeddedness)
#
n_bins <- 12
min_bucket_n = 250

df_emb <- df %>%
  filter(dyad_n_common_friends > 0) %>%
  mutate(dyad_n_common_friends_bin = log10_bin(dyad_n_common_friends, n_breaks = n_bins)) %>%
  group_by(parent_tox, child_tox, dyad_n_common_friends_bin) %>%
  summarise(n = n()) %>%
  na.omit

df_emb <- df_emb %>%
  group_by() %>%
  mutate(child_tox = ifelse(child_tox == T, 'child_tox', 'child_nontox')) %>%
  spread(child_tox, n) %>%
  mutate(
    n_child = child_tox + child_nontox,
    p_child_tox = child_tox / n_child,
    p_child_ci = binomial_95_ci(p_child_tox, n_child)
  ) %>%
  filter(n_child > min_bucket_n)

df_emb <- df_emb %>%
  group_by() %>%
  mutate(
    parent_tox = case_when(parent_tox == T ~ "post = toxic", parent_tox == F ~ "post = nontoxic"),
    parent_tox = factor(parent_tox, c("post = toxic", "post = nontoxic"))
  )

write_csv(df_emb, sprintf("%s/dyad_emb_%s.csv", plt_csv_dir, ds))

# END