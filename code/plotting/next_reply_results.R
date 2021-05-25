
library(tidyverse)
library(cowplot)


# NB: setwd to the project root directory

# paths
plots_out_dir <- "plots/"

# read data
df <- read_csv("data/modeling/next_reply/runs_csvs/res_nested.csv")

df <- df %>% select(-outer_n_folds, -inner_n_folds)

nrow(df)
names(df)

# gather metrics => [dataset, feature_groups, clf_name,] metric, mean, sem
df <- df %>%
  gather(key, val, -dataset, -feature_groups, -clf_name) %>%
  extract(key, c("metric", "stat"), "(.+)__(.+)") %>%
  spread(stat, val)


#
# Performance Bar Plot
#
selected_metrics <- c(
  "test_accuracy",
  "test_roc_auc",
  "test_f1"
)

df_bar <- df %>%
  filter(
    metric %in% selected_metrics,
    clf_name == "GB"
  ) %>%
  mutate(
    feature_groups = case_when(
      feature_groups == "all" ~ "All",
      feature_groups == "conversation_state" ~ "Conversation State",
      feature_groups == "all/conversation_state" ~ "All \\ Conversation State",
      feature_groups == "dyad_up" ~ "User-Parent Dyad",
      feature_groups == "embeddedness_toxicity" ~ "Toxic Embeddedness",
      feature_groups == "dyad_ur" ~ "User-Root Dyad",
      feature_groups == "reply_di_ud_emb" ~ "Reply Graph",
      feature_groups == "follow_di_ud_emb" ~ "Follow Graph",
      feature_groups == "tree" ~ "Reply Tree",
      feature_groups == "embeddedness_all" ~ "Overall Embeddedness",
      feature_groups == "alignments" ~ "Political Alignment",
      feature_groups == "user_info" ~ "User Info",
      T ~ feature_groups
    ),
    feature_group_order = case_when(
      feature_groups == "All" ~ 1,
      feature_groups == "All \\ Conversation State" ~ 2,
      feature_groups == "Conversation State" ~ 3,
      feature_groups == "User-Parent Dyad" ~ 4,
      feature_groups == "Toxic Embeddedness" ~ 5,
      feature_groups == "Reply Graph" ~ 6,
      feature_groups == "User-Root Dyad" ~ 7,
      feature_groups == "Reply Tree" ~ 8,
      feature_groups == "Follow Graph" ~ 9,
      feature_groups == "User Info" ~ 10,
      feature_groups == "Overall Embeddedness" ~ 11,
      feature_groups == "Political Alignment" ~ 12
    ),    
    metric = case_when(
      metric == "test_accuracy" ~ "ACC",
      metric == "test_roc_auc" ~ "AUC",
      metric == "test_f1" ~ "F1"
    )
  )

# news
plt_news <- df_bar %>%
  filter(dataset == "news") %>%
  ggplot(aes(
    x = fct_reorder(feature_groups, -feature_group_order),
    y = mean
  )) +
  geom_bar(stat = "identity", fill = "#ddebf7", width = 0.84) +
  geom_errorbar(
    aes(ymin = mean - 1.96 * sem, ymax = mean + 1.96 * sem),
    width = 0.4,
    color = "grey70",
    size = 0.4
  ) +
  geom_text(aes(y = 0.25, label = sprintf("%0.3f", mean)), size = 3.6) +
  scale_y_continuous(breaks = c(0.0, 0.5)) +
  coord_flip() +
  labs(title = "News", x = NULL, y = NULL) +
  facet_grid(. ~ metric) +
  theme_light() +
  theme(
    plot.title = element_text(
      size = 13,
      hjust = 0.02,
      margin = margin(t = 1, b = 5, unit = "pt")
    ),
    strip.background = element_blank(),
    strip.text = element_text(
      color = "grey30",
      size = 11,
      hjust = 0.0,
      margin = margin(
        l = 2.5,
        t = 0,
        b = 2,
        unit = "pt"
      )
    ),
    panel.spacing = unit(0, "mm"),
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 8)
  )

print(plt_news)


# midterms
plt_midterms <- df_bar %>%
  filter(dataset == "midterms") %>%
  ggplot(aes(
    x = fct_reorder(feature_groups, -feature_group_order),
    y = mean
  )) +
  geom_bar(stat = "identity", fill = "#DAE7E0", width = 0.84) +
  geom_errorbar(
    aes(ymin = mean - 1.96 * sem, ymax = mean + 1.96 * sem),
    width = 0.4,
    color = "grey70",
    size = 0.4
  ) +
  geom_text(aes(y = 0.25, label = sprintf("%0.3f", mean)), size = 3.6) +
  scale_y_continuous(breaks = c(0.0, 0.5)) +
  coord_flip() +
  labs(title = "Midterms", x = NULL, y = NULL) +
  facet_grid(. ~ metric) +
  theme_light() +
  theme(
    plot.title = element_text(
      size = 13,
      hjust = 0.02,
      margin = margin(t = 1, b = 5, unit = "pt")
    ),
    strip.background = element_blank(),
    strip.text = element_text(
      color = "grey30",
      size = 11,
      hjust = 0.0,
      margin = margin(
        l = 2.5,
        t = 0,
        b = 2,
        unit = "pt"
      )
    ),
    panel.spacing = unit(0, "mm"),
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 8)
  )

print(plt_midterms)


# grid
plt_grid <- plot_grid(
  plt_news + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_midterms + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  nrow = 1,
  align = "h",
  rel_widths = c(1.56, 1)
)

print(plt_grid)

ggsave(
  str_c(plots_out_dir, "next_reply.pdf"),
  plot = plt_grid,
  width = 6,
  height = 3
)

# END