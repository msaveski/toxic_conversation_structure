
library(tidyverse)
library(cowplot)
library(ggthemes)
library(ggrepel)
library(lemon)


# paths
plots_out_dir <- "plots/"

# read data
df <- read_csv("data/modeling/prefix/runs_csvs/res_q50_nested.csv")

# remove constant fields
df <- df %>% select(-outcome, -n_samples, -inner_n_folds, -outer_n_folds)

# stats
nrow(df)
names(df)

# gather metrics => [dataset, prefix, feature_groups, clf_name,] metric, mean, sem
df <- df %>%
  gather(key, val, -dataset, -prefix, -feature_groups, -clf_name) %>%
  extract(key, c("metric", "stat"), "(.+)__(.+)") %>%
  spread(stat, val)

#
# Plots
#
df_plts <- df %>% 
  filter(
    clf_name == "GB",
    metric == "test_accuracy"
  ) %>% 
  mutate(
    metric = "ACC",
    ci = 1.96 * sem,
    # Adding spaces at the ends: a bit hacky trick to help geom_text_repel
    feature_groups = case_when(
      feature_groups == "all" ~ "         All         ",
      feature_groups == "all/toxicity" ~ "All \\ Content Toxicity",
      feature_groups == "toxicity" ~ "Content Toxicity",
      feature_groups == "tree" ~ "Reply Tree",
      feature_groups == "reply_graph" ~ "Reply Graph",
      feature_groups == "follow_graph" ~ "Follow Graph",
      feature_groups == "subgraph" ~ "     Subgraph     ",
      feature_groups == "embeddedness" ~ "Embeddedness",
      feature_groups == "polarization" ~ "     Alignment     ",
      feature_groups == "arrival_seq" ~ "Arrival Seq.",
      feature_groups == "rate" ~ "        Rate            ",
      T ~ feature_groups
    )
  )

# labels
df_plts <- df_plts %>%
  mutate(
    label_left = if_else(prefix == 10, str_trim(feature_groups, side ="right"), NA_character_),
    label_right = if_else(prefix == 100, str_trim(feature_groups, side ="left"), NA_character_)
  ) 

# news
plt_news <- df_plts %>% 
  filter(dataset == "news") %>% 
  ggplot(aes(x = prefix, y = mean, color = feature_groups)) +
  geom_errorbar(
    aes(ymin = mean - ci, ymax = mean + ci),
    colour = "grey",
    width = 0.3,
    size = 0.4
  ) +
  geom_line(size = 0.25) +
  geom_text_repel(
    aes(label = label_left),
    na.rm = TRUE,
    size = 3.5,
    hjust = 1,
    direction = "y",
    nudge_x = -13,
    segment.color = "grey",
    segment.size = 0.2
  ) +
  geom_text_repel(
    aes(label = label_right),
    na.rm = TRUE,
    size = 3.5,
    hjust = 0,
    direction = "y",
    nudge_x = 13,
    segment.color = "grey",
    segment.size = 0.2
  ) +
  geom_point(size = 0.75) +
  scale_y_continuous(
    breaks = seq(0.5, 0.65, 0.05),
    limits = c(0.5, 0.65)
  ) +
  scale_x_continuous(
    breaks = seq(10, 100, 10),
    limits = c(-30, 140),
    minor_breaks = NULL
  ) +
  labs(title = "News", x = "Prefix Size", y = "Accuracy") +
  coord_capped_cart(bottom = 'both', left = 'both') +
  theme_light() + 
  theme(
    plot.title = element_text(hjust = 0.5, margin = margin(t = 1, b = -20, unit = "pt")), 
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.border = element_blank(),
    axis.line.y = element_line(color = "grey"),
    axis.line.x = element_line(color = "black"),
    axis.ticks.y = element_line(color = "grey", size = 0.5),
    axis.ticks.x = element_line(color = "black", size = 0.5),
    plot.margin = grid::unit(c(0, 0, 0, 0), "mm")
  ) +
  scale_color_stata()

print(plt_news)

#
# Midterms
#
plt_midterms <- df_plts %>% 
  filter(dataset == "midterms") %>% 
  ggplot(aes(x = prefix, y = mean, color = feature_groups)) +
  geom_errorbar(
    aes(ymin = mean - ci, ymax = mean + ci),
    colour = "grey",
    width = 0.3,
    size = 0.4
  ) +
  geom_line(size = 0.25) +
  geom_text_repel(
    aes(label = label_left),
    na.rm = TRUE,
    size = 3.5,
    hjust = 1,
    direction = "y",
    nudge_x = -13,
    segment.color = "grey",
    segment.size = 0.2
  ) +
  geom_text_repel(
    aes(label = label_right),
    na.rm = TRUE,
    size = 3.5,
    hjust = 0,
    direction = "y",
    nudge_x = 13,
    segment.color = "grey",
    segment.size = 0.2
  ) +
  geom_point(size = 0.75) +
  scale_y_continuous(
    breaks = seq(0.5, 0.65, 0.05),
    limits = c(0.5, 0.65)
  ) +  
  scale_x_continuous(
    breaks = seq(10, 100, 10),
    limits = c(-30, 140),
    minor_breaks = NULL
  ) +
  labs(title = "Midterms", x = "Prefix Size", y = "Accuracy") +
  coord_capped_cart(bottom = 'both', left = 'both') +
  theme_light() + 
  theme(
    plot.title = element_text(hjust = 0.5, margin = margin(t = 1, b = -20, unit = "pt")), 
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.border = element_blank(),
    axis.line.y = element_line(color = "grey"),
    axis.line.x = element_line(color = "black"),
    axis.ticks.y = element_line(color = "grey", size = 0.5),
    axis.ticks.x = element_line(color = "black", size = 0.5),
    plot.margin = grid::unit(c(0, 0, 0, 0), "mm")
  ) +
  scale_color_stata()

print(plt_midterms)

#
# Grid
#
plt_grid <- plot_grid(
  plt_news + theme(plot.margin = grid::unit(c(1, 0, 0, 0), "mm")), 
  plt_midterms, 
  nrow = 1, 
  align = "h"
)

ggsave(str_c(plots_out_dir, "prefix.pdf"), plot=plt_grid, width = 14.5, height = 3.3)

# END