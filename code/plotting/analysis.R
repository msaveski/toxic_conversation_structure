
library(tidyverse)
library(scales)
library(cowplot)
library(latex2exp)


# NB: setwd to the project root directory

# load utils
source("code/analyses/utils.R", echo=F)

# paths
csv_in_dir <- "data/plots_csvs/"
plots_out_dir <- "plots/"

#
# colors
#
tox_colors <- c(
  "#EE0000FF", # red
  "#3B4992FF"  # blue
)

ds_colors <- c(
  "#008B45FF", # green
  "#631879FF"  # purple
)

ds_news_shades <- c('#33004a', '#691e7f', '#994dae', '#ca7bdf', '#ffaeff') # purple
ds_midterms_shades <- c('#002e00', '#005c1b', '#058e48', '#50c075', '#85f4a5') # green

hex_with_alpha <- function(hex, alpha){
  # PDFs with transperancy may not be displayed properly on older PDF viewers, so 
  #   instead we are derving the lighter color rather than using transparancy
  #   (ACM camera-ready requirement)
  
  # NB: assumes white background
  
  # hex -> rgb
  rgb_col <- col2rgb(hex)
  r <- rgb_col[1,]
  g <- rgb_col[2,]
  b <- rgb_col[3,]
  
  new_hex <- rgb(
    (1 - alpha) * 255 + alpha * r, 
    (1 - alpha) * 255 + alpha * g, 
    (1 - alpha) * 255 + alpha * b, 
    maxColorValue = 255
  )
  
  return(new_hex)
}

tox_colors_trans <- hex_with_alpha(tox_colors, 0.3)
ds_colors_trans <- hex_with_alpha(ds_colors, 0.4)


#
# Individual level [num tweets (tox/all) | fraction of toxicity | toxicity rate]
# 

# number of tweets
df_ego_n_tweets <- rbind(
  read_csv(str_c(csv_in_dir, "ego_n_tweets_news.csv"), col_types="ddc") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "ego_n_tweets_midterms.csv"), col_types="ddc") %>% mutate(ds="Midterms")
)

plt_ego_n_tweets <- df_ego_n_tweets %>%
  ggplot(aes(x = bin, y = n_users, color = set)) +
  geom_line(size = 0.3) +
  geom_point(size = 2.25) +
  facet_grid(fct_rev(ds) ~ .) +
  scale_x_log10(breaks = log10_breaks, labels = log10_labels) + 
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x, n=4), labels = log10_labels) + 
  scale_color_manual(values = tox_colors) +
  guides(color = guide_legend(
    keyheight = 3.5,
    keywidth = 5.5,
    default.unit = "mm",
    reverse = T
  )) +
  labs(x="Num. of Tweets / User (log)", y="Number of Users (log)") +
  theme_light() +
  # legend inside
  theme(
    legend.text = element_text(size = 7.5),
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.995, .997),
    legend.margin = margin(-5, 3, 3, 3)
  ) + 
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.background = element_blank(),
    strip.text.y = element_blank()
  )

print(plt_ego_n_tweets)

# fraction of toxicity
df_ego_frac_tox <- rbind(
  read_csv(str_c(csv_in_dir, "ego_frac_tox_news.csv"), col_types="ddddddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "ego_frac_tox_midterms.csv"), col_types="ddddddd") %>% mutate(ds="Midterms")
)

plt_ego_frac_tox <- df_ego_frac_tox %>%
  ggplot(aes(x = n_tox_tweets_bin, y = n_tox_tweets_bin_f_all)) +
  geom_segment(
    aes(
      x = n_tox_tweets_bin,
      xend = n_tox_tweets_bin,
      y = 0,
      yend = n_tox_tweets_bin_f_all
    ),
    color = tox_colors[1],
    size = 0.3
  ) +
  geom_point(color = tox_colors[1], size = 3) +
  facet_grid(fct_rev(ds) ~ .) +
  scale_x_log10(breaks = log10_breaks, labels = log10_labels) +
  scale_y_continuous(breaks = c(0.0, 0.1, 0.2), limits = c(0, 0.225)) +
  labs(x = "Num. of Toxic Tweets / User (log)  ", y = "Fraction of Overall Toxicity") +
  theme_light()  + 
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.background = element_blank(),
    strip.text.y = element_blank()
  )

print(plt_ego_frac_tox)

# toxicity rate
df_ego_tox_rate <- rbind(
  read_csv(str_c(csv_in_dir, "ego_rate_tox_news.csv"), col_types="ddddddddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "ego_rate_tox_midterms.csv"), col_types="ddddddddd") %>% mutate(ds="Midterms")
)

plt_ego_tox_rate <- df_ego_tox_rate %>%
  ggplot(aes(x=n_tweets_bin, y=f_tox_tweets_avg)) +
  geom_ribbon(
    aes(
      ymin = f_tox_tweets_avg - f_tox_tweets_avg_95_ci,
      ymax = f_tox_tweets_avg + f_tox_tweets_avg_95_ci
    ),
    fill = hex_with_alpha(tox_colors[2], 0.35)
  ) +  
  geom_line(color =  tox_colors[2]) +
  geom_point(size = 2.25, color = tox_colors[2]) +  
  facet_grid(fct_rev(ds) ~ .) +
  scale_x_log10(breaks = log10_breaks, labels = log10_labels) +
  scale_y_continuous(breaks = c(0.0, 0.1, 0.2), limits = c(0, 0.225)) +
  labs(x = "Num. of Tweets / User (log)", y = "Average Fraction of Toxic Tweets") +  
  theme_light() + 
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.background = element_rect(fill="gray90"),
    strip.text.y = element_text(size = 11, color = "black", face = "bold")
  )

print(plt_ego_tox_rate)

# ego grid
plt_ego <- plot_grid(
  plt_ego_n_tweets + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_ego_frac_tox + theme(plot.margin = unit(c(0, 1, 0, 1), "mm")),
  plt_ego_tox_rate + theme(plot.margin = unit(c(0, 0, 0, 1), "mm")),
  nrow = 1,
  align = "h",
  rel_widths = c(1.05, 1.07, 1.155)
)

ggsave(str_c(plots_out_dir, "ego.pdf"), plot=plt_ego, width = 8, height = 3.5)

#
# Dyad level [edge type | influence gap | embeddedness]
#
y_lims <- c(0, 0.384)

# edge type
df_dyad_edge_type <- rbind(
  read_csv(str_c(csv_in_dir, "dyad_edgetype_news.csv"), col_types="ccddddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "dyad_edgetype_midterms.csv"), col_types="ccddddd") %>% mutate(ds="Midterms")
)

plt_dyad_edge_type <- df_dyad_edge_type %>%
  ggplot(aes(
    x = fct_reorder(dyad_type, p_child_tox),
    y = p_child_tox,
    fill = fct_rev(parent_tox)
  )) +
  geom_bar(position = position_dodge(), stat = "identity") +
  geom_errorbar(
    aes(ymin = p_child_tox - p_child_ci, ymax = p_child_tox + p_child_ci),
    position = position_dodge(0.9),
    width = 0.5
  ) +
  scale_x_discrete(labels = c(
    "O  O" = "O   O",
    "O->O" = expression("O" %->% "O"),
    "O<-O" = expression("O" %<-% "O"),
    "O==O" = "O==O"
  )) +
  scale_y_continuous(limits = y_lims) +
  scale_fill_manual(values = hex_with_alpha(tox_colors, 0.9)) +
  guides(fill = guide_legend(
    keyheight = 3.7,
    keywidth = 4,
    default.unit = "mm"
  )) +   
  labs(
    x = "Edge Type",
    y = "p(reply = toxic | post)"
  ) +
  facet_grid(fct_rev(ds) ~ ., scales = "fixed") +
  theme_light() +
  theme(
    legend.text = element_text(size = 8.5),
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.57, .485),
    legend.margin = margin(-5, 3, 3, 3)
  ) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(color = "black"),
    strip.text.y = element_blank()
  )  

print(plt_dyad_edge_type)

# influence gap
df_dyad_inf_gap <- rbind(
  read_csv(str_c(csv_in_dir, "dyad_inf_news.csv"), col_types="cdddddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "dyad_inf_midterms.csv"), col_types="cdddddd") %>% mutate(ds="Midterms")
)

plt_dyad_inf_gap <- df_dyad_inf_gap %>%
  ggplot(aes(
    x = pc_bin,
    y = p_child_tox,
    color = fct_rev(parent_tox),
    fill = fct_rev(parent_tox)
  )) +
  geom_ribbon(
    aes(
      ymin = p_child_tox - p_child_ci,
      ymax = p_child_tox + p_child_ci
    ),
    # alpha = 0.3, 
    color = NA
  ) +  
  geom_vline(xintercept = 0, linetype = "dashed") +
  expand_limits(y = 0) +
  geom_line() +
  geom_point(size = 2.25) +
  facet_grid(fct_rev(ds) ~ .) +
  scale_y_continuous(limits = y_lims) +
  scale_color_manual(values = tox_colors) +
  scale_fill_manual(values = hex_with_alpha(tox_colors, 0.25)) +
  guides(color = guide_legend(
    keyheight = 3.7,
    keywidth = 4.5,
    default.unit = "mm"
  ),
  fill = "none") +
  labs(x = "Parent / Child Followers (log)",
       y = "p(reply = toxic | post)") +
  theme_light() +
  theme(
    legend.text = element_text(size = 8.5),
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.995, .10),
    legend.margin = margin(-5, 1, 2, 0.5)
  ) +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    strip.background = element_blank(),
    strip.text.y = element_blank()
  ) 

print(plt_dyad_inf_gap)

# embeddedness
df_dyad_emb <- rbind(
  read_csv(str_c(csv_in_dir, "dyad_emb_news.csv"), col_types="cdddddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "dyad_emb_midterms.csv"), col_types="cdddddd") %>% mutate(ds="Midterms")
)

plt_dyad_emb <- df_dyad_emb %>%
  ggplot(aes(
    x = dyad_n_common_friends_bin,
    y = p_child_tox,
    color = fct_rev(parent_tox),
    fill = fct_rev(parent_tox)
  )) +
  geom_ribbon(
    aes(
      ymin = p_child_tox - p_child_ci,
      ymax = p_child_tox + p_child_ci
    ),
    # alpha = 0.3, 
    color = NA
  ) +  
  expand_limits(y = 0) +
  geom_line() +
  geom_point(size = 2.25) +
  facet_grid(fct_rev(ds) ~ .) +
  scale_x_log10(breaks = log10_breaks, labels = log10_labels) +
  scale_y_continuous(limits = y_lims) +
  scale_color_manual(values = tox_colors) +
  # scale_fill_manual(values = tox_colors) +
  scale_fill_manual(values = hex_with_alpha(tox_colors, 0.25)) +
  guides(color = guide_legend(
    keyheight = 3.7,
    keywidth = 5.5,
    default.unit = "mm"
  ), fill = "none") +  
  labs(x = "Embeddedness (log)",
       y = "p(reply = toxic | post)") +
  theme_light() +
  theme(
    legend.text = element_text(size = 8.5),
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.995, .1),
    legend.margin = margin(-5, 2, 2, 1)
  ) +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    strip.background = element_rect(fill = "gray90"),
    strip.text.y = element_text(size = 10.5, color = "black", face = "bold")
  )

print(plt_dyad_emb)


# dyad grid
plt_dyad <- plot_grid(
  plt_dyad_edge_type + theme(plot.margin = unit(c(1, 1, 0, 0), "mm")),
  plt_dyad_inf_gap + theme(plot.margin = unit(c(0, 1, 0, 1), "mm")),
  plt_dyad_emb + theme(plot.margin = unit(c(0, 0, 0, 1), "mm")),
  nrow = 1,
  align = "h",
  rel_widths = c(1.01, 0.87, 0.94)
)

ggsave(str_c(plots_out_dir, "dyad.pdf"), plot = plt_dyad, width = 7, height = 3.75)


#
# Reply trees [1: size | depth | width]
#
df_rt_size <- rbind(
  read_csv(str_c(csv_in_dir, "rt_size_news.csv"), col_types="dcddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "rt_size_midterms.csv"), col_types="dcddd") %>% mutate(ds="Midterms")
)

df_rt_depth <- rbind(
  read_csv(str_c(csv_in_dir, "rt_depth_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "rt_depth_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

df_rt_width <- rbind(
  read_csv(str_c(csv_in_dir, "rt_width_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "rt_width_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

# combine dfs
df_rt1 <- rbind(
  df_rt_size %>% 
    filter(var == "Number of Tweets") %>% 
    rename(bin = x) %>% 
    select(-var) %>% 
    mutate(var = "Size (log)", var_order = 1),
  #
  df_rt_depth %>% 
    rename(bin = depth_bin) %>%
    mutate(var = "Depth (log)", var_order = 2),
  #
  df_rt_width %>% 
    rename(bin = width_bin) %>%
    mutate(var = "Width (log)", var_order = 3)
)

plt_rt1 <- df_rt1 %>%
  ggplot(aes(x = bin, y = bin_tox_mean, color = ds, fill = ds)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4,
    color = NA
  ) +  
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  expand_limits(y = 0) +
  facet_grid(. ~ fct_reorder(var, var_order), scales = "free_x", switch = 'x') +
  scale_x_log10(breaks=log10_breaks, labels=log10_labels) + 
  scale_y_continuous(breaks = pretty_breaks(n = 3)) + 
  scale_color_manual(values = ds_colors) +
  # scale_fill_manual(values = ds_colors) +
  scale_fill_manual(values = ds_colors_trans) +
  guides(
    color = guide_legend(
      keyheight = 5,
      default.unit = "mm",
      reverse = T
    ),
    fill = "none"
  ) +
  labs(x = "", y = "Mean fraction of toxic tweets") +
  theme_light() +
  # legend inside
  theme(
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.998, .22),
    legend.margin = margin(-5, 3, 3, 3)
  ) +
  # strips as axis labels
  theme(
    strip.placement = "outside",
    axis.title.x = element_blank(),
    strip.text = element_text(color = "black", size = 11, margin = margin(t=0, b=2)), 
    strip.background = element_blank()
  ) +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "mm")
  )

print(plt_rt1)

ggsave(str_c(plots_out_dir, "rt1.pdf"), plot=plt_rt1, width = 6, height = 2.5)


#
# Reply trees [1: weiner index | wi x size, news | wi x size, midterms]
#
x_breaks <- c(1, 10^0.4, 10^0.8, 10^1.2)
x_lims <- c(1, 10^1.22)
y_breaks <- c(0, 0.1, 0.2, 0.3)
y_lims <- c(0, 0.33)

df_rt_wi <- rbind(
  read_csv(str_c(csv_in_dir, "rt_wiener_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "rt_wiener_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

plt_rt_wi <- df_rt_wi %>%
  ggplot(aes(x = sv_bin, y = bin_tox_mean, color = ds, fill = ds)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) +
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  expand_limits(y = 0) +
  scale_x_log10(breaks = x_breaks, labels = log10_labels, limits = x_lims) + 
  scale_y_continuous(breaks = y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_colors) +
  # scale_fill_manual(values = ds_colors) +
  scale_fill_manual(values = ds_colors_trans) +
  guides(color = guide_legend(
    keyheight = 3.5,
    default.unit = "mm",
    reverse = T
  ),
  fill = "none") +
  labs(x="", y="Mean fraction of toxic tweets") +
  theme_light() +
  # legend inside
  theme(
    legend.text = element_text(size = 7.5), 
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.995, .16),
    legend.margin = margin(-5, 3, 3, 3)
  ) +
  # remove minor gridlines
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank()
  )

print(plt_rt_wi)

# wi x size | news
df_rt_wi_size_news <- read_csv(str_c(csv_in_dir, "rt_wiener_size_news.csv"), col_types="cdddd")

df_rt_wi_size_news <- df_rt_wi_size_news %>%
  mutate(
    n_tweets_bin = factor(n_tweets_bin), 
    n_tweets_bin = fct_recode(
      n_tweets_bin,
      "[1, 5]" = "[1,5.4972]", 
      "(5, 30]" = "(5.4972,30.22]", 
      "(30, 166]"  = "(30.22,166.12]", 
      "(166, 913]" = "(166.12,913.22]", 
      "(913, 5020]" = "(913.22,5020.2]"
    ),
    n_tweets_bin = fct_relevel(
      n_tweets_bin,
      rev(c("[1, 5]", "(5, 30]", "(30, 166]", "(166, 913]", "(913, 5020]"))
    )
  )

plt_rt_wi_size_news <- df_rt_wi_size_news %>%
  ggplot(aes(x = sv_bin, y = bin_tox_mean, color = n_tweets_bin, fill = n_tweets_bin)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) + 
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  annotate(
    "label",
    x = Inf,
    y = Inf,
    label = "News",
    size = 3.5,
    label.r = unit(0, "lines"),
    label.padding = unit(0.3, "lines"),
    vjust = 1,
    hjust = 1
  ) +
  expand_limits(y = 0) +
  scale_x_log10(breaks = x_breaks, labels = log10_labels, limits = x_lims) + 
  scale_y_continuous(breaks = y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_news_shades) +
  scale_fill_manual(values = hex_with_alpha(ds_news_shades, 0.4)) +
  guides(
    color = guide_legend(
      keyheight = 2.9,
      keywidth = 4.7,
      default.unit = "mm",
      reverse = T
    ),
    fill = "none"
  ) +
  labs(x = "Wiener index (log)", y = NULL, color="Num. of tweets") +
  theme_light() +
  # legend inside
  theme(
    legend.text = element_text(size = 7.5), 
    legend.title = element_text(size = 7.5, margin = margin(0, 0,-1, 0, unit = "mm")), 
    legend.justification = c("right", "top"),
    legend.position = c(.994, .347),
    legend.margin = margin(1, 1, 1, 1)
  ) + 
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  # remove minor gridlines
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank()
  )

print(plt_rt_wi_size_news)

# wi x size | midterms
df_rt_wi_size_midterms <- read_csv(str_c(csv_in_dir, "rt_wiener_size_midterms.csv"), col_types="cdddd")

df_rt_wi_size_midterms <- df_rt_wi_size_midterms %>%
  mutate(
    n_tweets_bin = factor(n_tweets_bin), 
    n_tweets_bin = fct_recode(
      n_tweets_bin,
      "[1, 5]" = "[1,5.8865]",
      "(5, 34]" = "(5.8865,34.65]",
      "(34, 203]" = "(34.65,203.97]",
      "(203, 1200]" = "(203.97,1200.7]",
      "(1200, 7067]" = "(1200.7,7067.6]"
    ),
    n_tweets_bin = fct_relevel(
      n_tweets_bin,
      rev(c("[1, 5]", "(5, 34]", "(34, 203]", "(203, 1200]", "(1200, 7067]"))
    )
  )

plt_rt_wi_size_midterms <- df_rt_wi_size_midterms %>%
  ggplot(aes(x = sv_bin, y = bin_tox_mean, color = n_tweets_bin, fill = n_tweets_bin)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) +
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  annotate(
    "label",
    x = Inf,
    y = Inf,
    label = "Midterms",
    size = 3.5,
    label.r = unit(0, "lines"),
    label.padding = unit(0.3, "lines"),
    vjust = 1,
    hjust = 1
  ) +  
  expand_limits(y = 0) +
  scale_x_log10(breaks = x_breaks, labels = log10_labels, limits = x_lims) + 
  scale_y_continuous(breaks = y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_midterms_shades) +
  # scale_fill_manual(values = ds_midterms_shades) +
  scale_fill_manual(values = hex_with_alpha(ds_midterms_shades, 0.4)) +
  guides(
    color = guide_legend(
      keyheight = 2.9,
      keywidth = 4.7,
      default.unit = "mm",
      reverse = T
    ),
    fill = "none"
  ) +
  labs(x = NULL, y = NULL, color="Num. of tweets") +
  theme_light() +
  # legend inside
  theme(
    legend.text = element_text(size = 7.5), 
    legend.title = element_text(size = 7.5, margin = margin(0, 0,-1, 0, unit = "mm")), 
    legend.justification = c("right", "top"),
    legend.position = c(.994, .347),
    legend.margin = margin(1, 1, 1, 1)
  ) + 
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  # remove minor gridlines
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank()
  )

print(plt_rt_wi_size_midterms)

# rt2 grid
plt_rt2 <- plot_grid(
  plt_rt_wi + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_rt_wi_size_news + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_rt_wi_size_midterms + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  nrow = 1,
  align = "h",
  rel_widths = c(1.2, 1, 1)
)

ggsave(str_c(plots_out_dir, "rt2.pdf"), plot=plt_rt2, width = 6, height = 2.5)

#
# Follow graph [density | nCC | modularity]
# 
y_breaks <- c(0, 0.1, 0.2, 0.3)
y_lims <- c(0, 0.3)

# density
df_fg_density <- rbind(
  read_csv(str_c(csv_in_dir, "fg_density_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "fg_density_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

plt_fg_density <- df_fg_density %>%
  ggplot(aes(x = density_bin, y = bin_tox_mean, color = ds, fill = ds)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) +
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  expand_limits(y = 0) +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x)10 ^ x, n = 4),
    labels = log10_labels,
    limit = c(10^-3.25, 1)
  ) +
  scale_y_continuous(breaks=y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_colors) +
  scale_fill_manual(values = ds_colors_trans) +
  guides(color = "none", fill = "none") +
  labs(x="Density (log)", y="Mean fraction of toxic tweets") +
  theme_light() +
  # remove minor gridlines
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank()
  )

print(plt_fg_density)

# n CC
df_fg_ncc <- rbind(
  read_csv(str_c(csv_in_dir, "fg_nCC_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "fg_nCC_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

plt_fg_ncc <- df_fg_ncc %>%
  ggplot(aes(x = n_CC_bin, y = bin_tox_mean, color = ds, fill = ds)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) +
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  expand_limits(y = 0) +
  scale_x_log10(breaks=trans_breaks("log10", function(x) 10^x, n=3), labels=log10_labels) +
  scale_y_continuous(breaks=y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_colors) +
  scale_fill_manual(values = ds_colors_trans) +
  guides(color = "none", fill = "none") +
  labs(x = "Num. of CCs (log)", y = NULL) +
  theme_light() + 
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

print(plt_fg_ncc)

# modularity
df_fg_modularity <- rbind(
  read_csv(str_c(csv_in_dir, "fg_modularity_news.csv"), col_types="dddd") %>% mutate(ds="News"),
  read_csv(str_c(csv_in_dir, "fg_modularity_midterms.csv"), col_types="dddd") %>% mutate(ds="Midterms")
)

plt_fg_modularity <- df_fg_modularity %>%
  ggplot(aes(x = modularity_bin, y = bin_tox_mean, color = ds, fill = ds)) +
  geom_ribbon(
    aes(ymin = bin_tox_mean - bin_tox_ci, ymax = bin_tox_mean + bin_tox_ci),
    # alpha = 0.4, 
    color = NA
  ) +
  geom_line(size = 0.3) +
  geom_point(size = 2.5) +
  expand_limits(y = 0) +
  scale_x_continuous() +
  scale_y_continuous(breaks = y_breaks, limits = y_lims) + 
  scale_color_manual(values = ds_colors) +
  scale_fill_manual(values = ds_colors_trans) +
  guides(
    color = guide_legend(
      keyheight = 5,
      default.unit = "mm",
      reverse = T
    ),
    fill = "none"
  ) +
  labs(x = "Modularity", y = NULL) +
  theme_light() + 
  theme(
    legend.text = element_text(size = 9), 
    legend.title = element_blank(),
    legend.justification = c("right", "top"),
    legend.position = c(.995, .215),
    legend.margin = margin(-5, 3, 3, 3)
  ) +
  theme(
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(margin = margin(t = 4.5, r = 0, b = 0, l = 0))
  ) 

print(plt_fg_modularity)

plt_fg <- plot_grid(
  plt_fg_density + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_fg_ncc + theme(plot.margin = unit(c(0, 1, 0, 0), "mm")),
  plt_fg_modularity + theme(plot.margin = unit(c(0, 0, 0, 0), "mm")),
  nrow = 1,
  align = "h",
  rel_widths = c(1.2, 1, 1)
)

ggsave(str_c(plots_out_dir, "fg.pdf"), plot=plt_fg, width = 6, height = 2.5)

# END