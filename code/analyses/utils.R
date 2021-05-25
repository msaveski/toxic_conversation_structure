
# UTILS

#
# log 10 breaks
#
log10_breaks <- scales::trans_breaks("log10", function(x) 10^x)
log10_labels <- scales::trans_format("log10", scales::math_format(10^.x))


#
# Confidance intervals
#
binomial_95_ci <- function(p, n) {
  return(1.96 * sqrt(p * (1.0 - p) / n))
}

gaussian_mean_95_ci <- function(sample) {
  n <- length(sample)
  sample_std <- sd(sample)
  return(1.96 * sample_std / sqrt(n))
}

#
# Binning
#

lin_bin <- function(values, n_breaks) {
  min_value <- min(values)
  max_value <- max(values)
  
  breaks <- seq(min_value, max_value, length.out = n_breaks)
  
  # see where the values fall within the breaks (returns index of the break)
  bins_inds = findInterval(values, breaks) 
  values_bins = breaks[bins_inds]
  
  return(values_bins)
}

log10_bin <- function(values, n_breaks) {
  min_log10_value <- log10(min(values))
  max_log10_value <- log10(max(values))
  
  log10_breaks <- seq(min_log10_value, max_log10_value, length.out = n_breaks)
  breaks <- 10^log10_breaks
  
  # see where the values fall within the breaks (returns index of the break)
  bins_inds = findInterval(values, breaks, all.inside = T) 
  values_bins = breaks[bins_inds]
  
  return(values_bins)
}

log10_bin_factor <- function(values, n_breaks, dig.lab=5) {
  min_log10_value <- log10(min(values))
  max_log10_value <- log10(max(values) + 1)
  
  log10_breaks <- seq(min_log10_value, max_log10_value, length.out = n_breaks)
  breaks <- 10^log10_breaks
  
  values_bins = cut(values, breaks = breaks, include.lowest=T, right=T, dig.lab=dig.lab)
  
  return(values_bins)
}

# END