#!/usr/bin/env Rscript
# Camera-trap distance sampling (CTDS) density/abundance estimation.
#
# Port of wcf-pps-p3/scripts/PSS_P3_Chimp_CTDS_all.R: fits an activity model
# (activity::fitact) on independent detection events, fits a family of
# distance-sampling detection functions (Distance::ds, point-transect
# "snapshot" design), auto-selects a model (best QAIC within each key-function
# family, then lowest chi2_select criterion across families), and produces
# density/abundance estimates via Distance::dht2 with the activity rate as a
# "creation" availability multiplier.
#
# Usage:
#   Rscript ctds_abundance.R --flatfile ctds_flatfile.csv \
#       --activity activity_times.csv --config ctds_config.yaml \
#       --out-dir OUT_DIR [--model auto|hn0|hn1|hn2|uni1|uni2|hr0|hr1] \
#       [--activity-rate R --activity-se S]
#
# If --activity-rate/--activity-se are given, they override fitting the
# activity model from --activity (required if activity_times.csv is empty).

suppressMessages({
  library(Distance)
  library(dplyr)
  library(activity)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  args[idx + 1]
}

flatfile_path <- get_arg("--flatfile")
activity_path <- get_arg("--activity")
config_path <- get_arg("--config")
out_dir <- get_arg("--out-dir")
model_choice <- get_arg("--model", "auto")
activity_rate_override <- get_arg("--activity-rate")
activity_se_override <- get_arg("--activity-se")

if (is.null(flatfile_path) || is.null(activity_path) || is.null(config_path) || is.null(out_dir)) {
  stop(paste(
    "Usage: ctds_abundance.R --flatfile F --activity A --config C --out-dir D",
    "[--model auto|hn0|hn1|hn2|uni1|uni2|hr0|hr1]",
    "[--activity-rate R --activity-se S]"
  ))
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

config <- yaml::read_yaml(config_path)
conversion <- convert_units("meter", NULL, "square kilometer")

## ---- Activity model -------------------------------------------------------

if (!is.null(activity_rate_override) && !is.null(activity_se_override)) {
  activity_rate <- as.numeric(activity_rate_override)
  activity_se <- as.numeric(activity_se_override)
  cat(sprintf("Using activity override: rate=%.6f SE=%.6f\n", activity_rate, activity_se))
} else {
  activity_df <- tryCatch(read.csv(activity_path, stringsAsFactors = FALSE), error = function(e) NULL)
  if (is.null(activity_df) || nrow(activity_df) == 0 || !("new_event" %in% names(activity_df))) {
    stop(
      "activity_times.csv is empty or missing 'new_event'; supply ",
      "--activity-rate and --activity-se to override the activity model."
    )
  }
  events <- activity_df[activity_df$new_event == 1, , drop = FALSE]
  events$rtime <- gettime(events$time_hm, tryFormats = "%H:%M", scale = "radian")

  mod1 <- fitact(
    events$rtime,
    reps = config$activity$reps,
    sample = config$activity$sample,
    adj = config$activity$adj
  )
  activity_rate <- unname(mod1@act[1])
  activity_se <- unname(mod1@act[2])
  cat(sprintf("Fitted activity: rate=%.6f SE=%.6f (n independent events=%d)\n",
              activity_rate, activity_se, nrow(events)))

  png(file.path(out_dir, "activity.png"), width = 900, height = 500, res = 100)
  plot(mod1, cex.lab = 1.2,
       main = sprintf("Activity = %.3f (SE %.3f)", activity_rate, activity_se))
  dev.off()
}

camera.operation.per.day <- config$camera_operation_hours_per_day
prop.camera.time <- camera.operation.per.day / 24
avail <- list(creation = data.frame(
  rate = activity_rate / prop.camera.time,
  SE = activity_se / prop.camera.time
))

## ---- Distance data ---------------------------------------------------------

dist <- read.csv(flatfile_path, stringsAsFactors = FALSE)
dist$distance <- as.numeric(dist$distance)
dist$object <- suppressWarnings(as.integer(dist$object))

trunc.list <- list(left = as.numeric(config$truncation$left), right = as.numeric(config$truncation$right))
mybreaks <- as.numeric(unlist(config$cutpoints))

png(file.path(out_dir, "distance_histogram.png"), width = 900, height = 500, res = 100)
in_range <- dist$distance[!is.na(dist$distance) & dist$distance >= trunc.list$left & dist$distance <= trunc.list$right]
hist(in_range, breaks = mybreaks, main = "Distance data (truncated)", xlab = "Radial distance (m)")
dev.off()

fit_model <- function(key, adjustment = NULL, nadj = NULL) {
  if (is.null(adjustment)) {
    ds(dist, transect = "point", key = key, adjustment = NULL,
       cutpoints = mybreaks, truncation = trunc.list, convert_units = conversion)
  } else {
    ds(dist, transect = "point", key = key, adjustment = adjustment, nadj = nadj,
       cutpoints = mybreaks, truncation = trunc.list, convert_units = conversion)
  }
}

fit_safe <- function(label, key, adjustment = NULL, nadj = NULL) {
  tryCatch(
    fit_model(key, adjustment, nadj),
    error = function(e) {
      cat(sprintf("WARNING: model %s failed to fit: %s\n", label, conditionMessage(e)))
      NULL
    }
  )
}

cat("Fitting candidate models...\n")
all_models <- list(
  hn0 = fit_safe("hn0", "hn", NULL, NULL),
  hn1 = fit_safe("hn1", "hn", "cos", 1),
  hn2 = fit_safe("hn2", "hn", "cos", 2),
  uni1 = fit_safe("uni1", "unif", "cos", 1),
  uni2 = fit_safe("uni2", "unif", "cos", 2),
  hr0 = fit_safe("hr0", "hr", NULL, NULL),
  hr1 = fit_safe("hr1", "hr", "poly", 1)
)
fitted_ok <- !sapply(all_models, is.null)
all_models <- all_models[fitted_ok]

## ---- Model selection --------------------------------------------------------
## Best QAIC within each key-function family, then lowest chi2_select
## criterion across the family winners (families with no successfully-fit
## model are skipped).

families <- list(hn = c("hn0", "hn1", "hn2"), unif = c("uni1", "uni2"), hr = c("hr0", "hr1"))

family_qaic <- list()
family_best_names <- character(0)
for (fam in names(families)) {
  present <- intersect(families[[fam]], names(all_models))
  if (length(present) == 0) next
  if (length(present) == 1) {
    family_best_names <- c(family_best_names, present)
    next
  }
  qaic_df <- do.call(QAIC, unname(all_models[present]))
  rownames(qaic_df) <- present
  family_qaic[[fam]] <- qaic_df
  family_best_names <- c(family_best_names, present[which.min(qaic_df$QAIC)])
}

if (length(family_best_names) >= 2) {
  chats <- do.call(chi2_select, unname(all_models[family_best_names]))
  rownames(chats) <- family_best_names
  auto_selected_name <- family_best_names[order(chats$criteria)][1]
} else {
  chats <- data.frame(criteria = NA_real_, row.names = family_best_names)
  auto_selected_name <- family_best_names[1]
}

if (model_choice == "auto") {
  selected_name <- auto_selected_name
} else {
  if (!(model_choice %in% names(all_models))) {
    stop(sprintf("Unknown --model '%s'; must be one of: %s, or 'auto'",
                  model_choice, paste(names(all_models), collapse = ", ")))
  }
  selected_name <- model_choice
}
selected_model <- all_models[[selected_name]]
cat(sprintf("Model auto-selected (family QAIC winners + chi2_select): %s\n", auto_selected_name))
cat(sprintf("Model used for abundance estimation: %s\n", selected_name))

qaic_lookup <- function(name) {
  for (qaic_df in family_qaic) {
    if (name %in% rownames(qaic_df)) return(qaic_df[name, "QAIC"])
  }
  NA_real_
}
chi2_lookup <- function(name) {
  if (name %in% rownames(chats)) return(chats[name, "criteria"])
  NA_real_
}

model_meta <- data.frame(
  model = c("hn0", "hn1", "hn2", "uni1", "uni2", "hr0", "hr1"),
  key = c("hn", "hn", "hn", "unif", "unif", "hr", "hr"),
  adjustment = c("none", "cos1", "cos2", "cos1", "cos2", "none", "poly1"),
  stringsAsFactors = FALSE
)
model_selection <- model_meta %>%
  mutate(
    fitted = model %in% names(all_models),
    QAIC_within_family = sapply(model, qaic_lookup),
    chi2_criterion = sapply(model, chi2_lookup),
    family_qaic_winner = model %in% family_best_names,
    selected = model == selected_name
  )
write.csv(model_selection, file.path(out_dir, "model_selection.csv"), row.names = FALSE)

png(file.path(out_dir, "detection_function.png"), width = 900, height = 500, res = 100)
par(mfrow = c(1, 2), cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.3, mar = c(5, 5, 2, 2), oma = c(0, 0, 4, 0))
plot(selected_model, xlab = "Distance (m)", showpoints = FALSE, lwd = 3)
plot(selected_model, xlab = "Distance (m)", pdf = TRUE, showpoints = FALSE, lwd = 3)
mtext(sprintf("Chimpanzee - %s", selected_name), outer = TRUE, cex = 1.2, line = 1.5)
dev.off()

## ---- Abundance / density estimation -----------------------------------------

result <- dht2(selected_model, flatfile = dist, strat_formula = ~Region.Label,
                sample_fraction = 1, multipliers = avail, convert_units = conversion)

report_txt <- capture.output(print(result, report = "both"))
writeLines(report_txt, file.path(out_dir, "dht2_report.txt"))

density_attr <- attr(result, "density")
prop_var <- attr(result, "prop_var")

estimates <- data.frame(
  model = selected_name,
  n = result$n,
  k = result$k,
  ER = result$ER,
  cv.ER = result$ER_CV,
  D = density_attr$Density,
  D_se = density_attr$Density_se,
  D_cv = density_attr$Density_CV,
  D_lcl = density_attr$LCI,
  D_ucl = density_attr$UCI,
  N = result$Abundance,
  N_se = result$Abundance_se,
  N_cv = result$Abundance_CV,
  N_lcl = result$LCI,
  N_ucl = result$UCI,
  activity_rate = activity_rate,
  activity_se = activity_se,
  pct_var_detection = if (!is.null(prop_var)) prop_var$Detection else NA,
  pct_var_ER = if (!is.null(prop_var)) prop_var$ER else NA,
  pct_var_multipliers = if (!is.null(prop_var)) prop_var$Multipliers else NA
)
write.csv(estimates, file.path(out_dir, "abundance_estimates.csv"), row.names = FALSE)

cat("Done. Outputs written to", out_dir, "\n")
print(result, report = "both")
