getwd()
###############################################################################
#  Neural Network Analysis of U.S. House Prices (2019-2025)
#  --------------------------------------------------------
#  Uses the CSV in its ORIGINAL wide format (7 rows x 15,098 columns).
#
#  Feature matrix X (SAME for every county model):
#     All non-house-price, non-year columns = 12,235 predictors
#        National   (2) : mortgage_rate_30yr, construction_cost_index
#        State    (153) : 51 x 3 (income, unemployment, net migration)
#        County (12,080) : ~3,020 x 4 (population, growth rate,
#                                       building permits, unemployment)
#
#  Response y (different for each of the 2,862 counties):
#     house_price_<County>_<ST>
#
#  Each county model:  X is 7 x 12,235  |  y is 7 x 1
#  Train: 2019-2024 (6 obs)  |  Test: 2025 (1 obs)
#
#  Uses the nnet package (single hidden layer neural network).
#  With p >> n, the hidden layer size is kept small and weight
#  decay (L2 regularisation) is applied to prevent overfitting.
#
#  Outputs:
#     - nn_results_all_counties.csv
#     - nn_actual_vs_predicted.png
#     - nn_residual_distribution.png
#     - nn_top_bottom_counties.png
###############################################################################

# -- 0.  Packages --------------------------------------------------------------
required_pkgs <- c("nnet", "data.table", "ggplot2")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cran.r-project.org")
}
library(nnet)
library(data.table)
library(ggplot2)

# -- 1.  Read data -------------------------------------------------------------
# *** UPDATE this path to your CSV location ***
csv_path <- "house_price_data_2019_2025.csv"
wide     <- fread(csv_path)
cat("Data loaded:", nrow(wide), "rows x", ncol(wide), "columns\n\n")

# -- 2.  Separate predictors (X) from responses (house_price_*) ----------------
hp_cols <- grep("^house_price_", names(wide), value = TRUE)
x_cols  <- setdiff(names(wide), c("year", hp_cols))

cat("Predictor columns (X):", length(x_cols), "\n")
cat("House-price response columns:", length(hp_cols), "\n")

# Build the shared predictor matrix (7 rows x 12,235 columns)
X_all <- as.data.frame(wide[, ..x_cols])

# Median-impute any NAs in predictors
for (v in names(X_all)) {
  nas <- is.na(X_all[[v]])
  if (any(nas)) X_all[[v]][nas] <- median(X_all[[v]], na.rm = TRUE)
}

cat("Predictor matrix X:", nrow(X_all), "rows x", ncol(X_all), "columns\n")

# -- 3.  Standardise predictors (zero mean, unit variance) ---------------------
#  Neural networks are sensitive to feature scale.
#  We compute mean/sd from training rows only, then apply to all rows.
train_rows <- which(wide$year < 2025)   # 1-6
test_row   <- which(wide$year == 2025)  # 7

x_means <- colMeans(X_all[train_rows, ])
x_sds   <- apply(X_all[train_rows, ], 2, sd)
# Avoid division by zero for constant columns
x_sds[x_sds == 0] <- 1

X_scaled <- as.data.frame(scale(X_all, center = x_means, scale = x_sds))

X_train <- X_scaled[train_rows, ]
X_test  <- X_scaled[test_row, , drop = FALSE]

cat("X_train (scaled):", nrow(X_train), "x", ncol(X_train), "\n")
cat("X_test  (scaled):", nrow(X_test),  "x", ncol(X_test),  "\n")

# -- 4.  Parse county keys from house_price columns ---------------------------
parse_key <- function(col) {
  key <- sub("^house_price_", "", col)
  st  <- sub(".*_", "", key)
  cty <- sub("_[A-Z]{2}$", "", key)
  list(hp_col = col, key = key, county = cty, state = st)
}
keys <- rbindlist(lapply(hp_cols, parse_key))

# -- 5.  Fit one neural network per county -------------------------------------
results <- vector("list", nrow(keys))

cat("\nFitting neural network for", nrow(keys), "counties...\n")
cat("Each model: 6 train obs x", ncol(X_train), "predictors\n")
cat("Architecture: input(", ncol(X_train), ") -> hidden(3) -> output(1)\n")
cat("Progress: ")

set.seed(42)
t0 <- Sys.time()

for (i in seq_len(nrow(keys))) {

  k      <- keys[i]
  hp_col <- k$hp_col

  # Response vector (all 7 years)
  y_all <- wide[[hp_col]]

  # Skip if response has NAs in training period
  if (any(is.na(y_all[train_rows]))) {
    results[[i]] <- data.table(
      county = k$county, state = k$state,
      actual_2025 = y_all[test_row],
      predicted_2025 = NA_real_,
      train_rmse = NA_real_
    )
    next
  }

  y_train <- y_all[train_rows]
  y_test  <- y_all[test_row]

  # Standardise response (helps nnet converge)
  y_mean <- mean(y_train)
  y_sd   <- sd(y_train)
  if (y_sd == 0) y_sd <- 1
  y_train_scaled <- (y_train - y_mean) / y_sd

  # Prepare training data frame
  train_df <- data.frame(y = y_train_scaled, X_train)

  # Fit neural network with multiple random starts, pick best
  # - size = 3: 3 hidden neurons (small to regularise with n=6)
  # - decay: L2 weight penalty (regularisation)
  # - maxit: max iterations
  # - linout = TRUE: linear output for regression
  best_fit  <- NULL
  best_val  <- Inf

  for (start in 1:5) {
    fit <- tryCatch(
      nnet(
        y ~ .,
        data    = train_df,
        size    = 3,
        decay   = 0.1,
        maxit   = 500,
        linout  = TRUE,
        trace   = FALSE,
        MaxNWts = 100000
      ),
      error = function(e) NULL
    )
    if (!is.null(fit) && fit$value < best_val) {
      best_val <- fit$value
      best_fit <- fit
    }
  }

  if (is.null(best_fit)) {
    results[[i]] <- data.table(
      county = k$county, state = k$state,
      actual_2025 = y_test,
      predicted_2025 = NA_real_,
      train_rmse = NA_real_
    )
    next
  }

  # In-sample predictions (unscale)
  y_hat_train_scaled <- predict(best_fit, newdata = data.frame(X_train))
  y_hat_train <- y_hat_train_scaled * y_sd + y_mean
  train_rmse  <- sqrt(mean((y_train - y_hat_train)^2))

  # Out-of-sample prediction (unscale)
  y_hat_test_scaled <- predict(best_fit, newdata = data.frame(X_test))
  pred_2025 <- y_hat_test_scaled * y_sd + y_mean

  results[[i]] <- data.table(
    county         = k$county,
    state          = k$state,
    actual_2025    = y_test,
    predicted_2025 = as.numeric(pred_2025),
    train_rmse     = train_rmse
  )

  if (i %% 100 == 0) cat(i, "")
}

elapsed <- round(difftime(Sys.time(), t0, units = "mins"), 1)
cat("\nDone!", elapsed, "minutes elapsed.\n")

# -- 6.  Assemble results -----------------------------------------------------
res <- rbindlist(results)
res_valid <- res[!is.na(predicted_2025)]
cat("\nCounties successfully modeled:", nrow(res_valid), "/", nrow(res), "\n")

# -- 7.  Overall accuracy (2025 hold-out) --------------------------------------
res_valid[, residual  := actual_2025 - predicted_2025]
res_valid[, pct_error := residual / actual_2025 * 100]

overall_rmse <- sqrt(mean(res_valid$residual^2))
overall_mae  <- mean(abs(res_valid$residual))
overall_mape <- mean(abs(res_valid$pct_error))
overall_r2   <- 1 - sum(res_valid$residual^2) /
                    sum((res_valid$actual_2025 - mean(res_valid$actual_2025))^2)

cat("\n======================================================\n")
cat("   Neural Network  -  2025 Hold-Out Performance\n")
cat("   Features per model:", ncol(X_train), "\n")
cat("   Architecture: input ->", 3, "hidden -> output\n")
cat("======================================================\n")
cat("  Counties evaluated :", nrow(res_valid), "\n")
cat("  RMSE               : $", formatC(overall_rmse, format = "f",
                                         big.mark = ",", digits = 0), "\n")
cat("  MAE                : $", formatC(overall_mae,  format = "f",
                                         big.mark = ",", digits = 0), "\n")
cat("  MAPE               :", round(overall_mape, 2), "%\n")
cat("  R-squared          :", round(overall_r2, 4), "\n")
cat("======================================================\n")

# -- 8.  Save per-county results -----------------------------------------------
fwrite(res, "nn_results_all_counties.csv")
cat("\nSaved: nn_results_all_counties.csv\n")

# -- 9.  Actual vs Predicted scatter -------------------------------------------
p1 <- ggplot(res_valid, aes(x = actual_2025, y = predicted_2025)) +
  geom_point(alpha = 0.3, size = 1.2, colour = "darkorange") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "red") +
  labs(title = "Neural Network: Actual vs Predicted House Prices (2025)",
       subtitle = paste0("12,235 features | 3 hidden neurons | ",
                         nrow(res_valid), " counties"),
       x = "Actual ($)", y = "Predicted ($)") +
  theme_minimal(base_size = 14) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma)
ggsave("nn_actual_vs_predicted.png", p1, width = 8, height = 7, dpi = 150)
cat("Saved: nn_actual_vs_predicted.png\n")

# -- 10. Residual histogram ----------------------------------------------------
p2 <- ggplot(res_valid, aes(x = residual)) +
  geom_histogram(bins = 60, fill = "darkorange", colour = "white", alpha = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "red") +
  labs(title = "Neural Network: Distribution of 2025 Prediction Residuals",
       x = "Residual: Actual - Predicted ($)", y = "Count") +
  theme_minimal(base_size = 14) +
  scale_x_continuous(labels = scales::comma)
ggsave("nn_residual_distribution.png", p2, width = 8, height = 5, dpi = 150)
cat("Saved: nn_residual_distribution.png\n")

# -- 11. Top / Bottom 20 counties by absolute error ----------------------------
res_valid[, abs_error := abs(residual)]
top20 <- rbind(
  res_valid[order(-abs_error)][1:20][, group := "Largest errors"],
  res_valid[order(abs_error)][1:20][, group := "Smallest errors"]
)
top20[, label := paste0(county, ", ", state)]

p3 <- ggplot(top20, aes(x = reorder(label, abs_error), y = abs_error)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  facet_wrap(~ group, scales = "free_y") +
  labs(title = "Neural Network: Counties with Largest & Smallest Errors (2025)",
       x = NULL, y = "Absolute Error ($)") +
  theme_minimal(base_size = 11) +
  scale_y_continuous(labels = scales::comma)
ggsave("nn_top_bottom_counties.png", p3, width = 12, height = 8, dpi = 150)
cat("Saved: nn_top_bottom_counties.png\n")

# -- 12. Summary by state -----------------------------------------------------
state_summary <- res_valid[, .(
  n_counties     = .N,
  mean_actual    = round(mean(actual_2025)),
  mean_predicted = round(mean(predicted_2025)),
  mean_abs_error = round(mean(abs_error)),
  mean_pct_error = round(mean(abs(pct_error)), 2)
), by = state][order(mean_abs_error)]

cat("\n-- Performance Summary by State --\n")
print(state_summary)

# -- 13. Compare with BART if results exist ------------------------------------
if (file.exists("bart_results_all_counties.csv")) {
  cat("\n======================================================\n")
  cat("   BART vs Neural Network Comparison\n")
  cat("======================================================\n")

  bart_res <- fread("bart_results_all_counties.csv")
  bart_valid <- bart_res[!is.na(predicted_2025)]
  bart_valid[, residual := actual_2025 - predicted_2025]

  bart_rmse <- sqrt(mean(bart_valid$residual^2))
  bart_mae  <- mean(abs(bart_valid$residual))
  bart_mape <- mean(abs(bart_valid$residual / bart_valid$actual_2025 * 100))
  bart_r2   <- 1 - sum(bart_valid$residual^2) /
                   sum((bart_valid$actual_2025 - mean(bart_valid$actual_2025))^2)

  compare_df <- data.frame(
    Metric = c("RMSE ($)", "MAE ($)", "MAPE (%)", "R-squared", "Counties"),
    BART   = c(formatC(bart_rmse, format = "f", big.mark = ",", digits = 0),
               formatC(bart_mae,  format = "f", big.mark = ",", digits = 0),
               round(bart_mape, 2),
               round(bart_r2, 4),
               nrow(bart_valid)),
    NeuralNet = c(formatC(overall_rmse, format = "f", big.mark = ",", digits = 0),
                  formatC(overall_mae,  format = "f", big.mark = ",", digits = 0),
                  round(overall_mape, 2),
                  round(overall_r2, 4),
                  nrow(res_valid))
  )
  print(compare_df, row.names = FALSE)

  # Comparison scatter: BART vs NN predictions
  merged <- merge(
    bart_valid[, .(county = county, state = state,
                   bart_pred = predicted_2025, actual = actual_2025)],
    res_valid[, .(county = county, state = state,
                  nn_pred = predicted_2025)],
    by = c("county", "state")
  )

  if (nrow(merged) > 0) {
    p4 <- ggplot(merged, aes(x = bart_pred, y = nn_pred)) +
      geom_point(alpha = 0.3, size = 1.2, colour = "purple") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "red") +
      labs(title = "BART vs Neural Network: 2025 Predictions Compared",
           x = "BART Predicted ($)", y = "Neural Net Predicted ($)") +
      theme_minimal(base_size = 14) +
      scale_x_continuous(labels = scales::comma) +
      scale_y_continuous(labels = scales::comma)
    ggsave("bart_vs_nn_comparison.png", p4, width = 8, height = 7, dpi = 150)
    cat("Saved: bart_vs_nn_comparison.png\n")
  }
} else {
  cat("\n(Run the BART script first to enable side-by-side comparison)\n")
}

cat("\nDone! Neural network analysis complete.\n")
cat("Output files:\n")
cat("  nn_results_all_counties.csv\n")
cat("  nn_actual_vs_predicted.png\n")
cat("  nn_residual_distribution.png\n")
cat("  nn_top_bottom_counties.png\n")
if (file.exists("bart_results_all_counties.csv")) {
  cat("  bart_vs_nn_comparison.png\n")
}
