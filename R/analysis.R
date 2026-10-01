# Classifying Vinho Verde wine quality from physicochemical lab measurements.
#
# Run from the project root:
#   Rscript R/analysis.R
#
# Writes model metrics to results/ and figures to figures/.

suppressPackageStartupMessages({
  library(class)         # KNN
  library(MASS)          # LDA, QDA
  library(tree)          # classification trees
  library(randomForest)  # bagging
  library(e1071)         # SVM
})

dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

COL_NO   <- "#2a78d6"  # "not good" class
COL_YES  <- "#eb6834"  # "good" class
COL_INK  <- "#3b3b38"
COL_GRID <- "#e4e3dc"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Quality is scored 0-10 by expert tasters. A wine is "good" if it scores 7+.
load_wine <- function(path) {
  d <- read.csv(path)
  d$good <- factor(ifelse(d$quality >= 7, "Yes", "No"), levels = c("No", "Yes"))
  d$quality <- NULL
  d
}

wines <- list(
  red   = load_wine("data/winequality-red.csv"),
  white = load_wine("data/winequality-white.csv")
)

# 70/30 train/test split. Both splits are drawn from one seeded stream so the
# partitions match the original course project exactly.
set.seed(1)
splits <- lapply(wines, function(d) {
  idx <- sample(nrow(d), 0.7 * nrow(d))
  list(train = d[idx, ], test = d[-idx, ])
})

predictors <- setdiff(names(wines$red), "good")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Error alone is misleading here: most wines are "not good", so always guessing
# "No" already scores well. Sensitivity (share of good wines caught) and
# balanced accuracy show whether a model learned anything about good wine.
score <- function(pred, truth) {
  pred <- factor(pred, levels = c("No", "Yes"))
  sens <- mean(pred[truth == "Yes"] == "Yes")
  spec <- mean(pred[truth == "No"] == "No")
  c(error = mean(pred != truth), sensitivity = sens, specificity = spec,
    balanced_accuracy = (sens + spec) / 2)
}

# Choose k by leave-one-out CV on the training set.
choose_k <- function(x, y, ks = 1:25) {
  set.seed(1)
  cv_err <- sapply(ks, function(k) mean(knn.cv(x, y, k = k) != y))
  ks[which.min(cv_err)]
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

fit_all <- function(wine) {
  train <- splits[[wine]]$train
  test  <- splits[[wine]]$test
  xtr <- train[, predictors]
  xte <- test[, predictors]
  results <- list()
  params  <- list()
  fits    <- list()

  add <- function(model, pred_train, pred_test) {
    results[[model]] <<- data.frame(
      wine = wine, model = model,
      train_error = unname(score(pred_train, train$good)["error"]),
      t(score(pred_test, test$good)), row.names = NULL
    )
  }

  # Baseline: always predict the majority class
  add("Baseline (always 'not good')", rep("No", nrow(train)), rep("No", nrow(test)))

  # KNN on raw features. Distances are dominated by the sulfur dioxide
  # columns, which are measured on a scale ~100x larger than the others.
  k_raw <- choose_k(xtr, train$good)
  set.seed(1)
  add("KNN (raw features)",
      knn(xtr, xtr, train$good, k = k_raw),
      knn(xtr, xte, train$good, k = k_raw))

  # KNN on standardized features (scaled with training-set moments only)
  ctr <- colMeans(xtr)
  sds <- apply(xtr, 2, sd)
  ztr <- scale(xtr, ctr, sds)
  zte <- scale(xte, ctr, sds)
  k_std <- choose_k(ztr, train$good)
  set.seed(1)
  add("KNN (standardized)",
      knn(ztr, ztr, train$good, k = k_std),
      knn(ztr, zte, train$good, k = k_std))

  # LDA / QDA
  lda_fit <- lda(good ~ ., data = train)
  add("LDA", predict(lda_fit, train)$class, predict(lda_fit, test)$class)
  qda_fit <- qda(good ~ ., data = train)
  add("QDA", predict(qda_fit, train)$class, predict(qda_fit, test)$class)

  # Classification tree, unpruned
  tree_fit <- tree(good ~ ., data = train, model = TRUE)
  add("Tree (unpruned)",
      predict(tree_fit, train, type = "class"),
      predict(tree_fit, test, type = "class"))

  # Pruned tree: size chosen by 10-fold CV on misclassification. Among the
  # sizes with the lowest CV error, take the smallest with at least 2 leaves.
  set.seed(1)
  cv <- cv.tree(tree_fit, FUN = prune.misclass)
  ok <- cv$size > 1
  best_size <- min(cv$size[ok][cv$dev[ok] == min(cv$dev[ok])])
  pruned <- prune.misclass(tree_fit, best = best_size)
  add("Tree (pruned)",
      predict(pruned, train, type = "class"),
      predict(pruned, test, type = "class"))

  # Bagging: a random forest that considers every predictor at each split
  set.seed(1)
  bag_fit <- randomForest(good ~ ., data = train, mtry = length(predictors),
                          ntree = 500, importance = TRUE)
  add("Bagged trees", predict(bag_fit, train), predict(bag_fit, test))

  # SVM (radial kernel), cost tuned by 10-fold CV
  costs <- c(0.1, 0.5, 1, 2, 5, 10)
  set.seed(1)
  svm2 <- tune(svm, good ~ alcohol + volatile.acidity, data = train,
               kernel = "radial", ranges = list(cost = costs))$best.model
  add("SVM (alcohol + volatile acidity)", predict(svm2, train), predict(svm2, test))

  set.seed(1)
  svm_all <- tune(svm, good ~ ., data = train,
                  kernel = "radial", ranges = list(cost = costs))$best.model
  add("SVM (all predictors)", predict(svm_all, train), predict(svm_all, test))

  params <- data.frame(
    wine = wine,
    parameter = c("k (KNN, raw)", "k (KNN, standardized)", "unpruned tree leaves",
                  "pruned tree leaves", "SVM cost (2 predictors)",
                  "SVM cost (all predictors)", "SVM support vectors (2 predictors)",
                  "SVM support vectors (all predictors)"),
    value = c(k_raw, k_std, sum(tree_fit$frame$var == "<leaf>"), best_size,
              svm2$cost, svm_all$cost, svm2$tot.nSV, svm_all$tot.nSV)
  )

  list(metrics = do.call(rbind, results), params = params,
       lda = lda_fit, pruned = pruned, bag = bag_fit, svm2 = svm2,
       train = train, test = test)
}

fits <- lapply(names(splits), fit_all)
names(fits) <- names(splits)

metrics <- do.call(rbind, lapply(fits, `[[`, "metrics"))
params  <- do.call(rbind, lapply(fits, `[[`, "params"))
rownames(metrics) <- rownames(params) <- NULL

class_balance <- do.call(rbind, lapply(names(wines), function(w) data.frame(
  wine = w,
  n = nrow(wines[[w]]),
  n_train = nrow(splits[[w]]$train),
  n_test = nrow(splits[[w]]$test),
  good_train = sum(splits[[w]]$train$good == "Yes"),
  share_good = mean(wines[[w]]$good == "Yes")
)))

importance <- do.call(rbind, lapply(names(fits), function(w) {
  imp <- importance(fits[[w]]$bag, type = 2)
  data.frame(wine = w, variable = rownames(imp), mean_decrease_gini = imp[, 1],
             row.names = NULL)
}))

write.csv(metrics, "results/metrics.csv", row.names = FALSE)
write.csv(params, "results/parameters.csv", row.names = FALSE)
write.csv(class_balance, "results/class_balance.csv", row.names = FALSE)
write.csv(importance, "results/importance.csv", row.names = FALSE)

print(class_balance)
print(params)
print(metrics, digits = 3)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

png_fig <- function(file, width = 1600, height = 900) {
  png(file.path("figures", file), width = width, height = height, res = 200)
  par(family = "sans", col.axis = COL_INK, col.lab = COL_INK, col.main = COL_INK,
      fg = COL_INK, bty = "n", las = 1, mgp = c(2.2, 0.6, 0), tcl = -0.3)
}

title_case <- function(w) paste0(toupper(substring(w, 1, 1)), substring(w, 2))

# 1. Test error by model, one panel per wine, with the majority-class baseline
png_fig("test_error_by_model.png", width = 2000, height = 1000)
par(mfrow = c(1, 2), mar = c(3.5, 13.5, 3.5, 1.5))
for (w in names(fits)) {
  m <- fits[[w]]$metrics
  base <- m$error[1]
  m <- m[-1, ]
  m <- m[order(m$error, decreasing = TRUE), ]
  y <- seq_len(nrow(m))
  xmax <- 0.25
  plot(NA, xlim = c(0, xmax), ylim = c(0.5, nrow(m) + 0.5), yaxt = "n",
       xlab = "Test error rate", ylab = "", main = "", xaxs = "i")
  title(paste(title_case(w), "wine"), line = 2)
  abline(v = seq(0, xmax, 0.05), col = COL_GRID)
  abline(v = base, lty = 2, lwd = 2, col = COL_INK)
  mtext(sprintf("always 'not good' = %.3f", base), side = 3, at = base,
        line = 0.3, cex = 0.65, col = COL_INK)
  segments(0, y, m$error, y, col = COL_NO, lwd = 2)
  points(m$error, y, pch = 19, col = COL_NO, cex = 1.3)
  axis(2, at = y, labels = m$model, tick = FALSE, cex.axis = 0.75)
}
dev.off()

# 2. LDA discriminant scores by class
for (w in names(fits)) {
  f <- fits[[w]]
  ld <- predict(f$lda, f$train)$x[, 1]
  brks <- seq(floor(min(ld)), ceiling(max(ld)), by = 0.25)
  png_fig(paste0("lda_", w, ".png"))
  par(mar = c(3.5, 3.5, 2.5, 1))
  h_no  <- hist(ld[f$train$good == "No"],  breaks = brks, plot = FALSE)
  h_yes <- hist(ld[f$train$good == "Yes"], breaks = brks, plot = FALSE)
  plot(h_no, freq = FALSE, col = adjustcolor(COL_NO, 0.45), border = "white",
       ylim = c(0, max(h_no$density, h_yes$density)),
       main = paste0("LDA discriminant scores, ", w, " wine (training set)"),
       xlab = "LD1", ylab = "Density")
  # "Good" drawn as an outline so overlap with "not good" stays readable
  lines(rep(brks, each = 2), c(0, rep(h_yes$density, each = 2), 0),
        col = COL_YES, lwd = 2.5)
  legend("topright", c("Not good (< 7)", "Good (7+)"),
         fill = c(adjustcolor(COL_NO, 0.45), NA), border = c(NA, COL_YES),
         bty = "n", cex = 0.8)
  dev.off()
}

# 3. Pruned classification trees
for (w in names(fits)) {
  png_fig(paste0("tree_", w, ".png"), width = 1800, height = 1100)
  par(mar = c(1, 1, 2.5, 1))
  plot(fits[[w]]$pruned, col = COL_INK)
  text(fits[[w]]$pruned, pretty = 0, cex = 0.75, col = COL_INK)
  title(paste0("Pruned classification tree, ", w, " wine"))
  dev.off()
}

# 4. Variable importance from the bagged trees
png_fig("importance.png", width = 1800, height = 900)
par(mfrow = c(1, 2), mar = c(3.5, 11, 2.5, 1))
for (w in names(fits)) {
  imp <- importance[importance$wine == w, ]
  imp <- imp[order(imp$mean_decrease_gini), ]
  y <- seq_len(nrow(imp))
  plot(NA, xlim = c(0, max(imp$mean_decrease_gini) * 1.05), ylim = c(0.5, nrow(imp) + 0.5),
       yaxt = "n", xlab = "Mean decrease in Gini impurity", ylab = "",
       main = paste(title_case(w), "wine"))
  abline(v = pretty(c(0, max(imp$mean_decrease_gini))), col = COL_GRID)
  segments(0, y, imp$mean_decrease_gini, y, col = COL_NO, lwd = 2)
  points(imp$mean_decrease_gini, y, pch = 19, col = COL_NO, cex = 1.2)
  axis(2, at = y, labels = gsub("\\.", " ", imp$variable), tick = FALSE, cex.axis = 0.75)
}
dev.off()

# 5. Two-predictor SVM decision regions, white wine
png_fig("svm_white_2d.png", width = 1600, height = 1100)
f <- fits$white
plot(f$svm2, f$train[, c("good", "alcohol", "volatile.acidity")],
     alcohol ~ volatile.acidity,
     symbolPalette = c(COL_NO, COL_YES),
     color.palette = function(n) c("#f4f3ee", "#fbe1d6")[seq_len(n)],
     main = "SVM decision regions, white wine (alcohol + volatile acidity)")
dev.off()

cat("\nDone. Results in results/, figures in figures/.\n")
