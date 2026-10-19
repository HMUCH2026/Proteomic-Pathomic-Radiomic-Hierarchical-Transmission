
args <- commandArgs(trailingOnly = FALSE)
script <- grep("^--file=", args, value = TRUE)
ROOT <- if (length(script)) dirname(normalizePath(sub("^--file=", "", script[1]))) else getwd()
OUT <- file.path(ROOT, "results", "qc")
ICC_TYPE <- NA_character_  
IMPUTATION <- NA_character_  
K <- 5L
ICC_MIN <- .75
MAX_MISSING <- .20
FREQ_CUT <- 95/5
UNIQUE_CUT <- 10
SEED <- 00

read_features <- function(path) {
  d <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                 colClasses = c(case_id = "character"))
  if (anyDuplicated(d$case_id)) stop("error")
  rownames(d) <- d$case_id; d$case_id <- NULL
  if (!all(vapply(d, is.numeric, logical(1)))) stop("特error")
  if (any(is.infinite(as.matrix(d)))) stop("error")
  d
}

icc_point <- function(y, type) {
  if (nrow(y) < 3L || anyNA(y)) return(NA_real_)
  n <- nrow(y); k <- ncol(y)
  grand <- mean(y); rows <- rowMeans(y); cols <- colMeans(y)
  msr <- k * sum((rows - grand)^2) / (n - 1)
  msc <- n * sum((cols - grand)^2) / (k - 1)
  residual <- y - rows - rep(cols, each = n) + grand
  mse <- sum(residual^2) / ((n - 1) * (k - 1))
  denominator <- switch(type, A1 = msr + (k-1)*mse + k*(msc-mse)/n,
                         C1 = msr + (k-1)*mse, Ak = msr + (msc-mse)/n, Ck = msr)
  if (is.null(denominator) || denominator <= 0) return(NA_real_)
  (msr - mse) / denominator
}

impute_matrix <- function(target, reference, method, medians, k = K, training = FALSE) {
  target <- as.matrix(target); reference <- as.matrix(reference)
  if (!anyNA(target)) return(target)
  if (method == "none") stop("error")
  if (method == "median_train") {
    for (j in seq_len(ncol(target))) target[is.na(target[, j]), j] <- medians[j]
  } else if (method == "knn_train") {
    center <- colMeans(reference, na.rm = TRUE)
    scale <- apply(reference, 2, sd, na.rm = TRUE)
    scale[!is.finite(scale) | scale == 0] <- 1
    zref <- sweep(sweep(reference, 2, center, "-"), 2, scale, "/")
    original <- target
    ztarget <- sweep(sweep(original, 2, center, "-"), 2, scale, "/")
    for (i in which(rowSums(is.na(target)) > 0)) {
      delta <- sweep(zref, 2, ztarget[i, ], "-")
      distance <- rowMeans(delta^2, na.rm = TRUE)
      distance[rowSums(is.finite(delta)) == 0] <- Inf
      if (training && rownames(target)[i] %in% rownames(reference))
        distance[match(rownames(target)[i], rownames(reference))] <- Inf
      for (j in which(is.na(target[i, ]))) {
        eligible <- which(is.finite(reference[, j]) & is.finite(distance))
        if (!length(eligible)) stop("error")
        neighbors <- head(eligible[order(distance[eligible])], k)
        target[i, j] <- mean(reference[neighbors, j])
      }
    }
  } else stop("error")
  if (any(!is.finite(target))) stop("error")
  target
}

process_domain <- function(domain, training_ids) {
  input <- file.path(ROOT, "results", "features")
  final <- read_features(file.path(input, paste0(domain, "_final.csv")))
  r1 <- read_features(file.path(input, paste0(domain, "_rater1.csv")))
  r2 <- read_features(file.path(input, paste0(domain, "_rater2.csv")))
  if (!all(training_ids %in% rownames(final))) stop(domain, "error")

  paired <- training_ids[training_ids %in% rownames(r1) & training_ids %in% rownames(r2)]
  if (length(paired) < 3L) stop(domain, "error")
  if (!all(colnames(final) %in% colnames(r1)) || !all(colnames(final) %in% colnames(r2))) stop("error")
  icc <- vapply(colnames(final), function(f) icc_point(cbind(r1[paired, f], r2[paired, f]), ICC_TYPE), numeric(1))
  missing <- colMeans(is.na(final[training_ids, , drop = FALSE]))
  keep <- names(icc)[is.finite(icc) & icc >= ICC_MIN & missing <= MAX_MISSING]
  if (!length(keep)) stop(domain, "error")
  raw_train <- final[training_ids, keep, drop = FALSE]
  medians <- vapply(raw_train, median, numeric(1), na.rm = TRUE)
  imputed_train <- impute_matrix(raw_train, raw_train, IMPUTATION, medians, training = TRUE)
  all_imputed <- impute_matrix(final[, keep, drop = FALSE], raw_train, IMPUTATION, medians)
  
  all_imputed[match(training_ids, rownames(final)), ] <- imputed_train
  nzv <- caret::nearZeroVar(as.data.frame(imputed_train), freqCut = FREQ_CUT, uniqueCut = UNIQUE_CUT)
  passed <- if (length(nzv)) keep[-nzv] else keep
  if (!length(passed)) stop(domain, "error")
  means <- colMeans(imputed_train[, passed, drop = FALSE])
  sds <- apply(imputed_train[, passed, drop = FALSE], 2, sd)
  if (any(sds <= 0 | !is.finite(sds))) stop("error")
  scaled <- sweep(sweep(all_imputed[, passed, drop = FALSE], 2, means, "-"), 2, sds, "/")
  audit <- data.frame(feature = names(icc), ICC = icc, missing_fraction = missing,
                       retained = names(icc) %in% passed, n_icc_pairs = length(paired))
  write.csv(audit, file.path(OUT, paste0(domain, "_qc.csv")), row.names = FALSE)
  write.csv(data.frame(case_id = rownames(final), scaled, check.names = FALSE),
             file.path(OUT, paste0(domain, "_scaled.csv")), row.names = FALSE)
  jsonlite::write_json(list(ICC_type = ICC_TYPE, ICC_threshold = ICC_MIN,
      ICC_population = "training_only_independent_pre-adjudication_annotations",
      imputation = IMPUTATION, knn_k = K, frequency_cut = FREQ_CUT, unique_cut = UNIQUE_CUT,
      retained_features = passed, mean = as.list(means), sd = as.list(sds),
      medians = as.list(medians), training_n = length(training_ids), icc_n = length(paired)),
      file.path(OUT, paste0(domain, "_parameters.json")), auto_unbox = TRUE, pretty = TRUE)
}

main <- function() {
  if (!requireNamespace("caret", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) stop("error")
  if (is.na(ICC_TYPE) || !ICC_TYPE %in% c("A1", "C1", "Ak", "Ck") || is.na(IMPUTATION))
    stop("error")
  set.seed(SEED)
  clinical <- read.csv(file.path(ROOT, "input/clinical.csv"), stringsAsFactors = FALSE,
                         colClasses = c(case_id = "character"))
  pathways <- read.csv(file.path(ROOT, "results/pathways/training_scores.csv"),
                         colClasses = c(case_id = "character"), check.names = FALSE)
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  process_domain("radiomics", clinical$case_id[clinical$cohort == "training"])
  process_domain("pathomics", pathways$case_id)
  writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
}

if (sys.nframe() == 0L) main()
