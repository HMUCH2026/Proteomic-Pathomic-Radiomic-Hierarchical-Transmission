

args <- commandArgs(trailingOnly = FALSE)
script <- grep("^--file=", args, value = TRUE)
ROOT <- if (length(script)) dirname(normalizePath(sub("^--file=", "", script[1]))) else getwd()
INPUT <- file.path(ROOT, "input", "proteomics")
OUT <- file.path(ROOT, "results", "proteomics")
DETECTION_FRACTION <- 0.70
K <- 5L
SEED <- 00
KNN_AXIS <- NA_character_  #
MAPPING_RELEASE <- NA_character_  
COHORTS <- c("training", "external1")

read_table <- function(path) read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                                    colClasses = c(case_id = "character"))

prepare_cohort <- function(abundance, mapping, samples) {
  ids <- samples$case_id
  d <- merge(abundance[abundance$case_id %in% ids, ], mapping, by = "protein_id", all.x = TRUE)
  d <- d[!is.na(d$gene_symbol) & nzchar(d$gene_symbol), ]
  d$abundance[!is.finite(d$abundance) | d$abundance <= 0] <- NA_real_
  med <- function(x) if (all(is.na(x))) NA_real_ else median(x, na.rm = TRUE)
  d <- aggregate(d["abundance"], d[c("gene_symbol", "case_id")], med)
  genes <- sort(unique(d$gene_symbol))
  mat <- matrix(NA_real_, length(genes), length(ids), dimnames = list(genes, ids))
  mat[cbind(match(d$gene_symbol, genes), match(d$case_id, ids))] <- d$abundance
  keep <- rowMeans(!is.na(mat)) >= DETECTION_FRACTION
  mat <- log2(mat[keep, , drop = FALSE])
  if (!nrow(mat) || any(colSums(!is.na(mat)) == 0)) stop("error")
  if (anyNA(mat)) {
    work <- if (KNN_AXIS == "genes") mat else t(mat)
    if (nrow(work) <= K) stop("error")
    work <- impute::impute.knn(work, k = K, rowmax = 1, colmax = 1,
                               rng.seed = SEED)$data
    mat <- if (KNN_AXIS == "genes") work else t(work)
  }
  mat <- sweep(mat, 2, apply(mat, 2, median), "-")
  batch <- factor(samples$batch)
  if (anyNA(batch)) stop("error")
  if (nlevels(batch) > 1L) {
    if (any(table(batch) < 2L)) stop("error")
    
    mat <- sva::ComBat(dat = mat, batch = batch, mod = matrix(1, nrow = ncol(mat), ncol = 1),
                       par.prior = TRUE, prior.plots = FALSE)
  }
  if (any(!is.finite(mat))) stop("error")
  list(matrix = mat, detected = data.frame(gene_symbol = genes, retained = keep))
}

main <- function() {
  required <- c("impute", "sva", "jsonlite")
  absent <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent)) stop("error: ", paste(absent, collapse = ", "))
  if (is.na(KNN_AXIS) || !KNN_AXIS %in% c("genes", "samples") || is.na(MAPPING_RELEASE))
    stop("error")
  abundance <- read_table(file.path(INPUT, "abundance_long.csv"))
  mapping <- read_table(file.path(INPUT, "protein_gene_mapping.csv"))
  samples <- read_table(file.path(INPUT, "samples.csv"))
  if (anyDuplicated(mapping$protein_id)) stop("error")
  if (anyDuplicated(samples$case_id) || !all(samples$pCR %in% c(0, 1))) stop("error")
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  for (cohort in COHORTS) {
    s <- samples[samples$cohort == cohort, , drop = FALSE]
    if (!nrow(s)) stop("error: ", cohort)
    result <- prepare_cohort(abundance, mapping, s)
    output <- data.frame(gene_symbol = rownames(result$matrix), result$matrix, check.names = FALSE)
    write.csv(output, file.path(OUT, paste0(cohort, "_expression.csv")), row.names = FALSE)
    write.csv(result$detected, file.path(OUT, paste0(cohort, "_gene_retention.csv")), row.names = FALSE)
  }
  jsonlite::write_json(list(detection_fraction = DETECTION_FRACTION, knn_k = K,
      knn_axis = KNN_AXIS, seed = SEED, mapping_release = MAPPING_RELEASE,
      mapping_md5 = unname(tools::md5sum(file.path(INPUT, "protein_gene_mapping.csv"))),
      transform = "log2 -> KNN -> sample median centering -> within-cohort ComBat",
      combat_design = "intercept_only", package_versions = as.list(setNames(
        vapply(required, function(p) as.character(packageVersion(p)), character(1)), required))),
      file.path(OUT, "parameters.json"), auto_unbox = TRUE, pretty = TRUE)
  writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
}

if (sys.nframe() == 0L) main()
