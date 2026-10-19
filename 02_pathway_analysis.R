args <- commandArgs(trailingOnly = FALSE)
script <- grep("^--file=", args, value = TRUE)
ROOT <- if (length(script)) dirname(normalizePath(sub("^--file=", "", script[1]))) else getwd()
OUT <- file.path(ROOT, "results", "pathways")
GMT <- file.path(ROOT, "input", "proteomics", "h.all.v2026.1.Hs.symbols.gmt")
MIN_GENES <- 16L
MAX_GENES <- 200L
N_PERMUTATIONS <- 1000L
SEED <- 00
Q_THRESHOLD <- .05
VIF_THRESHOLD <- 5
SSGSEA_ALPHA <- .25
SSGSEA_NORMALIZE <- TRUE
REQUIRE_ALL_EXTERNAL_GENES <- TRUE  # 

read_expression <- function(path) {
  d <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (anyDuplicated(d$gene_symbol)) stop("error")
  x <- as.matrix(d[, -1, drop = FALSE]); storage.mode(x) <- "double"
  rownames(x) <- d$gene_symbol
  if (any(!is.finite(x))) stop("error")
  x
}

read_gmt <- function(path) {
  records <- strsplit(readLines(path, warn = FALSE), "\t", fixed = FALSE)
  records <- records[lengths(records) >= 3L]
  names <- vapply(records, `[`, character(1), 1L)
  if (anyDuplicated(names)) stop("error")
  setNames(lapply(records, function(z) unique(z[-c(1L, 2L)])), names)
}

pathway_vif <- function(x) {
  if (ncol(x) == 1L) return(setNames(1, colnames(x)))
  vapply(seq_len(ncol(x)), function(i) {
    y <- x[, i]; design <- cbind(1, x[, -i, drop = FALSE])
    residual <- lm.fit(design, y)$residuals
    proportion <- sum(residual^2) / sum((y - mean(y))^2)
    if (proportion <= 1e-12) Inf else 1 / proportion
  }, numeric(1)) |> setNames(colnames(x))
}

ssgsea <- function(x, sets) {
  parameter <- GSVA::ssgseaParam(x, sets, minSize = MIN_GENES, maxSize = MAX_GENES,
      alpha = SSGSEA_ALPHA, normalize = SSGSEA_NORMALIZE)
  as.matrix(GSVA::gsva(parameter, verbose = FALSE, BPPARAM = BiocParallel::SerialParam()))
}

write_scores <- function(scores, path) {
  write.csv(data.frame(case_id = colnames(scores), t(scores), check.names = FALSE), path, row.names = FALSE)
}

main <- function() {
  required <- c("clusterProfiler", "GSVA", "BiocParallel", "jsonlite")
  if (!all(vapply(required, requireNamespace, logical(1), quietly = TRUE))) stop("error")
  set.seed(SEED)
  BiocParallel::register(BiocParallel::SerialParam())
  train <- read_expression(file.path(ROOT, "results/proteomics/training_expression.csv"))
  metadata <- read.csv(file.path(ROOT, "input/proteomics/samples.csv"), stringsAsFactors = FALSE,
                       colClasses = c(case_id = "character"))
  s <- metadata[match(colnames(train), metadata$case_id), , drop = FALSE]
  if (anyNA(s$case_id) || !all(s$cohort == "training") || !setequal(unique(s$pCR), c(0, 1))) stop("error")
  all_sets <- read_gmt(GMT)
  represented <- lapply(all_sets, intersect, y = rownames(train))
  sizes <- lengths(represented)
  sets <- represented[sizes >= MIN_GENES & sizes <= MAX_GENES]
  if (!length(sets)) stop("error")
  ranks <- rowMeans(train[, s$pCR == 1, drop = FALSE]) - rowMeans(train[, s$pCR == 0, drop = FALSE])
  
  ranks <- ranks[order(-ranks, names(ranks))]
  term2gene <- do.call(rbind, lapply(names(sets), function(n) data.frame(term = n, gene = sets[[n]])))
  gsea <- clusterProfiler::GSEA(geneList = ranks, TERM2GENE = term2gene,
      minGSSize = MIN_GENES, maxGSSize = MAX_GENES, pvalueCutoff = 1,
      pAdjustMethod = "BH", by = "fgsea", nPerm = N_PERMUTATIONS,
      seed = TRUE, verbose = FALSE)
  gsea_table <- as.data.frame(gsea)
  if (!nrow(gsea_table)) stop("error")
  raw_scores <- ssgsea(train, sets)
  means <- rowMeans(raw_scores); sds <- apply(raw_scores, 1, sd)
  if (any(sds <= 0 | !is.finite(sds))) stop("error")
  standardized <- sweep(sweep(raw_scores, 1, means, "-"), 1, sds, "/")
  p <- apply(standardized, 1, function(v) wilcox.test(v[s$pCR == 1], v[s$pCR == 0],
                                                   alternative = "two.sided", exact = FALSE)$p.value)
  q <- p.adjust(p, "BH")
  combined <- data.frame(pathway = names(p), ssgsea_p = p, ssgsea_q = q,
                         gsea_q = gsea_table$p.adjust[match(names(p), gsea_table$ID)])
  candidates <- combined$pathway[!is.na(combined$gsea_q) & combined$gsea_q < Q_THRESHOLD &
                                  combined$ssgsea_q < Q_THRESHOLD]
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  write.csv(gsea_table, file.path(OUT, "gsea_results.csv"), row.names = FALSE)
  write.csv(combined, file.path(OUT, "joint_pathway_tests.csv"), row.names = FALSE)
  write_scores(standardized, file.path(OUT, "training_all_scores.csv"))
  write.csv(data.frame(gene_symbol = names(ranks), log2_fold_change = unname(ranks)),
             file.path(OUT, "ranked_genes.csv"), row.names = FALSE)
  write.csv(term2gene, file.path(OUT, "eligible_gene_sets.csv"), row.names = FALSE)
  if (!length(candidates)) stop("error")
  repeat {
    v <- pathway_vif(t(standardized[candidates, , drop = FALSE]))
    if (max(v) < VIF_THRESHOLD) break
    candidates <- setdiff(candidates, names(v)[which.max(v)])
    if (!length(candidates)) stop("error")
  }
  write_scores(standardized[candidates, , drop = FALSE], file.path(OUT, "training_scores.csv"))
  score_rows <- function(scores, meta, cohort) {
    data.frame(case_id = rep(colnames(scores), each = nrow(scores)),
               pathway = rep(rownames(scores), ncol(scores)), score = as.vector(scores),
               pCR = rep(meta$pCR, each = nrow(scores)), cohort = cohort)
  }
  long <- score_rows(standardized[candidates, , drop = FALSE], s, "training")
  external_path <- file.path(ROOT, "results/proteomics/external1_expression.csv")
  ext <- read_expression(external_path)
  missing <- setdiff(unique(unlist(sets[candidates])), rownames(ext))
  if (length(missing) && REQUIRE_ALL_EXTERNAL_GENES) stop("error: ", length(missing))
  
  ext_scores <- ssgsea(ext, sets)
  if (!all(candidates %in% rownames(ext_scores))) stop("error")
  ext_standardized <- sweep(sweep(ext_scores, 1, means[rownames(ext_scores)], "-"), 1,
                            sds[rownames(ext_scores)], "/")
  ext_selected <- ext_standardized[candidates, , drop = FALSE]
  write_scores(ext_selected, file.path(OUT, "external1_scores.csv"))
  ext_meta <- metadata[match(colnames(ext), metadata$case_id), , drop = FALSE]
  if (anyNA(ext_meta$case_id) || !all(ext_meta$cohort == "external1")) stop("error")
  long <- rbind(long, score_rows(ext_selected, ext_meta, "external1"))
  write.csv(long, file.path(OUT, "pathway_scores_long.csv"), row.names = FALSE)
  jsonlite::write_json(list(xcore = candidates, mean = as.list(means), sd = as.list(sds),
      min_genes = MIN_GENES, max_genes = MAX_GENES, permutations = N_PERMUTATIONS,
      seed = SEED, ssgsea_alpha = SSGSEA_ALPHA, ssgsea_normalize = SSGSEA_NORMALIZE,
      GMT_md5 = unname(tools::md5sum(GMT)), package_versions = as.list(setNames(
        vapply(required, function(p) as.character(packageVersion(p)), character(1)), required))),
      file.path(OUT, "parameters.json"), auto_unbox = TRUE, pretty = TRUE)
  saveRDS(gsea, file.path(OUT, "gsea_result.rds"))
  writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
}

if (sys.nframe() == 0L) main()
