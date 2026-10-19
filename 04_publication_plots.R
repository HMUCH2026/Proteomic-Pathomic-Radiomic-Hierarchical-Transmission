# 统计结果绘图，不读取/写入预设的论文数值。Rscript --vanilla 04_publication_plots.R
# 依赖ggplot2/jsonlite；桑基图需要可选ggalluvial。输入前面脚本的results表。
# 每幅图从对应结果计算流程读取；缺少某步骤则明确跳过，不用演示数据替代。
# 输出独立PDF/PNG以及可编辑CSV；组织切片、MRI、条带原图由真实素材另外排版。
# 这些是可复用科学绘图实现，不宣称逐像素复制现稿排版。

args <- commandArgs(trailingOnly = FALSE)
script <- grep("^--file=", args, value = TRUE)
ROOT <- if (length(script)) dirname(normalizePath(sub("^--file=", "", script[1]))) else getwd()
RESULTS <- file.path(ROOT, "results")
OUT <- file.path(RESULTS, "publication_plots")
FONT_FAMILY <- "serif"
BASE_SIZE <- 10
DPI <- 600L
WIDTH <- 9
HEIGHT <- 6
PALETTE <- c("Comp-Trans" = "#228833", "Part-Trans" = "#CC9900", "Direct-Dom" = "#AA3377",
             "Trans-Null" = "#999999", "Data-Driven" = "#4477AA")

read_result <- function(relative) {
  path <- file.path(RESULTS, relative)
  if (!file.exists(path)) { message("跳过（无结果文件）: ", relative); return(NULL) }
  d <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  # pandas CSV使用True/False；不能将字符列直接用于R行筛选。
  for (name in intersect(c("assessable", "dominant", "replicated", "retained", "chosen"), names(d))) {
    value <- tolower(as.character(d[[name]]))
    if (any(!is.na(value) & !value %in% c("true", "false", "1", "0"))) stop("无效布尔结果列: ", name)
    converted <- rep(NA, length(value))
    converted[!is.na(value)] <- value[!is.na(value)] %in% c("true", "1")
    d[[name]] <- converted
  }
  d
}

save_plot <- function(p, name, width = WIDTH, height = HEIGHT) {
  ggplot2::ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = width, height = height)
  ggplot2::ggsave(file.path(OUT, paste0(name, ".png")), p, width = width, height = height,
                   dpi = DPI, bg = "white")
}

enrichment_curve <- function(ranks, genes) {
  hit <- names(ranks) %in% genes
  if (!any(hit) || all(hit)) stop("无法计算该基因集运行曲线")
  weights <- abs(ranks) * hit
  if (sum(weights) <= 0) stop("命中基因排名权重之和为0")
  cumsum(weights / sum(weights) - (!hit) / sum(!hit))
}

plot_pathways <- function() {
  gsea <- read_result("pathways/gsea_results.csv")
  if (!is.null(gsea)) {
    d <- gsea[is.finite(gsea$pvalue), ]
    d$display <- factor(d$Description, levels = unique(d$Description[order(d$NES)]))
    save_plot(ggplot2::ggplot(d, ggplot2::aes(-log10(pmax(pvalue, .Machine$double.xmin)), display,
        size = setSize, color = NES)) + ggplot2::geom_point() +
        ggplot2::scale_color_gradient2(low = "#2166AC", mid = "white", high = "#B2182B") +
        ggplot2::labs(x = expression(-log[10](P)), y = NULL), "GSEA_bubble", height = max(5, .18*nrow(d)))
  }
  scores <- read_result("pathways/pathway_scores_long.csv")
  if (!is.null(scores)) {
    scores$response <- factor(scores$pCR, levels = c(0, 1), labels = c("non-pCR", "pCR"))
    save_plot(ggplot2::ggplot(scores, ggplot2::aes(response, score, fill = response)) +
        ggplot2::geom_violin(trim = FALSE) + ggplot2::geom_boxplot(width = .15, outlier.shape = NA) +
        ggplot2::facet_grid(cohort ~ pathway) + ggplot2::scale_fill_manual(values = c("#4477AA", "#CC3311")) +
        ggplot2::labs(x = NULL, y = "Training-standardized ssGSEA score"), "pathway_scores")
    ranks <- read_result("pathways/ranked_genes.csv")
    sets <- read_result("pathways/eligible_gene_sets.csv")
    if (!is.null(ranks) && !is.null(sets)) {
      v <- setNames(ranks$log2_fold_change, ranks$gene_symbol)
      running <- do.call(rbind, lapply(unique(scores$pathway), function(pathway) {
        data.frame(pathway = pathway, rank = seq_along(v),
                   running_ES = enrichment_curve(v, sets$gene[sets$term == pathway]))
      }))
      save_plot(ggplot2::ggplot(running, ggplot2::aes(rank, running_ES)) + ggplot2::geom_line() +
          ggplot2::facet_wrap(~pathway) + ggplot2::geom_hline(yintercept = 0, linetype = 2), "GSEA_running_curves")
    }
  }
}

plot_transmission <- function() {
  t <- read_result("transmission/transmission_features.csv")
  if (!is.null(t)) {
    save_plot(ggplot2::ggplot(t, ggplot2::aes(direct, transmitted, color = type, shape = type)) +
        ggplot2::geom_point(alpha = .8) + ggplot2::scale_color_manual(values = PALETTE) +
        ggplot2::labs(x = "Direct component (incremental R2)", y = "Transmitted component"), "transmission_components")
    save_plot(ggplot2::ggplot(t, ggplot2::aes(reorder(feature, transmitted), transmitted, color = type)) +
        ggplot2::geom_linerange(ggplot2::aes(ymin = transmitted_lower, ymax = transmitted_upper)) +
        ggplot2::geom_point(size = .7) + ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        ggplot2::coord_flip() + ggplot2::scale_color_manual(values = PALETTE) +
        ggplot2::labs(x = NULL, y = "Transmitted component with percentile 95% CI"),
        "transmission_feature_intervals", height = max(7, .13*nrow(t)))
    meta <- read_result("features/feature_metadata.csv")
    if (!is.null(meta) && requireNamespace("ggalluvial", quietly = TRUE)) {
      flow <- merge(t[c("feature", "type")], meta, by = "feature")
      save_plot(ggplot2::ggplot(flow, ggplot2::aes(axis1 = type, axis2 = compartment,
          axis3 = source, axis4 = family, y = 1)) +
          ggalluvial::geom_alluvium(ggplot2::aes(fill = type), alpha = .55) +
          ggalluvial::geom_stratum(width = .15, color = "grey50") +
          ggplot2::geom_text(stat = "stratum", ggplot2::aes(label = ggplot2::after_stat(stratum)), size = 2.5) +
          ggplot2::scale_fill_manual(values = PALETTE) +
          ggplot2::scale_x_discrete(limits = c("Transmission type", "Compartment", "MRI source", "Family")),
          "feature_composition_alluvial", width = 12)
    }
  }
  for (filename in c("pathway_pathomic_coefficients.csv", "pathomic_radiomic_coefficients.csv")) {
    d <- read_result(paste0("transmission/", filename))
    if (!is.null(d)) {
      p <- ggplot2::ggplot(d, ggplot2::aes(outcome, predictor, fill = beta)) + ggplot2::geom_tile() +
          ggplot2::scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B") +
          ggplot2::geom_point(data = d[d$q < .05, ], shape = 8, size = 1) +
          ggplot2::labs(x = NULL, y = NULL, fill = "Coefficient")
      if (length(unique(d$outcome)) > 30) p <- p + ggplot2::theme(axis.text.x = ggplot2::element_blank())
      save_plot(p, sub(".csv", "_heatmap", filename, fixed = TRUE), width = 12)
    }
  }
  chains <- read_result("transmission/supported_chains.csv")
  if (!is.null(chains) && !is.null(t)) {
    chains <- chains[chains$feature %in% t$feature[t$type == "Comp-Trans"], ]
    if (nrow(chains)) {
      layers <- list(sort(unique(chains$pathway)), sort(unique(chains$pathomic)), sort(unique(chains$feature)))
      nodes <- do.call(rbind, lapply(seq_along(layers), function(i) data.frame(
          label = layers[[i]], x = i, y = seq(0, 1, length.out = length(layers[[i]])))))
      edges1 <- unique(chains[c("pathway", "pathomic", "beta_xm")])
      edges2 <- unique(chains[c("pathomic", "feature", "beta_my")])
      edge_table <- rbind(data.frame(from = edges1$pathway, to = edges1$pathomic, beta = edges1$beta_xm),
                          data.frame(from = edges2$pathomic, to = edges2$feature, beta = edges2$beta_my))
      edge_table$x <- nodes$x[match(edge_table$from, nodes$label)]
      edge_table$y <- nodes$y[match(edge_table$from, nodes$label)]
      edge_table$xend <- nodes$x[match(edge_table$to, nodes$label)]
      edge_table$yend <- nodes$y[match(edge_table$to, nodes$label)]
      save_plot(ggplot2::ggplot() + ggplot2::geom_segment(data = edge_table,
          ggplot2::aes(x, y, xend = xend, yend = yend, color = beta > 0), alpha = .4) +
          ggplot2::geom_point(data = nodes, ggplot2::aes(x, y), size = 2) +
          ggplot2::geom_text(data = nodes, ggplot2::aes(x, y, label = label), size = 1.8, vjust = -1) +
          ggplot2::scale_x_continuous(breaks = 1:3, labels = c("Pathway", "Pathomic", "Radiomic")) +
          ggplot2::labs(x = NULL, y = NULL, color = "Positive coefficient"), "supported_chain_network", width = 14, height = 10)
    }
  }
  external <- read_result("transmission_external/feature_type_agreement.csv")
  if (!is.null(external)) {
    assessed <- external[external$assessable, ]
    counts <- as.data.frame(table(assessed$type_train, assessed$type_external))
    names(counts) <- c("training", "external", "n")
    save_plot(ggplot2::ggplot(counts, ggplot2::aes(external, training, fill = n)) +
        ggplot2::geom_tile() + ggplot2::geom_text(ggplot2::aes(label = n)) +
        ggplot2::scale_fill_gradient(low = "white", high = "#4477AA"), "type_migration_matrix")
    for (component in c("transmitted", "direct")) {
      d <- data.frame(training = assessed[[paste0(component, "_train")]], external = assessed[[paste0(component, "_external")]],
                      type = assessed$type_train)
      save_plot(ggplot2::ggplot(d, ggplot2::aes(training, external, color = type)) +
          ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) + ggplot2::geom_point() +
          ggplot2::scale_color_manual(values = PALETTE), paste0(component, "_external_concordance"))
    }
  }
}

plot_performance <- function() {
  d <- read_result("performance/curves.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(x, y, color = model)) +
      ggplot2::geom_path() + ggplot2::facet_grid(curve ~ cohort) +
      ggplot2::scale_color_manual(values = PALETTE) + ggplot2::labs(x = "FPR (ROC) / recall (PR)",
          y = "TPR (ROC) / precision (PR)"), "ROC_PR", width = 14, height = 7)
  d <- read_result("performance/calibration.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(predicted, observed, color = model)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) + ggplot2::geom_line() +
      ggplot2::geom_point() + ggplot2::facet_wrap(~cohort) + ggplot2::scale_color_manual(values = PALETTE), "calibration")
  d <- read_result("performance/dca.csv")
  if (!is.null(d)) {
    all <- unique(d[c("cohort", "threshold", "treat_all")])
    save_plot(ggplot2::ggplot(d, ggplot2::aes(threshold, net_benefit, color = model)) +
        ggplot2::geom_line() + ggplot2::geom_line(data = all,
            ggplot2::aes(threshold, treat_all), inherit.aes = FALSE, color = "black") +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) + ggplot2::facet_wrap(~cohort) +
        ggplot2::scale_color_manual(values = PALETTE), "decision_curves")
  }
  for (filename in c("metrics.csv", "comparisons.csv")) {
    d <- read_result(paste0("performance/", filename))
    if (!is.null(d)) {
      d$estimate_CI <- sprintf("%.3f (%.3f, %.3f)", d$estimate, d$lower, d$upper)
      write.csv(d, file.path(OUT, paste0("formatted_", filename)), row.names = FALSE)
    }
  }
  d <- read_result("subgroup_sensitivity/subgroup_curves.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(x, y, color = subgroup)) +
      ggplot2::geom_path() + ggplot2::facet_wrap(~grouping) + ggplot2::labs(x = "1-specificity", y = "Sensitivity"), "subgroup_ROC")
  d <- read_result("subgroup_sensitivity/classifier_curves.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(x, y, color = classifier)) +
      ggplot2::geom_path() + ggplot2::facet_wrap(~cohort) + ggplot2::labs(x = "1-specificity", y = "Sensitivity"), "classifier_sensitivity_ROC")
}

plot_replication_shap <- function() {
  d <- read_result("replication/set_intervals.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(model, estimate, color = model, shape = cohort)) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower, ymax = upper), width = .1,
                             position = ggplot2::position_dodge(.45)) +
      ggplot2::geom_point(position = ggplot2::position_dodge(.45)) +
      ggplot2::facet_wrap(~metric, scales = "free_y") + ggplot2::scale_color_manual(values = PALETTE) +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1)), "replication_and_AUC")
  d <- read_result("replication/trends_concordance_contrasts.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(statistic, estimate)) +
      ggplot2::geom_hline(yintercept = 0, linetype = 2) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower, ymax = upper), width = .1) +
      ggplot2::geom_point() + ggplot2::coord_flip(), "replication_trends_and_contrasts", height = 9)
  d <- read_result("shap/importance.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(reorder(feature, mean_abs_shap), mean_abs_shap)) +
      ggplot2::geom_col(fill = "#228833") + ggplot2::coord_flip() + ggplot2::labs(x = NULL, y = "Mean absolute SHAP"), "SHAP_importance")
  d <- read_result("shap/group_contributions.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(enrichment, category)) +
      ggplot2::geom_vline(xintercept = 1, linetype = 2) + ggplot2::geom_point() +
      ggplot2::facet_grid(grouping ~ ., scales = "free_y", space = "free_y") +
      ggplot2::labs(x = "SHAP share / feature-count share", y = NULL), "SHAP_enrichment", height = 9)
  d <- read_result("chain_attribution/feature_concordance.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(transmission_rank, importance_rank, color = compartment)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) + ggplot2::geom_point() +
      ggplot2::labs(x = "Transmission rank (1=highest)", y = "SHAP importance rank (1=highest)"), "rank_concordance")
  d <- read_result("chain_attribution/chain_allocations.csv")
  if (!is.null(d)) {
    d$chain <- paste(d$pathway, d$pathomic, sep = " -> ")
    save_plot(ggplot2::ggplot(d, ggplot2::aes(chain, feature, color = signed_strength > 0, size = absolute_strength)) +
        ggplot2::geom_point() + ggplot2::geom_point(data = d[d$dominant, ], shape = 1, color = "black") +
        ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)) +
        ggplot2::labs(x = NULL, y = NULL), "chain_strengths", width = 12, height = 8)
  }
  d <- read_result("chain_attribution/pathway_shares.csv")
  if (!is.null(d)) save_plot(ggplot2::ggplot(d, ggplot2::aes(pathway, importance_share)) +
      ggplot2::geom_col(fill = "#AA3377") + ggplot2::geom_text(ggplot2::aes(label = sprintf("%.1f%%", 100*importance_share)),
        vjust = -.3) + ggplot2::labs(x = NULL, y = "Descriptively attributed SHAP share"), "pathway_SHAP_shares")
  d <- read_result("mouse/table2_counts.csv")
  if (!is.null(d)) {
    rates <- do.call(rbind, lapply(c("m_over_N", "n_over_m", "n_over_N"), function(name)
      data.frame(type = d$type, rate = d[[name]], definition = name)))
    save_plot(ggplot2::ggplot(rates, ggplot2::aes(type, rate, fill = type)) +
        ggplot2::geom_col() + ggplot2::facet_wrap(~definition) +
        ggplot2::scale_fill_manual(values = PALETTE) + ggplot2::labs(x = NULL), "mouse_response_rates")
  }
  for (filename in c("univariable.csv", "multivariable.csv")) {
    d <- read_result(paste0("logistic/", filename))
    if (!is.null(d)) {
      d$OR_CI <- sprintf("%.3f (%.3f, %.3f)", d$OR, d$lower, d$upper)
      write.csv(d, file.path(OUT, paste0("formatted_logistic_", filename)), row.names = FALSE)
    }
  }
}

main <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) stop("需要ggplot2/jsonlite")
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  ggplot2::theme_set(ggplot2::theme_bw(base_family = FONT_FAMILY, base_size = BASE_SIZE))
  plot_pathways(); plot_transmission(); plot_performance(); plot_replication_shap()
  writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
  message("绘图完成；缺少输入的模块已明确跳过，未使用任何示意结果替代。")
}

if (sys.nframe() == 0L) main()
