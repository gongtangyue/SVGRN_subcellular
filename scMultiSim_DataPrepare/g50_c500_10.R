library(scMultiSim)

data(GRN_params_100)

target_col <- if ("regulated.gene" %in% colnames(GRN_params_100)) {
  "regulated.gene"
} else if ("target" %in% colnames(GRN_params_100)) {
  "target"
} else {
  stop("Cannot find target column in GRN_params_100")
}

reg_col <- if ("regulator.gene" %in% colnames(GRN_params_100)) {
  "regulator.gene"
} else if ("regulator" %in% colnames(GRN_params_100)) {
  "regulator"
} else {
  stop("Cannot find regulator column in GRN_params_100")
}

# 只保留 1-50 号 gene 内部的 GRN edge
GRN_params <- GRN_params_100[
  GRN_params_100[[target_col]] <= 50 &
    GRN_params_100[[reg_col]] <= 50,
]

lig_params <- data.frame(
  target    = c(42, 43),
  regulator = c(44, 45),
  effect    = c(5.2, 5.9)
)

cell_num <- 500

spatial_options <- function(...) {
  cci_opt <- list(
    params = lig_params,
    max.neighbors = 4,
    cell.type.interaction = "random"
  )
  
  list(
    rand.seed = 0,
    GRN = GRN_params,
    num.genes = 50,
    num.cells = cell_num,
    num.cifs = 50,
    tree = Phyla1(),
    diff.cif.fraction = 0.8,
    do.velocity = FALSE,
    speed.up = TRUE,
    intrinsic.noise = 1,
    dynamic.GRN = list(
      num.steps = 1,
      cell.per.step = 1,
      num.changing.edges = 5,
      weight.mean = 0,
      weight_sd = 4
    ),
    cci = c(cci_opt, list(...))
  )
}

results <- sim_true_counts(
  spatial_options(layout = "layers")
)

out_dir <- "F:/st/SVGRN_subcellular/in_sim/g50_c05_n1"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "cell_specific_GRN"), recursive = TRUE, showWarnings = FALSE)

write.csv(results[["cci_locs"]],
          file.path(out_dir, "cell_loc.csv"),
          row.names = TRUE)
print("locs saved")

write.csv(t(results[["counts"]]),
          file.path(out_dir, "raw_count.csv"),
          row.names = TRUE)
print("counts saved")

for (x in 1:cell_num) {
  write.csv(results[["cell_specific_grn"]][[x]],
            file.path(out_dir, "cell_specific_GRN", paste0("cell", x, ".csv")),
            row.names = TRUE)
}

print("cell_specific_grn saved")

normalizePath("F:/st/SVGRN_subcellular/in_sim/g50_c05_n1")

