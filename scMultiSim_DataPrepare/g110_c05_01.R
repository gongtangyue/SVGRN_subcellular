library(scMultiSim)

data(GRN_params_100)
GRN_params <- GRN_params_100

lig_params <- data.frame(
  target    = c(102, 103),
  regulator = c(104, 105),
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
    num.genes = 110,
    num.cells = cell_num,
    num.cifs = 50,
    tree = Phyla1(),
    diff.cif.fraction = 0.8,
    do.velocity = FALSE,
    speed.up = TRUE,
    intrinsic.noise = 0.1,
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

out_dir <- "in_sim/g110_c05_n01"
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