#!/usr/bin/env Rscript

# compute ICC using lme4 and write summary table
library(yaml)
library(lme4)

cfg <- yaml::read_yaml(file.path(dirname(sys.frame(1)$ofile), "00_config.yaml"))
df <- read.csv(cfg$data$raw)
target <- cfg$target
group <- cfg$group_var

f <- as.formula(paste(target, "~ 1 + (1|", group, ")"))
mod <- lmer(f, data = df, REML = TRUE)
vc <- as.data.frame(VarCorr(mod))
# assume first row is group variance, last row residual
group_var <- vc$vcov[1]
resid_var <- attr(VarCorr(mod), "sc")^2
icc <- group_var / (group_var + resid_var)

out <- data.frame(group_variance=group_var, residual_variance=resid_var, icc=icc)

out_dir <- cfg$outputs$tables
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "icc_summary.csv")
write.csv(out, out_path, row.names = FALSE)
