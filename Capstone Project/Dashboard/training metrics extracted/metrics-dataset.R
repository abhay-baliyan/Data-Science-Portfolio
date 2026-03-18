root_path <- rstudioapi::getActiveDocumentContext()$path
root_folder <- dirname(root_path)
dashboard_folder <- dirname(root_folder)
project_folder <- dirname(dashboard_folder)
results_path <- paste(project_folder,"/Crop Disease Prediction System/results",sep="")



crop_folders <- list.dirs(
  results_path,
  full.names = TRUE,
  recursive = FALSE
)

all_files <- lapply(crop_folders, function(p) {
  list.files(
    p,
    pattern = "classification_report.csv",
    full.names = TRUE,
    ignore.case = TRUE
  )
})

all_files <- unlist(all_files)
print(all_files)



library(dplyr)
library(readr)

combined_df <- lapply(all_files, function(f) {

  df <- read_csv(f, show_col_types = FALSE)

  crop_name <- basename(dirname(f))

  df$crop <- crop_name

  df
}) %>%
  bind_rows()

output_path <- file.path(root_folder, "all_crops_classification_metrics.csv")

write_csv(combined_df, output_path)
colnames(combined_df)[1] <- "class"
output_path








# ---- fix column name if needed
if ("...1" %in% names(combined_df)) {
  combined_df <- combined_df %>%
    rename(class = ...1)
}

# ---- split rows

overall_metrics <- combined_df %>%
  filter(class %in% c("accuracy", "macro avg", "weighted avg"))

class_metrics <- combined_df %>%
  filter(!class %in% c("accuracy", "macro avg", "weighted avg"))

# ---- save to root results folder

write_csv(
  class_metrics,
  file.path(dashboard_folder, "predicition_metrics.csv")
)

write_csv(
  overall_metrics,
  file.path(dashboard_folder, "accuracy_metrics.csv")
)

# show paths
file.path(root_folder, "all_crops_class_level_metrics.csv")
file.path(root_folder, "all_crops_overall_metrics.csv")
