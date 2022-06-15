setwd("E:/corona2020/Python Scripts_/DysTacMap/")
library(tidyverse)
library(scales)

#####LINEAR Simple#####
coef_means <- read_csv("SVC_linear_coef_means.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "avg") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

coef_lower_ci <- read_csv("SVC_linear_coef_lower_ci.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "lower_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp)%>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

coef_upper_ci <- read_csv("SVC_linear_coef_upper_ci.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "upper_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp)%>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

clusters_lut <- tibble(cluster = coef_means$cluster %>%
                         unique(),
                       cluster_name = c("L Crus-II",
                                        "R MFG/IFG",
                                        "L Cereb VIIa/b",
                                        "R LOC/AG",
                                        "R SFG/MFG",
                                        "R SFG",
                                        "L Ins",
                                        "R STG",
                                        "R Ins",
                                        "R Puta",
                                        "R SFG",
                                        "L Crus-I/II"))
modality_lut <- tibble(modality = c("falff", "localcorr", "globalcorr", "gm", "wm"),
                       new_mod = c("fALFF", "Local Corr", "Global Corr", "GM Vol", "WM Vol"))

coef_ci <- left_join(coef_lower_ci, coef_upper_ci) %>% 
  left_join(clusters_lut) %>%
  left_join(modality_lut)%>%
  mutate(Group = if_else(Group == "DYS",
                         "DD",
                         Group))

coef_means <- left_join(coef_means, clusters_lut) %>%
  left_join(modality_lut) %>%
  mutate(Group = if_else(Group == "DYS",
                         "DD",
                         Group))
  
coef_means %>%
  left_join(coef_ci) %>%
  select(new_mod, cluster_name, Group, avg, lower_ci, upper_ci) %>%
  mutate(to_print = str_glue("{round(avg,2)}\n[{round(lower_ci,2)},{round(upper_ci,2)}]")) %>%
  select(-avg, -contains("ci")) %>%
  pivot_wider(id_cols = c(new_mod, cluster_name), names_from = Group, values_from = to_print) %>%
  arrange(new_mod) %>%
  select(new_mod, cluster_name, COM, DCD, DD) %>%
  write_csv2("coef_table_simple.csv")

cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_plot <- ggplot(coef_means, aes(x = cluster_name, y = avg, fill = Group)) +
  geom_bar(stat = "identity", 
           position = position_dodge(.9)) +
  geom_errorbar(data = coef_ci, 
                aes(ymin = lower_ci, ymax = upper_ci, y = NULL),
                position = position_dodge(.9),
                width = .5) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette) +
  xlab("Cluster") +
  ylab("SVM Weights")

png("coef_from_linear_SVC.png", h = 900, w = 1300)
coef_plot
dev.off()

coef_plot_leg <- ggplot(coef_means, aes(x = cluster_name, y = avg, fill = Group)) +
  geom_bar(stat = "identity", 
           position = position_dodge(.9)) +
  geom_errorbar(data = coef_ci, 
                aes(ymin = lower_ci, ymax = upper_ci, y = NULL),
                position = position_dodge(.9),
                width = .5) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette) +
  xlab("Cluster") +
  ylab("SVM Weights")

png("coef_from_linear_SVC_leg.png", h = 900, w = 1300)
coef_plot_leg
dev.off()

coef_means


#####Linear Upsampling#####
coef_upsampling_means <- read_csv("SVC_upsampling_linear_coef_means.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "avg") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

coef_upsampling_lower_ci <- read_csv("SVC_upsampling_linear_coef_lower_ci.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "lower_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp)%>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

coef_upsampling_upper_ci <- read_csv("SVC_upsampling_linear_coef_upper_ci.csv") %>%
  mutate(discrimination = if_else(X1 == 0, "dcd", 
                                  if_else(X1 == 1, "dys", "com"))) %>%
  select(-X1) %>%
  pivot_longer(cols = -discrimination, names_to = "temp", values_to = "upper_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp)%>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  mutate(Group = toupper(discrimination))

coef_upsampling_ci <- left_join(coef_upsampling_lower_ci, coef_upsampling_upper_ci) %>% 
  left_join(clusters_lut) %>%
  left_join(modality_lut)

coef_upsampling_means <- left_join(coef_upsampling_means, clusters_lut) %>%
  left_join(modality_lut)

cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_upsampling_plot <- ggplot(coef_upsampling_means, aes(x = cluster_name, y = avg, fill = Group)) +
  geom_bar(stat = "identity", 
           position = position_dodge(.9)) +
  geom_errorbar(data = coef_upsampling_ci, 
                aes(ymin = lower_ci, ymax = upper_ci, y = NULL),
                position = position_dodge(.9),
                width = .5) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette)+ 
  xlab("Cluster") +
  ylab("SVM Weights")

png("coef_from_linear_SVC_upsampling.png", h = 900, w = 1300)
coef_upsampling_plot
dev.off()

#####RF#####
coef_RF_means <- read_csv("RF_linear_coef_means.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "avg") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_lower_ci <- read_csv("RF_linear_coef_lower_ci.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "lower_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_upper_ci <- read_csv("RF_linear_coef_upper_ci.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "upper_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_ci <- left_join(coef_RF_lower_ci, coef_RF_upper_ci) %>% 
  left_join(clusters_lut) %>%
  left_join(modality_lut)

coef_RF_means <- left_join(coef_RF_means, clusters_lut) %>%
  left_join(modality_lut)

cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_RF_plot <- ggplot(coef_RF_means, aes(x = cluster_name, y = avg)) +
  geom_bar(stat = "identity") +
  geom_errorbar(data = coef_RF_ci, 
                aes(ymin = lower_ci, ymax = upper_ci, y = NULL),
                width = .5) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette) +
  xlab("Cluster") +
  ylab("SVM Weights")

png("coef_from_RF.png", h = 900, w = 1300)
coef_RF_plot
dev.off()


#####RF UPSAMPLING#####
coef_RF_upsampling_means <- read_csv("RF_upsampling_linear_coef_means.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "avg") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_upsampling_lower_ci <- read_csv("RF_upsampling_linear_coef_lower_ci.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "lower_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_upsampling_upper_ci <- read_csv("RF_upsampling_linear_coef_upper_ci.csv") %>%
  select(-X1) %>%
  pivot_longer(cols = everything(), names_to = "temp", values_to = "upper_ci") %>%
  mutate(cluster = str_replace_all(temp, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  select(-temp) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_RF_upsampling_ci <- left_join(coef_RF_upsampling_lower_ci, coef_RF_upsampling_upper_ci)

left_join(coef_RF_upsampling_ci, coef_RF_upsampling_means) %>%
  left_join(modality_lut) %>%
  left_join(clusters_lut) %>%
  mutate(statistic = str_glue("{round(avg,2)}[{round(lower_ci,3)} - {round(upper_ci,3)}]")) %>%
  arrange(desc(avg)) %>%
  select(new_mod, cluster_name, statistic) %>%
  write_csv2("RF_from_upsampling_table.csv")

left_join(coef_RF_upsampling_ci, coef_RF_upsampling_means) %>%
  left_join(modality_lut) %>%
  left_join(clusters_lut) %>%
  mutate(statistic = str_glue("{round(avg,2)}")) %>%
  arrange(desc(avg)) %>%
  select(new_mod, cluster_name, statistic) %>%
  write_csv2("RF_from_upsampling_table_no_ci.csv")


cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_RF_upsampling_plot <- ggplot(coef_RF_upsampling_means, aes(x = cluster, y = avg)) +
  geom_bar(stat = "identity") +
  geom_errorbar(data = coef_RF_upsampling_ci, 
                aes(ymin = lower_ci, ymax = upper_ci, y = NULL),
                width = .5) +
  facet_wrap(~modality, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette)

png("coef_from_RF_upsampling.png", h = 900, w = 1300)
coef_RF_upsampling_plot
dev.off()

#####Linear and RF#####
coef_RF_means_rescaled <- coef_RF_means %>%
  mutate(avg = rescale(avg, range(abs(coef_means$avg))))
coef_means_linear_rf <- bind_rows(coef_means, coef_RF_means_rescaled) %>%
  mutate(Group = if_else(is.na(Group), "RF", Group))


cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_linear_rf_plot <- ggplot(coef_means_linear_rf, aes(x = cluster_name, y = avg, fill = Group)) +
  geom_bar(stat = "identity",
           position = position_dodge(.9)) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette) +
  xlab("Cluster") +
  ylab("Weights")

png("coef_from_linear_and_rf.png", h = 900, w = 1300)
coef_linear_rf_plot
dev.off()

coef_linear_rf_plot_leg <- ggplot(coef_means_linear_rf, aes(x = cluster_name, y = avg, fill = Group)) +
  geom_bar(stat = "identity",
           position = position_dodge(.9)) +
  facet_wrap(~new_mod, nrow = 2, scales = "free_x") + 
  theme_bw() + 
  theme(panel.grid.minor = element_blank(),
        text = element_text(size = 45),
        axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = cbbPalette) +
  xlab("Cluster") +
  ylab("Weights")

png("coef_from_linear_and_rf_leg.png", h = 900, w = 1300)
coef_linear_rf_plot_leg
dev.off()


#####Descriptive stat#####
coef_means %>% left_join(coef_ci) %>%
  filter(discrimination == "dcd") %>%
  mutate(abs_w = abs(avg)) %>%
  arrange(desc(abs_w)) %>%
  select(modality, cluster_name, avg, lower_ci, upper_ci)

coef_means %>% left_join(coef_ci) %>%
  filter(discrimination == "dys") %>%
  mutate(abs_w = abs(avg)) %>%
  arrange(desc(abs_w)) %>%
  select(modality, cluster_name, avg, lower_ci, upper_ci)

coef_means %>% left_join(coef_ci) %>%
  filter(discrimination == "com") %>%
  mutate(abs_w = abs(avg)) %>%
  arrange(desc(abs_w)) %>%
  select(modality, cluster_name, avg, lower_ci, upper_ci)


coef_means %>% left_join(coef_ci) %>%
  mutate(abs_w = abs(avg)) %>%
  arrange(desc(Group), desc(abs_w)) %>%
  select(Group, new_mod, cluster_name, avg, lower_ci, upper_ci) %>%
  mutate(to_print = str_glue("{round(avg,2)}[{round(lower_ci,2)} - {round(upper_ci,2)}]")) %>%
  select(-avg, -lower_ci, -upper_ci) %>%

  write_csv2("weights_table.csv")