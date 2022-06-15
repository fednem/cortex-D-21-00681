setwd("E:/corona2020/Python Scripts_/DysTacMap/")
library(tidyverse)
library(scales)
library(weights)

#####LINEAR Simple#####
coef_means <- read_csv("SVC_upsample_linear_coef_means_two_by_two.csv") %>%
  select(-X1) %>%
  mutate(cluster = str_replace_all(features, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ")

coef_lower_ci <- read_csv("SVC_upsample_linear_coef_lower_ci_two_by_two.csv") %>%
  select(-X1) %>%
  mutate(cluster = str_replace_all(features, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  rename("lower_ci" = "median")

coef_upper_ci <- read_csv("SVC_upsample_linear_coef_upper_ci_two_by_two.csv") %>%
  select(-X1) %>%
  mutate(cluster = str_replace_all(features, "xx", " - ") %>%
           str_replace_all("_", " ")) %>%
  separate(cluster, c("modality", "cluster"), sep = " - ") %>%
  rename("upper_ci" = "median")

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
  left_join(modality_lut) %>%
  mutate(discriminations = str_replace_all(discriminations, "_", " ") %>%
           str_replace_all(., "dys", "DD") %>%
           toupper(.)) %>%
  rename("Model" = "discriminations")%>%
  mutate(Model = str_replace(Model, "VS", "vs"))

coef_means <- left_join(coef_means, clusters_lut) %>%
  left_join(modality_lut) %>%
  mutate(discriminations = str_replace_all(discriminations, "_", " ") %>%
           str_replace_all(., "dys", "DD") %>%
           toupper(.)) %>%
  rename("Model" = "discriminations") %>%
  mutate(Model = str_replace(Model, "VS", "vs"))



  

cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

coef_plot <- ggplot(coef_means, aes(x = cluster_name,
                                    y = median,
                                    fill = Model)) +
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

png("coef_from_upsample_linear_SVC_two_by_two.png", h = 900, w = 1300)
coef_plot
dev.off()

coef_plot_leg <- ggplot(coef_means, aes(x = cluster_name,
                                    y = median,
                                    fill = Model)) +
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

png("coef_from_upsample_linear_SVC_two_by_two_leg.png", h = 900, w = 1300)
coef_plot_leg
dev.off()

##NB: in sklearn la label positiva è la prima incontrata
#dcd vs dys: dcd positive - dys negative
#dys vs com: dys positive - com negative


####TABLE####
#dcd vs com: dcd positive - dys negative