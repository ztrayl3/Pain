library(ggpubr)
library(car)
library(multcomp)
library(tidyverse)
library(rstatix)
library(rcompanion)
library(emmeans)
library(readr)
library(ggplot2)
library(afex)

gamma1 <- read_csv("Data/Perception_freq_G.csv", col_types = cols(...1 = col_skip(), 
                                                                  Sex = col_factor(levels = c("male", "female")), 
                                                                  Stimulus = col_factor(levels = c("1", "2", "3"))))
gamma1$Condition <- "Perception"
gamma2 <- read_csv("Data/EDA_freq_G.csv", col_types = cols(...1 = col_skip(), 
                                                           Sex = col_factor(levels = c("male", "female")), 
                                                           Stimulus = col_factor(levels = c("1", "2", "3"))))
gamma2$Condition <- "EDA"
gamma3 <- read_csv("Data/Motor_freq_G.csv", col_types = cols(...1 = col_skip(), 
                                                             Sex = col_factor(levels = c("male", "female")), 
                                                             Stimulus = col_factor(levels = c("1", "2", "3"))))
gamma3$Condition <- "Motor"
gamma4 <- read_csv("Data/Control_freq_G.csv", col_types = cols(...1 = col_skip(), 
                                                               Sex = col_factor(levels = c("male", "female")), 
                                                               Stimulus = col_factor(levels = c("1", "2", "3"))))
gamma4$Condition <- "Control"
gamma <- do.call("rbind", list(gamma1, gamma2, gamma3, gamma4))

gamma <- gamma[!duplicated(gamma), ]  # remove duplicate rows
gamma <- subset(gamma, gamma$ID!=18)  # exclude subject 18 (DNF motor)

# baseline correction, if desired
new <- data.frame(ID=NA,
                  Sex=NA,
                  Stimulus=NA,
                  Condition=NA,
                  Gamma=NA)
for (subject in unique(gamma$ID)) {  # for each subject
  for (stimulus in unique(gamma$Stimulus)) {  # for each stimulus level
    for (condition in unique(gamma$Condition)) {  # for each condition
      selection <- subset(gamma, (gamma$ID==subject & gamma$Stimulus==stimulus & gamma$Condition==condition))
      
      B <- subset(selection, selection$Component=="Baseline_Amp")$Value
      G <- subset(selection, selection$Component=="Gamma_Amp")$Value
      P <- ((G - B) / B) * 100  # convert gamma power to % change from baseline
      
      row <- data.frame(ID=subject,
                        Sex=selection$Sex,
                        Stimulus=stimulus,
                        Condition=condition,
                        Gamma=P)
      
      new <- rbind(new, row)
    }
  }
}
new <- na.omit(new)  # remove extra NA row at the top
gamma <- new  # replace ERP with baseline % values