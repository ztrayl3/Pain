library(readxl)
library(ggpubr)
library(car)
library(lme4)
library(multcomp)
library(tidyverse)
library(rstatix)
library(lmerTest)
library(rcompanion)
library(emmeans)
library(readr)
library(ggplot2)
require(gridExtra)
library(ggsignif)

#### ANOVA on Pain Threshold Data ####

# Load Pain Threshold data
threshold <- read_excel("Statistics.xlsx", 
                         sheet = "Gender & Pain Threshold",
                         col_types = c("numeric", "text", "numeric", 
                                       "numeric", "skip", "skip", "skip", 
                                       "skip", "skip", "skip", "skip", "skip", 
                                       "skip", "skip", "skip"))

# Assign variables as factors and rename them
threshold$Sex <- as.factor(threshold$`Sex Name`)
threshold$Threshold <- threshold$`Pain Threshold (mJ)`

threshold <- subset(threshold, threshold$Subject != 33)  # remove single outlier

# Visualize distributions
ggboxplot(threshold, x = "Sex", y = "Threshold", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Pain Threshold (mJ)", xlab = "Sex")
ggqqplot(threshold, "Threshold", facet.by = "Sex")

# Compute the analysis of variance
thresh.aov <- aov(Threshold ~ Sex, data = threshold)

# Levene's for homogeneity of variance (significant = non-homogeneous)
leveneTest(Threshold ~ Sex, data = threshold)
# Extract the residuals
aov_residuals <- residuals(object = thresh.aov)
# Run Shapiro-Wilk test to check for normality (significant = non-normal)
shapiro.test(x = aov_residuals)

# Summary of the analysis
summary(thresh.aov)


#### Within Subjects ANCOVA on Verbal Pain Rating Data ####

ratings <- read_excel("Statistics.xlsx",
                      sheet = "Gender & Pain Rating",
                      col_types = c("numeric","text", "text", "text",
                                    "skip", "numeric", "skip", "skip", "skip"))

# Rename and factor variables
ratings$Sex <- as.factor(ratings$Sex)
ratings$Pain <- as.numeric(ratings$`Verbal Pain Rating`)
ratings$Stim <- as.factor(ratings$`Stimulus Level`)
# omit NAs, where participant did not report within time limit
ratings <- na.omit(ratings)

# visualize the data
ggboxplot(ratings, x = "Sex", y = "Pain", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Verbal Pain Rating", xlab = "Sex")

ggboxplot(ratings, x = "Stim", y = "Pain", 
          color = "Stim", palette = c("#00AFBB", "#E7B800", "#A03D41"),
          ylab = "Verbal Pain Rating", xlab = "Stimulus Level",
          order = c("low", "med", "high")) + 
          geom_signif(comparisons = list(c("low", "med")), 
              map_signif_level=TRUE,
              y_position = 100) + 
          geom_signif(comparisons = list(c("low", "high")), 
              map_signif_level=TRUE,
              y_position = 110) + 
          geom_signif(comparisons = list(c("high", "med")), 
              map_signif_level=TRUE,
              y_position = 105)

# model Pain Ratings as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)
pain.model <- lmer(Pain ~ Sex * Stim + (1|Subject), data = ratings, REML = TRUE)
anova(pain.model)
rand(pain.model)
post.hoc <- emmeans(pain.model, ~ Sex * Stim)
pairs(post.hoc, adjust = "tukey")

# Check the residuals for normality
R = residuals(pain.model)
qqnorm(R)
qqline(R)
ggqqplot(R)
# Run Shapiro-Wilk test to check for normality (significant = non-normal)
shapiro.test(R)

# a visualization of this significance when random effects are ignored
ggboxplot(ratings, x = "Sex", y = "Pain", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Verbal Pain Rating", xlab = "Sex",
          facet.by = "Stim") + 
          geom_point(color = "grey") +
          stat_summary(fun = mean, shape = 18)

#### ERP Component vs Baseline ####

# Load and clean data
erp <- read_csv("erp.csv", col_types = cols(...1 = col_skip(), 
                Sex = col_factor(levels = c("male", "female")), 
                Stimulus = col_factor(levels = c("1", "2", "3"))))
erp <- na.omit(erp)
erp$Value <- abs(erp$Value)

# Do a full paired-samples T-test to be sure, ignoring any missing subjects
difference <- function(df, title, ignore){
  baseline <- subset(df, df$Component=="Baseline_Amp" & !(df$ID %in% ignore))
  test <- subset(df, df$Component==title & !(df$ID %in% ignore))
  
  data <- rbind(baseline, test)
  
  # check that the difference is close enough to normal
  d <- baseline$Value - test$Value
  hist(d)
  
  t.test(Value ~ Component, data = data, paired = TRUE)
}

# N1 vs Baseline
difference(erp, "N1_Amp", ignore=c(10, 19))  

# N2 vs Baseline
difference(erp, "N2_Amp", ignore=c(3, 19, 31, 32))

# P2 vs Baseline
difference(erp, "P2_Amp", ignore=c(5, 8, 15, 23, 43))

#### Comparison of ERP Component Attributes ####

# model ERP component as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)

model <- function(df, title) {
  temp <- subset(df, df$Component==title)
  erp.model <- lmer(Value ~ Sex * Stimulus + (1|ID), data = temp, REML = TRUE)
  print(anova(erp.model))
  print(rand(erp.model))
  post.hoc <- emmeans(erp.model, ~ Sex * Stimulus)
  print(pairs(post.hoc, adjust = "tukey"))
  
  R = residuals(erp.model)
  p1 <- ggqqplot(R)
  p2 <- ggboxplot(temp, x = "Sex", y = "Value", 
            color = "Stimulus", palette = c("#00AFBB", "#E7B800", "#A3Bf39"),
            ylab = title, xlab = "Sex")
  grid.arrange(p1, p2)
}

# N1 LATENCY
model(erp, "N1_Lat")

# N1 AMPLITUDE
model(erp, "N1_Amp")

# N2 LATENCY
model(erp, "N2_Lat")

# N2 AMPLITUDE
model(erp, "N2_Amp")

# P2 LATENCY
model(erp, "P2_Lat")

# P2 AMPLITUDE
model(erp, "P2_Amp")

#### Gamma band analysis ####

freq <- read_csv("freq.csv", col_types = cols(...1 = col_skip(), 
                  Sex = col_factor(levels = c("male", "female")), 
                  Stimulus = col_factor(levels = c("1", "2", "3"))))

# Compare to baseline
difference(freq, "Gamma_Amp", ignore=c())

# Compare across genders

# first, convert to % change in gamma band
for (i in seq_len(length(unique(freq$ID)))){  # for every subject
  for (j in seq_len(length(unique(freq$Stimulus)))){  # for every stimulus
    temp <- subset(freq, freq$ID == i & freq$Stimulus == j)  # grab that data
    if (length(temp$ID) > 3){
      temp <- head(temp, -3)  # sometimes it is 6, should be 3
    }
    
    g <- temp$Value[temp$Component == "Gamma_Amp"]
    b <- temp$Value[temp$Component == "Baseline_Amp"]
    percentage <- (abs(g - b) / b) * 100  # calculate gamma change as % of baseline
    newRow <- data.frame(temp$ID, temp$Sex, temp$Stimulus, "Percentage", percentage)
    names(newRow) <- names(temp)
    if (length(newRow$ID) > 1){
      newRow <- head(newRow, -2)  # sometimes it is 3, should be 1
    }

    freq <- rbind(freq, newRow)  # add to the original data frame
  }
}

model(freq, "Percentage")

#### Can we replicate the original paper with simpler models? ####

# Amplitudes vs. Baseline (ERP)
ggboxplot(subset(erp, grepl("Amp", erp$Component, fixed = TRUE)),
          x="Component", y="Value",
          color = "Component", palette = c("#00AFBB", "#E7B800", "#A03D41", "#C130A2"),
          ylab = "Amplitude", xlab = "Component",
          order = c("Baseline_Amp", "N1_Amp", "N2_Amp", "P2_Amp")) +
  geom_signif(comparisons = list(c("Baseline_Amp", "N1_Amp")),
              map_signif_level=TRUE,
              y_position = 18) +
  geom_signif(comparisons = list(c("Baseline_Amp", "N2_Amp")),
              map_signif_level=TRUE,
              y_position = 19) +
  geom_signif(comparisons = list(c("Baseline_Amp", "P2_Amp")),
              map_signif_level=TRUE,
              y_position = 20)

# Amplitudes vs. Baseline (Gamma)
ggboxplot(subset(freq, grepl("Amp", freq$Component, fixed = TRUE)),
          x="Component", y="Value",
          color = "Component", palette = c("#00AFBB", "#E7B800"),
          ylab = "Amplitude", xlab = "Component",
          order = c("Baseline_Amp", "Gamma_Amp")) +
  geom_signif(comparisons = list(c("Baseline_Amp", "Gamma_Amp")),
              map_signif_level=TRUE,
              y_position = -104)

simple_model <- function(df, title) {
  temp <- subset(df, df$Component==title)
  erp.model <- aov(Value ~ Stimulus, data = temp)
  post.hoc <- tukey_hsd(erp.model)
  print(summary(erp.model))
  print(post.hoc)
  
  ggboxplot(temp, x = "Stimulus", y = "Value",
            color = "Stimulus", palette = c("#00AFBB", "#E7B800", "#A03D41"),
            ylab = title, xlab = "Stimulus")
}

# N1_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "N1_Amp")

# N2_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "N2_Amp")

# P2_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "P2_Amp")

# Gamma_Amp as a factor of Stimulus intensity alone?
simple_model(freq, "Gamma_Amp")
simple_model(freq, "Percentage")
