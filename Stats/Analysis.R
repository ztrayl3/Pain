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

# model Pain Ratings as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)
pain.model <- lmer(Pain ~ Sex * Stim + (1|Subject), data = ratings, REML = TRUE)
anova(pain.model)
rand(pain.model)
post.hoc <- emmeans(pain.model, ~ Sex * Stim)
pairs(post.hoc, adjust = "tukey")

# a visualization of this significance when random effects are ignored
ggboxplot(ratings, x = "Sex", y = "Pain", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Verbal Pain Rating", xlab = "Sex",
          facet.by = "Stim") + 
          geom_point(color = "grey") +
          stat_summary(fun = mean, shape = 18)


#### Comparison of ERP Component Attributes ####
# TODO:
# -Extract ERP latencies and amplitudes for each stim level and sex
# -Repeat analysis from Verbal Pain Ratings for each ERP DV

