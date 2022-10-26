"
Amplitudes of all analyzed responses were significant in comparison to a 1s 
prestimulus baseline (dependent samples t-tests; p<0.01 for all comparisons).
Moreover, amplitudes were modulated by stimulus intensity as indicated by 
3 (intensity levels)x3 (conditions) repeated-measures ANOVAs. As expected, 
amplitudes of all responses either increased (P2 and gamma) or decreased 
(N1 and N2) with increasing stimulus intensity (N1: F(2, 80)=13.66, p<0.001;
N2: F(2, 76)=6.30, p=0.003; P2: F(2, 82)=30.26, p=0.001;
gamma: F(2, 98)=13.78, p<0.001; Greenhouse-Geisser corrected where necessary).
In addition, amplitudes of N1 (F(2, 98)=4.89, p<0.015) and gamma
(F(1, 66)=7.21, p=0.001) but not of N2 (F(2, 87)=1.97, p=0.15) 
and P2 (F(2, 98)=0.12, p=0.90) responses were influenced by condition. 
Post hoc pairwise comparisons confirmed a significantly more negative N1 
response amplitude in the motor than in the perception (t(49)=4.37, p=0.001)
and autonomic conditions (t(49)=−4.29, p<0.001) as well as stronger gamma 
responses in the motor than in the autonomic condition (t(49)=3.02, p=0.01; 
all p-values Bonferroni-corrected). Taken together, noxious stimuli elicited a 
well-known pattern of electrophysiological responses, including N1, N2, and P2
waves5,6, and gamma oscillations7, which were influenced by stimulus intensity 
and in part by condition.
"


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

# Check the residuals for normality
R = residuals(pain.model)
qqnorm(R)
qqline(R)
ggqqplot(R)
ggqqplot(ratings, "Pain", facet.by = "Sex")
# Run Shapiro-Wilk test to check for normality (significant = non-normal)
shapiro.test(R)

# a visualization of this significance when random effects are ignored
ggboxplot(ratings, x = "Sex", y = "Pain", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Verbal Pain Rating", xlab = "Sex",
          facet.by = "Stim") + 
          geom_point(color = "grey") +
          stat_summary(fun = mean, shape = 18)


#### Comparison of ERP Component Attributes ####

# Load and clean data
erp <- read_csv("erp.csv", col_types = cols(...1 = col_skip(), 
                Sex = col_factor(levels = c("male", "female")), 
                Stimulus = col_factor(levels = c("1", "2", "3"))))
erp <- na.omit(erp)
erp$Value <- abs(erp$Value)

# model ERP component as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)

model <- function(df, title) {
  temp <- subset(erp, erp$Component==title)
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
model(erp, "N1_Lat")  # NO SIGNIFICANT GENDER DIFFERENCE

# N1 AMPLITUDE
model(erp, "N1_Amp")  # NO SIGNIFICANT GENDER DIFFERENCE

# N2 LATENCY
model(erp, "N2_Lat")  # SIGNIFICANT INTERACTION: Male-Low > Female-Low only

# N2 AMPLITUDE
model(erp, "N2_Amp")  # NO SIGNIFICANT GENDER DIFFERENCE

# P2 LATENCY
model(erp, "P2_Lat")  # SIGNIFICANT EFFECT OF SEX, Female-Low < Male-High

# P2 AMPLITUDE
model(erp, "P2_Amp") # SIGNIFICANT INTERACTION: Only between female levels...
