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


#### Comparison of ERP Component Attributes ####

# Load and clean data
erp <- read_csv("erp.csv", col_types = cols(...1 = col_skip(), 
                Sex = col_factor(levels = c("male", "female")), 
                Stimulus = col_factor(levels = c("1", "2", "3"))))
erp <- na.omit(erp)
erp$Value <- abs(erp$Value)

# model ERP component as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)

# ggboxplot(subset(erp, erp$Component=="N1_Amp"), x = "Stimulus", y = "Value", 
#           color = "Stimulus", palette = c("#00AFBB", "#E7B800", "#A03D41"),
#           ylab = "N1 Amplitude", xlab = "Stimulus Level",
#           order = c("1", "2", "3")) + 
#   geom_signif(comparisons = list(c("1", "2")), 
#               map_signif_level=TRUE,
#               y_position = 15) + 
#   geom_signif(comparisons = list(c("1", "3")), 
#               map_signif_level=TRUE,
#               y_position = 17) + 
#   geom_signif(comparisons = list(c("3", "2")), 
#               map_signif_level=TRUE,
#               y_position = 16)


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
