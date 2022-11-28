#### Imports and Libraries ####
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

threshold <- read_csv("Thresholds.csv", col_types = cols(Sex = col_factor(
                                                       levels = c("male", "female"))))

# Visualize distributions
ggboxplot(threshold, x = "Sex", y = "Threshold", 
          color = "Sex", palette = c("#00AFBB", "#E7B800"),
          ylab = "Pain Threshold (mJ)", xlab = "Sex")
ggqqplot(threshold, "Threshold", facet.by = "Sex")


threshold <- subset(threshold, threshold$Subject != 33)  # remove single outlier
ggqqplot(threshold, "Threshold", facet.by = "Sex")

# Compute the analysis of variance
thresh.aov <- aov(Threshold ~ Sex, data = threshold)

# Levene's for homogeneity of variance (significant = non-homogeneous)
leveneTest(Threshold ~ Sex, data = threshold)
# Extract the residuals
aov_residuals <- residuals(object = thresh.aov)
# Run Shapiro-Wilk test to check for normality (significant = non-normal)
shapiro.test(x = aov_residuals)
# Independence of observations: each subject is only male or female, so we're good

# Summary of the analysis
summary(thresh.aov)



#### Within Subjects ANCOVA on Verbal Pain Rating Data ####

R1 <- read_csv("Perception_ratings.csv", col_types = cols(Sex = col_factor(
                                                          levels = c("male", "female")),
                                                          Stim = col_factor(
                                                          levels = c("low", "med", "high"))))
R2 <- read_csv("Control_ratings.csv", col_types = cols(Sex = col_factor(
                                                       levels = c("male", "female")),
                                                       Stim = col_factor(
                                                       levels = c("low", "med", "high"))))
ratings <- rbind(R1, R2)
ratings <- ratings[sample(nrow(ratings), 5000), ]  # downsample, as we have too many ratings (>6000)
ratings$Stim_Num <- as.numeric(ratings$Stim)  # useful for later

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
emm_options(pbkrtest.limit = 5000)
post.hoc <- emmeans(pain.model, ~ Sex * Stim)
pairs(post.hoc, adjust = "tukey")

# Check the assumptions
R = residuals(pain.model)
ggqqplot(R)
# Run Shapiro-Wilk test to check for normality (significant = non-normal)
shapiro.test(R)
# Levene's test for homogeneity of variance
leveneTest(R ~ ratings$Sex * ratings$Stim)
## NOTE: both of these tests are significant, but our sample size is big (Shapiro)
## and there isn't a group imbalance (Levene), so this should be okay

# Assumptions of homogeneity of regression slopes (should have no significant interaction)
Anova(aov(Pain ~ Sex * Stim, data = ratings), type = 3)
# Assumption of linearity (relationship between Pain and Stim at each Sex should be linear)
ggplot(ratings, aes(Stim_Num, Pain, colour = Sex)) + geom_point(size = 3) + 
  geom_smooth(method = "lm", aes(fill = Sex), alpha = 0.1) + theme(legend.position="top")



#### ERP Component vs Baseline ####

# Load and clean data
erp <- read_csv("erp.csv", col_types = cols(...1 = col_skip(), 
                Sex = col_factor(levels = c("male", "female")), 
                Stimulus = col_factor(levels = c("1", "2", "3"))))
erp <- na.omit(erp)
erp$Value <- abs(erp$Value)

# Amplitudes vs. Baseline (ERP)
ggboxplot(subset(erp, grepl("Amp", erp$Component, fixed = TRUE)),
          x="Component", y="Value",
          color = "Component", palette = c("#00AFBB", "#E7B800", "#A03D41", "#C130A2"),
          ylab = "Amplitude", xlab = "Component",
          order = c("Baseline_Amp", "N1_Amp", "N2_Amp", "P2_Amp"))

# Do a full paired-samples T-test to be sure, ignoring any missing subjects
difference <- function(df, title, ignore){
  baseline <- subset(df, df$Component=="Baseline_Amp" & !(df$ID %in% ignore))
  test <- subset(df, df$Component==title & !(df$ID %in% ignore))
  
  data <- rbind(baseline, test)
  
  # check that the difference is close enough to normal
  d <- baseline$Value - test$Value
  hist(d)
  
  pairwise_t_test(data, Value ~ Component, paired = TRUE, p.adjust.method = "bonferroni")
}

# N1 vs Baseline
difference(erp, "N1_Amp", ignore=c(10, 19))  

# N2 vs Baseline
difference(erp, "N2_Amp", ignore=c(3, 19, 31, 32))

# P2 vs Baseline
difference(erp, "P2_Amp", ignore=c(5, 8, 15, 23, 43))

#### Gamma Activity vs Baseline ####

freq <- read_csv("freq.csv", col_types = cols(...1 = col_skip(), 
                                              Sex = col_factor(levels = c("male", "female")), 
                                              Stimulus = col_factor(levels = c("1", "2", "3"))))
freq$Value <- abs(freq$Value)

# Compare to baseline
difference(freq, "Gamma_Amp", ignore=c())

# Amplitudes vs. Baseline (Gamma)
ggboxplot(subset(freq, grepl("Amp", freq$Component, fixed = TRUE)),
          x="Component", y="Value",
          color = "Component", palette = c("#00AFBB", "#E7B800"),
          ylab = "Amplitude", xlab = "Component",
          order = c("Baseline_Amp", "Gamma_Amp"))

#### Can we replicate the original paper with its simpler models? ####

simple_model <- function(df, title, ignore) {
  temp <- subset(df, df$Component==title & !(df$ID %in% ignore))
  temp <- temp[!duplicated(temp), ]  # remove any duplicated rows
  erp.model <- anova_test(data = temp, dv = Value, wid = ID, within = Stimulus)
  post.hoc <- pairwise_t_test(temp, Value ~ Stimulus, paired = TRUE, p.adjust.method = "bonferroni")
  post.hoc <- add_xy_position(post.hoc, x = "Stimulus")
  print(erp.model)
  
  height <- max(temp$Value)
  ggboxplot(temp, x = "Stimulus", y = "Value",
            color = "Stimulus", palette = c("#00AFBB", "#E7B800", "#A03D41"),
            ylab = title, xlab = "Stimulus") + 
    stat_summary(fun=mean, colour="black", aes(group=1),
                 geom="line", lwd=1, lty=1) +
    stat_pvalue_manual(post.hoc, tip.length = 0, hide.ns = FALSE, y.position = c(height+1, height+2, height+3)) +
    labs(
      subtitle = get_test_label(erp.model, detailed = TRUE),
      caption = get_pwc_label(post.hoc)
    )
}

# N1_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "N1_Amp", ignore=c(10, 19))

# N2_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "N2_Amp", ignore=c(3, 19, 31, 32))

# P2_Amp as a factor of Stimulus intensity alone?
simple_model(erp, "P2_Amp", ignore=c(5, 8, 15, 23, 43))

# Gamma_Amp as a factor of Stimulus intensity alone?
simple_model(freq, "Gamma_Amp", ignore=c())

#### Comparison Between Genders ####

# model ERP component as a factor of Sex, controlling for Stim level and with
# Subject as a random effect (since we have subject-specific pain thresholds)

signif <- function(p) {
  val <- list()
  for (i in seq_len(length(p))){
    if (p[i] < 0.001) {val <- append(val, "***")}
    else if (p[i] < 0.01) {val <- append(val, "**")}
    else if (p[i] < 0.05) {val <- append(val, "*")}
    else if (p[i] < 0.1) {val <- append(val, ".")}
    else if (p[i] >= 0.1) {val <- append(val, " ")}
  }
  return(val)
}

model <- function(df, title) {
  temp <- subset(df, df$Component==title)
  erp.model <- lmer(Value ~ Sex * Stimulus + (1|ID), data = temp, REML = TRUE)
  print(anova(erp.model))  # look for main effects/interactions
  print(rand(erp.model))  # look at the significance of our random effect (subject)
  
  post.hoc <- data.frame(pairs(emmeans(erp.model, ~ Sex * Stimulus), adjust = "bonferroni"))
  post.hoc$Signif <- signif(post.hoc$p.value)
  print(post.hoc)
  
  # shape post.hoc so that we can plot it easier
  height <- max(temp$Value)
  columns <- c("Stimulus", "group1", "group2", "p", "y.position")
  stats <- data.frame(matrix(nrow = 0, ncol = length(columns))) 
  colnames(stats) = columns
  
  # check residuals
  #R = residuals(erp.model)
  #ggqqplot(R)
  
  # plot results
  ggboxplot(temp, x = "Sex", y = "Value",
            color = "Stimulus", palette = c("#028090", "#C64191", "#157F1F"),
            ylab = title, xlab = "Sex", facet.by = "Stimulus") +
    stat_summary(fun=mean, colour="red", aes(group=1),
                 geom="line", lwd=1, lty=1) + 
    stat_summary(fun=mean, colour="black", aes(group=1),
                 geom="point", size=2)
}

# N1 LATENCY
#model(erp, "N1_Lat")

# N1 AMPLITUDE
model(erp, "N1_Amp")

# N2 LATENCY
#model(erp, "N2_Lat")

# N2 AMPLITUDE
model(erp, "N2_Amp")

# P2 LATENCY
#model(erp, "P2_Lat")

# P2 AMPLITUDE
model(erp, "P2_Amp")

# Gamma LATENCY
#model(freq, "Gamma_Lat")

# Gamma AMPLITUDE
model(freq, "Gamma_Amp")
