library(readxl)
library(ggplot2)
library(ggpubr)
library(tidyr)
library(cramer)

rm(list=ls())

# ---- Read the unsequenced dataframes ----
bsi.df <- read.csv("data/processed-spreadsheets/NORM_data.csv")
old.df <- bsi.df[bsi.df$Year<2011,]
new.df <- bsi.df[bsi.df$Year>=2011,]

atbs <- c("Ciprofloxacin","Ceftazidim","Gentamicin")

# Define masks for ST131-C isolates
old.mask <- old.df$Group=="131-C1" | old.df$Group=="131-C2"
new.mask <- new.df$Group=="131-C1" | new.df$Group=="131-C2"
whole.mask <- bsi.df$Group=="131-C1" | bsi.df$Group=="131-C2"

# Compute Cramer test statistics for each group
old.test <- cramer.test(x=as.matrix(old.df[old.mask,atbs]),
                        y=as.matrix(old.df[!old.mask,atbs]))
new.test <- cramer.test(x=as.matrix(new.df[new.mask,atbs]),
                        y=as.matrix(new.df[!new.mask,atbs]))
whole.test <- cramer.test(x=as.matrix(bsi.df[whole.mask,atbs]),
                          y=as.matrix(bsi.df[!whole.mask,atbs]))



