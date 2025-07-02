library(readxl)
library(ggplot2)
library(ggpubr)


rm(list=ls())


# ---- Load the dataframes ----
## Load the BSI data
bsi.df = as.data.frame(read_excel("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx"))
rownames(bsi.df) <- bsi.df$`Run accession`
bsi.df$`Run accession` <- NULL

## Load the UTI data
uti.df <- as.data.frame(read_excel('/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/Table S1 Edited Ørjan Rebecca FINAL.xlsx'))
rownames(uti.df) <- uti.df$`Sample accession`
bsi.df$`Sample accession` <- NULL

uti.metadata <- read_excel('/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/FINAL Rev Table S1.xlsx',
                           sheet="Metadata")
uti.metadata <- as.data.frame(uti.metadata)
rownames(uti.metadata) <- uti.metadata$`Sample accession`
uti.df[rownames(uti.df),"Clade"] <- uti.metadata[rownames(uti.df),"ST131 clade"]


## Preprocess the BSI dataframe
mask_gent <- bsi.df$Gentamicin_ResType == "Sonediameter"
mask_cip <- bsi.df$Ciprofloxacin_ResType == "Sonediameter"
mask_cip[is.na(mask_cip)] <- FALSE
mask_cef <- bsi.df$Ceftazidim_ResType == "Sonediameter"
mask <- mask_gent & mask_cip & mask_cef

columns = c('Year','ST',"Clade",'Ciprofloxacin','Gentamicin','Ceftazidim')
bsi.df = bsi.df[mask,columns]
colnames(bsi.df) <- c('Year','ST',"Clade",'Ciprofloxacin','Gentamicin','Ceftazidime')
bsi.df$Ciprofloxacin = as.numeric(bsi.df$Ciprofloxacin)
bsi.df$Gentamicin = as.numeric(bsi.df$Gentamicin)
bsi.df$Ceftazidime = as.numeric(bsi.df$Ceftazidime)
bsi.df$ST = as.character(bsi.df$ST)
bsi.df$Clade = as.character(bsi.df$Clade)

## Preprocess the UTI dataframe
uti.cols <- c("ST","Clade","Cipro         DD","Genta        DD", "Ceftazidim DD")
uti.df <- uti.df[,uti.cols]
colnames(uti.df) <- c("ST","Clade","Ciprofloxacin","Gentamicin","Ceftazidime")
uti.df$Ciprofloxacin <- as.numeric(uti.df$Ciprofloxacin)
uti.df$Gentamicin <- as.numeric(uti.df$Gentamicin)
uti.df$Ceftazidime <- as.numeric(uti.df$Ceftazidime)
# uti.df %>% drop_na(Gentamicin)

# uti.df$Clade <- NA
# uti.df[rownames(uti.df),"Clade"] <- uti.metadata[rownames()]


# ---- Combine ST and Clade labels for grouping ----
bsi.df <- bsi.df %>% unite("Group", c("ST","Clade"), sep="-", na.rm=TRUE)
bsi.df$Group[bsi.df$Group == '131-C1'] <- '131-C'
bsi.df$Group[bsi.df$Group == '131-C2'] <- '131-C'

uti.df <- uti.df %>% unite("Group", c("ST","Clade"), sep="-", na.rm=TRUE)
uti.df$Group[uti.df$Group == '131-C1'] <- '131-C'
uti.df$Group[uti.df$Group == '131-C2'] <- '131-C'

# ---- Isolate the top 5 most common STs ----
top.bsi.sts = tail(names(sort(table(bsi.df$Group))), 20)
top.bsi.sts = c(top.bsi.sts, "other")

top.uti.sts = tail(names(sort(table(uti.df$Group))), 20)
top.uti.sts = c(top.uti.sts, "other")
# top.uti.sts <- top.bsi.sts

## Add an indicator column
bsi.df$Type <- "BSI"
uti.df$Type <- "UTI"
uti.df$Year <- NA

# ---- Plot the top few STs of BSI isolates ----
temp.df <- bsi.df
temp.df$Group <- ifelse(temp.df$Group %in% top.bsi.sts, temp.df$Group, 'other')


p1 <- ggplot(temp.df, aes(x = factor(Group,levels=top.bsi.sts), y = Ciprofloxacin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p2 <- ggplot(temp.df, aes(x = factor(Group,levels=top.bsi.sts), y = Gentamicin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p3 <- ggplot(temp.df, aes(x = factor(Group,levels=top.bsi.sts), y = Ceftazidime,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")

plot <- ggarrange(p1, p2, p3, ncol=3)
ggsave("output/boxplots_ST_DD_bsi.png", height=3, width=8, dpi=400)
print(plot)



# ---- Plot the top few STs of UTI isolates ----
temp.df <- uti.df
temp.df$Group <- ifelse(temp.df$Group %in% top.uti.sts, temp.df$Group, 'other')

p1 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Ciprofloxacin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p2 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Gentamicin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p3 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Ceftazidime,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")

plot <- ggarrange(p1, p2, p3, ncol=3)
ggsave("output/boxplots_ST_DD_uti.png", height=3, width=8, dpi=400)
print(plot)


# ---- ANOVA test ----
# labs = factor(bsi.df$Group)
# # labs = factor(ifelse(bsi.df$Group=="131-C", "131-C", "other"))
# Y = cbind(bsi.df$Ciprofloxacin, bsi.df$Gentamicin, bsi.df$Ceftazidime)
# fit = manova(Y ~ labs, data=bsi.df)



# ---- Plot AST data by year ----
# year.df <- read.csv("data/processed-spreadsheets/BSI_data.csv")
# year.df <- bsi.df[bsi.df$Year >= 2013,]
year.df <- bsi.df
year.df$Year <- factor(year.df$Year)
g1 <- ggplot(year.df, aes(x = Year, y = Ciprofloxacin,
                          fill = Year)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Year") + theme(legend.position="none")
g2 <- ggplot(year.df, aes(x = Year, y = Gentamicin,
                          fill = Year)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Year") + theme(legend.position="none")
g3 <- ggplot(year.df, aes(x = Year, y = Ceftazidime,
                          fill = Year)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Year") + theme(legend.position="none")

glob <- ggarrange(g1, g2, g3, ncol=3)
ggsave("output/violinplots_year_DD_bsi.png", height=3, width=6, dpi=400)
glob

## Compute the mean of each distribution for each ATB for each year
years <- sort(unique(year.df$Year))
means.df <- data.frame(matrix(ncol=4, nrow=length(years)), row.names=years)
colnames(means.df) <- c("Year","Ciprofloxacin","Gentamicin","Ceftazidime")
means.df$Year <- years
for (yr in years){
  yr.mask <- year.df$Year == yr
  means.df[yr,"Ciprofloxacin"] <- median(year.df[yr.mask,"Ciprofloxacin"])
  means.df[yr,"Gentamicin"] <- median(year.df[yr.mask,"Gentamicin"])
  means.df[yr,"Ceftazidime"] <- median(year.df[yr.mask,"Ceftazidime"])
}


# ---- Plot UTI vs BSI data ----
# rownames(bsi.df) <- bsi.df$`Sample accession`
# uti.cols <- c("Sample accession", "ST","Cipro         DD","Genta        DD", "Ceftazidim DD")
# uti.df <- uti.df[,uti.cols]
# colnames(uti.df) <- c("Run accession","Group","Ciprofloxacin","Gentamicin","Ceftazidime")
# uti.df$Gentamicin <- as.numeric(uti.df$Gentamicin)



temp.df <- rbind(uti.df, bsi.df[bsi.df$Year>2013,])

## Plot them
b1 <- ggplot(temp.df, aes(x = Type, y = Ciprofloxacin, 
                          fill = Type)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(legend.position="none", axis.title.x=element_blank())
b2 <- ggplot(temp.df, aes(x = Type, y = Gentamicin, 
                          fill = Type)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(legend.position="none", axis.title.x=element_blank())
b3 <- ggplot(temp.df, aes(x = Type, y = Ceftazidime, 
                          fill = Type)) +
  geom_violin(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(legend.position="none", axis.title.x=element_blank())

boxes <- ggarrange(b1, b2, b3, ncol=3) 
ggsave("output/violinplots_bsi_uti.png", width=4, height=3, dpi=400)
boxes


# ---- Plot 
