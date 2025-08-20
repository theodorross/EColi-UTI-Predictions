library(readxl)
library(ggplot2)
library(ggpubr)
library(tidyr)
library(cramer)
library(VennDiagram)

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

# uti.df <- uti.df %>% unite("Group", c("PP","Clade"), sep="-", na.rm=TRUE)
uti.df$Group <- uti.df$ST
uti.df$Group[uti.df$Clade == 'C1'] <- '131-C'
uti.df$Group[uti.df$Clade == 'C2'] <- '131-C'
uti.df$ST <- NULL
uti.df$Clade <- NULL
# uti.df$Group[uti.df$Group == '2-C1'] <- '131-C'
# uti.df$Group[uti.df$Group == '2-C2'] <- '131-C'

# ---- Isolate the top 5 most common STs ----
top.bsi.sts = tail(names(sort(table(bsi.df$Group))), 20)
top.bsi.sts = c(top.bsi.sts, "other")

top.uti.sts = tail(names(sort(table(uti.df$Group))), 20)
top.uti.sts = c(top.uti.sts, "other")
# top.uti.sts <- top.bsi.sts

## Add an indicator column
bsi.df$Type <- "BSI"
uti.df$Type <- "UTI"
uti.df$Year <- 2019

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

# Save the plot limits
cip.lim <- layer_scales(p1)$y$range$range
gen.lim <- layer_scales(p2)$y$range$range
cef.lim <- layer_scales(p3)$y$range$range


# ---- Plot the top few STs of UTI isolates ----
temp.df <- uti.df
temp.df$Group <- ifelse(temp.df$Group %in% top.uti.sts, temp.df$Group, 'other')

p1 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Ciprofloxacin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none") + 
  ylim(cip.lim)
p2 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Gentamicin,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none") + 
  ylim(gen.lim)
p3 <- ggplot(temp.df, aes(x = factor(Group,levels=top.uti.sts), y = Ceftazidime,
                          fill = Group)) +
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none") + 
  ylim(cef.lim)

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

## Test the distributions of each
temp.df <- na.omit(temp.df[,c("Ciprofloxacin","Gentamicin","Ceftazidime","Type")])
m1 <- as.matrix(temp.df[temp.df$Type=="BSI",c("Ciprofloxacin","Gentamicin","Ceftazidime")])
m2 <- as.matrix(temp.df[temp.df$Type=="UTI",c("Ciprofloxacin","Gentamicin","Ceftazidime")])
# c.test <- cramer.test(x=m2, y=m1)

## Compare the values of ST131-C
uti.131c <- na.omit(uti.df[uti.df$Group=="131-C",c("Ciprofloxacin","Gentamicin","Ceftazidime")])
bsi.131c <- na.omit(bsi.df[bsi.df$Group=="131-C",c("Ciprofloxacin","Gentamicin","Ceftazidime")])

std.err <- function(x) c(mean(x) , sd(x)/sqrt(length(x)) )
uti.out <- lapply(uti.131c, std.err)
bsi.out <- lapply(bsi.131c, std.err)
c.test <- cramer.test(as.matrix(uti.131c), as.matrix(bsi.131c))



# ---- Find overlaps between Q1 and Q3 ----
quartile.df <- data.frame(Ciprofloxacin=quantile(bsi.131c$Ciprofloxacin, prob=c(0.25,0.75)),
                          Ceftazidime=quantile(bsi.131c$Ceftazidime, prob=c(0.25,0.75)),
                          Gentamicin=quantile(bsi.131c$Gentamicin, prob=c(0.25,0.75)))

## Fraction of UTI 131-C in interquartile ranges
uti.non131c.mask <- uti.df$Group != "131-C"
uti.non131c <- uti.df[uti.non131c.mask,]
antibiotics <- c("Ciprofloxacin","Ceftazidime","Gentamicin")
in.out.df <- data.frame(row.names=c("uti.131c.in","uti.non.131c.in"))


## Initialize dataframes for tracking which isolates fall between Q1 and Q3
overlap.df.non.131c <- data.frame(row.names=rownames(uti.df))
overlap.df.131c <- data.frame(row.names=rownames(uti.df))


for (atb in antibiotics){
  # Find UTI 131c in
  mask.up <- uti.131c[,atb] >= quartile.df["25%",atb]
  mask.dn <- uti.131c[,atb] <= quartile.df["75%",atb]
  uti.131.in.mask <- mask.up & mask.dn
  in.out.df["uti.131c.in",atb] <- mean(uti.131.in.mask, na.rm=TRUE)

  # Find UTI non-131c withing
  mask.up <- uti.non131c[,atb] >= quartile.df["25%",atb]
  mask.dn <- uti.non131c[,atb] <= quartile.df["75%",atb]
  uti.non131.in.mask <- mask.up & mask.dn
  in.out.df["uti.non.131c.in",atb] <- mean(uti.non131.in.mask, na.rm=TRUE)
  
  # Populate the tracker dataframes
  non.ix <- rownames(uti.non131c)[uti.non131.in.mask]
  non.ix <- non.ix[!is.na(non.ix)]
  overlap.df.non.131c[,atb] <- FALSE
  overlap.df.non.131c[non.ix,atb] <- TRUE
  
  ix <- rownames(uti.131c)[uti.131.in.mask]
  ix <- ix[!is.na(ix)]
  overlap.df.131c[,atb] <- FALSE
  overlap.df.131c[ix,atb] <- TRUE
  
  # Loop through common STs
  for (st in top.uti.sts){
    if (!(st %in% c("131-C","other"))){
      st.mask <- uti.df$Group == st
      mask.up <- uti.131c[st.mask,atb] >= quartile.df["25%",atb]
      mask.up <- uti.131c[st.mask,atb] <= quartile.df["75%",atb]
      uti.st.in.mask <- mask.up & mask.dn
      in.out.df[st,atb] <- mean(uti.st.in.mask, na.rm=TRUE)
    }
  }

}


## Draw Venn Diagram of overlapping non-131C isolates
venn.list.non.131 <- list(Ciprofloxacin=rownames(overlap.df.non.131c)[overlap.df.non.131c$Ciprofloxacin],
                          Gentamicin=rownames(overlap.df.non.131c)[overlap.df.non.131c$Gentamicin],
                          Ceftazidime=rownames(overlap.df.non.131c)[overlap.df.non.131c$Ceftazidime])

venn.list.131 <- list(Ciprofloxacin=rownames(overlap.df.131c)[overlap.df.131c$Ciprofloxacin],
                      Gentamicin=rownames(overlap.df.131c)[overlap.df.131c$Gentamicin],
                      Ceftazidime=rownames(overlap.df.131c)[overlap.df.131c$Ceftazidime])


venn.diagram(venn.list.131,
             filename="output/venn_diagram_uti_non-131C.png",
             category.names=c("CIP","GEN","CEF"),
             col=c("#5d9864","#c1437a","#6088df"),
             cat.col=c("#5d9864","#c1437a","#6088df"),
             height=1000,
             width=1000,
             dpi=300,
             cex=0.5, cat.cex=0.5,
             imagetype = "png",
             disable.logging=T)

venn.diagram(venn.list.non.131,
             filename="output/venn_diagram_uti_131C.png",
             category.names=c("CIP","GEN","CEF"),
             col=c("#5d9864","#c1437a","#6088df"),
             cat.col=c("#5d9864","#c1437a","#6088df"),
             height=1000,
             width=1000,
             dpi=300,
             cex=0.5, cat.cex=0.5,
             imagetype = "png",
             disable.logging=T)


