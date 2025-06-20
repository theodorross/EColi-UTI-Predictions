library(readxl)
library(ggplot2)
library(ggpubr)


rm(list=ls())


# ---- Load the dataframe ----
df = read_excel("data/raw-spreadsheets/per_isolate_AST_DD_SIR_v4.xlsx")
# rownames(df) <- df$`Run accession`

# ---- filter by test type ----
mask_gent <- df$Gentamicin_ResType == "Sonediameter"
mask_cip <- df$Ciprofloxacin_ResType == "Sonediameter"
mask_cip[is.na(mask_cip)] <- FALSE
mask_cef <- df$Ceftazidim_ResType == "Sonediameter"
mask <- mask_gent & mask_cip & mask_cef

columns = c('Run accession','ST',"Clade",'Ciprofloxacin','Gentamicin','Ceftazidim')
df = df[mask,columns]
colnames(df) <- c('Run accession','ST',"Clade",'Ciprofloxacin','Gentamicin','Ceftazidime')
df$Ciprofloxacin = as.numeric(df$Ciprofloxacin)
df$Gentamicin = as.numeric(df$Gentamicin)
df$Ceftazidime = as.numeric(df$Ceftazidime)
df$ST = as.character(df$ST)
df$Clade = as.character(df$Clade)

# ---- Combine ST and Clade labels for grouping ----
df <- df %>% unite("Group", c("ST","Clade"), sep="-", na.rm=TRUE)
df$Group[df$Group == '131-C1'] <- '131-C'
df$Group[df$Group == '131-C2'] <- '131-C'

# ---- Isolate the top 5 most common STs ----
top.sts = tail(names(sort(table(df$Group))), 20)
top.sts = c(top.sts, "other")

# ---- Plot the top few STs ----
plots = list()
temp.df <- df
temp.df$Group <- ifelse(temp.df$Group %in% top.sts, temp.df$Group, 'other')


p1 <- ggplot(temp.df, aes(x = factor(Group,levels=top.sts), y = Ciprofloxacin, 
                          fill = Group)) + 
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p2 <- ggplot(temp.df, aes(x = factor(Group,levels=top.sts), y = Gentamicin, 
                          fill = Group)) + 
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")
p3 <- ggplot(temp.df, aes(x = factor(Group,levels=top.sts), y = Ceftazidime, 
                          fill = Group)) + 
  geom_boxplot(outliers=F) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("ST-Clade") + theme(legend.position="none")

# grid.arrange(p1, p2, p3, ncol=3)
# plot <- ggarrange(p1, p2, p3, ncol=3, common.legend=TRUE, legend=NA)
plot <- ggarrange(p1, p2, p3, ncol=3) 
ggsave("output/ST_DD_boxplots.png", height=3, width=8, dpi=400)
print(plot)


# ---- ANOVA test ----
labs = factor(df$Group)
# labs = factor(ifelse(df$Group=="131-C", "131-C", "other"))
Y = cbind(df$Ciprofloxacin, df$Gentamicin, df$Ceftazidime)
fit = manova(Y ~ labs, data=df)




