library("biom")

# dat_file = system.file("data/Subramanian_et_al_otu_table.biom")

# read in file
raw_dat = read_biom("data/Subramanian_et_al_otu_table.biom")
# singleton mat with depth of 2000
# singletons_dat = read_biom("data/only_Healthy_Singletons_d2000.biom")
rarefied_dat = read_biom("data/otu_table_over_onetenthpercent.biom") # one-tenth percent rarefied

# get data matrix
raw_mat = biom_data(raw_dat)
rarefied_mat = biom_data(rarefied_dat)

map_mat = read.table("data/Subramanian_et_al_mapping_file.txt", header=FALSE, row.names=1, sep="\t")

colnames(map_mat) <- c("BarcodeSequence", "LinkerPrimerSequence", "PersonID", "FamilyID", "ENA.SampleID", 
    "ENA.LibraryName", "Health_Analysis_Groups", "tax_id", "scientific_name", "common_name", "anonymized_name",
    "sample_title", "sample_description", "investigation type", "project name", "target gene",
    "sequencing method", "collection date", "host subject id", "age_in_months", "sex", "environmental package",
    "geographic location_latitude", "geographic location_longitude", "geographic location_country",
    "environment_biome", "environment_feature", "environment_material", "Description")

dat_df = data.frame(t(as.matrix(dat_mat)), check.names=FALSE)

rarefied_df = data.frame(t(as.matrix(rarefied_mat)), check.names=FALSE)
map_df = data.frame(map_mat)
 
# [rowname, colname] 

# ======================================================
# Missing data:
# Mapping file: Bgsng7035.m1, h37B.1, h37A.1, h10A.1
# ======================================================

# delete missing data from dat data frame
# dat_row = rownames(dat_df)
# dat_df = dat_df[-match("Bgsng7035.m1", dat_row),]
dat_row = rownames(dat_df)
dat_df = dat_df[-match("h37B.1", dat_row),]
dat_row = rownames(dat_df)
dat_df = dat_df[-match("h37A.1", dat_row),]
dat_row = rownames(dat_df)
dat_df = dat_df[-match("h10A.1", dat_row),]

# make sample id a column rather than row names
dat_df$id <- rownames(dat_df) 

rarefied_df$id <- rownames(rarefied_df)
map_df$id <- rownames(map_df)

# rownames(dat_df) <- NULL
# rownames(map_df) <- NULL

# adding rows (not used)
# total <- rbind(data frameA, data frameB)

# merge the two tables
# tot_mat <- merge(dat_df, map_df,by="id")
all_mat <- merge(rarefied_df, map_df,by="id")

write.table(all_mat, "all_mat.csv", sep=",", row.names=FALSE)

# write tot_mat to file
# write.csv(tot_mat, file = "tot_mat.csv")

# svm library
library("e1071")

# add age factor vector 
all_mat$age_factor <- factor(all_mat$age_in_months)

# get all subtables
h_singletons_mat <- subset(all_mat, Health_Analysis_Groups == "Healthy Singletons") 
h_tt_mat <- subset(all_mat, Health_Analysis_Groups == "Healthy Twins Triplets")

write.table(h_singletons_mat, "singletons.csv", sep=",", row.names=FALSE)
write.table(h_tt_mat, "twins.csv", sep=",", row.names=FALSE)

# extract the otu columns [2:79598]
heads = colnames(h_singletons_mat)
otu_lst = heads[2:82]

# get healthy singletons
sub_col_lst <- append(otu_lst, "age_factor")

sub_singletons_mat <- subset(h_singletons_mat, select=sub_col_lst) # Health_Analysis_Groups == "Healthy Singletons", 

single_train = subset(sub_singletons_mat, select = -age_factor)
single_age = sub_singletons_mat$age_factor

svm_model <- svm(single_train, single_age, type='C-classification', scale=FALSE)
# summary(svm_model)

# predict same singleton data
sing_pred <- predict(svm_model, single_train)

sing_pred_nums <- as.numeric(as.character(sing_pred))
sing_actual_nums <- as.numeric(as.character(single_age))

plot(sing_pred_nums, sing_actual_nums)

# predict twin data
sub_tt_mat <- subset(h_tt_mat, select=sub_col_lst)
tt_train <- subset(sub_tt_mat, select= -age_factor)
tt_age <- sub_tt_mat$age_factor

tt_pred <- predict(svm_model, tt_train)

tt_pred_nums <- as.numeric(as.character(tt_pred))
tt_actual_nums <- as.numeric(as.character(tt_age))

plot(tt_actual_nums, tt_pred_nums)




