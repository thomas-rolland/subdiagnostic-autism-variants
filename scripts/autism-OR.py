#!/bin/python3
# Author: Thomas Rolland	jerikoo75@gmail.com
# Date: 03/30/2022

import pandas as pd
from random import sample
from numpy import mean, sqrt, log10
from plotnine import *
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import statsmodels.stats as sm

#===================================================================================================
# autism-OR.py
# Calculate gene-level autism-OR
#
# Input files and required columns (and corresponding variable format or accepted values):
# - individuals.tsv = required columns 'IID' (str), 'FID' (str), 'status' ('case' or 'control')
# - variants.tsv = required columns 'IID' (str), 'FID' (str), 'chr:pos:ref:alt' (str), 'gene' (str)
# - genes.tsv = required columns 'gene' (str), 'avg_PxRy' with x in [1, 4] and y in [1, 8] (float)
# NOTE1: autism-OR is only calculated on genes with variants among cases in variants.tsv file
# NOTE2: correlation with brain expression is not calculated for P1R4 because of too small sample size
#
# Output files
# - autism-OR.tsv = gene, number of carriers and number of individuals for cases and controls, autism-OR, p-value, 95% CI of p-value
# - autism-OR.pdf = scatterplot of number of carriers among cases vs. autism-OR of each gene
# - brain_correlation.tsv = period/region, period, region, pearson R, p-value and corrected p-value for correlation betweem autism-OR and brain expression
# - brain_correlation.pdf = heatmap of pearson R for each period/region, labeled with corrected p-value
#===================================================================================================

def get_autismOR(individuals, variants, n_subsamplings, size_subsample):

	# Filtering variants in cases to singletons (one individual or several members of the same family)
	individuals_cases = individuals[(individuals["status"] == "case")].copy()
	variants_cases = variants[variants["IID"].isin(individuals_cases["IID"]) == True].copy()
	variants_cases_tmp = variants_cases[["chr:pos:ref:alt", "FID"]].drop_duplicates().reset_index(drop = True)
	PT_cases = pd.pivot_table(variants_cases_tmp, index = "chr:pos:ref:alt", aggfunc = lambda x: x.nunique())
	singleton_variants_cases = PT_cases.index[PT_cases["FID"] == 1]
	variants_cases = variants_cases[(variants_cases["chr:pos:ref:alt"].isin(pd.Series(singleton_variants_cases)) == True)].copy()

	# Getting number of carriers among cases for each gene
	IID2Gene = variants_cases[["IID", "gene"]].copy()
	IID2Gene.columns = ["IID", "Gene"]
	individuals_cases = individuals_cases[individuals_cases["IID"].isin(IID2Gene["IID"]) == True]
	individuals_cases = pd.merge(IID2Gene, individuals_cases, on = "IID", how = "left")
	individuals_cases.loc[individuals_cases["Gene"].isna(), "Gene"] = "none"
	individuals_cases = individuals_cases[["IID", "status", "Gene"]].copy()
	PT_cases = pd.pivot_table(individuals_cases, index = "Gene", values = "IID", aggfunc = lambda x: x.nunique())
	PT_cases["gene"] = PT_cases.index
	PT_cases["N_cases"] = individuals[(individuals["status"] == "case")].IID.nunique()
	PT_cases.columns = ["NC_cases", "gene", "N_cases"]
	counts_cases = PT_cases.drop_duplicates().reset_index(drop = True)

	# Sub-sampling procedure to match number of controls to number of cases
	counts_controls_samp = pd.DataFrame()
	individuals = individuals[individuals["status"] != "case"]
	individuals_IIDs = list(individuals["IID"].unique())
	for sampling in range(0, n_subsamplings):
		# Sampling as many control IIDs as case IIDs
		sampled_iids = sample(individuals_IIDs, size_subsample)

		# Filtering variants in controls to singletons (one individual or several members of the same family)
		individuals_controls = individuals[(individuals["IID"].isin(pd.Series(sampled_iids)) == True)].copy()
		variants_controls = variants[(variants["IID"].isin(pd.Series(sampled_iids)) == True)].copy()
		variants_controls_tmp = variants_controls[["chr:pos:ref:alt", "FID"]].drop_duplicates().reset_index(drop = True)
		PT_controls = pd.pivot_table(variants_controls_tmp, index = "chr:pos:ref:alt", aggfunc = lambda x: x.nunique())
		singleton_variants_controls = PT_controls.index[PT_controls["FID"] == 1]
		variants_controls = variants_controls[(variants_controls["chr:pos:ref:alt"].isin(pd.Series(singleton_variants_controls)) == True)].copy()

		# Getting number of carriers among controls for each gene
		IID2Gene = variants_controls[["IID", "gene"]].copy()
		IID2Gene.columns = ["IID", "Gene"]
		individuals_controls = individuals_controls[individuals_controls["IID"].isin(IID2Gene["IID"]) == True]
		individuals_controls = pd.merge(IID2Gene, individuals_controls, on = "IID", how = "left")
		individuals_controls.loc[individuals_controls["Gene"].isna(), "Gene"] = "none"
		individuals_controls = individuals_controls[["IID", "status", "Gene"]].copy()
		PT_controls = pd.pivot_table(individuals_controls, index = "Gene", values = "IID", aggfunc = lambda x: x.nunique())
		PT_controls["gene"] = PT_controls.index
		PT_controls["N_controls"] = len(sampled_iids)
		PT_controls["sampling"] = sampling
		PT_controls.columns = ["NC_controls", "gene", "N_controls", "sampling"]
		counts_controls = PT_controls.drop_duplicates().reset_index(drop = True)
		counts_controls_samp = pd.concat([counts_controls_samp, counts_controls])

	# Calculating average number of carriers per gene in all sub-samplings
	counts_controls = pd.DataFrame()
	for gene in counts_cases.gene.unique():
		counts_controls_samp_gene = counts_controls_samp[counts_controls_samp["gene"] == gene].copy().reset_index(drop = True)
		# Add rows when 0 carriers were found among controls
		if counts_controls_samp_gene.shape[0] < n_subsamplings:
			n = n_subsamplings - counts_controls_samp_gene.shape[0]
			chunk = pd.concat([pd.Series([0] * n), pd.Series([gene] * n), pd.Series([size_subsample] * n), pd.Series(range(0, n))], axis = 1)
			chunk.columns = ["NC_controls", "gene", "N_controls", "sampling"]
			counts_controls_samp_gene = pd.concat([counts_controls_samp_gene, chunk])
		NC_controls = mean(counts_controls_samp_gene["NC_controls"])
		N_controls = mean(counts_controls_samp_gene["N_controls"])
		chunk = pd.concat([pd.Series(NC_controls), pd.Series(gene), pd.Series(N_controls)], axis = 1)
		counts_controls = pd.concat([counts_controls, chunk])
	counts_controls.columns = ["NC_controls", "gene", "N_controls"]

	# Calculating odds ratio for autism (autism-OR)
	counts = pd.merge(counts_cases, counts_controls, on = "gene", how = "left")
	counts = counts[counts["gene"] != "none"]
	counts["autism-OR"] = pd.NA
	for gene in counts.gene.unique():
		NC_cases = list(counts["NC_cases"][counts["gene"] == gene])[0]
		N_cases = list(counts["N_cases"][counts["gene"] == gene])[0]
		NC_controls = list(counts["NC_controls"][counts["gene"] == gene])[0]
		N_controls = list(counts["N_controls"][counts["gene"] == gene])[0]
		if NC_controls > 0:
			observed_penetrance = (NC_cases * (N_controls - NC_controls)) / ((N_cases - NC_cases) * NC_controls)
		else:
			observed_penetrance = float("inf")
		counts.loc[counts["gene"] == gene, "autism-OR"] = observed_penetrance

	# Reordering columns and return
	counts = counts[["gene", "NC_cases", "N_cases", "NC_controls", "N_controls", "autism-OR"]]
	return (counts)


##############################################################################################################################
######################################################### Parameters #########################################################
##############################################################################################################################

individuals = pd.read_csv("individuals.tsv", sep = "\t", header = 0, dtype = {'FID': 'str', 'IID': 'str'})
variants = pd.read_csv("variants.tsv", sep = "\t", header = 0, dtype = {'FID': 'str', 'IID': 'str'})
genes = pd.read_csv("genes.tsv", sep = "\t", dtype = str, header = 0)

n_subsamplings = 100
n_bootstraps = 100
size_subsample = individuals[(individuals["status"] == "case")].IID.nunique()

adjust_text_dict = {'arrowprops': { 'arrowstyle': '-', 'color': 'black' }, 'expand_points': (1.5, 1.5), 'lim': 100, 'precision': 5.5, 'force_text': (1, 1) }#

output_autismOR_tsv = "autism-OR.tsv"
output_autismOR_pdf = "autism-OR.pdf"
output_brainCorr_tsv = "autism-OR_brain-exp_correlation.tsv"
output_brainCorr_pdf = "autism-OR_brain-exp_correlation.pdf"

##############################################################################################################################



##############################################################################################################################
################################################### Calculate autism-OR ######################################################
##############################################################################################################################

print (">>>>>>>>>>> Observed autism-OR by sub-sampling controls")
counts = get_autismOR(individuals, variants, n_subsamplings, size_subsample)

print (">>>>>>>>>>> P-value of autism-OR using bootstrap procedure based on sub-sampling")
df_bootstrap = pd.DataFrame()
for bootstrap in range(0, n_bootstraps):
	print ("\tBootstrap", bootstrap, "over", n_bootstraps)
	individuals_tmp = individuals.copy()
	sampled_iids = sample(list(individuals_tmp["IID"]), size_subsample)
	individuals_tmp.loc[individuals_tmp["IID"].isin(pd.Series(sampled_iids)) == True, "status"] = "case"
	individuals_tmp.loc[individuals_tmp["IID"].isin(pd.Series(sampled_iids)) == False, "status"] = "control"
	counts_boot = get_autismOR(individuals_tmp, variants, n_subsamplings, size_subsample)
	counts_boot["bootstrap"] = bootstrap
	df_bootstrap = pd.concat([df_bootstrap, counts_boot])

print (">>>>>>>>>>> Measuring significance of autism-OR by gene ...")
counts["p"] = pd.NA
counts["p_CI5"] = pd.NA
counts["p_CI95"] = pd.NA
for gene in counts["gene"].unique():
	observed_autismOR = list(counts["autism-OR"][counts["gene"] == gene].astype("float"))[0]
	n_expected = df_bootstrap[(df_bootstrap["gene"] == gene) & (df_bootstrap["autism-OR"] >= observed_autismOR)].shape[0]
	p = (n_expected + 1)/(n_bootstraps + 1)
	CIl = p - (1.96 * sqrt((p * (1 - p))/(n_bootstraps + 1)))
	CIh = p + (1.96 * sqrt((p * (1 - p))/(n_bootstraps + 1)))
	counts.loc[counts["gene"] == gene, "p"] = p
	counts.loc[counts["gene"] == gene, "p_CI5"] = CIl
	counts.loc[counts["gene"] == gene, "p_CI95"] = CIh
counts.to_csv(output_autismOR_tsv, sep = "\t", header = True, index = False)

print (">>>>>>>>>>> Plot number of carriers among autistic individuals vs. autism-OR by gene ...")
counts = pd.read_csv(output_autismOR_tsv, sep = "\t", header = 0, dtype = {'NC_cases': 'int', 'autism-OR': 'float', 'gene': "str", 'p_CI95': "float"})
counts["above_exp"] = "No"
counts.loc[counts["p_CI95"] < 0.05, "above_exp"] = "Yes"
ymax = max(counts["autism-OR"][counts["autism-OR"] < float("inf")]) * 1.5
counts.loc[counts["autism-OR"] == float("inf"), "autism-OR"] = ymax
plot = ggplot(counts) + aes(x = 'NC_cases', y = 'autism-OR', label = 'gene') + scale_y_log10(limits = [0.5, ymax]) + scale_x_log10(minor_breaks = range(1, 20)) + geom_point(aes(color = "above_exp"), alpha = 0.5) + theme_matplotlib(rc={'pdf.fonttype':42}) + geom_text(aes(label='gene'),data=counts, size=8, adjust_text=adjust_text_dict)
plot.save(output_autismOR_pdf, height = 8, width = 8, format = "pdf")



##############################################################################################################################
############################################# Correlation with brain expression ##############################################
##############################################################################################################################

counts = pd.read_csv(output_autismOR_tsv, sep = "\t", header = 0, dtype = {'NC_cases': 'int', 'autism-OR': 'float', 'gene': "str", 'p_CI95': "float"})

# Set infinite autism-OR values to maximum observed for all genes
print (">>>>>>>>>>> Set infinite autism-OR values to maximum observed for all genes...")
ymax = max(counts["autism-OR"][counts["autism-OR"] < float("inf")])
counts.loc[counts["autism-OR"] == float("inf"), "autism-OR"] = ymax

# Select columns in genes.tsv file containing brain expression values
print (">>>>>>>>>>> Select columns in genes.tsv file containing brain expression values...")
cols = [r for r in genes.columns if (("avg_P" in r) & ("R" in r) & ("." not in r) & ("P1R4" not in r)) | ("gene_symbol" in r)]
genes = genes[cols]
genes.columns = ["gene", 'avg_P1R1', 'avg_P1R2', 'avg_P1R3', 'avg_P2R1', 'avg_P2R2', 'avg_P2R3', 'avg_P2R4', 'avg_P3R1', 'avg_P3R2', 'avg_P3R3', 'avg_P3R4', 'avg_P4R1', 'avg_P4R2', 'avg_P4R3', 'avg_P4R4', 'avg_P5R1', 'avg_P5R2', 'avg_P5R3', 'avg_P5R4', 'avg_P6R1', 'avg_P6R2', 'avg_P6R3', 'avg_P6R4', 'avg_P7R1', 'avg_P7R2', 'avg_P7R3', 'avg_P7R4', 'avg_P8R1', 'avg_P8R2', 'avg_P8R3', 'avg_P8R4']
cols = ['avg_P1R1', 'avg_P1R2', 'avg_P1R3', 'avg_P2R1', 'avg_P2R2', 'avg_P2R3', 'avg_P2R4', 'avg_P3R1', 'avg_P3R2', 'avg_P3R3', 'avg_P3R4', 'avg_P4R1', 'avg_P4R2', 'avg_P4R3', 'avg_P4R4', 'avg_P5R1', 'avg_P5R2', 'avg_P5R3', 'avg_P5R4', 'avg_P6R1', 'avg_P6R2', 'avg_P6R3', 'avg_P6R4', 'avg_P7R1', 'avg_P7R2', 'avg_P7R3', 'avg_P7R4', 'avg_P8R1', 'avg_P8R2', 'avg_P8R3', 'avg_P8R4']
counts = pd.merge(counts, genes, on = "gene")

# Z-score brain expression values by region/developmental period
print (">>>>>>>>>>> Z-score brain expression values by region/developmental period...")
scaler = StandardScaler()
for col in cols:
	counts[col + "_zscored"] = scaler.fit_transform(counts[[col]])

# Calculating correlation
print (">>>>>>>>>>> Calculating correlation...")
correlation = pd.DataFrame()
for P in range(1, 9):
	for R in range(1, 5):
		if P == 1 and R == 4:
			continue
		PR = "avg_P" + str(P) + "R" + str(R)
		counts[PR] = counts[PR].astype(float)
		counts_tmp = counts[(counts[PR].isnull() == False) & (counts[PR] > 0) & (counts["autism-OR"].isnull() == False)].copy()
		test = st.pearsonr(log10(list(counts_tmp["autism-OR"])), log10(list(counts_tmp[PR])))
		chunk = pd.concat([pd.Series("P" + str(P) + "R" + str(R)), pd.Series(P), pd.Series(R), pd.Series(test[0]), pd.Series(test[1])], axis = 1)
		correlation = pd.concat([correlation, chunk])
correlation.columns = ["PR", "P", "R", "coef", "pval"]
correlation["pval_adj"] = sm.multitest.multipletests(list(correlation["pval"]), method = "fdr_bh", alpha = 0.05)[1]
correlation.to_csv(output_brainCorr_tsv, sep = "\t", header = True, index = False)

# Plotting heatmap of correlations
print (">>>>>>>>>>> Plotting heatmap of correlations...")
correlation = correlation.round({'pval_adj' : 5})
plot = ggplot(correlation, aes(x = 'factor(P)', y = 'factor(R)', fill = 'coef', label = 'pval_adj')) + geom_tile(color='white',size=.1) + geom_label(aes(label = 'pval_adj'), fill = "white") + theme_matplotlib(rc={'pdf.fonttype':42})
plot.save(output_brainCorr_pdf, height = 8, width = 8)
