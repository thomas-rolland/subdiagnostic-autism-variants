#!/bin/python3
# Author: Thomas Rolland	jerikoo75@gmail.com
# Date: 03/31/2022

import pandas as pd
from numpy import mean
from pyliftover import LiftOver

#=========================================================================
# variant-pext.py
# Annotate pext score for each variant of tabular file
# NOTE: This scripts is based on Hg38 version of the human reference genome
#
# Input files and required columns (and corresponding variable format or accepted values):
# - variants.tsv = required columns 'chromosome' (str), 'position' (int), 'ref' (str), 'alt' (str), 'consequence' (str), 'gene' (str)
# - all.baselevel.pext.tsv = The base-level pext score from the gnomAD website (GRCh37)
#
# Output files
# - variants-pext.tsv = Variants with corresponding GRCh37 position and annotated pext
#=========================================================================


##############################################################################################################################
######################################################### Parameters #########################################################
##############################################################################################################################

input_file = "variants.tsv"
input_pext = "all.baselevel.pext.tsv"
output_file = "variants-pext.tsv"


####################################################################################################################################
########################################## Loading liftover chain because pext is on hg19 ##########################################
####################################################################################################################################

print (">>>>>>>>>>> Loading liftover chain because pext is on hg19 ...")
lo = LiftOver('hg38', 'hg19')


#####################################################################################################################################################################
########################################## Loading hg38-based variants into dataframe and create column with hg19 position ##########################################
#####################################################################################################################################################################

print (">>>>>>>>>>> Loading hg38-based variants into dataframe and create column with hg19 position...")

VCF = pd.read_csv(input_file, sep = "\t", dtype = str)
VCF["hg19_position"] = "NA"
for i in range(0, VCF.shape[0]):
	position = int(VCF["position"][i])
	# Specify closest exon for splice donor/acceptor variants
	if (VCF["consequence"][i] == "splice_donor_variant"):
		if VCF["strand"][i] == "1":
			position = position - 3
		else:
			position = position + 3
	if (VCF["consequence"][i] == "splice_acceptor_variant"):
		if VCF["strand"][i] == "1":
			position = position + 3
		else:
			position = position - 3
	# Get hg19-based variant position
	hg19_position = lo.convert_coordinate('chr' + VCF["chromosome"][i], position)[0][1]
	VCF["hg19_position"][i] = str(hg19_position)

# Format dataframe
vcf = pd.DataFrame(pd.concat([VCF["chromosome"].reset_index(drop=True), VCF["hg19_position"].reset_index(drop=True), VCF["ref"].reset_index(drop=True), VCF["alt"].reset_index(drop=True), VCF["consequence"].reset_index(drop=True), VCF["gene"].reset_index(drop=True)], axis = 1))
vcf.columns = "chromosome position ref alt consequence gene".split()
vcf["chrpos"] = vcf["chromosome"] + ":" + vcf["position"]
vcf = vcf.drop_duplicates()


################################################################################################################################
########################################## Loading pext score for variants identified ##########################################
################################################################################################################################

print (">>>>>>>>>>> Loading pext score for variants identified...")

brain = ["Brain_FrontalCortex_BA9_", "Brain_Hippocampus", "Brain_Nucleusaccumbens_basalganglia_", "Brain_Spinalcord_cervicalc_1_", "Brain_CerebellarHemisphere", "Brain_Cerebellum", "Brain_Cortex", "Brain_Substantianigra", "Brain_Anteriorcingulatecortex_BA24_", "Brain_Putamen_basalganglia_", "Brain_Caudate_basalganglia_", "Brain_Amygdala"]
iter_csv = pd.read_csv(input_pext, iterator = True, chunksize = 1000000, sep = "\t", dtype = str)
variants_with_pext = pd.DataFrame()
for chunk in iter_csv:
	# Getting position and matching to identified variants
	position = chunk["locus"].str.split(':', n = 2, expand = True)
	chunk["chr"] = position[0]
	chunk["position"] = position[1]
	chunk["chrpos"] = chunk["chr"] + ":" + chunk["position"]
	chunk = chunk[chunk["chrpos"].isin(vcf["chrpos"])].reset_index(drop=True)
	if (chunk.shape[0] == 0):
		continue

	# Measuring average pext over brain tissues
	chunk["pext"] = "NA"
	for i in range(0, chunk.shape[0]):
		mn = []
		for j in brain:
			if (pd.isna(chunk[j][i])):
				continue
			value = chunk[j][i]
			if (value != "NaN"):
				mn.append(float(value))
		if len(mn) > 0:
			chunk["pext"][i] = max(mn)

	# Recording variants found with their corresponding pext
	chunk = pd.DataFrame(pd.concat([chunk["chr"].reset_index(drop = True), chunk["position"].reset_index(drop = True), chunk["pext"].reset_index(drop = True), chunk["chrpos"].reset_index(drop = True)], axis = 1))
	variants_with_pext = pd.concat([variants_with_pext, chunk])
	variants_with_pext = variants_with_pext.drop_duplicates()


################################################################################################################################
########################################## Writing out variants identified with corresponding pext ##########################################
################################################################################################################################

print (">>>>>>>>>>> Writing out variants identified with corresponding pext...")

VCF["chrpos_hg19"] = VCF["chromosome"] + ":" + VCF["hg19_position"]
VCF["pext"] = "NA"
for i in range(0, VCF.shape[0]):
	if (len(variants_with_pext["pext"][variants_with_pext["chrpos"].isin([VCF["chrpos_hg19"][i]])]) > 0):
		VCF["pext"][i] = variants_with_pext["pext"][variants_with_pext["chrpos"].isin([VCF["chrpos_hg19"][i]])].reset_index(drop=True)[0]
VCF.to_csv(output_file, sep = "\t", header = True, index = False)
