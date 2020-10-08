"""
Lists of features

Every list of features is a numpy array
"""

# External imports
import numpy as np

# Internal imports
import dataset

# BRCA features
brca = dataset.brca.columns.to_numpy() if dataset.brca else np.array([])

# TCGA features
tcga = dataset.tcga.columns.to_numpy() if dataset.tcga else np.array([])

# Article genes 
# TODO: add reference
good_genes = np.array(['ABCA1', 'ADAM10', 'ADAM17', 'ADAMTS1', 'ADIPOQ', 'AOPEP', 'APAF1', 'APC', 'ARNTL', 'ASAP2', 'ASF1B', 'ASPM', 'ATAD2', 'ATF1', 'ATF2', 'ATF3', 'ATF6', 'ATM', 'ATP2A2', 'ATP2C1', 'AURKA', 'AURKB', 'BACE1', 'BARD1', 'BAX', 'BCL2', 'BCL2L11', 'BDP1', 'BIRC2', 'BIRC5', 'BRCA1', 'BRIP1', 'BTAF1', 'BTK', 'CAD', 'CAV1', 'CBX5', 'CBX7', 'CCL5', 'CCNA2', 'CCNB1', 'CCND2', 'CCNE1', 'CCR1', 'CD163', 'CD2AP', 'CD36', 'CD40', 'CD6', 'CD86', 'CD8A', 'CDC25A', 'CDC25B', 'CDC5L', 'CDC6', 'CDH5', 'CDK1', 'CEACAM1', 'CEBPB', 'CEBPZ', 'CFLAR', 'CHD8', 'CHEK1', 'CIITA', 'CLOCK', 'COL27A1', 'CREB1', 'CREBBP', 'CRY1', 'CSF1', 'CSF2RA', 'CTCF', 'CTNNB1', 'CTSK', 'CTSS', 'CXCL10', 'CXCL12', 'CYBB', 'DEK', 'DLG1', 'DNAJC3', 'DNMT1', 'DR1', 'DYRK1A', 'E2F1', 'E2F6', 'E2F8', 'EDN1', 'EGFR', 'EGR1', 'EGR2', 'EGR3', 'ELF1', 'ELK4', 'ENG', 'EP300', 'EPAS1', 'ERAP1', 'ERCC2', 'ERG', 'ESR1', 'ETF1', 'ETS1', 'ETS2', 'ETV3', 'ETV5', 'EVL', 'EZH2', 'F13A1', 'FABP4', 'FAS', 'FCGR1A', 'FLI1', 'FLT1', 'FMR1', 'FOS', 'FOXA1', 'FOXC1', 'FOXF2', 'FOXM1', 'FOXO1', 'FOXO3', 'FOXP3', 'FST', 'GABPA', 'GATA3', 'GPNMB', 'GSK3B', 'GSPT1', 'GTF2I', 'HBP1', 'HCK', 'HDAC4', 'HECA', 'HIC1', 'HLA-A', 'HLA-DMA', 'HLA-DMB', 'HLA-DOA', 'HLA-DOB', 'HLA-DPB1', 'HLA-DQB1', 'HLA-DRA', 'HLA-DRB1', 'HLA-DRB5', 'HNRNPA2B1', 'HOXA5', 'HOXB3', 'HOXC4', 'HTT', 'ICAM1', 'ICAM2', 'ID4', 'IFI16', 'IFIT3', 'IFNAR1', 'IGF2', 'IL18', 'INO80', 'IRF1', 'IRF3', 'IRF7', 'IRF8', 'IRF9', 'ITGA2', 'ITGA5', 'ITGAM', 'ITGAV', 'ITGAX', 'ITGB2', 'JAG1', 'JUN', 'JUND', 'KAT2B', 'KIF2C', 'KLF2', 'KLF4', 'KLF6', 'KRT15', 'LCK', 'LDLR', 'LIF', 'LIPE', 'LMO2', 'LMO4', 'LRIG2', 'LYL1', 'MAD2L1', 'MAF', 'MAFB', 'MAPK14', 'MAT2A', 'MAT2B', 'MCL1', 'MCM5', 'MDC1', 'MEF2A', 'MEOX2', 'MEST', 'MITF', 'MLH1', 'MMP13', 'MMP2', 'MNDA', 'MSC', 'MSH2', 'MSH6', 'MSR1', 'MTF1', 'MYBL2', 'NCF2', 'NCOA2', 'NF1', 'NFKB1', 'NFYA', 'NOTCH4', 'NR2C2', 'NR3C1', 'NUPR1', 'PCNA', 'PCYT1A', 'PIK3CA', 'PLAU', 'PLD1', 'PLK1', 'PLRG1', 'PLSCR1', 'PML', 'POU2F1', 'PPARG', 'PRDM1', 'PSEN1', 'PSMB10', 'PSMB9', 'PTEN', 'PTN', 'PTPRN2', 'PTTG1', 'RACGAP1', 'RAD51', 'RB1', 'RBL1', 'RBL2', 'RBPJ', 'REST', 'RNF111', 'RORA', 'RPS6KA3', 'RRM2', 'RRM2B', 'RUNX2', 'RUNX3', 'SERPINF1', 'SGPL1', 'SIRT1', 'SLC31A1', 'SMAD2', 'SMAD4', 'SMARCA4', 'SNAI2', 'SOCS1', 'SOX7', 'SOX9', 'SP1', 'SP2', 'SP3', 'SP4', 'SPI1', 'SPRY1', 'SPRY2', 'SREBF1', 'SREBF2', 'SRPX', 'STAT1', 'STAT2', 'STAT3', 'TAF1', 'TAP1', 'TAP2', 'TCF19', 'TCF4', 'TCF7L2', 'TGFBR2', 'TLR3', 'TLR4', 'TMPO', 'TNFRSF10B', 'TNFSF12', 'TNFSF13B', 'TOP2A', 'TRIM22', 'TWIST1', 'TWIST2', 'TXNIP', 'TYMS', 'UBA7', 'UHRF1', 'USF2', 'USP7', 'UTRN', 'VCAN', 'VIM', 'VWF', 'WAS', 'WRN', 'XAF1', 'XIAP', 'XPC', 'YY1', 'ZEB1', 'ZEB2', 'ZNF143'])

# TODO: add description
super_good_genes = np.array(['CDK1', 'FOXM1', 'LRIG2', 'MSH2', 'PLK1', 'RACGAP1', 'RRM2', 'TMPO'])

# TODO: add description
oncotype_dx = np.array(['GRB7', 'ERBB2', 'ESR1', 'PGR', 'BCL2', 'SCUBE2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'CTSL2', 'MMP11', 'CD68', 'GSTM1', 'BAG1'])

# Article genes
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2999994/
mammaprint = np.array(['BBC3', 'EGLN1', 'TGFB3', 'ESM1', 'IGFBP5', 'FGF18', 'SCUBE2', 'TGFB3', 'WISP1', 'FLT1', 'HRASLS', 'STK32B', 'RASSF7', 'DCK', 'MELK', 'EXT1', 'GNAZ', 'EBF4', 'MTDH', 'PITRM1', 'QSCN6L1', 'CCNE2', 'ECT2', 'CENPA', 'LIN9', 'KNTC2', 'MCM6', 'NUSAP1', 'ORC6L', 'TSPYL5', 'RUNDC1', 'PRC1', 'RFC4', 'RECQL5', 'CDCA7', 'DTL', 'COL4A2', 'GPR180', 'MMP9', 'GPR126', 'RTN4RL1', 'DIAPH3', 'CDC42BPA', 'PALM2', 'ALDH4A1', 'AYTL2', 'OXCT1', 'PECI', 'GMPS', 'GSTM3', 'SLC2A3', 'FLT1', 'FGF18', 'COL4A2', 'GPR180', 'EGLN1', 'MMP9', 'LOC100288906', 'C9orf30', 'ZNF533', 'C16orf61', 'SERF1A', 'C20orf46', 'LOC730018', 'LOC100131053', 'AA555029_RC', 'LGP2', 'NMU', 'UCHL5', 'JHDM1D', 'AP2B1', 'MS4 A7', 'RAB6B'])
