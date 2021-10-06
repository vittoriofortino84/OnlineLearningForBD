import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from utils import cross_validate

data = pd.read_csv('sw_breast_gdat_2.csv')
data = data.drop('Unnamed: 0', axis=1, errors='ignore')

pheno = pd.read_csv('sw_breast_pheno.csv')
pheno = pheno.drop('Unnamed: 0', axis=1, errors='ignore')

pam50 = data['pam50']
data = data.drop('pam50', axis=1, errors='ignore')

class2idx = {
    'LumA':0,
    'LumB':1,
    'Her2':2,
    'Basal':3,
    'Normal':4
}

idx2class = {v: k for k, v in class2idx.items()}

# replacing labels
pam50.replace(class2idx, inplace=True)

scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

pam50sig = ["ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6","CDH3","CENPF","CEP55","CXXC5",
            "EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT",
            "MDM2","MELK","MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","PGR","PHGDH","PTTG1","RRM2","SFRP1",
            "SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T", "ORC6L", "KNTC2", "CDCA1"]
# The list contains also 'ORC6L', 'KNTC2', 'CDCA1'

all_feats = ['DNAJC12', 'ABCC8', 'CKAP2L', 'CDC25A', 'CDCA3', 'KLK6', 'SOX11', 'C5orf30', 'FOXA1', 'NUDT12', 'SKA3',
             'CLSPN', 'CENPI', 'FOXC1', 'HAPLN3', 'KRT14', 'DEPDC1', 'SKA1', 'TPX2', 'MKI67', 'SERPINA11', 'ARSG',
             'MPP6', 'ASPM', 'CMBL', 'ANKRA2', 'CDCA2', 'LRRC48', 'GATA3', 'TCEAL1', 'NDC80', 'ZNF695', 'MICALL1',
             'PLEKHG1', 'APH1B', 'RRM2', 'CEP55', 'MAPT', 'YBX1', 'DIAPH3', 'ROPN1B', 'KIF18A', 'KRT16', 'GABRP',
             'MCM10', 'BUB1B', 'ACADSB', 'ANLN', 'CENPN', 'KRT5', 'TROAP', 'AURKA', 'FAM72B', 'EXO1', 'KIF20A',
             'BCL11A', 'UBE2T', 'CENPO', 'TTC8', 'NEK2', 'RUNDC1', 'NUF2', 'MLPH', 'FANCA', 'CDK1', 'NCAPD2', 'CDT1',
             'FAM174A', 'C10orf32', 'GINS1', 'NCAPG', 'CCDC125', 'SPTBN4', 'FAM47E', 'TUBA1C', 'SUV39H2', 'TACC3',
             'CCNA2', 'CLSTN2', 'SLC39A6', 'CRYAB', 'TBC1D9', 'CPLX1', 'GPM6B', 'BUB1', 'CCNB2', 'SPC25', 'DLGAP5',
             'CPEB2', 'ERBB4', 'CDC45', 'CDCA5', 'STIL', 'UGT8', 'UBE2C', 'SFRP1', 'CT62', 'REEP6', 'FAM171A1',
             'NUSAP1', 'ADCY9', 'HMMR', 'PTTG1', 'TTC36', 'LRRC56', 'ANXA9', 'SUSD3', 'KLK5', 'CA12', 'PLK1', 'ROPN1',
             'CCNE1', 'PRR15', 'E2F1', 'SPAG5', 'NCAPH', 'IL6ST', 'RHOB', 'XBP1', 'PARD6B', 'AGR3', 'SCUBE2', 'FSIP1',
             'LRRC46', 'PRR11', 'TRIM29', 'TTLL4', 'CCDC96', 'SGOL1', 'FERMT1', 'CDC20', 'ABAT', 'MYBL2', 'ANKRD42',
             'ERGIC1', 'FOXM1', 'KIF18B', 'TTK', 'MELK', 'LEPREL1', 'AGR2', 'POLQ', 'FAM64A', 'MAGED2', 'PDSS1',
             'LEMD1', 'BIRC5', 'PGAP3', 'GTSE1', 'UBXN10', 'FZD9', 'TLE3', 'CENPW', 'NAT1', 'AURKB', 'IFRD1', 'PTPRT',
             'CELSR1', 'C20orf26', 'WWP1', 'KIFC1', 'C6orf211', 'WDR19', 'ESPL1', 'UBE2S', 'PSAT1', 'CENPA', 'RARA',
             'BLM', 'KCMF1', 'CACNA1D', 'RAD51', 'SLC7A8', 'E2F2', 'KCNJ11', 'PGR', 'EZH2', 'RGMA', 'LRTOMT', 'TENC1',
             'SCN4B', 'CDKN3', 'DYNLRB2', 'LMX1B', 'PGK1', 'IRX1', 'FAM83D', 'CHEK1', 'MYB', 'ZNF703', 'ESR1',
             'C9orf116', 'DEPDC1B', 'ZNF552', 'STAC', 'B3GNT5', 'SPDEF', 'SPARCL1', 'DNAL1', 'DEGS2', 'CCNB1',
             'C7orf63', 'KDM4B', 'TCF19', 'KRT17', 'TRIP13', 'BCL2', 'PRC1', 'KIAA1467', 'RERG', 'KIF14', 'CDCA7',
             'MIA', 'SLC22A5', 'KRT6B', 'RAD54L', 'ZMYND10', 'SYTL4', 'GPR160', 'KIF11', 'BBS1', 'RGS22', 'ERBB2',
             'KIF4A', 'VGLL1', 'GSG2', 'AFF3', 'RABEP1', 'TFF1', 'KIF15', 'CDC6', 'SOX10', 'KIF2C', 'TCEAL4', 'MTHFD1L',
             'SHCBP1', 'MAD2L1', 'HJURP', 'IGF1R', 'THSD4', 'CKS1B', 'CDCA8', 'LONRF2', 'PPP1R14C', 'RAD51AP1',
             'SLC7A13', 'APOBEC3B']

selected_data = data.loc[:, [c in all_feats for c in data.columns.values.tolist()]]
pam_50_data = data.loc[:, [c in pam50sig for c in data.columns.values.tolist()]]

y_cox = []
for index, row in pheno.iterrows():
    y_cox.append((row['OverallSurv'], row['SurvDays']))
y_cox = np.array(y_cox, dtype=[('event', bool), ('time', int)])

estimator_selected = CoxPHSurvivalAnalysis().fit(selected_data, y_cox)
estimator_pam50 = CoxPHSurvivalAnalysis().fit(pam_50_data, y_cox)

for a in [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]:
    print("alpha: " + str(a))
    selected_score = cross_validate(x=selected_data, y=y_cox, n_folds=10, alpha=a)
    print("selected score: " + str(selected_score))
    pam50_score = cross_validate(x=pam_50_data, y=y_cox, n_folds=10, alpha=a)
    print("pam50 score: " + str(pam50_score))

