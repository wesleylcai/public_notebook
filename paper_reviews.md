# Table of Contents
[Meta-analysis of tumor- and T cell-intrinsic mechanisms of sensitization to checkpoint inhibition](#meta-analysis-of-tumor--and-T-cell-intrinsic-mechanisms-of-sensitization-to-checkpoint-inhibition)

[A review of author name disambiguation techniques for the PubMed bibliographic database](#a-review-of-author-name-disambiguation-techniques-for-the-pubmed-bibliographic-database)

[Statistics versus machine learning](#statistics-versus-machine-learning)
<!---toc--->



# Meta-analysis of tumor- and T cell-intrinsic mechanisms of sensitization to checkpoint inhibition
PMID: [33508232](https://pubmed.ncbi.nlm.nih.gov/33508232/)

First and last authors: Kevin Litchfield, Charles Swanton
##### Main points
1. CPI1000+: 1008 CPI-treated (CTLA-4, PD-1, PD-L1)tumors from 12 individual cohorts from 7 tumor types (met uorthelial, melanoma, HNSCC, NSCLC, RCC, CRC, BRCA)
2. Benchmarked 55 biomarkers across 723 articles (includes TMB, tobacco mut, UV sig, etc). Z score standardized mutations and expression for comparison.
3. Clonal TMB strongest biomarker (estimated per cell mutation #)
4. All measured biomarkers only account for 60% of the variation, so 40% left undiscovered.
5. XGBoost to derive single score of biomarkers that achieved significance (11). Compared score to FDA-approved TMB biomarker. 3 test cohorts not in CPI1000+. Better AUC across all test cohorts compared to TMB.
6. CXCL9 expression and clonal TMB performs better than TMB alone.
7. COSMIC mutational signatures v2: 1A_aging, 4_tobacco, 7_UV, 10_POLE, 2_13_APOBEC were significant
8. Loss of 9q34 was associated with sensitization - TRAF2 gene is important. Selective pressure comes from other genes on chr9, including CDKN2A
9. CCND1 amplification associated with CPI resistance

# A review of author name disambiguation techniques for the PubMed bibliographic database

[Article link](https://journals.sagepub.com/doi/full/10.1177/0165551519888605?casa_token=kIW_km4OtaoAAAAA%3AzGXblIrEvk8RCOqVCQ_401mD5J0rasgpq0v7RlXetAri640TU994wWUO2eAhzzQldLDYkULB4Or7) 

## Main points

1. Problem definition
    1. Records contain at least title and author list
    2. Email, affiliation, venue, abstract, keywords, MeSH terms
    3. S={s_1, ..., s_N} into set of M(<=N) clusters C={C_1, ..., C_M} so each cluster C_i contains all and only all references to same real-world author
2. Evaluation measures used in literature
    0. TP = total number of pairs correctly assigned to same cluster, TN = total number of pairs correctly assigned to different clusters, FP = total number of pairs incorrectly assigned to same cluster, FN = total number of pairs incorrectly assigned to different clusters. S = TP + TN + FP + FN
    1. Accuracy = (TP + TN)/S
    2. Pairwise precision pp = TP/(TP + FP)
        1. Fraction of pairs in cluster being  co-referent
    3. Pairwise recall pr = TP/(TP + FN)
        1. Fraction of co-referent pairs put into same cluster
    4. Pairwise F1-score pf1 = (2 X pp X pr)/(ppp + pr)
        1. Harmonic mean of pp and pr
        
## Ideas for the future

1. Feature learning to discover which features most prominently identify authors
    1. The tendency for author to include email, institution (presence/absence of information)
    2. How collaborative an author is
    3. Likelihood of author appearing last or second to last

# Statistics versus machine learning

PMID: [30100822](https://pubmed.ncbi.nlm.nih.gov/30100822/)

## Main points


