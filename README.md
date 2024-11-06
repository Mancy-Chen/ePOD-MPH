Code and documentation behind the publication: 

# Prediction of methylphenidate treatment response for ADHD using conventional and radiomics T1 and DTI features: secondary analysis of a randomized clinical trial
*Mingshi Chen; Zarah van der Pal; Maarten G. Poirot; Anouk Schrantee; Marco Bottelier; Sandra J. J. Kooij; Henk A. Marquering; Liesbeth Reneman; Matthan W.A. Caan*

*Submitting 06 November 2024*

## Abstract

**Background:** Attention-Deficit/Hyperactivity Disorder (ADHD) is commonly treated with methylphenidate (MPH). Although highly effective, MPH treatment still has a relatively high non-response rate of around 30%, highlighting the need for a better understanding of treatment response. Radiomics of T1-weighted images and Diffusion Tensor Imaging (DTI) combined with machine learning approaches could offer a novel method for assessing MPH treatment response.

**Purpose:** To evaluate the accuracy of both conventional and radiomics approaches in predicting treatment response based on baseline T1 and DTI data in stimulant-naive ADHD participants. 

**Methods:** We performed a secondary analysis of a randomized clinical trial (ePOD-MPH), involving 47 stimulant-naive ADHD participants (23 boys aged 11.4 ± 0.4 years, 24 men aged 28.1 ± 4.3 years) who underwent 16 weeks of treatment with MPH. Baseline T1-weighted and DTI MRI scans were acquired. Treatment response was assessed at 8 weeks (during treatment) and one week after cessation of 16-week treatment (post-treatment) using the Clinical Global Impressions - Improvement scale as our primary outcome. We compared prediction accuracy using a conventional model and a radiomics model. The conventional approach included the volume of bilateral caudate, putamen, pallidum, accumbens, and hippocampus, and for DTI the mean fractional anisotropy (FA) of the entire brain white matter, bilateral Anterior Thalamic Radiation (ATR), and the splenium of the corpus callosum, totaling 14 regional features. For the radiomics approach, 380 features (shape-based and first-order statistics) were extracted from these 14 regions. XGBoost models with nested cross-validation were used and constructed for the total cohort (n = 47), as well as children (n = 23) and adults (n = 24) separately. Exact binomial tests were employed to compare model performance.

**Results:** For the conventional model, balanced accuracy (bAcc) in predicting treatment response during treatment was 63% for the total cohort, 32% for children, and 36% for adults (Area Under the Receiver Operating Characteristic Curve (AUC-ROC): 0.69, 0.33, 0.41 respectively). Radiomics models demonstrated bAcc’s of 68%, 64%, and 64% during treatment (AUC-ROCs of 0.73, 0.62, 0.69 respectively). These predictions were better than chance for both conventional and radiomics models in the total cohort (p = 0.04, p = 0.003 respectively). The radiomics models outperformed the conventional models during treatment in children only (p = 0.02). At post-treatment, performance was markedly reduced. 

**Conclusion:** While conventional and radiomics models performed equally well in predicting clinical improvement across children and adults during treatment, radiomics features offered enhanced structural information beyond conventional region-based volume and FA averages in children. Prediction of symptom improvement one week after treatment cessation was poor, potentially due to the transient effects of stimulant treatment on symptom improvement. 

![Fig 2](https://github.com/user-attachments/assets/dc725a0a-a3b8-45a2-9b3f-b5104d8e3341)

