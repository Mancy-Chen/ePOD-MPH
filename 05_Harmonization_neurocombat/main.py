#Mancy Chen 24/05/2023
################################################################################################################
# Merge data: Radiomics per ROI
import pandas as pd
import numpy as np
import glob
import os
from combat.pycombat import pycombat

# Specify the path to the directory containing the CSV files
input_path = 'enter_the_input_path'
output_path = "enter_the_output_path'"
# Get a list of all CSV file paths in the directory
csv_files = glob.glob(input_path + '*.csv')

# Create an empty list to store the dataframes
dfs = []
renamed_columns_list = []

# Iterate over the CSV files and load them into dataframes
for file in csv_files:
    # Load the CSV file into a dataframe, adjust the column range as needed
    # df = pd.read_csv(file, usecols=range(25, 57))  # FastSurfer
    #df = pd.read_csv(file, usecols=range(26, 58))  # SynthSeg
    # df = pd.read_csv(file, usecols= [40, 41, 44, 45, 46, 47, 50, 51, 52, 53, 54, 56]) # DTI TBSS
    df = pd.read_csv(file, usecols=[40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57])  # DTI TBSS new
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    df_columns = df.columns.tolist()  # Convert the columns to a list
    for col in df_columns:
        # Rename the columns with the original header plus file name
        renamed_columns = file_name.split('_')[0] +'_'+ file_name.split('_')[1] + '_' + col.split('_')[1] + '_' + col.split('_')[2]
        renamed_columns_list.append(renamed_columns)
    # Append the dataframe to the list
    dfs.append(df)

# Merge the dataframes based on a common column (e.g., patient ID)
merged_df = pd.concat(dfs, axis=1)

# Assign the new column names to the merged dataframe
merged_df.columns = renamed_columns_list

# Save the merged dataframe to a new CSV file
output_file = output_path + 'DTI_Radiomics_per_ROIs.csv'
merged_df.to_csv(output_file, index=False)

################################################################################################################
# Neurocombat pandas example
print(__doc__)
from neurocombat_sklearn import CombatModel
sys.path.append('/scratch/mchen/miniconda3/lib/python3.10/site-packages')
from neuroCombat import neuroCombat
import pandas as pd


# Loading data
data = pd.read_csv('enter_the_input_X_path.csv') # Independent variable
covars = pd.read_csv('enter_your_covariate_path.csv')

# Save the Header
data_columns = data.columns.tolist()

# Creating model
model = CombatModel()

# Fitting model
# make sure that your inputs are 2D, e.g. shape [n_samples, n_discrete_covariates]
data_combat = model.fit(data = data, sites = covars[['Group']], discrete_covariates=None,
                                    # continuous_covariates = covars[['DTI Motion Score']]
                                  )

# Harmonize data
# could be performed together with fitt by using .fit_transform method
data_combat = model.transform(data=data, sites=covars[['Group']], discrete_covariates=None,
                              # continuous_covariates=covars[['DTI Motion Score']]
                              )

# Save the Harmonized dataframe to a new CSV file
data_combat = pd.DataFrame(data_combat)
data_combat.columns = data_columns
print(data_combat)
output_file = 'enter_your_output_path.csv'
data_combat.to_csv(output_file, index=False)

##############################################################################################################
# Best bayesian estimate
import best
#from best.bayesian_estimator import BEST
from best import analyze_two, plot_all
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# T1
Seg = 'FastSurfer'  # 'FastSurfer' or 'SynthSeg'
data_route = '.../T1/' + Seg + '/T1_TP1_' + Seg + '_Voxel_Volume_Merged_Cerebellum.csv'

# DTI
data_route = '.../DTI/DTI_TP1_Mean_FA.csv'

# T1 ComBat
# data_route = '.../Output/99_FastSurfer_ComBat.csv'
data_route = '.../46_47_CGI6_tier1_BEST/X3.csv'

# DTI ComBat
data_route = '.../47_DTI_ComBat.csv'

# Load data
data = pd.read_csv(data_route)
MPH_CGI6_R = data[(data['Medication'] == 1) & (data['CGI6'] == 1)]
MPH_CGI6_NR = data[(data['Medication'] == 1) & (data['CGI6'] == 0)]
# Placebo_CGI6_R = data[(data['Medication'] == 2) & (data['CGI6'] == 1)]
# Placebo_CGI6_NR = data[(data['Medication'] == 2) & (data['CGI6'] == 0)]
MPH_CGI7_R = data[(data['Medication'] == 1) & (data['CGI7'] == 1)]
MPH_CGI7_NR = data[(data['Medication'] == 1) & (data['CGI7'] == 0)]
# Placebo_CGI7_R = data[(data['Medication'] == 2) & (data['CGI7'] == 1)]
# Placebo_CGI7_NR = data[(data['Medication'] == 2) & (data['CGI7'] == 0)]



# T1
for i in ['Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus','Left-Accumbens-area',  \
          'Right-Caudate', 'Right-Putamen','Right-Pallidum', 'Right-Hippocampus', 'Right-Accumbens-area']:
    Group1 = MPH_CGI6_R[i]
    Group2 = MPH_CGI6_NR[i]
    best_out = best.analyze_two(Group1, Group2)
    fig = best.plot_all(best_out)
    fig_name = 'T1_' + Seg + '_MPH_CGI6_' + i + '.png'
    fig.savefig(fig_name)
    Group3 = Placebo_CGI6_R[i]
    Group4 = Placebo_CGI6_NR[i]
    best_out = best.analyze_two(Group3, Group4)
    fig = best.plot_all(best_out)
    fig_name = 'T1_' + Seg + '_Placebo_CGI6_' + i + '.png'
    fig.savefig(fig_name)
    Group5 = MPH_CGI7_R[i]
    Group6 = MPH_CGI7_NR[i]
    best_out = best.analyze_two(Group5, Group6)
    fig = best.plot_all(best_out)
    fig_name = 'T1_' + Seg + '_MPH_CGI7_' + i + '.png'
    fig.savefig(fig_name)
    Group7 = Placebo_CGI7_R[i]
    Group8 = Placebo_CGI7_NR[i]
    best_out = best.analyze_two(Group7, Group8)
    fig = best.plot_all(best_out)
    fig_name = 'T1_' + Seg + '_Placebo_CGI7_' + i + '.png'
    fig.savefig(fig_name)

# DTI
for i in ['ATR_Left', 'ATR_Right', 'Forceps_Major', 'Whole_Brain']:
    Group1 = MPH_CGI6_R[i]
    Group2 = MPH_CGI6_NR[i]
    best_out = best.analyze_two(Group1, Group2)
    fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
    fig_name = 'DTI_MPH_CGI6_' + i + '.png'
    fig.savefig(fig_name)
    # Group3 = Placebo_CGI6_R[i]
    # Group4 = Placebo_CGI6_NR[i]
    # best_out = best.analyze_two(Group3, Group4)
    # fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
    # fig_name = 'DTI_Placebo_CGI6_' + i + '.png'
    # fig.savefig(fig_name)
    Group5 = MPH_CGI7_R[i]
    Group6 = MPH_CGI7_NR[i]
    best_out = best.analyze_two(Group5, Group6)
    fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
    fig_name = 'DTI_MPH_CGI7_' + i + '.png'
    fig.savefig(fig_name)
    # Group7 = Placebo_CGI7_R[i]
    # Group8 = Placebo_CGI7_NR[i]
    # best_out = best.analyze_two(Group7, Group8)
    # fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
    # fig_name = 'DTI_Placebo_CGI7_' + i + '.png'
    # fig.savefig(fig_name)

# T1 ComBat
ROI = ['Left_Caudate_shape_VoxelVolume', 'Left_Putamen_shape_VoxelVolume', 'Left_Pallidum_shape_VoxelVolume', \
          'Left_Hippocampus_shape_VoxelVolume','Left_Accumbens_Area_shape_VoxelVolume',  \
          'Right_Caudate_shape_VoxelVolume', 'Right_Putamen_shape_VoxelVolume','Right_Pallidum_shape_VoxelVolume', \
          'Right_Hippocampus_shape_VoxelVolume', 'Right_Accumbens_Area_shape_VoxelVolume']
Group_name = ['MPH_CGI6', 'Placebo_CGI6', 'MPH_CGI7', 'Placebo_CGI7']
for j in Group_name:
    for i in ROI:
        Group1 = globals()[f"{j}_R"][i]
        Group2 = globals()[f"{j}_NR"][i]
        best_out = best.analyze_two(Group1, Group2)
        fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
        directory = '.../T1_BEST/SynthSeg/' + j + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        #fig_name = 'T1_ComBat_FastSurfer_' + j + '_' + i.split('_')[0] + '_' + i.split('_')[1] + '.png'
        fig_name = 'T1_ComBat_SynthSeg_' + j + '_' + i.split('_')[0] + '_' + i.split('_')[1] + '.png'
        fig.savefig(os.path.join(directory, fig_name))

# Save the voxel volume of SynthSeg
selected_columns = [
    'Left_Caudate_shape_VoxelVolume',
    'Left_Putamen_shape_VoxelVolume',
    'Left_Pallidum_shape_VoxelVolume',
    'Left_Hippocampus_shape_VoxelVolume',
    'Left_Accumbens_Area_shape_VoxelVolume',
    'Right_Caudate_shape_VoxelVolume',
    'Right_Putamen_shape_VoxelVolume',
    'Right_Pallidum_shape_VoxelVolume',
    'Right_Hippocampus_shape_VoxelVolume',
    'Right_Accumbens_Area_shape_VoxelVolume']
selected_data = data[selected_columns]  # Assuming 'data' is your DataFrame containing the desired columns
selected_data.to_csv('SynthSeg_Voxel_Volume_ComBat.csv', index=False)

# DTI ComBat
for i in ['ATR_Left', 'ATR_Right', 'Forceps_Major', 'Whole_Brain']:
    Group1 = MPH_CGI6_R[i]
    Group2 = MPH_CGI6_NR[i]
    best_out = best.analyze_two(Group1, Group2)
    fig = best.plot_all_two(best_out,group1_name='Respondents',group2_name='Non-Respondents')
    fig_name = 'DTI_MPH_CGI6_' + i + '.png'
    directory = '.../New_DTI_BEST/' + i + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, fig_name))
    Group5 = MPH_CGI7_R[i]
    Group6 = MPH_CGI7_NR[i]
    best_out2 = best.analyze_two(Group5, Group6)
    fig2 = best.plot_all_two(best_out2,group1_name='Respondents',group2_name='Non-Respondents')
    fig_name2 = 'DTI_MPH_CGI7_' + i + '.png'
    fig2.savefig(os.path.join(directory, fig_name2))

# BEST per column feature
data_route = 'enter_your_X.csv'
data = pd.read_csv(data_route)
y = pd.read_csv('enter_your_y.csv', header=None)
y = y.squeeze()
# Reset the index if necessary to ensure alignment
y.reset_index(drop=True, inplace=True)
data.reset_index(drop=True, inplace=True)
# Check alignment just to be sure
assert y.index.equals(data.index), "The indices of y and data do not match."
# Specify the directory to save the plots
directory = 'enter_your_output_folder_path'
# Make sure the directory exists, if not create it
os.makedirs(directory, exist_ok=True)
# Loop through each column in data and perform the analysis
for column_name in data.columns:
    # Select the column corresponding to the current feature
    feature_data = data[column_name]
    # Split the feature data into responders and non-responders based on y
    responders = feature_data[y == 1]
    non_responders = feature_data[y == 0]
    # Perform BEST analysis
    best_analysis = best.analyze_two(non_responders, responders)
    # Plot the analysis
    best.plot_all(best_analysis)
    # Save the plot with the feature name as the filename
    plt.savefig(os.path.join(directory, f'{column_name}.png'))
    plt.close()