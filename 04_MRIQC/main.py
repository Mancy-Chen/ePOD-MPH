# Mancy Chen 25/04/2023
# MRIQC
import pandas as pd
import os
import matplotlib.pyplot as plt
list_of_input = ['001', '003', '004', '006', '007', '008', '010', '011', '017', '018', '020',
                 '021', '022', '024', '027', '029', '030', '032', '033', '034', '035', '036',
                 '037', '038', '039', '040', '041', '042', '044', '045', '047', '049', '050',
                 '051', '052', '054', '056', '059', '062', '063', '064', '065', '067', '068',
                 '069', '070', '071', '072', '074', '075', '101', '102', '103', '104', '105',
                 '106', '107', '108', '110', '111', '112', '113', '114', '115', '116', '117',
                 '118', '119', '120', '121', '122', '123', '124', '125', '126', '128', '130',
                 '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143',
                 '144', '145', '146', '147', '148', '149', '150', '152', '153', '155', '156']

#################################################################################################
# Merge all T1 TP1 voxel volume into one file
data_list = []
index_list = []
error_list = []
merged_data = pd.DataFrame()
for i in label_list:
#for i in [7]: # test code
  Label_name = label_name[str(i)]
  file_path_1 = '.../Extract_features_ROI/T1/TP1/Output_FastSurfer/' + \
            Label_name + '_radiomics_features.csv'  # /Output_FastSurfer/ or /Output_SynthSeg/
  output_file_path = ".../XGBoost/Output/T1_TP1_FastSurfer_Voxel_Volume.csv"
  if not os.path.exists(file_path_1):
      error_list.append(i)
      continue  # skip the file if it doesn't exist
  df1 = pd.read_csv(file_path_1, header=0)
  column = df1['original_shape_VoxelVolume']
  merged_data = pd.concat([merged_data, column], axis = 1)
  print('These files fail to merge: ', error_list)
headings = ['Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
          'Left-Caudate', 'Left-Putamen',  'Left-Pallidum',  'Left-Hippocampus',
          'Left-Accumbens-area', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
           'Right-Caudate', 'Right-Putamen',  'Right-Pallidum','Right-Hippocampus',
           'Right-Accumbens-area']
merged_data.columns = headings
# Write the merged data to a CSV file
merged_data.to_csv(output_file_path, index=False)
############################################################################################

# Set the file paths for input files
file_paths = ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"]

# Set the file path for output file
output_file_path = "path/to/output_TP1.csv" #TP1
#output_file_path = "path/to/output_TP2.csv" #TP2

# Initialize an empty list to store the merged data
data_list = []
index_list = []
error_list = []

# Loop through the file paths and extract the data from the second column of the second row in each file
for i in list_of_input:
    file_path = '.../MRIQC/out/sub-' + i + 'TP1_T1w/T1w.csv' #TP1
    #file_path = '.../MRIQC/out/sub-' + i + 'TP2_T1w/T1w.csv' #TP2
    if not os.path.exists(file_path):
        error_list.append(i)
        continue  # skip the file if it doesn't exist
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[1, 1]
    data_list.append(data)
    index_list.append(i)

# Create a new dataframe with the merged data
merged_df = pd.DataFrame(list(zip(index_list, data_list)), columns = ["Patient number", "Coefficient of Joint Variation"])

# Write the merged data to a CSV file
merged_df.to_csv(output_file_path, index=False)

print("Data merged successfully!")
print("These files don't exist: ", error_list)

#############################################################################################################
# Read data from the CJV and DTI motion score csv file
data = pd.read_csv('.../MRIQC/output_TP1.csv')
# Plot with histogram
plt.hist(data['Coefficient of Joint Variation'], bins=25) # Adjust the number of bins as needed
plt.xlabel('Coefficient of Joint Variation of TP1')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Plot the CJV with boxplot, split into treatment and placebo groups.
filtered_df = data[data['Treatment_Outcome'].isin([0, 1])]
# Create the box plot
plt.boxplot([filtered_df[filtered_df['Treatment_Outcome'] == 1]['Coefficient of Joint Variation'],
             filtered_df[filtered_df['Treatment_Outcome'] == 0]['Coefficient of Joint Variation']])
# Add labels and title
plt.xlabel('Groups')
plt.ylabel('Coefficient of Joint Variation')
plt.title('Box Plot of Treatment Outcome')

# Set x-axis tick labels
plt.xticks([1, 2], ['Respondents', 'Non-respondents'])
plt.ylabel('Coefficient of Joint Variation in TP1')
plt.title('Boxplot')
plt.show()

# Plot the DTI motion score with boxplot, split into treatment and placebo groups.
filtered_df = data[data['Treatment_Outcome1'].isin([0, 1])]
# Create the box plot
plt.boxplot([filtered_df[filtered_df['Treatment_Outcome1'] == 1]['DTI_Motion Score'],
             filtered_df[filtered_df['Treatment_Outcome1'] == 0]['DTI_Motion Score']])
# Add labels and title
plt.xlabel('Groups')
plt.ylabel('DTI Motion Score')
plt.title('Box Plot of Treatment Outcome')

# Set x-axis tick labels
plt.xticks([1, 2], ['Respondents', 'Non-respondents'])
plt.ylabel('DTI Motion Score in TP1')
plt.title('Boxplot')
plt.show()
#############################################################################################################

# Statistical Analysis
from scipy.stats import pointbiserialr
import pandas as pd
data1 = pd.read_csv('.../MRIQC/output_TP1.csv')
X = data1.iloc[:,2]
Y = data1.iloc[:,1]
pointbiserialr(X,Y)
#############################################################################################################
#MRI QA Voxel volume TP2-TP1
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
label_list = [7, 8, 11, 12, 13, 17, 26, 46, 47, 50, 51, 52, 53, 58]
label_name = {'7': 'Left-Cerebellum-White-Matter', '8': 'Left-Cerebellum-Cortex',
          '11':'Left-Caudate', '12': 'Left-Putamen', '13': 'Left-Pallidum', '17': 'Left-Hippocampus',
          '26': 'Left-Accumbens-area', '46': 'Right-Cerebellum-White-Matter', '47': 'Right-Cerebellum-Cortex',
          '50': 'Right-Caudate', '51': 'Right-Putamen', '52': 'Right-Pallidum', '53': 'Right-Hippocampus',
          '58': 'Right-Accumbens-area'}

data_list = []
index_list = []
error_list = []

for i in label_list:
#for i in [7]: # test code
  Label_name = label_name[str(i)]
  file_path_1 = '.../Extract_features_ROI/T1/TP1/Output_SynthSeg/' + \
            Label_name + '_radiomics_features.csv'
  output_file_path = ".../MRIQC/Output/SynthSeg/" + Label_name + "_Voxel_Volume_Subtraction.csv"
  if not os.path.exists(file_path_1):
      error_list.append(i)
      continue  # skip the file if it doesn't exist
  df1 = pd.read_csv(file_path_1, header=0)
  data_1 = df1.iloc[:, 22]
  file_path_2 = '.../Extract_features_ROI/T1/TP2/Output_SynthSeg/' + \
            Label_name + '_radiomics_features.csv'
  df2 = pd.read_csv(file_path_2, header=0)
  data_2 = df2.iloc[:, 19]
  data_s = data_2 - data_1
  data_list.append(data_s)
  # Create a new dataframe with the merged data
  merged_df = pd.DataFrame(zip(data_s), columns=[Label_name])
  # Write the merged data to a CSV file
  merged_df.to_csv(output_file_path, index=False)
  plt.hist(data_s, bins = 30)
  plt.xlabel(Label_name + ' Voxel Volume Subtraction')
  plt.ylabel('Frequency')
  plt.savefig(Label_name + "_Voxel_Volume.png")
  plt.show()



