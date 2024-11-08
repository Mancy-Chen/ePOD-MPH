# Mancy Chen 03/04/2023 ePOD extract features from ROI
from __future__ import print_function
import logging
import os
import sys
import pandas
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import nibabel as nib
import numpy as np
# label_list = [7, 8, 11, 12, 13, 17, 26, 46, 47, 50, 51, 52, 53, 58]
# label_name = {'7': 'Left-Cerebellum-White-Matter', '8': 'Left-Cerebellum-Cortex',
#           '11':'Left-Caudate', '12': 'Left-Putamen', '13': 'Left-Pallidum', '17': 'Left-Hippocampus',
#           '26': 'Left-Accumbens-area', '46': 'Right-Cerebellum-White-Matter', '47': 'Right-Cerebellum-Cortex',
#           '50': 'Right-Caudate', '51': 'Right-Putamen', '52': 'Right-Pallidum', '53': 'Right-Hippocampus',
#           '58': 'Right-Accumbens-area'}
# list_of_input = ['001', '003', '004', '006', '007', '008', '010', '011', '017', '018', '020',
#                  '021', '022', '024', '027', '029', '030', '032', '033', '034', '035', '036',
#                  '037', '038', '039', '040', '041', '042', '044', '045', '047', '049', '050',
#                  '051', '052', '054', '056', '059', '062', '063', '064', '065', '067', '068',
#                  '069', '070', '071', '072', '074', '075', '101', '102', '103', '104', '105',
#                  '106', '107', '108', '110', '111', '112', '113', '114', '115', '116', '117',
#                  '118', '119', '120', '121', '122', '123', '124', '125', '126', '128', '130',
#                  '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143',
#                  '144', '145', '146', '147', '148', '149', '150', '152', '153', '155', '156']
label_list = [11, 12, 13, 17, 26, 50, 51, 52, 53, 58]
label_name = {'11':'Left-Caudate', '12': 'Left-Putamen', '13': 'Left-Pallidum', '17': 'Left-Hippocampus',
          '26': 'Left-Accumbens-area', '50': 'Right-Caudate', '51': 'Right-Putamen', '52': 'Right-Pallidum',
          '53': 'Right-Hippocampus', '58': 'Right-Accumbens-area'}
list_of_input = ['001', '004', '006', '010', '011', '022', '024', '027', '030', '034',
                 '036', '038', '040', '041', '044', '049', '052', '054', '059', '063',
                 '065', '067', '070', '071', '075', '102', '104', '105', '108', '110',
                 '111', '113', '117', '118', '119', '121', '124', '125', '130', '134',
                 '135', '137', '138', '140', '143', '145', '148', '150', '152', '155']

#########################################################################################################
# FastSurfer
# Creat the input csv file path for batch extraction:
input_file_path = []
input_mask_path = []
for i in list_of_input:
  input_file = '.../FastSurfer/Output_TP1/' + i + '/mri/orig.nii.gz'
  input_mask = '.../FastSurfer/Output_TP1/' + i + '/mri/manually_adjusted.nii.gz'
  input_file_path.append(input_file)
  input_mask_path.append(input_mask)

list_of_input_1 = {'Image': input_file_path, 'Mask': input_mask_path}
lt = pandas.DataFrame(list_of_input_1)
lt.to_csv('.../Extract_features_ROI/T1/TP1/file_path_csv/FastSurfer_adjusted_path.csv')

# Convert orig.mgz to orig.nii.gz
for i in list_of_input:
  command_1 = 'cd .../FastSurfer/Output1/' + i + '/mri/                        '
  command_2 = 'mri_convert orig.mgz orig.nii.gz'
  command_3 = 'mri_convert aparc.DKTatlas+aseg.deep.mgz aparc.DKTatlas+aseg.deep.nii.gz'
  command = command_1 + '&&' + command_2 + '&&' + command_3
  os.system(command)
###########################################################################################################
# Merge Cerebellum Cortex and White Matter ROI into one ROI
for i in list_of_input:
#for i in ['001']: # test code
  nii_file = '.../SynthSeg/Output/' + i + '/orig_synthseg.nii.gz'
  image = nib.load(nii_file)
  data = image.get_fdata().astype(np.uint16)
  label7 = np.where(data == 7, 1, 0)
  label8 = np.where(data == 8, 1, 0)
  label46 = np.where(data == 46, 1, 0)
  label47 = np.where(data == 47, 1, 0)
  # Merge Left Cerebellum
  merged_mask = np.logical_or(label7, label8).astype(np.uint16)
  merged_image = nib.Nifti1Image(merged_mask, image.affine, image.header)
  output_file = '.../Extract_features_ROI/Merged_Cerebellum_ROI/SynthSeg/Left_Cerebellum/' + i + \
                '_SynthSeg_Left_Cerebellum.nii.gz'
  nib.save(merged_image, output_file)
  # Merge Right Cerebellum
  merged_mask1 = np.logical_or(label46, label47).astype(np.uint16)
  merged_image1 = nib.Nifti1Image(merged_mask1, image.affine, image.header)
  output_file1 = '.../Extract_features_ROI/Merged_Cerebellum_ROI/SynthSeg/Right_Cerebellum/' + i + \
                '_SynthSeg_Right_Cerebellum.nii.gz'
  nib.save(merged_image1, output_file1)

# Creat the input csv file path for batch extraction:
input_file_path = []
input_mask_path = []
for i in list_of_input:
  input_file = '.../SynthSeg/Output/' + i + '/orig.nii.gz'
  input_mask = '.../Extract_features_ROI/Merged_Cerebellum_ROI/SynthSeg/Left_Cerebellum/' + i + \
                '_SynthSeg_Left_Cerebellum.nii.gz' # FastSurfer or SynthSeg, Left_Cerebellum or Right_Cerebellum
  input_file_path.append(input_file)
  input_mask_path.append(input_mask)

list_of_input_1 = {'Image': input_file_path, 'Mask': input_mask_path}
lt = pandas.DataFrame(list_of_input_1)
lt.to_csv('SynthSeg_Left_Cerebellum_path.csv') # FastSurfer or SynthSeg, Left_Cerebellum or Right_Cerebellum
###########################################################################################################
# DTI ROI
# Load the DTI mask
dti_mask = nib.load('.../Extract_features_ROI/DTI/Input/JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz')
dti_mask_data = dti_mask.get_fdata().astype(np.uint16)
ATR_left = np.where(dti_mask_data == 1, 1, 0)
ATR_right = np.where(dti_mask_data == 2, 1, 0)
Forceps_Major = np.where(dti_mask_data == 9, 1, 0)
whole_brain = np.where(dti_mask_data > 0, 1, 0)
# Save the DTI mask
ATR_left_image = nib.Nifti1Image(ATR_left, dti_mask.affine, dti_mask.header)
ATR_right_image = nib.Nifti1Image(ATR_right, dti_mask.affine, dti_mask.header)
Forceps_Major_image = nib.Nifti1Image(Forceps_Major, dti_mask.affine, dti_mask.header)
whole_brain_image = nib.Nifti1Image(whole_brain, dti_mask.affine, dti_mask.header)
nib.save(ATR_left_image, '.../Extract_features_ROI/DTI/Input/mask/ATR_left.nii.gz')
nib.save(ATR_right_image, '.../Extract_features_ROI/DTI/Input/mask/ATR_right.nii.gz')
nib.save(Forceps_Major_image, '.../Extract_features_ROI/DTI/Input/mask/Forceps_Major.nii.gz')
nib.save(whole_brain_image, '.../Extract_features_ROI/DTI/Input/mask/whole_brain.nii.gz')
# Creat the input csv file path for batch extraction:
track_name = ['ATR_left', 'ATR_right', 'Forceps_Major', 'whole_brain']
ad = [f"{i:04}" for i in range(0, 48)]
ch = [f"{i:04}" for i in range(0, 44)]
for j in track_name:
    input_file_path = []
    input_mask_path = []
    for i in ad:
        input_file = '.../Extract_features_ROI/DTI/Input/split_TBSS/adhdAd' + str(i) + '.nii.gz'
        input_mask = '.../Extract_features_ROI/DTI/Input/mask/' + j + '.nii.gz'
        input_file_path.append(input_file)
        input_mask_path.append(input_mask)
    for k in ch:
        input_file = '.../Extract_features_ROI/DTI/Input/split_TBSS/adhdCh' + str(k) + '.nii.gz'
        input_mask = '.../Extract_features_ROI/DTI/Input/mask/' + j + '.nii.gz'
        input_file_path.append(input_file)
        input_mask_path.append(input_mask)
    list_of_input_1 = {'Image': input_file_path, 'Mask': input_mask_path}
    lt = pandas.DataFrame(list_of_input_1)
    lt.to_csv('.../Extract_features_ROI/DTI/file_path_csv/DTI_' + j + '_path.csv')
###########################################################################################################
# Create a general mask for all DTI TBSS images
import nibabel as nib
import os
# Directory where all the image masks are stored
mask_directory = ".../Extract_features_ROI/DTI/Input/split_TBSS/"
# Get a list of all files with the .nii.gz extension in the directory
image_paths = [os.path.join(mask_directory, f) for f in os.listdir(mask_directory) if f.endswith('.nii.gz')]
# Ensure there's at least one mask in the directory
if not image_paths:
    raise ValueError("No mask files found in the specified directory!")
# Load the first image mask to initialize the resulting mask
first_mask_nii = nib.load(image_paths[0])
resulting_mask_data = first_mask_nii.get_fdata().astype(bool)
# For each subsequent image mask, take the logical OR with the resulting mask
for path in image_paths[1:]:
    mask_nii = nib.load(path)
    mask_data = mask_nii.get_fdata().astype(bool)
    resulting_mask_data = resulting_mask_data | mask_data
# Convert the resulting mask to the same data type as the original mask for saving
resulting_mask_data = resulting_mask_data.astype(first_mask_nii.get_data_dtype())
# Save the resulting union mask
resulting_mask_nii = nib.Nifti1Image(resulting_mask_data, first_mask_nii.affine)
nib.save(resulting_mask_nii, ".../Extract_features_ROI/DTI/Input/mask/New_ROI/combined_mask.nii.gz")
###########################################################################################################
# Extract the overlapping region between two masks
import nibabel as nib
# 1. Load the image masks using Nibabel
mask_A_nii = nib.load(".../Extract_features_ROI/DTI/Input/mask/New_ROI/combined_mask.nii.gz")
mask_B_nii = nib.load(".../Extract_features_ROI/DTI/Input/mask/thr25/Forceps_Major.nii.gz")

# 2. Extract the data arrays from the loaded Nifti images
mask_A_data = mask_A_nii.get_fdata()
mask_B_data = mask_B_nii.get_fdata()

# 3. Use logical operations to get the overlap
overlap = (mask_A_data > 0) & (mask_B_data > 0)

# Convert overlap to the same data type as original mask for saving
overlap = overlap.astype(mask_A_data.dtype)

# 4. Save the overlapping region as a new Nifti image (optional)
overlap_nii = nib.Nifti1Image(overlap, mask_A_nii.affine)
nib.save(overlap_nii, ",,,/Extract_features_ROI/DTI/Input/mask/New_ROI/Forceps_Major.nii.gz")
###########################################################################################################
# Change DTI mask's label
import nibabel as nib
import numpy as np
# Step 1: Load the NIfTI mask
nifti_file_path = "path/to/your/mask.nii.gz"
img = nib.load(nifti_file_path)
# Step 2: Extract the data array
data = img.get_fdata()
# Step 3: Change values from 2 to 1
data[data == 2] = 1
# Step 4: Create a new NIfTI object
new_img = nib.Nifti1Image(data, img.affine, img.header)
# Step 5: Save the new NIfTI object
new_nifti_file_path = "path/to/your/new_mask.nii.gz"
nib.save(new_img, new_nifti_file_path)
###########################################################################################################
# Create a new whole brain mask where voxel > 0
import numpy as np
import nibabel as nib  # if you're working with NIfTI images

# Load your 3D data (replace 'your_image_file.nii.gz' with your image path)
img = nib.load('.../Extract_features_ROI/DTI/Input/split_TBSS/adhdAd0000.nii.gz')
data = img.get_fdata()

# Create the mask
mask = data > 0

# If you want to save the mask as a NIfTI image:
mask_img = nib.Nifti1Image(mask.astype(np.int16), img.affine)
nib.save(mask_img, '.../Extract_features_ROI/DTI/Input/mask/voxel_mask.nii.gz')
###########################################################################################
# Single File Extraction Test
from __future__ import print_function
import logging
import os
import six
import radiomics
from radiomics import featureextractor, getFeatureClasses

# Get some test data
imageName = '.../FastSurfer/Output_TP1/155/mri/orig.nii.gz'
maskName = '.../FastSurfer/Output_TP1/155/mri/manually_adjusted.nii.gz'
# Get the location of the example settings file
paramsFile = '.../Extract_features_ROI/Params.yml'

if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
  print('Error getting testcase!')
  exit()

# Regulate verbosity with radiomics.verbosity
# radiomics.setVerbosity(logging.INFO)

# Get the PyRadiomics logger (default log-level = INFO
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize feature extractor using the settings file
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)


featureClasses = getFeatureClasses()

print("Active features:")
for cls, features in six.iteritems(extractor.enabledFeatures):
  if features is None or len(features) == 0:
    features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
  for f in features:
    print(f)
    print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

print("Calculating features")
featureVector = extractor.execute(imageName, maskName)

for featureName in featureVector.keys():
  print("Computed %s: %s" % (featureName, featureVector[featureName]))
###########################################################################################
# DTI Batch Radiomics Feature Extraction
from __future__ import print_function
import collections
import csv
import logging
import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
# Notes: if mask label =/ 1, remember to add a new column in the csv file of 'Label' and assign the label number
Label_name_list = ['ATR_left', 'ATR_right', 'Forceps_Major', 'whole_brain']
for i in Label_name_list:
    Output_path = i + '_radiomics_features.csv'
    log_path = i + '_pyrad_log.txt'
    def main():
      outPath = r'.../Extract_features_ROI/DTI/Output/New_ROI_expand_skeleton/'
      inputCSV = '.../Extract_features_ROI/DTI/file_path_csv/New_ROI_expand_skeleton/DTI_' + i + '_path.csv'
      outputFilepath = os.path.join(outPath, Output_path)
      progress_filename = os.path.join(outPath, log_path)
      params = r'.../Extract_features_ROI/Params.yml'

      # Configure logging
      rLogger = logging.getLogger('radiomics')

      # Set logging level
      # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

      # Create handler for writing to log file
      handler = logging.FileHandler(filename=progress_filename, mode='w')
      handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
      rLogger.addHandler(handler)

      # Initialize logging for batch log messages
      logger = rLogger.getChild('batch')

      # Set verbosity level for output to stderr (default level = WARNING)
      radiomics.setVerbosity(logging.INFO)

      logger.info('pyradiomics version: %s', radiomics.__version__)
      logger.info('Loading CSV')

      flists = []
      try:
          with open(inputCSV, 'r') as inFile:
              cr = csv.DictReader(inFile, lineterminator='\n')
              flists = [row for row in cr]
      except Exception:
          logger.error('CSV READ FAILED', exc_info=True)

      logger.info('Loading Done')
      logger.info('Patients: %d', len(flists))

      if os.path.isfile(params):
          extractor = featureextractor.RadiomicsFeatureExtractor(params)
      else:  # Parameter file not found, use hardcoded settings instead
          settings = {}
          settings['binWidth'] = 25
          settings['resampledPixelSpacing'] = None  # [3,3,3]
          settings['interpolator'] = sitk.sitkBSpline
          settings['enableCExtensions'] = True

          extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
          # extractor.enableInputImages(wavelet= {'level': 2})

      logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
      logger.info('Enabled features: %s', extractor.enabledFeatures)
      logger.info('Current settings: %s', extractor.settings)

      headers = None

      for idx, entry in enumerate(flists, start=1):

          logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'],
                      entry['Mask'])

          imageFilepath = entry['Image']
          maskFilepath = entry['Mask']
          label = entry.get('Label', None)

          if str(label).isdigit():
              label = int(label)
          else:
              label = None

          if (imageFilepath is not None) and (maskFilepath is not None):
              featureVector = collections.OrderedDict(entry)
              featureVector['Image'] = os.path.basename(imageFilepath)
              featureVector['Mask'] = os.path.basename(maskFilepath)

              try:
                  featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

                  with open(outputFilepath, 'a') as outputFile:
                      writer = csv.writer(outputFile, lineterminator='\n')
                      if headers is None:
                          headers = list(featureVector.keys())
                          writer.writerow(headers)

                      row = []
                      for h in headers:
                          row.append(featureVector.get(h, "N/A"))
                      writer.writerow(row)
              except Exception:
                  logger.error('FEATURE EXTRACTION FAILED', exc_info=True)


    if __name__ == '__main__':
        main()
##############################################################################################################
# T1 Batch Radiomics Feature Extraction
#!/usr/bin/env python
from __future__ import print_function
import collections
import csv
import logging
import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

label_list = [11, 12, 13, 17, 26, 50, 51, 52, 53, 58]
label_name = {'11':'Left_Caudate', '12': 'Left_Putamen', '13': 'Left_Pallidum', '17': 'Left_Hippocampus',
          '26': 'Left_Accumbens_Area', '50': 'Right_Caudate', '51': 'Right_Putamen', '52': 'Right_Pallidum',
          '53': 'Right_Hippocampus', '58': 'Right_Accumbens_Area'}
list_of_input = ['001', '004', '006', '010', '011', '022', '024', '027', '030', '034',
                 '036', '038', '040', '041', '044', '049', '052', '054', '059', '063',
                 '065', '067', '070', '071', '075', '102', '104', '105', '108', '110',
                 '111', '113', '117', '118', '119', '121', '124', '125', '130', '134',
                 '135', '137', '138', '140', '143', '145', '148', '150', '152', '155']
for i in label_list:
  Label_name = label_name[str(i)]
  def main():
      outPath = os.path.join('.../Extract_features_ROI/FastSurfer_adjusted/Output',
                             Label_name)
      inputCSV = r'.../Extract_features_ROI/FastSurfer_adjusted/FastSurfer_adjusted_path.csv'
      outputFilepath = outPath + '_radiomics_features.csv'
      progress_filename = outPath + '_pyrad_log.txt'
      params = r'.../Extract_features_ROI/Params.yml'

      # Configure logging
      rLogger = logging.getLogger('radiomics')

      # Set logging level
      # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

      # Create handler for writing to log file
      handler = logging.FileHandler(filename=progress_filename, mode='w')
      handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
      rLogger.addHandler(handler)

      # Initialize logging for batch log messages
      logger = rLogger.getChild('batch')

      # Set verbosity level for output to stderr (default level = WARNING)
      radiomics.setVerbosity(logging.INFO)

      logger.info('pyradiomics version: %s', radiomics.__version__)
      logger.info('Loading CSV')

      flists = []
      try:
        with open(inputCSV, 'r') as inFile:
          cr = csv.DictReader(inFile, lineterminator='\n')
          flists = [row for row in cr]
      except Exception:
        logger.error('CSV READ FAILED', exc_info=True)

      logger.info('Loading Done')
      logger.info('Patients: %d', len(flists))

      if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
      else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        # extractor.enableInputImages(wavelet= {'level': 2})

      logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
      logger.info('Enabled features: %s', extractor.enabledFeatures)
      logger.info('Current settings: %s', extractor.settings)

      headers = None

      for idx, entry in enumerate(flists, start=1):

        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Mask'])

        imageFilepath = entry['Image']
        maskFilepath = entry['Mask']

        label = int(i)  # batch process the multiple label [Remember to recover this line if having multiple labels!]

        # label = entry.get('Label', None) # Mute this line if having multiple labels
        # if str(label).isdigit(): # Mute this line if having multiple labels
        #   label = int(label) # Mute this line if having multiple labels
        # else: # Mute this line if having multiple labels
        #   label = None # Mute this line if having multiple labels

        if (imageFilepath is not None) and (maskFilepath is not None):
          featureVector = collections.OrderedDict(entry)
          featureVector['Image'] = os.path.basename(imageFilepath)
          featureVector['Mask'] = os.path.basename(maskFilepath)

          try:
            featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

            with open(outputFilepath, 'a') as outputFile:
              writer = csv.writer(outputFile, lineterminator='\n')
              if headers is None:
                headers = list(featureVector.keys())
                writer.writerow(headers)

              row = []
              for h in headers:
                row.append(featureVector.get(h, "N/A"))
              writer.writerow(row)
          except Exception:
            logger.error('FEATURE EXTRACTION FAILED', exc_info=True)


  if __name__ == '__main__':
      main()

########################################################################################################################
# Batch Radiomics Extration
label_list = [11, 12, 13, 17, 26, 50, 51, 52, 53, 58]
label_name = {'11':'Left_Caudate', '12': 'Left_Putamen', '13': 'Left_Pallidum', '17': 'Left_Hippocampus',
          '26': 'Left_Accumbens_Area', '50': 'Right_Caudate', '51': 'Right_Putamen', '52': 'Right_Pallidum',
          '53': 'Right_Hippocampus', '58': 'Right_Accumbens_Area'}
list_of_input = ['001', '004', '006', '010', '011', '022', '024', '027', '030', '034',
                 '036', '038', '040', '041', '044', '049', '052', '054', '059', '063',
                 '065', '067', '070', '071', '075', '102', '104', '105', '108', '110',
                 '111', '113', '117', '118', '119', '121', '124', '125', '130', '134',
                 '135', '137', '138', '140', '143', '145', '148', '150', '152', '155']
# Label_name_list = ['ATR_left', 'ATR_right', 'Forceps_Major', 'whole_brain']
for i in label_list:
# for i in [1]: # test code
#for Label_name in Label_name_list:
  Label_name = label_name[str(i)]
  #Label_name = 'SynthSeg_Left_Cerebellum_' # 'SynthSeg' or 'FastSurfer', 'Left' or 'Right'
  # Label_name = 'ATR_left'
  Output_path = Label_name + '_radiomics_features.csv'
  log_path = Label_name + '_pyrad_log.txt'
  method_name = 'FastSurfer_adjusted'       # 'SynthSeg' or 'FastSurfer'
  method_path = method_name + '_path.csv'
  def main():

    outPath = r'.../Extract_features_ROI/FastSurfer_adjusted/'
    inputCSV = '.../Extract_features_ROI/FastSurfer_adjusted/FastSurfer_adjusted_path.csv'
    outputFilepath = os.path.join(outPath, Output_path)
    progress_filename = os.path.join(outPath, log_path)
    params = '.../Extract_features_ROI/Params.yml'

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

    try:
      # Use pandas to read and transpose ('.T') the input data
      # The transposition is needed so that each column represents one test case. This is easier for iteration over
      # the input cases
      flists = pandas.read_csv(inputCSV).T
    except Exception:
      logger.error('CSV READ FAILED', exc_info=True)
      exit(-1)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))

    if os.path.isfile(params):
      extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
      settings = {}
      settings['binWidth'] = 25
      settings['resampledPixelSpacing'] = None  # [3,3,3]
      settings['interpolator'] = sitk.sitkBSpline
      settings['enableCExtensions'] = True

      extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
      # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    # Instantiate a pandas data frame to hold the results of all patients
    results = pandas.DataFrame()

    for entry in flists:  # Loop over all columns (i.e. the test cases)
      logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                  entry + 1,
                  len(flists),
                  flists[entry]['Image'],
                  flists[entry]['Mask'])

      imageFilepath = flists[entry]['Image']
      maskFilepath = flists[entry]['Mask']

      label = int(i)  # batch process the multiple label [Remember to recover this line if having multiple labels!]

      # label = flists[entry].get('Label', None) # Mute this line if having multiple labels
      # if str(label).isdigit(): # Mute this line if having multiple labels
      #   label = int(label) # Mute this line if having multiple labels
      # else: # Mute this line if having multiple labels
      #   label = None # Mute this line if having multiple labels

      if (imageFilepath is not None) and (maskFilepath is not None):
        featureVector = flists[entry]  # This is a pandas Series
        featureVector['Image'] = os.path.basename(imageFilepath)
        featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
          # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
          # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
          # as the values in the rows.
          nii = nib.load('path')
          vol = nii.get_fdata()
          vol[:,:,:,:]
          result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
          featureVector = featureVector.append(result)
        except Exception:
          logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

        # To add the calculated features for this case to our data frame, the series must have a name (which will be the
        # name of the column.
        featureVector.name = entry
        # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
        # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
        # it is 'joined' with the empty data frame.
        results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

    logger.info('Extraction complete, writing CSV')
    # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
    results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
    logger.info('CSV writing complete')


  if __name__ == '__main__':
    main()


