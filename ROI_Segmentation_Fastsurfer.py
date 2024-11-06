#####################################################################################################################
# Clone FastSurfer repo
git clone -q https://github.com/deep-mi/fastsurfer.git
cd fastsurfer
export FASTSURFER_HOME=$PWD

# Install python dependencies
python3 -m pip install --user --trusted-host pypi.python.org --prefer-binary -r requirements.txt
##################################################################################################################
# Define your input image (the one you downloaded above)
input=/data/projects/CSC/code/FastSurfer/orig.mgz

# Define where you stored the output
outputDir=/data/projects/CSC/code/FastSurfer/Output_test

# Define where your local installation of FastSurfer can be found
export FASTSURFER_HOME=/data/projects/CSC/code/FastSurfer/pythonProject/fastsurfer/

# Final run command
$FASTSURFER_HOME/run_fastsurfer.sh --t1 $input \
                                   --sd $outputDir \
                                   --sid test3 \
                                   --seg_only
#####################################################
# test code
input=/data/projects/CSC/data/epod/analysis/subjects-5.3.0-STORAGE-freesurfer/001/001_TP2/mri/orig.mgz
outputDir=/data/projects/CSC/code/06_FastSurfer/Output1
export FASTSURFER_HOME=/data/projects/CSC/code/06_FastSurfer/pythonProject/fastsurfer
$FASTSURFER_HOME/run_fastsurfer.sh --t1 $input \
                                   --sd $outputDir \
                                   --sid 001 \
                                   --seg_only
#######################################################################################################################
# Batch segmentation
import os

list = {'001','017','030','039','050','064','074','107','116','124','136','144','153',
        '003','018','032','040','051','065','075','108','117','125','137','145','155',
        '004','020','033','041','052','067','101','110','118','126','138','146','156',
        '006','021','034','042','054','068','102','111','119','128','139','147',
        '007','022','035','044','056','069','103','112','120','130','140','148',
        '008','024','036','045','059','070','104','113','121','133','141','149',
        '010','027','037','047','062','071','105','114','122','134','142','150',
        '011','029','038','049','063','072','106','115','123','135','143','152'}

for i in list:
    i = '001' # test code
    path = '/data/projects/CSC/data/epod/analysis/subjects-5.3.0-STORAGE-freesurfer'
    file_1 = i + '/'  # 001/
    file_2 = i + '_TP2/'  # 001_TP2/
    rest = 'mri/orig.mgz'
    input = os.path.join(path, file_1, file_2, rest)
    outputDir= '/data/projects/CSC/code/06_FastSurfer/Output1/'
    env = 'export FASTSURFER_HOME=/data/projects/CSC/code/06_FastSurfer/pythonProject/fastsurfer'
    command = '$FASTSURFER_HOME/run_fastsurfer.sh --t1 ' + input  \
                                       + " --sd " + outputDir  \
                                       + ' --sid ' + i  \
                                       + ' --seg_only'
    # when running multiple commands, you need to use '&&' to link them
    full_command = env + ' && ' + command
    os.system(full_command)
    print ('Finish ', i)
