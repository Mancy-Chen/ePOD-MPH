
SDIR=.../MRIQC
BDIR=$SDIR/data
ODIR=$SDIR/out
WDIR=$SDIR/work

######
# copy data

IDIR=.../subjects-5.3.0-STORAGE-freesurfer

doit=1

# find all input files
cd $IDIR
for file in `find ./*/???_TP? -name "orig.mgz"` ; do
  ID=`echo $file | cut -f3 -d"/"`
  OID=sub-`echo $ID | sed s/_//g`_T1w
  echo $ID $OID
  TODIR=$BDIR/$OID/anat
  echo $TODIR

  echo "mri_convert $file $TODIR/$OID.nii.gz"

  if [ $doit == 1 ] ; then
    mkdir -p $TODIR
    mri_convert $file $TODIR/$OID.nii.gz
    fslreorient2std $TODIR/$OID.nii.gz $TODIR/$OID.nii.gz
  fi
  
done



###################
# START SINGULARITY
cd $BDIR
MRIQC_IMAGE=poldracklab_mriqc_0.11.0-2018-06-05-442554ee49a6.img
singularity run --cleanenv --bind $BDIR:/data --bind $ODIR:/out --bind $WDIR:/work $SDIR/$MRIQC_IMAGE


# subject analysis
for subjdir in /data/*; do
  subj=`echo $subjdir | cut -f3 -d"/"`
  echo $subj  
  # check for output
  if [ -f /out/$subj/T1w.csv ] ; then
    echo "exists"
  else
    RUNSTR="mriqc /data/$subj/anat /out/$subj participant --n_procs 8 --no-sub"
    echo $RUNSTR
    mriqc /data/$subj/anat /out/$subj participant --n_procs 8 --no-sub
  fi
done

# group analysis -> doesn't run
mriqc /data /out group --n_procs 8 --no-sub --participant_label sub-156TP2 






#################
# OLD
#
# run MRIQC for all subjects

#cd $BDIR
#MRIQC_IMAGE=poldracklab_mriqc_0.11.0-2018-06-05-442554ee49a6.img

# run single subject
#nice -n 19 singularity run --cleanenv --bind $BDIR:/data --bind $ODIR:/out $SDIR/$MRIQC_IMAGE \
# /data /out participant --participant_label `ls -d sub-*` \
#  --n_procs 8 --mem_gb 32 --no-sub

# group analysis
#nice -n 19 singularity run --cleanenv --bind $BDIR:/data --bind $ODIR:/out $SDIR/$MRIQC_IMAGE \
# /data /out group  --n_procs 8 --mem_gb 32 --no-sub
 






