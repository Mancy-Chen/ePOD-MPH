%% EPOD DTI ROI analysis
%
% Saves to tracts_mean_XXX.txt in maindir defined below
%
% Toolbox dependencies: 
% - DIPimage (dilation function can be reimplemented using other software if needed)
% - toolboxes-amc/gfx (optional, code at the end for visualization)
%
% maindir should point to TBSS-output folder

close all
clear all

%maindir='.../tbss/stats'
maindir='.../analysis/DTI/stats'
cd(maindir)

% specify FSL-atlas folder
fsldir='.../fsl-5.0.9/data/atlases/JHU'

% select thresholded probability file
tractsfile=fullfile(fsldir,'JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
t=load_untouch_nii(tractsfile);
tracts=t.img;

% load TBSS-output
m=load_untouch_nii(fullfile(maindir,('mean_FA_skeleton_mask.nii.gz')));
mask=m.img;

mfa=load_untouch_nii(fullfile(maindir,('mean_FA.nii.gz')));
meanfa=mfa.img;

%% set ROIs
IDs=[1 2 9 15 16]
tags={'ATR_L','ATR_R','splenium','SLF_L','SLF_R'}
IDs=[1 2 9 ]
tags={'ATR_L','ATR_R','splenium'}

%% select DTI-measure and calc mean values in tracts of interest
files=cellstr(spm_select('fplist',maindir,'all_.*_skeletonised.nii.gz$'));

for iFile=1:length(files)
  file=files{iFile}
  [~,name]=fileparts(file);
  [~,name]=fileparts(name); % remove .nii

  v=load_untouch_nii(file);
  fa=v.img;

  sz=size(fa);
  nsubj=size(fa,4);

  fa=reshape(fa,[],nsubj);

  clear meas_roi
  for ii=1:length(IDs)
    % dilate the mask - here using DIPimage function
    roi=double(bdilation(tracts==IDs(ii)));
    roi=roi.*mask;
    meas_roi(:,ii)=mean(fa(roi>0,:),1);
  end

  savename=fullfile(maindir,['tracts_mean_' name '.txt'])
  T=table([tags;num2cell(meas_roi)])
  writetable(T,savename);

end

%% plot ROIs
roi=zeros(size(tracts));
for iROI=1:length(IDs)
  thisroi=double(bdilation(tracts==IDs(iROI)));
  roi=roi+double(thisroi)*iROI;
end

cl=lines(length(IDs)+1);
cl(1,:)=[1 1 1];
out=roi.*mask+mask;
sz=size(out);
out=out(:);
outcl=zeros(length(out),3);
outcl(out>0,:)=cl(out(out>0),:);
outcl(~out,:)=meanfa(~out)*[1 1 1];
outcl=reshape(outcl,[sz 3]);
outcl=mshow(outcl);
outcl=autocrop(outcl);
%%
arr(rgb(autocrop(outcl(:,:,45:4:70,:))),10)

% legend
figure
plot(zeros(20,length(tags)+1),'linewidth',2)
colormap(cl)
legend([{''},tags])

