import nilearn as nilearn
from nilearn import image
import numpy as np
import pandas as pd
import nibabel as nb

#check:https://nilearn.github.io/modules/generated/nilearn.signal.clean.html
###based on the file structure we need to loop across session and participants

def postproc_script(img_files, img_mask_files, conf_files):
    postprocess_track_dict={}
    for subject, img_file_paths in img_files.items():
        
        img_mask_file_paths = img_mask_files.get(subject, [])
        conf_file_paths = conf_files.get(subject, [])

        design_out_list=[]
        for img_file, img_mask_file, conf_file in zip(img_file_paths, img_mask_file_paths, conf_file_paths):
            img = nb.load(img_file)
            img_mask = nb.load(img_mask_file)
            conf_fil = conf_file

            dread = pd.read_csv(conf_fil,
                                sep = '\t')

            # prepare the nuisance factors: we should discuss this: https://neurostars.org/t/confounds-from-fmriprep-which-one-would-you-use-for-glm/326
            df = pd.concat([dread['trans_x'], 
                            dread['trans_y'],
                            dread['trans_z'],
                            dread['rot_x'],
                            dread['rot_y'],
                            dread['rot_z'],
                            dread['framewise_displacement'],
                            dread['a_comp_cor_00'],
                            dread['a_comp_cor_01'],
                            dread['a_comp_cor_02'],
                            dread['a_comp_cor_03'],
                            dread['a_comp_cor_04'],
                            dread['a_comp_cor_05']],
                            axis=1)

            design_out = conf_fil[:-4] + '_small.csv'
            design_out_list.append(design_out)

            df.to_csv(design_out, 
                    sep='\t',
                    index=False,
                    header=False)

            df_removed = df.loc[5:,] #remove 5 TRs from the csv file
            img_removed = nilearn.image.index_img(img, slice(5, np.size(img, 3))) # remive 5 slices/TRs from the image

            img_cleaned = nilearn.image.clean_img(img_removed, detrend = True, 
                                                standardize = True, 
                                                confounds = df_removed,  
                                                low_pass = 1/10,
                                                high_pass = 1/100,
                                                t_r = 2.1, mask_img = img_mask) # detrending, motion correction, low pass filter, high-pass filter 

            cleaned_filename = img_file[:-7]+'_postproc.nii.gz'
            smoothed_filename = cleaned_filename[:-7]+'_smooth.nii.gz'
            img_cleaned.to_filename(cleaned_filename)
            img_cleaned_smoothed = nilearn.image.smooth_img(img_cleaned, fwhm = [4, 4, 4])
            img_cleaned_smoothed.to_filename(smoothed_filename)
        
        
        postprocess_track_dict[subject] = design_out_list
        print(f"{subject} successfully postprocessed")
    sizes_match = all(len(conf_files[key]) == len(postprocess_track_dict[key]) for key in conf_files)

    if sizes_match:
        print("The sizes of arrays for each key are the same.")
    else:
        print("The sizes of arrays for each key are not the same.")
    
    return(postprocess_track_dict)