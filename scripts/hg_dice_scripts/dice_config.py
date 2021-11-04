from fs_lut import fs_lut

SYNTHSEG_PRJCT = '/space/calico/1/users/Harsha/SynthSeg'
SYNTHSEG_RESULTS = f'{SYNTHSEG_PRJCT}/results'

UW_HARD_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard'
UW_SOFT_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft'
UW_MRI_SCAN = '/cluster/vive/UW_photo_recon/FLAIR_Scan_Data'

MRI_SCANS = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans'
MRI_SCANS_SEG = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations'
MRI_SCANS_REG = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.registered'
MRI_SCANS_SEG_RESAMPLED = MRI_SCANS_SEG + '.resampled'
MRI_SCANS_SEG_REG_RES = MRI_SCANS_SEG_RESAMPLED + '.registered'

HARD_RECONS = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon'
HARD_RECON_SYNTHSEG = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei'
HARD_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.hard.samseg.segmentations'
HARD_MANUAL_LABELS_MERGED = f'{SYNTHSEG_RESULTS}/UW.photos.hard.manual.labels'

SOFT_RECONS = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon'
SOFT_RECON_REG = SOFT_RECONS + '.registered'
SOFT_RECON_SYNTHSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei'
SOFT_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.samseg.segmentations'
SOFT_MANUAL_LABELS_MERGED = f'{SYNTHSEG_RESULTS}/UW.photos.soft.manual.labels'

# Note: All of these are in photo RAS space (just resampling based on reference)
MRI_SYNTHSEG_IN_SAMSEG_SPACE = MRI_SCANS_SEG_REG_RES + '.in_samseg_space'
MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = MRI_SCANS_SEG_REG_RES + '.in_softsamseg_space'
HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE = HARD_RECON_SYNTHSEG + '.in_samseg_space'
HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = HARD_RECON_SYNTHSEG + '.in_mri_space'
SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE = SOFT_RECON_SYNTHSEG + '.in_samseg_space'

mri_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations.csv'
soft_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei.csv'
hard_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei.csv'

#### Extract SAMSEG Volumes
HARD_SAMSEG_STATS = f'{UW_HARD_RECON}/SAMSEG/'
SOFT_SAMSEG_STATS = f'{UW_SOFT_RECON}/SAMSEG/'

ALL_LABELS = [
    0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 17, 18, 26, 28, 41, 42, 43, 44, 49, 50,
    51, 52, 53, 54, 58, 60
]
IGNORE_LABELS = [0, 5, 14, 26, 28, 44, 58, 60]
ADDL_IGNORE_LABELS = [7, 8, 15, 16, 46, 47]
LABEL_PAIRS = [(2, 41), (3, 42), (4, 43), (10, 49), (11, 50), (12, 51),
               (13, 52), (17, 53), (18, 54)]
LABEL_PAIR_NAMES = [
    'White Matter', 'Cortex', 'Ventricle', 'Thalamus', 'Caudate', 'Putamen',
    'Pallidum', 'Hippocampus', 'Amygdala'
]
IGNORE_SUBJECTS = ['18-1343', '18-2260', '19-0019']

LUT, REVERSE_LUT = fs_lut()
