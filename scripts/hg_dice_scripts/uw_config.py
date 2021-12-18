#!/usr/bin/env python
# coding: utf-8
"""
Author: HG
Acknowledgements: JEI, BB, and HT
Notes: Contains code to reproduce dice plots from https://arxiv.org/abs/2009.05596
Running Instructions:
1. https://github.com/hvgazula/SynthSeg/tree/photos
2. Create environment using the requirements.txt
3. source activate /space/calico/1/users/Harsha/venvs/synthseg-venv &&
4. export PYTHONPATH='/space/calico/1/users/Harsha/SynthSeg'

TODO:
1. Write a generic function for reading volumes from SAMSEG stats files
2. Write a generic function for reading volumes from SynthSeg volume files
"""

SAMSEG_GATHER_DICT = {
    "hard_samseg": {
        "source": "SAMSEG_OUTPUT_HARD_C0",
        "destination": "HARD_SAMSEG_C0",
        "expr": ["seg.mgz"],
        "message": "Hard SAMSEG C0",
    },
    "soft_samseg": {
        "source": "SAMSEG_OUTPUT_SOFT_C0",
        "destination": "SOFT_SAMSEG_C0",
        "expr": ["seg.mgz"],
        "message": "Soft SAMSEG C0",
    },
    "hard_samsegc1": {
        "source": "SAMSEG_OUTPUT_HARD_C1",
        "destination": "HARD_SAMSEG_C1",
        "expr": ["seg.mgz"],
        "message": "Hard SAMSEG C1",
    },
    "soft_samsegc1": {
        "source": "SAMSEG_OUTPUT_SOFT_C1",
        "destination": "SOFT_SAMSEG_C1",
        "expr": ["seg.mgz"],
        "message": "Soft SAMSEG C1",
    },
    "hard_samsegc2": {
        "source": "SAMSEG_OUTPUT_HARD_C2",
        "destination": "HARD_SAMSEG_C2",
        "expr": ["seg.mgz"],
        "message": "Hard SAMSEG C2",
    },
    "soft_samsegc2": {
        "source": "SAMSEG_OUTPUT_SOFT_C2",
        "destination": "SOFT_SAMSEG_C2",
        "expr": ["seg.mgz"],
        "message": "Soft SAMSEG C2",
    },
}

VOLUMES_LIST = [
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_HARD_C0",
        "tag": "Hard-Samseg-C0"
    },
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_SOFT_C0",
        "tag": "Soft-Samseg-C0"
    },
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_HARD_C1",
        "tag": "Hard-Samseg-C1"
    },
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_SOFT_C1",
        "tag": "Soft-Samseg-C1"
    },
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_HARD_C2",
        "tag": "Hard-Samseg-C2"
    },
    {
        "type": "samseg",
        "source": "SAMSEG_OUTPUT_SOFT_C2",
        "tag": "Soft-Samseg-C2"
    },
    {
        "type": "samseg",
        "source": "HARD_SAMSEG_STATS",
        "tag": "Hard-Samseg-Old"
    },
    {
        "type": "samseg",
        "source": "SOFT_SAMSEG_STATS",
        "tag": "Soft-Samseg-Old"
    },
    {
        "type": "synthseg",
        "source": "mri_synthseg_vols_file",
        "tag": "MRI-Synthseg"
    },
    {
        "type": "synthseg",
        "source": "hard_synthseg_vols_file",
        "tag": "Hard-Synthseg"
    },
    {
        "type": "synthseg",
        "source": "soft_synthseg_vols_file",
        "tag": "Soft-Synthseg"
    },
]

CORRELATIONS_LIST = [
    [["MRI-Synthseg", "Hard-Samseg-C0", "Hard-Synthseg"], "hard", "c0"],
    [["MRI-Synthseg", "Soft-Samseg-C0", "Soft-Synthseg"], "soft", "c0"],
    [["MRI-Synthseg", "Hard-Samseg-C0", "Hard-Synthseg"], "hard", "c0"],
    [["MRI-Synthseg", "Soft-Samseg-C0", "Soft-Synthseg"], "soft", "c0"],
    [["MRI-Synthseg", "Hard-Samseg-C1", "Hard-Synthseg"], "hard", "c1"],
    [["MRI-Synthseg", "Soft-Samseg-C1", "Soft-Synthseg"], "soft", "c1"],
    [["MRI-Synthseg", "Hard-Samseg-C2", "Hard-Synthseg"], "hard", "c2"],
    [["MRI-Synthseg", "Soft-Samseg-C2", "Soft-Synthseg"], "soft", "c2"],
    [["MRI-Synthseg", "Soft-Samseg-Old", "Soft-Synthseg"], "soft", "old"],
    [["MRI-Synthseg", "Hard-Samseg-Old", "Hard-Synthseg"], "hard", "old"],
]

merge = [1]
DICE2D_LIST = [
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "hard_manual_vs_hard_synth_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG",
        "output_name": "hard_manual_vs_hard_sam_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG_C0",
        "output_name": "hard_manual_vs_hard_sam_c0_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG_C1",
        "output_name": "hard_manual_vs_hard_sam_c1_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG_C2",
        "output_name": "hard_manual_vs_hard_sam_c2_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "soft_manual_vs_soft_synth_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG",
        "output_name": "soft_manual_vs_soft_sam_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG_C0",
        "output_name": "soft_manual_vs_soft_sam_c0_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG_C1",
        "output_name": "soft_manual_vs_soft_sam_c1_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG_C2",
        "output_name": "soft_manual_vs_soft_sam_c2_in_sam_space",
        "slice": 1,
        "merge": merge,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
]

# TODO:
# DICE2D_LIST = [dict(**{"merge":val}, **item) for val in merge for item in DICE2D_LIST_TEMPLATE]

PLOTS_LIST = [
    [
        "hard_manual_vs_hard_sam_in_sam_space_merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_merge.json",
        1,
        "hard",
        "hard_sam_old_merge.png",
    ],
    [
        "hard_manual_vs_hard_sam_c0_in_sam_space_merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_merge.json",
        1,
        "hard",
        "hard_sam_c0_merge.png",
    ],
    [
        "hard_manual_vs_hard_sam_c1_in_sam_space_merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_merge.json",
        1,
        "hard",
        "hard_sam_c1_merge.png",
    ],
    [
        "hard_manual_vs_hard_sam_c2_in_sam_space_merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_merge.json",
        1,
        "hard",
        "hard_sam_c2_merge.png",
    ],
    [
        "soft_manual_vs_soft_sam_in_sam_space_merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_merge.json",
        1,
        "soft",
        "soft_sam_old_merge.png",
    ],
    [
        "soft_manual_vs_soft_sam_c0_in_sam_space_merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_merge.json",
        1,
        "soft",
        "soft_sam_c0_merge.png",
    ],
    [
        "soft_manual_vs_soft_sam_c1_in_sam_space_merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_merge.json",
        1,
        "soft",
        "soft_sam_c1_merge.png",
    ],
    [
        "soft_manual_vs_soft_sam_c2_in_sam_space_merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_merge.json",
        1,
        "soft",
        "soft_sam_c2_merge.png",
    ],
]
