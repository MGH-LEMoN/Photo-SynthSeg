import glob
import os

import scripts.config as config

NUM_COPIES = 50


def create_destination_name(source, idx):
    """Create destination file names given source

    Args:
        source (string): source file name 
        idx (int): integer to append (while creating a copy)

    Returns:
        string: destination file name
    """
    base_dir, file_name_with_ext = os.path.split(source)
    file_name, ext = file_name_with_ext.split(os.extsep, 1)

    file_name = '_'.join([file_name, 'copy', f'{idx:02}'])
    new_file_name = os.extsep.join([file_name, ext])

    return os.path.join(base_dir, new_file_name)


def create_symlink(source, num_copies=10):
    """Create num_copies number of symlinks for source

    Args:
        source (string): file to be copied
        num_copies (int, optional): number of copies to create. Defaults to 10.
    """
    list(
        map(lambda x: os.symlink(source, create_destination_name(source, x)),
            range(1, num_copies)))

    return


def main():
    subject_files = sorted(
        glob.glob(os.path.join(config.LABEL_MAPS_DIR, 'subject*.nii.gz')))

    for file in subject_files:
        print(f'Creating copies of {os.path.split(file)[-1]}')
        create_symlink(file, NUM_COPIES)

    return


if __name__ == '__main__':
    main()
