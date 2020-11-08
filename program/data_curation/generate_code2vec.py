# This program takes positive and negative samples generated from learning data generation
# and put it through code2vec to obtain trained vectors for smell analysis
# We put 1/3rd of each positive and negative set of samples to train, validate, and test folders
# code2vec generate trained vectors only for validate samples so we rotate the 1/3rd of samples in each iteration

# --- imports ---
import os
import shutil
import subprocess
from subprocess import call

# -- Parameters --
DATA_BASE_PATH = r'D:\research\smellDetectionML\data'
SAMPLE_BASE = DATA_BASE_PATH + r'\training_data'
C2V_OUT_BASE = DATA_BASE_PATH + r'\c2v_vectors'
TRAIN_DIR = r'D:\research\smellDetectionML\code2vec_data\train'
VAL_DIR = r'D:\research\smellDetectionML\code2vec_data\validate'
TEST_DIR = r'D:\research\smellDetectionML\code2vec_data\test'
VECTOR_FILE = r'D:\research\smellDetectionML\code2vec\data\cs_dataset\cs_dataset.val.c2v.vectors'
C2V_FILE = r'D:\research\smellDetectionML\code2vec\data\cs_dataset\cs_dataset.val.c2v'
C2V_BASE = r'D:\research\smellDetectionML\code2vec'

def run_c2v():
    print('Running preprocess')
    # preprocess_file = os.path.join(C2V_BASE, 'preprocess_csharp.sh')
    # bash_exe = r'bash.exe'
    # result_code = call([bash_exe, preprocess_file])
    # print('Complete running preprocess with result: ' + str(result_code))
    # print('Running train')
    # train_file = os.path.join(C2V_BASE, 'train.sh')
    # result_code = call([bash_exe, train_file])
    # print('Complete running train with result: ' + str(result_code))


def get_filename(folder):
    i = 1
    while True:
        filepath = os.path.join(folder, 'samples' + str(i) + '.vec')
        if not os.path.exists(filepath):
            return filepath
        i = i + 1


def check_filesize(folder, filename):
    if os.path.getsize(filename) > 52428800:
        return get_filename(folder)
    return filename


def write_file(filename, list):
    with open(filename, 'a') as file:
        file.writelines(list)


def segregate_vectors(dest_folder):
    print('start segregate_vectors')
    pos_dest_folder = os.path.join(dest_folder, "Positive")
    if not os.path.exists(pos_dest_folder):
        os.makedirs(pos_dest_folder)
    neg_dest_folder = os.path.join(dest_folder, "Negative")
    if not os.path.exists(neg_dest_folder):
        os.makedirs(neg_dest_folder)

    with open(C2V_FILE, "r") as file:
        all_lines = file.readlines()
    num_lines = len(all_lines)
    print('Total num lines: ' + str(num_lines))
    with open(VECTOR_FILE, "r") as file:
        all_vectors = file.readlines()

    assert (num_lines, len(all_vectors))

    neg_list = []
    pos_list = []
    pos_filename = get_filename(pos_dest_folder)
    neg_filename = get_filename(neg_dest_folder)
    for line in range(num_lines):
        cur_line = all_lines[line]
        cur_vector = all_vectors[line]
        tokens = cur_line.split()
        if len(tokens) > 0:
            if tokens[0] == 'False':
                neg_list.append(cur_vector)
            if tokens[0] == 'True':
                pos_list.append(cur_vector)
            if len(neg_list) > 100:
                print('Writing to neg vector file')
                write_file(neg_filename, neg_list)
                neg_list = []
                neg_filename = check_filesize(neg_dest_folder, neg_filename)
            if len(pos_list) > 100:
                print('Writing to pos vector file')
                write_file(pos_filename, pos_list)
                pos_list = []
                pos_filename = check_filesize(neg_dest_folder, pos_filename)

    if len(neg_list) > 0:
        write_file(neg_filename, neg_list)
    if len(pos_list) > 0:
        write_file(pos_filename, pos_list)
    print('End segregate_vectors')


def rename_folders_rotate(folder1, folder2, folder3):
    temp_folder = folder1 + 'temp'
    os.rename(folder1, temp_folder)
    os.rename(folder2, folder1)
    os.rename(folder3, folder2)
    os.rename(temp_folder, folder3)


def generate_c2v():
    for sample_type in ["Positive", "Negative"]:
        cur_src_folder = os.path.join(src_folder, sample_type)
        train_dest_folder = os.path.join(TRAIN_DIR, sample_type)
        val_dest_folder = os.path.join(VAL_DIR, sample_type)
        test_dest_folder = os.path.join(TEST_DIR, sample_type)
        # empty_folder(train_dest_folder)
        # empty_folder(test_dest_folder)
        # empty_folder(val_dest_folder)
        #
        # copy_samples(cur_src_folder, test_dest_folder, train_dest_folder, val_dest_folder)
    # run_c2v()
    segregate_vectors(dest_folder)

    rename_folders_rotate(TEST_DIR, TRAIN_DIR, VAL_DIR)
    run_c2v()
    segregate_vectors(dest_folder)

    rename_folders_rotate(TEST_DIR, TRAIN_DIR, VAL_DIR)
    run_c2v()
    segregate_vectors(dest_folder)


def copy_samples(cur_src_folder, test_dest_folder, train_dest_folder, val_dest_folder):
    print('Copying from ' + cur_src_folder)
    total_samples = len(os.listdir(cur_src_folder))
    samples = os.listdir(cur_src_folder)
    print('Copying training samples...')
    for i in range(int(total_samples / 3)):
        shutil.copy(os.path.join(cur_src_folder, samples[i]), train_dest_folder)
    print('Copying test samples...')
    for i in range(int(total_samples / 3), int((total_samples * 2) / 3)):
        shutil.copy(os.path.join(cur_src_folder, samples[i]), test_dest_folder)
    print('Copying val samples...')
    for i in range(int((total_samples * 2) / 3), total_samples):
        shutil.copy(os.path.join(cur_src_folder, samples[i]), val_dest_folder)
    print('Copying done.')


# def empty_folder(folder_path):
#     if os.path.exists(folder_path) and os.path.isdir(folder_path):
#         print('Deleting ' + folder_path)
#         shutil.rmtree(folder_path, ignore_errors=True)
#     os.makedirs(folder_path)


if __name__ == "__main__":
    # list = ["ComplexConditional", "ComplexMethod", "MultifacetedAbstraction", "FeatureEnvy"]
    smells_list = ["ComplexConditional", "ComplexMethod"]
    for smell in smells_list:
        src_folder = os.path.join(SAMPLE_BASE, smell)
        dest_folder = os.path.join(C2V_OUT_BASE, smell)
        # empty_folder(dest_folder)
        generate_c2v()

