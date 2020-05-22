import os
import subprocess

def _get_solution_file(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".sln"):
                return os.path.join(root, file)
    return None


def _run_code_split(folder_name, folder_path, code_split_result_folder, code_split_exe_path, code_split_mode):
    sln_file = _get_solution_file(folder_path)
    if (sln_file != None):
        out_folder = os.path.join(code_split_result_folder, folder_name)
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
            subprocess.call([code_split_exe_path, sln_file, code_split_mode, out_folder])


def cs_code_split(repo_base_folder, code_split_result_folder, code_split_mode, code_split_exe_path):
    if not os.path.exists(code_split_result_folder):
        os.makedirs(code_split_result_folder)
    for dir in os.listdir(repo_base_folder):
        print("Processing " + dir)
        if os.path.exists(os.path.join(code_split_result_folder, dir)):
            print ("\t.. skipping.")
        else:
            _run_code_split(dir, os.path.join(repo_base_folder, dir),
                            code_split_result_folder, code_split_exe_path, code_split_mode)
    print("Done.")
