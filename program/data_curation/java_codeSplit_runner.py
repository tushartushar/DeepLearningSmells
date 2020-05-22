import os
import subprocess

def _run_code_split(folder_name, folder_path, code_split_result_folder, code_split_exe_path, code_split_mode):
    out_folder = os.path.join(code_split_result_folder, folder_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    subprocess.call(["java", "-jar", code_split_exe_path,
                     "-i", folder_path, "-o", out_folder, "-m", code_split_mode])

def java_code_split(repo_source_folder, code_split_mode, code_split_result_folder, code_split_exe_path):
    assert code_split_mode == "method" or code_split_mode == "class"

    for dir in os.listdir(repo_source_folder):
        print("Processing " + dir)
        if os.path.exists(os.path.join(code_split_result_folder, dir)):
            print ("\t.. skipping.")
        else:
            _run_code_split(dir, os.path.join(repo_source_folder, dir),
                            code_split_result_folder, code_split_exe_path, code_split_mode)
    print("Done.")