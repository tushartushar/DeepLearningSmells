import os
import subprocess
from subprocess import Popen, PIPE

# java -jar Designite.jar -i <path of the input source folder> -o <path of the output folder>

def _run_designite_java(folder_name, folder_path, designiteJava_jar_path, smells_results_folder):
    out_folder = os.path.join(smells_results_folder, folder_name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    # logfile = os.path.join(out_folder, "log.txt")
    proc = Popen(["java", "-jar", designiteJava_jar_path, "-i", folder_path, "-o", out_folder])
    proc.wait()

def analyze_repositories(repo_source_folder, smells_results_folder, designiteJava_jar_path):
    for dir in os.listdir(repo_source_folder):
        print("Processing " + dir)
        if os.path.exists(os.path.join(smells_results_folder, dir)):
            print ("\t.. skipping.")
        else:
            _run_designite_java(dir, os.path.join(repo_source_folder, dir), designiteJava_jar_path, smells_results_folder)
    print("Done.")