import os
import shutil

def _get_positive_src_file_list(lines, solution, smell_type,
                           code_split_out_folder_class, code_split_out_folder_method):
    pos_src_file_list = list()

    for line in lines:
        tokens = line.split(",")
        if smell_type == "Design":
            file = os.path.join(os.path.join(os.path.join(os.path.join(
                os.path.join(code_split_out_folder_class, solution), solution), str(_filter_str(tokens[1]))),
                str(_filter_str(tokens[2])).replace('<', '').replace('>', '') + ".code"))
        else:
            file = os.path.join(os.path.join(os.path.join(
                os.path.join(os.path.join(code_split_out_folder_method, solution), solution),
                                    str(_filter_str(tokens[1]))),
                                str(_filter_str(tokens[2])).replace('<', '').replace('>', '')),
                str(_filter_str(tokens[3])) + ".code")
        if os.path.exists(file):
            if file not in pos_src_file_list:
                pos_src_file_list.append(file)
        else:
            print("Path doesn't exists: " + str(file))
    return pos_src_file_list


def _put_files_in_right_bucket(pos_source_file_list, solution, positive_cases_folder, negative_cases_folder,
                               smell_type, code_split_out_folder_class, code_split_out_folder_method):
    if not os.path.exists(positive_cases_folder):
        os.makedirs(positive_cases_folder)
    if not os.path.exists(negative_cases_folder):
        os.makedirs(negative_cases_folder)

    total_files_copied = 0
    base_folder_path = os.path.join(code_split_out_folder_class, solution) if smell_type == "Design" \
        else os.path.join(code_split_out_folder_method, solution)

    pos_counter = 0
    neg_couter = 0
    for root, dirs, files in os.walk(base_folder_path):
        for file in files:
            src_file_path = os.path.join(root, file)

            if smell_type == "Design":
                namespace = root.replace(code_split_out_folder_class + os.path.sep, "").replace(os.path.sep, "_")
            else:
                namespace = root.replace(code_split_out_folder_method + os.path.sep, "").replace(os.path.sep, "_")

            if _is_present(pos_source_file_list, src_file_path):

                dest_file_path = os.path.join(positive_cases_folder, namespace + str(pos_counter) + file)
                if not os.path.exists(dest_file_path):
                    try:
                        shutil.copyfile(src_file_path, dest_file_path)
                    except:
                        pass
                    total_files_copied += 1
                    pos_counter += 1
                else:
                    print("File already exists: " + str(dest_file_path))  # This should not be the case
            else:
                dest_file_path = os.path.join(negative_cases_folder, namespace + str(neg_couter) + file)
                if not os.path.exists(dest_file_path):
                    try:
                        shutil.copyfile(src_file_path, dest_file_path)
                    except:
                        pass
                    total_files_copied += 1
                    neg_couter += 1
                else:
                    print("File already exists: " + str(dest_file_path))  # This should not be the case
    return total_files_copied


def _is_present(an_list, item):
    if item in an_list:
        return True

    item_upper = item.upper()
    for obj in an_list:
        if obj.upper() == item_upper:
            return True
    return False


def _filter_str(token):
    #line = token.decode('utf-8', 'ignore').encode("utf-8")
    line = bytes(token, 'utf-8').decode('utf-8', 'ignore')
    return line


def _scan_solution(solution, positive_cases_folder, negative_cases_folder,
                   smell_name_str, smell_type, smells_results_folder,
                           code_split_out_folder_class, code_split_out_folder_method):
    print("Processing solution: " + solution)
    if smell_type == "Impl":
        solution_folder = os.path.join(code_split_out_folder_method, solution)
    else:
        solution_folder = os.path.join(code_split_out_folder_class, solution)

    if os.path.exists(solution_folder):
        total_file_count = sum([len(files) for r, d, files in os.walk(solution_folder)])
        if total_file_count == 0:
            return
    else:
        return

    pos_source_file_list = list()
    for root, dirs, files in os.walk(os.path.join(smells_results_folder, solution)):
        for file in files:
            if smell_type == "Design":
                if not file.endswith("DesignSmells.csv"):
                    continue
            else:
                if not file.endswith("ImplementationSmells.csv"):
                    continue
            print("Processing file: " + os.path.join(root, file))
            lines = []
            with open(os.path.join(root, file), encoding="utf8", errors='ignore') as fp:
                for line in fp:
                    if smell_name_str in line:
                        lines.append(line)

            pos_source_files = _get_positive_src_file_list(lines, solution, smell_type,
                           code_split_out_folder_class, code_split_out_folder_method)
            pos_source_file_list.extend(pos_source_files)
    total_copied_files = _put_files_in_right_bucket(pos_source_file_list, solution, positive_cases_folder, negative_cases_folder,
                               smell_type, code_split_out_folder_class, code_split_out_folder_method)
    # Sanity check
    assert (total_file_count == total_copied_files)


def generate_data(smells_results_folder, code_split_out_folder_class, code_split_out_folder_method, learning_data_folder_path):
    SMELL_NAME_LIST = ["ComplexConditional", "ComplexMethod", "MultifacetedAbstraction", "FeatureEnvy"]
    SMELL_NAME_STR_LIST = ["Complex Conditional", "Complex Method", "Multifaceted Abstraction", "Feature Envy"]
    SMELL_TYPE_LIST = ["Impl", "Impl", "Design", "Design"]

    if not os.path.exists(learning_data_folder_path):
        os.makedirs(learning_data_folder_path)
    for smell in range(len(SMELL_NAME_LIST)):
        print("Generating samples for {0} smell...".format(SMELL_NAME_LIST[smell]))
        positive_cases_folder = os.path.join(learning_data_folder_path, SMELL_NAME_LIST[smell], "Positive")
        negative_cases_folder = os.path.join(learning_data_folder_path, SMELL_NAME_LIST[smell], "Negative")

        for solution in os.listdir(smells_results_folder):
            _scan_solution(solution, positive_cases_folder, negative_cases_folder,
                           SMELL_NAME_STR_LIST[smell], SMELL_TYPE_LIST[smell], smells_results_folder,
                           code_split_out_folder_class, code_split_out_folder_method)
