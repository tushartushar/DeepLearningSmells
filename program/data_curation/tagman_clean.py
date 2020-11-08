'''
This script takes an exported annotation file (csv) in the following form
 Filename	User	entity	annotation
 m187.cs	tushar	cc	FALSE
and prepares another file with following form
 Filename cc cm [cc] [cm]
 m187.cs FALSE FALSE 
'''

INPUT_FILE = r'/Users/Tushar/Documents/Research/smellDetectionML/data/manual annotations/exported.csv'
OUTPUT_FILE = r'/Users/Tushar/Documents/Research/smellDetectionML/data/manual annotations/exported_clean.csv'

tag_dict = dict() # key: filename, value: dict
with open(INPUT_FILE, 'r', errors='ignore') as reader:
    with open(OUTPUT_FILE, 'w') as writer:
        writer.write('Filename,CC,CM,CC,CM\n')
        is_header = True
        for line in reader.readlines():
            if is_header:
                is_header = False
                continue
            tokens = line.split(',')
            if len(tokens) > 3:
                if not tokens[0] in tag_dict:
                    smell_dict = dict()
                    smell_dict[tokens[2]] = [tokens[3].strip()]
                    tag_dict[tokens[0]] = smell_dict
                else:
                    smell_dict = tag_dict[tokens[0]]
                    if tokens[2] in smell_dict:
                        smell_dict[tokens[2]].append(tokens[3].strip())
                    else:
                        smell_dict[tokens[2]] = [tokens[3].strip()]
        for key,value in tag_dict.items():
            line = key
            cc_list = value['cc'] if 'cc' in value else list()
            cm_list = value['cm'] if 'cm' in value else list()
            line += ',' + cc_list[0] if len(cc_list)>0 else ''
            line += ',' + cm_list[0] if len(cm_list)>0 else ''
            line += ',' + cc_list[1] if len(cc_list) > 1 else ''
            line += ',' + cm_list[1] if len(cm_list) > 1 else ''
            line += '\n'
            writer.write(line)
