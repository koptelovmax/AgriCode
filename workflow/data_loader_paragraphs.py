import csv
import json

import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from collections import Counter
#%%
# Load list of files:
f = open("data/ecozept/filelist.txt", "r")

fileNames = []
for line in f:
    try:
        fileNames.append(line.strip())
                
    except ValueError:
        print('Invalid input:',line)

f.close()
#%%
# Multiple file loader (multiple codes per each paragraph):
segments_all = []
paragraphs_all = []

for f_name in fileNames:
    
    paragraphs = []
    f = open("data/ecozept/paragraphs/"+f_name[15:]+".csv", "r")
    
    title = ''
    block = ''
    question = ''
    tab = csv.DictReader(f,delimiter=';')
    for line in tab:
        if line["main title"] !='':
            title = line["main title"]
        if line["block name"] != '':
            block = line["block name"]
        if line["question"] != '':
            question = line["question"]
        paragraphs.append([f_name,title,block,question,line["paragraph"],[]])

    f.close()
    
    
    f = open("data/ecozept/"+f_name+".csv", "r")
    
    tab = csv.DictReader(f,delimiter=';')
    error_list = []
    for line in tab:
        segm_text = line["Segment"].replace('\n',' ')
        segments_all.append([f_name,segm_text,line["Code"].strip()])
        flag = False
        for i in range(len(paragraphs)):
            if segm_text.strip() in paragraphs[i][4]:
                if not flag:
                    if paragraphs[i][5] == []:
                        paragraphs[i][5] = [line["Code"].strip()]
                    elif line["Code"].strip() not in paragraphs[i][5]:
                        paragraphs[i][5].append(line["Code"].strip())
                    flag = True
                else:
                    print('Error! Matching second time',segm_text)
        if not flag:
            print('Error! Matching not found:', f_name, segm_text)
            error_list.append(segm_text)

    f.close() 

    paragraphs_all += paragraphs
#%%
# NEW STRATEGY: ignore overlapping classes+threat 'company' and 'experts' as the same class:

# First level codes only (remove dublicates):
para_all_one_level = []
for i in range(len(paragraphs_all)):
    if len(paragraphs_all[i][5]) > 0:
        # get 1st level code only:
        para_codes = []
        for item in paragraphs_all[i][5]:
            item_code = item.split('>')[0].strip()
            if (item_code == 'company') or (item_code == 'Experts'):
                item_code = 'company+Experts'
            if item_code not in para_codes:
                para_codes.append(item_code)
        if len(para_codes) == 1:
            para_all_one_level.append([paragraphs_all[i][0],paragraphs_all[i][1],paragraphs_all[i][2],paragraphs_all[i][3],paragraphs_all[i][4],para_codes[0]])
    else:
        para_all_one_level.append([paragraphs_all[i][0],paragraphs_all[i][1],paragraphs_all[i][2],paragraphs_all[i][3],paragraphs_all[i][4],'Other (not pertinent)'])
#%%
Counter([item[5] for item in para_all_one_level])
#Counter({'Other (not pertinent)': 126,
#         'limitations and barriers': 102,
#         'Stakeholders’ expectations': 98,
#         'market opportunities': 96,
#         'valorization': 54,
#         'company+Experts': 30,
#         'type of stream': 25})
#%%
# Save the result to CSV:
f_out = open('data/ecozept_processed/paragraphs_7-classes.csv', 'w', newline='', encoding='UTF8')

tab = csv.writer(f_out,delimiter=';')

header = ['Document','Main title','Block name','Question','Paragraph','Code']
tab.writerow(header)

for row in para_all_one_level:
    tab.writerow(row)

f_out.close()
#%
# Save data into JSON format:
data_dict = {}

for i in range(len(para_all_one_level)):
    data_dict[i] = {
            'Document': para_all_one_level[i][0],
            'Main title': para_all_one_level[i][1],
            'Block name': para_all_one_level[i][2],
            'Question': para_all_one_level[i][3],
            'Paragraph': para_all_one_level[i][4],
            'Code': para_all_one_level[i][5]
            }

with open("data/ecozept_processed/paragraphs_7-classes.json", "w") as outfile:
    json.dump(data_dict, outfile)
#%%
# Split data into train and test:
    
# Fix random seed to make results reproducible:
np.random.seed(78)

# Split data into train and test sets:
val_ratio = 0.2

true_codes = [item[5] for item in para_all_one_level]

# Indices of the train and validation splits stratified by labels:
train_index, test_index = train_test_split(
    np.arange(len(para_all_one_level)),
    test_size = val_ratio,
    shuffle = True,
    stratify = true_codes)

data = []
for i in range(len(para_all_one_level)):
    if true_codes[i] == 'Other (not pertinent)':
        label = 0
    elif true_codes[i] == 'limitations and barriers':
        label = 1
    elif true_codes[i] == 'Stakeholders’ expectations':
        label = 2
    elif true_codes[i] == 'market opportunities':
        label = 3
    elif true_codes[i] == 'valorization':
        label = 4
    elif true_codes[i] == 'company+Experts':
        label = 5
    elif true_codes[i] == 'type of stream':
        label = 6
    else:
        print("Error!",true_codes[i])
    
    data.append([para_all_one_level[i][0],para_all_one_level[i][4],label])
#%%
# Check some statistics:
print('\nNumber of examples in train and test sets:', len(train_index), len(test_index))
print('Class 0 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==0 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==0 and (i in test_index)]))
print('Class 1 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==1 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==1 and (i in test_index)]))
print('Class 2 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==2 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==2 and (i in test_index)]))
print('Class 3 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==3 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==3 and (i in test_index)]))
print('Class 4 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==4 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==4 and (i in test_index)]))
print('Class 5 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==5 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==5 and (i in test_index)]))
print('Class 6 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==6 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==6 and (i in test_index)]))
# Number of examples in train and test sets: 424 107
# Class 0 in train and test sets: 101 25
# Class 1 in train and test sets: 81 21
# Class 2 in train and test sets: 78 20
# Class 3 in train and test sets: 77 19
# Class 4 in train and test sets: 43 11
# Class 5 in train and test sets: 24 6
# Class 6 in train and test sets: 20 5
#%%
# Save results:
data_test = [data[i] for i in range(len(data)) if i in test_index]
data_train = [data[i] for i in range(len(data)) if i in train_index]

# Save test set as a binary file:
f_out = open("data/7-classes/test_set_paragraphs.pkl","wb")
pickle.dump(data_test,f_out)
f_out.close()

# Save training set as a binary file:
f_out = open("data/7-classes/train_set_paragraphs.pkl","wb")
pickle.dump(data_train,f_out)
f_out.close()
#%%
# Multiple codes per each paragraph -> one code per each paragraph + first and second level codes:
para_all_two_levels = []
for i in range(len(paragraphs_all)):
    if len(paragraphs_all[i][5]) > 0:
        # get 1st and 2nd level codes:
        para_codes = []
        for item in paragraphs_all[i][5]:
            code_levels = item.split('>')
            if len(code_levels) == 1:
                item_code = code_levels[0].strip()
            elif len(code_levels) > 1:
                item_code = code_levels[0].strip()+" > "+code_levels[1].strip()
            # some code preprocessing:
            if item_code == 'valorization':
                item_code = 'valorization > current structures'
            elif 'type of stream' in item_code:
                item_code = 'type of stream'
            elif 'company' in item_code:
                item_code = 'company'
            if (item_code == 'company') or (item_code == 'Experts'):
                item_code = 'company+Experts'
            # memorise the code:
            if item_code not in para_codes:
                para_codes.append(item_code)
        # memorise results:
        if len(para_codes) == 1:
            para_all_two_levels.append([paragraphs_all[i][0],paragraphs_all[i][1],paragraphs_all[i][2],paragraphs_all[i][3],paragraphs_all[i][4],para_codes[0]])
    else:
        para_all_two_levels.append([paragraphs_all[i][0],paragraphs_all[i][1],paragraphs_all[i][2],paragraphs_all[i][3],paragraphs_all[i][4],'Other (not pertinent)'])
#%%
Counter([item[5] for item in para_all_two_levels])
#Counter({'Other (not pertinent)': 126,
#         'Stakeholders’ expectations > valorization/ PHA-applications': 65,
#         'limitations and barriers > valorization /PHA-applications': 59,
#         'market opportunities > PHA MO': 45,
#         'market opportunities > PHA-Applications MO': 35,
#         'valorization > current structures': 32,
#         'company+Experts': 30,
#         'limitations and barriers > Main issues and challenges for extracted/microbial protein': 26,
#         'type of stream': 25,
#         'Stakeholders’ expectations > PHA expectation': 22,
#         'limitations and barriers > Main issues and challenges for PHA': 17,
#         'market opportunities > MP MO': 16,
#         'Stakeholders’ expectations > MP': 11,
#         'valorization > satisfaction': 10,
#         'valorization > advantages': 9})
#%%
# Save the result to CSV:
f_out = open('data/ecozept_processed/paragraphs_15-classes.csv', 'w', newline='', encoding='UTF8')

tab = csv.writer(f_out,delimiter=';')

header = ['Document','Main title','Block name','Question','Paragraph','Code']
tab.writerow(header)

for row in para_all_two_levels:
    tab.writerow(row)

f_out.close()
#%
# Save data into JSON format:
data_dict = {}

for i in range(len(para_all_two_levels)):
    data_dict[i] = {
            'Document': para_all_one_level[i][0],
            'Main title': para_all_one_level[i][1],
            'Block name': para_all_one_level[i][2],
            'Question': para_all_one_level[i][3],
            'Paragraph': para_all_one_level[i][4],
            'Code': para_all_one_level[i][5]
            }

with open("data/ecozept_processed/paragraphs_15-classes.json", "w") as outfile:
    json.dump(data_dict, outfile)
#%%
# Split data into train and test:
    
# Fix random seed to make results reproducible:
np.random.seed(78)

# Split data into train and test sets:
val_ratio = 0.2

true_codes = [item[5] for item in para_all_two_levels]

# Indices of the train and validation splits stratified by labels:
train_index, test_index = train_test_split(
    np.arange(len(para_all_two_levels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = true_codes)

data = []
for i in range(len(para_all_two_levels)):
    if true_codes[i] == 'Other (not pertinent)':
        label = 0
    elif true_codes[i] == 'Stakeholders’ expectations > valorization/ PHA-applications':
        label = 1
    elif true_codes[i] == 'limitations and barriers > valorization /PHA-applications':
        label = 2
    elif true_codes[i] == 'market opportunities > PHA MO':
        label = 3
    elif true_codes[i] == 'market opportunities > PHA-Applications MO':
        label = 4
    elif true_codes[i] == 'valorization > current structures':
        label = 5
    elif true_codes[i] == 'company+Experts':
        label = 6
    elif true_codes[i] == 'limitations and barriers > Main issues and challenges for extracted/microbial protein':
        label = 7
    elif true_codes[i] == 'type of stream':
        label = 8
    elif true_codes[i] == 'Stakeholders’ expectations > PHA expectation':
        label = 9
    elif true_codes[i] == 'limitations and barriers > Main issues and challenges for PHA':
        label = 10
    elif true_codes[i] == 'market opportunities > MP MO':
        label = 11
    elif true_codes[i] == 'Stakeholders’ expectations > MP':
        label = 12
    elif true_codes[i] == 'valorization > satisfaction':
        label = 13
    elif true_codes[i] == 'valorization > advantages':
        label = 14
    else:
        print("Error!",true_codes[i])
    
    data.append([para_all_two_levels[i][0],para_all_two_levels[i][4],label])
#%%
# Check some statistics:
print('\nNumber of examples in train and test sets:', len(train_index), len(test_index))
print('Class 0 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==0 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==0 and (i in test_index)]))
print('Class 1 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==1 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==1 and (i in test_index)]))
print('Class 2 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==2 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==2 and (i in test_index)]))
print('Class 3 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==3 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==3 and (i in test_index)]))
print('Class 4 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==4 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==4 and (i in test_index)]))
print('Class 5 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==5 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==5 and (i in test_index)]))
print('Class 6 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==6 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==6 and (i in test_index)]))
print('Class 7 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==7 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==7 and (i in test_index)]))
print('Class 8 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==8 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==8 and (i in test_index)]))
print('Class 9 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==9 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==9 and (i in test_index)]))
print('Class 10 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==10 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==10 and (i in test_index)]))
print('Class 11 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==11 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==11 and (i in test_index)]))
print('Class 12 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==12 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==12 and (i in test_index)]))
print('Class 13 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==13 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==13 and (i in test_index)]))
print('Class 14 in train and test sets:', len([i for i in range(len(data)) if data[i][2]==14 and (i in train_index)]), len([i for i in range(len(data)) if data[i][2]==14 and (i in test_index)]))
# Number of examples in train and test sets: 422 106
# Class 0 in train and test sets: 101 25
# Class 1 in train and test sets: 52 13
# Class 2 in train and test sets: 47 12
# Class 3 in train and test sets: 36 9
# Class 4 in train and test sets: 28 7
# Class 5 in train and test sets: 25 7
# Class 6 in train and test sets: 24 6
# Class 7 in train and test sets: 21 5
# Class 8 in train and test sets: 20 5
# Class 9 in train and test sets: 17 5
# Class 10 in train and test sets: 14 3
# Class 11 in train and test sets: 13 3
# Class 12 in train and test sets: 9 2
# Class 13 in train and test sets: 8 2
# Class 14 in train and test sets: 7 2
#%%
# Save results:
data_test = [data[i] for i in range(len(data)) if i in test_index]
data_train = [data[i] for i in range(len(data)) if i in train_index]

# Save test set as a binary file:
f_out = open("data/15-classes/test_set_paragraphs.pkl","wb")
pickle.dump(data_test,f_out)
f_out.close()

# Save training set as a binary file:
f_out = open("data/15-classes/train_set_paragraphs.pkl","wb")
pickle.dump(data_train,f_out)
f_out.close()
#%%
