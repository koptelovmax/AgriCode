import csv
import json

import numpy as np
import pickle

from sklearn.model_selection import GroupShuffleSplit
from nltk.tokenize import sent_tokenize
from collections import Counter

def get_percentage(x,y):
    
    return str(np.round(x*100/(x+y),2))+'%'
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
# II. Sentences
#
# 2 versions:
#   a) one sentence per each segment
#   b) up to 3 sentences (including previous and following) for each segment
#
# Sentences with codes: b) up to 3 sentences (including previous and following) for each segment
#
sentences_all = []
par_id = 0
    
for f_name in fileNames:
    # Load paragraphs:
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
        paragraphs.append([f_name,title,block,question,line["paragraph"]])
    f.close()
    
    # Load segments:
    segm_sents_all = []
    segm_sents_txt = []
    f = open("data/ecozept/"+f_name+".csv", "r")
    tab = csv.DictReader(f,delimiter=';')
    for line in tab:
        segm_text = line["Segment"].replace('\n',' ')
        segm_sents = sent_tokenize(segm_text)
        for j in range(len(segm_sents)):
            segm_sents_all.append([segm_sents[j],line["Code"].strip()])
            segm_sents_txt.append(segm_sents[j])  
    f.close()
    
    # Separate paragraphs to sentences:
    for i in range(len(paragraphs)):
        para_sents = sent_tokenize(paragraphs[i][4])
        # Label sentences by codes:
        for j in range(len(para_sents)):
            # Determine code of current sentence:
            if para_sents[j] in segm_sents_txt:
                # Determine sentence index:
                sent_indx = segm_sents_txt.index(para_sents[j])
                sent_code = segm_sents_all[sent_indx][1]
            else:
                sent_code = 'Other (not pertinent)'
            # Add previous and next sentences if they have same code:
            if j==0: # the 1st sentence of paragraph
                if len(para_sents)>1:
                    # determine code of next sentence:
                    if para_sents[j+1] in segm_sents_txt:
                        next_sent_indx = segm_sents_txt.index(para_sents[j+1])
                        next_sent_code = segm_sents_all[next_sent_indx][1]
                    else:
                        next_sent_code = 'Other (not pertinent)'
                    # compare code of next sentence with current sentence:
                    if sent_code == next_sent_code:
                        # add current and next sentences:
                        segment = para_sents[j]+' '+para_sents[j+1]
                    else:
                        # add current sentence only:
                        segment = para_sents[j]
                else:
                    # add current sentence only:
                    segment = para_sents[j]
            elif j<len(para_sents)-1: # i-th sentence of paragraph
                # determine code of previous sentence:
                if para_sents[j-1] in segm_sents_txt:
                    prev_sent_indx = segm_sents_txt.index(para_sents[j-1])
                    prev_sent_code = segm_sents_all[prev_sent_indx][1]
                else:
                    prev_sent_code = 'Other (not pertinent)'
                # determine code of next sentence:
                if para_sents[j+1] in segm_sents_txt:
                    next_sent_indx = segm_sents_txt.index(para_sents[j+1])
                    next_sent_code = segm_sents_all[next_sent_indx][1]
                else:
                    next_sent_code = 'Other (not pertinent)'
                # compare code of previous sentence with current sentence:
                if prev_sent_code == sent_code:
                    # compare code of next sentence with current sentence:
                    if sent_code == next_sent_code:
                        # add previous, current and next sentences:
                        segment = para_sents[j-1]+' '+para_sents[j]+' '+para_sents[j+1]
                    else:
                        # add previous and current sentences:
                        segment = para_sents[j-1]+' '+para_sents[j]
                else:
                    # compare code of next sentence with current sentence:
                    if sent_code == next_sent_code:
                        # add current and next sentences:
                        segment = para_sents[j]+' '+para_sents[j+1]
                    else:
                        # add current sentence only:
                        segment = para_sents[j]
            else: # last sentence of paragraph
                # determine code of previous sentence:
                if para_sents[j-1] in segm_sents_txt:
                    prev_sent_indx = segm_sents_txt.index(para_sents[j-1])
                    prev_sent_code = segm_sents_all[prev_sent_indx][1]
                else:
                    prev_sent_code = 'Other (not pertinent)'
                # compare code of previous sentence with current sentence:
                if prev_sent_code == sent_code:
                    # add previous and current sentences:
                    segment = para_sents[j-1]+' '+para_sents[j]
                else:
                    # add current sentence only:
                    segment = para_sents[j]
            
            # Memorise the result:    
            sentences_all.append([paragraphs[i][0],paragraphs[i][1],paragraphs[i][2],paragraphs[i][3],segment,sent_code,par_id])
            
        # update paragraph id:
        par_id+=1
#%%
# First level codes only+threat 'company' and 'experts' as the same class:
sentences_one_level = []
for i in range(len(sentences_all)):
    # get 1st level code:
    code = sentences_all[i][5].split('>')[0].strip()
    # threat 'company' and 'experts' as the same class:
    if (code == 'company') or (code == 'Experts'):
        code = 'company+Experts'
    # memorise the result:
    sentences_one_level.append([sentences_all[i][0],sentences_all[i][1],sentences_all[i][2],sentences_all[i][3],sentences_all[i][4],code,sentences_all[i][6]])
#%%
# Get some stats:
Counter([item[5] for item in sentences_one_level])
#Counter({'Other (not pertinent)': 784,
#         'market opportunities': 314,
#         'limitations and barriers': 296,
#         'Stakeholders’ expectations': 247,
#         'valorization': 124,
#         'company+Experts': 112,
#         'type of stream': 58})
#%%
# Save the result to CSV:
f_out = open('data/ecozept_processed/sentences_7-classes_(sentences).csv', 'w', newline='', encoding='UTF8')

tab = csv.writer(f_out,delimiter=';')

header = ['Document','Main title','Block name','Question','Paragraph','Code']
tab.writerow(header)

for row in sentences_one_level:
    tab.writerow(row)

f_out.close()
#%
# Save data into JSON format:
data_dict = {}

for i in range(len(sentences_one_level)):
    data_dict[i] = {
            'Document': sentences_one_level[i][0],
            'Main title': sentences_one_level[i][1],
            'Block name': sentences_one_level[i][2],
            'Question': sentences_one_level[i][3],
            'Paragraph': sentences_one_level[i][4],
            'Code': sentences_one_level[i][5]
            }

with open("data/ecozept_processed/sentences_7-classes_(sentences).json", "w") as outfile:
    json.dump(data_dict, outfile)
#%%
## Split data into train and test the way that train and test data comes from different paragraphs:
true_codes = [item[5] for item in sentences_one_level]
groups = [item[6] for item in sentences_one_level]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=585)

for _, (i, j) in enumerate(gss.split(np.arange(len(sentences_one_level)), true_codes, groups)):
    train_index = i
    test_index = j
    
data = []
for i in range(len(sentences_one_level)):
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
    data.append([sentences_one_level[i][0],sentences_one_level[i][1]+"\n\n"+sentences_one_level[i][2]+"\n\n"+sentences_one_level[i][3]+"\n\n"+sentences_one_level[i][4],label,sentences_one_level[i][6]])
#%%
# Check some statistics:
train_all = len([i for i in range(len(data)) if i in train_index])
test_all = len([i for i in range(len(data)) if i in test_index])
print('\nNumber of examples in train and test sets:', train_all, test_all, '/', get_percentage(train_all,test_all), get_percentage(test_all,train_all))

train_0 = len([i for i in range(len(data)) if data[i][2]==0 and (i in train_index)])
test_0 = len([i for i in range(len(data)) if data[i][2]==0 and (i in test_index)])
print('Class 0 in train and test sets:', train_0, test_0, '/', get_percentage(train_0,test_0), get_percentage(test_0,train_0))

train_1 = len([i for i in range(len(data)) if data[i][2]==1 and (i in train_index)])
test_1 = len([i for i in range(len(data)) if data[i][2]==1 and (i in test_index)])
print('Class 1 in train and test sets:', train_1, test_1, '/', get_percentage(train_1,test_1), get_percentage(test_1,train_1))

train_2 = len([i for i in range(len(data)) if data[i][2]==2 and (i in train_index)])
test_2 = len([i for i in range(len(data)) if data[i][2]==2 and (i in test_index)])
print('Class 2 in train and test sets:', train_2, test_2, '/', get_percentage(train_2,test_2), get_percentage(test_2,train_2))

train_3 = len([i for i in range(len(data)) if data[i][2]==3 and (i in train_index)])
test_3 = len([i for i in range(len(data)) if data[i][2]==3 and (i in test_index)])
print('Class 3 in train and test sets:', train_3, test_3, '/', get_percentage(train_3,test_3), get_percentage(test_3,train_3))

train_4 = len([i for i in range(len(data)) if data[i][2]==4 and (i in train_index)])
test_4 = len([i for i in range(len(data)) if data[i][2]==4 and (i in test_index)])
print('Class 4 in train and test sets:', train_4, test_4, '/', get_percentage(train_4,test_4), get_percentage(test_4,train_4))

train_5 = len([i for i in range(len(data)) if data[i][2]==5 and (i in train_index)])
test_5 = len([i for i in range(len(data)) if data[i][2]==5 and (i in test_index)])
print('Class 5 in train and test sets:', train_5, test_5, '/', get_percentage(train_5,test_5), get_percentage(test_5,train_5))

train_6 = len([i for i in range(len(data)) if data[i][2]==6 and (i in train_index)])
test_6 = len([i for i in range(len(data)) if data[i][2]==6 and (i in test_index)])
print('Class 6 in train and test sets:', train_6, test_6, '/', get_percentage(train_6,test_6), get_percentage(test_6,train_6))
#%
# Number of examples in train and test sets: 1554 381 / 80.31% 19.69%
# Class 0 in train and test sets: 637 147 / 81.25% 18.75%
# Class 1 in train and test sets: 235 61 / 79.39% 20.61%
# Class 2 in train and test sets: 194 53 / 78.54% 21.46%
# Class 3 in train and test sets: 257 57 / 81.85% 18.15%
# Class 4 in train and test sets: 98 26 / 79.03% 20.97%
# Class 5 in train and test sets: 86 26 / 76.79% 23.21%
# Class 6 in train and test sets: 47 11 / 81.03% 18.97%
#%%
# Save results:
data_test = [data[i] for i in range(len(data)) if i in test_index]
data_train = [data[i] for i in range(len(data)) if i in train_index]

# Save test set as a binary file:
f_out = open("data/7-classes/test_set_sentences.pkl","wb")
pickle.dump(data_test,f_out)
f_out.close()

# Save training set as a binary file:
f_out = open("data/7-classes/train_set_sentences.pkl","wb")
pickle.dump(data_train,f_out)
f_out.close()
#%%
# First and second level codes+threat 'company' and 'experts' as the same class:
sentences_two_levels = []
for i in range(len(sentences_all)):
    # get 1st and 2nd level codes:
    code_levels = sentences_all[i][5].split('>')
    if len(code_levels) == 1:
        code = code_levels[0].strip()
    elif len(code_levels) > 1:
        code = code_levels[0].strip()+" > "+code_levels[1].strip()
    # some code preprocessing:
    if code == 'valorization':
        code = 'valorization > current structures'
    elif 'type of stream' in code:
        code = 'type of stream'
    elif 'company' in code:
        code = 'company'
    if (code == 'company') or (code == 'Experts'):
        code = 'company+Experts'
    # memorise the result:
    sentences_two_levels.append([sentences_all[i][0],sentences_all[i][1],sentences_all[i][2],sentences_all[i][3],sentences_all[i][4],code,sentences_all[i][6]])
#%%
# Get some stats:
Counter([item[5] for item in sentences_two_levels])
#Counter({'Other (not pertinent)': 784,
#         'market opportunities > PHA MO': 151,
#         'limitations and barriers > Main issues and challenges for extracted/microbial protein': 120,
#         'Stakeholders’ expectations > valorization/ PHA-applications': 117,
#         'company+Experts': 112,
#         'limitations and barriers > valorization /PHA-applications': 102,
#         'market opportunities > PHA-Applications MO': 82,
#         'valorization > current structures': 81,
#         'market opportunities > MP MO': 81,
#         'limitations and barriers > Main issues and challenges for PHA': 74,
#         'Stakeholders’ expectations > PHA expectation': 74,
#         'type of stream': 58,
#         'Stakeholders’ expectations > MP': 56,
#         'valorization > satisfaction': 29,
#         'valorization > advantages': 14})
#%%
# Save the result to CSV:
f_out = open('data/ecozept_processed/sentences_15-classes_(sentences).csv', 'w', newline='', encoding='UTF8')

tab = csv.writer(f_out,delimiter=';')

header = ['Document','Main title','Block name','Question','Paragraph','Code']
tab.writerow(header)

for row in sentences_two_levels:
    tab.writerow(row)

f_out.close()
#%
# Save data into JSON format:
data_dict = {}

for i in range(len(sentences_two_levels)):
    data_dict[i] = {
            'Document': sentences_two_levels[i][0],
            'Main title': sentences_two_levels[i][1],
            'Block name': sentences_two_levels[i][2],
            'Question': sentences_two_levels[i][3],
            'Paragraph': sentences_two_levels[i][4],
            'Code': sentences_two_levels[i][5]
            }

with open("data/ecozept_processed/sentences_15-classes_(sentences).json", "w") as outfile:
    json.dump(data_dict, outfile)
#%%
## Split data into train and test the way that train and test data comes from different paragraphs:
true_codes = [item[5] for item in sentences_two_levels]
groups = [item[6] for item in sentences_two_levels]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=4852)

for _, (i, j) in enumerate(gss.split(np.arange(len(sentences_two_levels)), true_codes, groups)):
    train_index = i
    test_index = j

data = []
for i in range(len(sentences_two_levels)):
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

    data.append([sentences_two_levels[i][0],sentences_two_levels[i][1]+"\n\n"+sentences_two_levels[i][2]+"\n\n"+sentences_two_levels[i][3]+"\n\n"+sentences_two_levels[i][4],label,sentences_two_levels[i][6]])
#%
# Check some statistics:
train_all = len([i for i in range(len(data)) if i in train_index])
test_all = len([i for i in range(len(data)) if i in test_index])
print('\nNumber of examples in train and test sets:', train_all, test_all, '/', get_percentage(train_all,test_all), get_percentage(test_all,train_all))

train_0 = len([i for i in range(len(data)) if data[i][2]==0 and (i in train_index)])
test_0 = len([i for i in range(len(data)) if data[i][2]==0 and (i in test_index)])
print('Class 0 in train and test sets:', train_0, test_0, '/', get_percentage(train_0,test_0), get_percentage(test_0,train_0))

train_1 = len([i for i in range(len(data)) if data[i][2]==1 and (i in train_index)])
test_1 = len([i for i in range(len(data)) if data[i][2]==1 and (i in test_index)])
print('Class 1 in train and test sets:', train_1, test_1, '/', get_percentage(train_1,test_1), get_percentage(test_1,train_1))

train_2 = len([i for i in range(len(data)) if data[i][2]==2 and (i in train_index)])
test_2 = len([i for i in range(len(data)) if data[i][2]==2 and (i in test_index)])
print('Class 2 in train and test sets:', train_2, test_2, '/', get_percentage(train_2,test_2), get_percentage(test_2,train_2))

train_3 = len([i for i in range(len(data)) if data[i][2]==3 and (i in train_index)])
test_3 = len([i for i in range(len(data)) if data[i][2]==3 and (i in test_index)])
print('Class 3 in train and test sets:', train_3, test_3, '/', get_percentage(train_3,test_3), get_percentage(test_3,train_3))

train_4 = len([i for i in range(len(data)) if data[i][2]==4 and (i in train_index)])
test_4 = len([i for i in range(len(data)) if data[i][2]==4 and (i in test_index)])
print('Class 4 in train and test sets:', train_4, test_4, '/', get_percentage(train_4,test_4), get_percentage(test_4,train_4))

train_5 = len([i for i in range(len(data)) if data[i][2]==5 and (i in train_index)])
test_5 = len([i for i in range(len(data)) if data[i][2]==5 and (i in test_index)])
print('Class 5 in train and test sets:', train_5, test_5, '/', get_percentage(train_5,test_5), get_percentage(test_5,train_5))

train_6 = len([i for i in range(len(data)) if data[i][2]==6 and (i in train_index)])
test_6 = len([i for i in range(len(data)) if data[i][2]==6 and (i in test_index)])
print('Class 6 in train and test sets:', train_6, test_6, '/', get_percentage(train_6,test_6), get_percentage(test_6,train_6))

train_7 = len([i for i in range(len(data)) if data[i][2]==7 and (i in train_index)])
test_7 = len([i for i in range(len(data)) if data[i][2]==7 and (i in test_index)])
print('Class 7 in train and test sets:', train_7, test_7, '/', get_percentage(train_7,test_7), get_percentage(test_7,train_7))

train_8 = len([i for i in range(len(data)) if data[i][2]==8 and (i in train_index)])
test_8 = len([i for i in range(len(data)) if data[i][2]==8 and (i in test_index)])
print('Class 8 in train and test sets:', train_8, test_8, '/', get_percentage(train_8,test_8), get_percentage(test_8,train_8))

train_9 = len([i for i in range(len(data)) if data[i][2]==9 and (i in train_index)])
test_9 = len([i for i in range(len(data)) if data[i][2]==9 and (i in test_index)])
print('Class 9 in train and test sets:', train_9, test_9, '/', get_percentage(train_9,test_9), get_percentage(test_9,train_9))

train_10 = len([i for i in range(len(data)) if data[i][2]==10 and (i in train_index)])
test_10 = len([i for i in range(len(data)) if data[i][2]==10 and (i in test_index)])
print('Class 10 in train and test sets:', train_10, test_10, '/', get_percentage(train_10,test_10), get_percentage(test_10,train_10))

train_11 = len([i for i in range(len(data)) if data[i][2]==11 and (i in train_index)])
test_11 = len([i for i in range(len(data)) if data[i][2]==11 and (i in test_index)])
print('Class 11 in train and test sets:', train_11, test_11, '/', get_percentage(train_11,test_11), get_percentage(test_11,train_11))

train_12 = len([i for i in range(len(data)) if data[i][2]==12 and (i in train_index)])
test_12 = len([i for i in range(len(data)) if data[i][2]==12 and (i in test_index)])
print('Class 12 in train and test sets:', train_12, test_12, '/', get_percentage(train_12,test_12), get_percentage(test_12,train_12))

train_13 = len([i for i in range(len(data)) if data[i][2]==13 and (i in train_index)])
test_13 = len([i for i in range(len(data)) if data[i][2]==13 and (i in test_index)])
print('Class 13 in train and test sets:', train_13, test_13, '/', get_percentage(train_13,test_13), get_percentage(test_13,train_13))

train_14 = len([i for i in range(len(data)) if data[i][2]==14 and (i in train_index)])
test_14 = len([i for i in range(len(data)) if data[i][2]==14 and (i in test_index)])
print('Class 14 in train and test sets:', train_14, test_14, '/', get_percentage(train_14,test_14), get_percentage(test_14,train_14))
#%
# Number of examples in train and test sets: 1541 394 / 79.64% 20.36%
# Class 0 in train and test sets: 635 149 / 80.99% 19.01%
# Class 1 in train and test sets: 94 23 / 80.34% 19.66%
# Class 2 in train and test sets: 80 22 / 78.43% 21.57%
# Class 3 in train and test sets: 121 30 / 80.13% 19.87%
# Class 4 in train and test sets: 67 15 / 81.71% 18.29%
# Class 5 in train and test sets: 68 13 / 83.95% 16.05%
# Class 6 in train and test sets: 86 26 / 76.79% 23.21%
# Class 7 in train and test sets: 100 20 / 83.33% 16.67%
# Class 8 in train and test sets: 40 18 / 68.97% 31.03%
# Class 9 in train and test sets: 50 24 / 67.57% 32.43%
# Class 10 in train and test sets: 63 11 / 85.14% 14.86%
# Class 11 in train and test sets: 65 16 / 80.25% 19.75%
# Class 12 in train and test sets: 42 14 / 75.0% 25.0%
# Class 13 in train and test sets: 19 10 / 65.52% 34.48%
# Class 14 in train and test sets: 11 3 / 78.57% 21.43%
#%%
# Save results:
data_test = [data[i] for i in range(len(data)) if i in test_index]
data_train = [data[i] for i in range(len(data)) if i in train_index]

# Save test set as a binary file:
f_out = open("data/15-classes/test_set_sentences.pkl","wb")
pickle.dump(data_test,f_out)
f_out.close()

# Save training set as a binary file:
f_out = open("data/15-classes/train_set_sentences.pkl","wb")
pickle.dump(data_train,f_out)
f_out.close()
#%%
