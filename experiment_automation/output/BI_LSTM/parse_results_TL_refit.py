import re
import numpy as np

import os
import glob

all = {}
counter = 0
main_work_dir = os.getcwd()
p = os.getcwd().split("/")
pleng = len(p)

datasets = ["iawe","redd","eco","ukdale"]

for current_dataset in datasets:
    print("..............................starting:",current_dataset)
    for subdir, dirs, files in os.walk(main_work_dir):

        #go trough dirs that are 2 subdirs lower 
        dirs_split = subdir.split("/")
        if len(dirs_split) == pleng+2:
            dataset = dirs_split[-2]
            method = dirs_split[-1]
            print("dataset",dataset)
            #use only refit TL 
            if dataset == current_dataset and method == "TL":
                os.chdir(subdir)
                for filename in glob.glob('*.ipynb'):
                    name = filename.split("_")[1]
                    print(filename.split("_")[0:3])
                    #print("files",filename.split("_"))
                    #if name == "refit":
                    with open(filename) as input_data:
                        # Skips text before the beginning of the interesting block:
                    
                        for line in input_data:
                            if 'Normalized confusion matrix' in line.strip():   # Or whatever test is needed
                                break

                        # Reads text until the end of the block:
                        for line in input_data:  # This keeps reading the file
                            if 'using dataset' in line.strip():
                                break
                            res = line.strip().replace("/"," ").split()
                            #print(line)
                            label = (' '.join(filter(str.isalpha,res[0:3])))
                            res = list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)))

                            valueToBeRemoved = 0
                            res = list(filter(lambda val: val !=  valueToBeRemoved, res))
                            if res:
                                    print(res)
                                    #dict = defaultdict(all)
                                    all.setdefault(label,[]).append(res)


    os.chdir(main_work_dir)
    f = open(""f"{current_dataset}_calc_TL_results.txt", "w")
    # for key,values in all.items():

    #     f.write(key)
    #     print(key)
    #     #print(values)
    #     avg = np.mean(np.asarray(values),axis=0).round(decimals=2)
    #     print(avg)
    #     f.write(str(avg))
    #     f.write("\n")

    # f.close()
    for key,values in all.items():
            
        f.write(key)
        print(key)
        np_vals = np.asarray(values)
        # np_vals = np.where(str(np_vals) == "1.", 100,np_vals)
        print(np_vals)
        if (np_vals.shape[0] == 3):
            print("shape ok, averaging number of seeds:",np_vals.shape[0])
            print(np.asarray(values))
            avg = np.mean(np_vals,axis=0).round(decimals=4)
            print(avg)
            f.write(" ")
            for el in avg:
                f.write(str(round(el,2)))
                f.write(" & ")
            #f.write(str(avg))
            f.write("\n")
        else:
            f.write("errrorrrrr!")
    f.close()

    
