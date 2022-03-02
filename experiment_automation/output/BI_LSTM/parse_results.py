import re
from collections import defaultdict
import numpy as np

import os
import glob

all = {}
counter = 0
main_dir = os.getcwd()
p = os.getcwd().split("/")
pleng = len(p)
for subdir, dirs, files in os.walk(main_dir):
    all = {}
    #go trough dirs that are 2 subdirs lower 
    dir_spl = subdir.split("/")
    if len(dir_spl) == pleng+2:
        #print(subdir)
        ds = dir_spl[-2]
        type = dir_spl[-1]
        
        if type == "BB" and ds=="iawe":
            print("using",ds,type)
            os.chdir(subdir)
            for filename in glob.glob('*.ipynb'):

                with open(filename) as input_data:
                    # Skips text before the beginning of the interesting block:
                    for line in input_data:
                        if 'Normalized confusion matrix' in line.strip():   # Or whatever test is needed
                            break
                        
                    # Reads text until the end of the block:
                    for line in input_data:  # This keeps reading the file
                        if ']' in line.strip():
                            break
                        res = line.strip().replace("/"," ").split()
                        
                        print(line)
                        label = (' '.join(filter(str.isalpha,res[0:3])))
                        res = list(map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)))

                        valueToBeRemoved = 0
                        res = list(filter(lambda val: val !=  valueToBeRemoved, res))
                        if res:
                                print(res)
                                #dict = defaultdict(all)
                                all.setdefault(label,[]).append(res)

            
            f = open("calc_results.txt", "w")
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

            