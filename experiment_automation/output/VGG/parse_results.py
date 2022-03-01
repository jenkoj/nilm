import re
from collections import defaultdict
import numpy as np

import os
import glob

all = {}
counter = 0

p = os.getcwd().split("/")
pleng = len(p)
for subdir, dirs, files in os.walk(os.getcwd()):

    #go trough dirs that are 2 subdirs lower 
    if len(subdir.split("/")) == pleng+2:
        print(subdir)

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
                    
                    #print(res)
                    label = (' '.join(filter(str.isalpha,res[0:3])))
                    res = list(map(float, re.findall(r'\d+', line)))

                    valueToBeRemoved = 0
                    res = list(filter(lambda val: val !=  valueToBeRemoved, res))
                    if res:
                            #print(res)
                            #dict = defaultdict(all)
                            all.setdefault(label,[]).append(res)

        
        f = open("calc_results.txt", "w")
        for key,values in all.items():
            
            f.write(key)
            print(key)
            #print(values)
            #avg = np.mean(np.asarray(values),axis=0).round(decimals=2)
            print(avg)
            f.write(str(avg))
            f.write("\n")

        f.close()

        