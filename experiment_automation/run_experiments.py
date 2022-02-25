import subprocess
from os import environ
import pathlib
import zipfile
import shutil

#AUTOMATE BASE MODEL (PRE SPLIT TO SPLIT)
#AUTOMATE TL (PRE SPLIT TO SPLIT)
#IMPLEMENT PROGRESS 

def unzip(file_name):
    print("unzipping ",file_name," ...")
    path = str(pathlib.Path().resolve())+"/../"
    with zipfile.ZipFile(path+"/data/"+file_name+".zip", 'r') as zip_ref:
        zip_ref.extractall(path+"/data/"+file_name+"/")

def clean_up(file_name):
    dirpath = pathlib.Path("../data/"+file_name).resolve()
    print("cleaing up..")
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(str(dirpath))

def executeJupyter(dataset,train_type,learn_type,epochs,batch_size,save_model,tl_datasets):

    environ['dataset'] = dataset
    environ['train_type'] = train_type
    environ['learn_type'] = learn_type
    environ['epochs'] = epochs
    environ['save_model'] =  save_model
    environ['tl_datasets'] =  tl_datasets
    environ['batch_size'] = batch_size
    seed= dataset.split("_R")[1][0:2]
    environ['seed'] =  seed

    pathlib.Path("output/"+learn_type+'/'+dataset.split("_")[0]+"/"+train_type).mkdir(parents=True, exist_ok=True) 
    
    subprocess.run(["jupyter","nbconvert" ,"--to" ,"notebook", "--execute", "../LSTM_gen.ipynb"])
    subprocess.run(['mv', '../LSTM_gen.nbconvert.ipynb','output/'+learn_type+'/'+dataset.split("_")[0]+"/"+train_type+"/"+dataset.split("_")[0]+"_"+seed+"_"+tl_datasets+'_out.nbconvert.ipynb'])

def listToStringWithoutBrackts(list1):
    return str(list1).replace('[','').replace(']','').replace("'","")


def get_tl_datasets(dataset,all_datasets):

    current_dataset = dataset.split("_")[0]
    all_datasets.remove(current_dataset)
    tl_datasets = listToStringWithoutBrackts(all_datasets)

    return tl_datasets


# datasets = [
#     "iawe_gen_GASF_13m_100S5X4A1545_R12_80-20",
#     "iawe_gen_GASF_13m_100S5X4A1545_R42_80-20",
#     "iawe_gen_GASF_13m_100S5X4A1545_R82_80-20",
#     "redd_gen_GASF_13m_100S5X5A4934_R82_80-20",
#     "redd_gen_GASF_13m_100S5X5A4934_R12_80-20", 
#     "redd_gen_GASF_13m_100S5X5A4934_R42_80-20",
#     "eco_gen_GASF_13m_100S5X11A38085_R12_80-20",
#     "eco_gen_GASF_13m_100S5X11A38085_R42_80-20",
#     "eco_gen_GASF_13m_100S5X11A38085_R82_80-20",
#     "ukdale_gen_GASF_13m_100S5X12A54480_R12_80-20",
#     "ukdale_gen_GASF_13m_100S5X12A54480_R42_80-20",
#     "ukdale_gen_GASF_13m_100S5X12A54480_R82_80-20",
#     "refit_gen_GASF_13m_100S5X_15A166006_R12_80-20",
#     "refit_gen_GASF_13m_100S5X_15A166006_R42-80-20",
#     "refit_gen_GASF_13m_100S5X_15A166006_R82_80-20"
#     ]
#CNN
datasets =[
"iawe_gen_GASF_60m_300S0X_R12_80-20",
"iawe_gen_GASF_60m_300S0X_R42_80-20",
"iawe_gen_GASF_60m_300S0X_R82_80-20",
"redd_gen_GASF_60m_300S0X_R12_80-20",
"redd_gen_GASF_60m_300S0X_R42_80-20",
"redd_gen_GASF_60m_300S0X_R82_80-20",
"eco_gen_GASF_60m_300S0X_2_R12_80-20",
"eco_gen_GASF_60m_300S0X_2_R42_80-20",
"eco_gen_GASF_60m_300S0X_2_R82_80-20",
"ukdale_gen_GASF_60m_300S0X_R42_80-20",
"ukdale_gen_GASF_60m_300S0X_R82_80-20",
"ukdale_gen_GASF_60m_300S0X_R12_80-20" ,
"refit_gen_GASF_60m_300S0X_15A157030N_R82_80-20-V1",
"refit_gen_GASF_60m_300S0X_15A157030N_R12_80-20-V1",
"refit_gen_GASF_60m_300S0X_15A157030N_R42_80-20-V1"
   ]

for dataset in datasets:

    unzip(dataset)

    #tl_datasets = get_tl_datasets(dataset, ["refit", "iawe", "eco", "redd", "ukdale"])

    tl_datasets = ""
    executeJupyter(dataset, "BB", "VGG", "100", "32", "True", tl_datasets)

    #executeJupyter(dataset, "TL", "CNN", "25", "32", "False", tl_datasets)

    clean_up(dataset)

    print("finished", dataset)

# #LSTM  
# datasets2 = [
#         "refit_gen_GASF_13m_100S5X_15A166006_R12_80-20",
#         "refit_gen_GASF_13m_100S5X_15A166006_R42-80-20",
#         "refit_gen_GASF_13m_100S5X_15A166006_R82_80-20"
#             ]

# # #CNN
# datasets2 = [
#         "refit_gen_GASF_60m_300S0X_15A157030N_R82_80-20-V1",
#         "refit_gen_GASF_60m_300S0X_15A157030N_R12_80-20-V1",
#         "refit_gen_GASF_60m_300S0X_15A157030N_R42_80-20-V1"
# #             ]
# for dataset in datasets2:

#     unzip(dataset)

#     batch = "32"
#     epochs = "20"
#     mode = "CNN"
#     # executeJupyter(dataset, "TL", mode, epochs, batch, "False", "ukdale")
#     executeJupyter(dataset, "TL", mode, epochs, batch, "False", "iawe")
#     executeJupyter(dataset, "TL", mode, epochs, batch, "False", "redd")
#     # executeJupyter(dataset, "TL", mode, epochs, batch, "False", "eco")

#     clean_up(dataset)

#     print("finished", dataset)
