gen - > models traind with generated data (generator)
classic -> model trained in a classical way (load it into RAM/np.array)

auto means that it was automated when testing. 

if model is not in any specific folder it belongs to REFIT, all other datasets should have their models stored in a folder with their name on it. 

REFIT:
auto_classic1 - tested with adding LSTM latyers, larger filters, more filters etc. see report 1 on tuning the model
auto_classic2 - tested with combining changes that worked fine in autoclassic_1. also see report 2 on tuning the model
auto_claasic3 - testing droupout on model2 from autoc_classic2 - see report 2 on tuning the model.

model_classic1 2 and 3. Tested with various number of samples.

Best model for LSTM3D and REFIT is model_classic3
Best model for tuned LSTM and REFIT is auto_classic_2/model2
