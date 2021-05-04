import pysindy
import pickle

##### ##### ########## ##### #####
## Sindy
##

sindyModel = None

with open("Pickle.sindy", "rb+") as f:
    sindyModel = pickle.load(f)

sindyModel.print()
