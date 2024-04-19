import os
import numpy as np
from derm_ita import get_ita
from PIL import Image
from derm_ita import get_kinyanjui_type

dataset_dir = "./samples/"

# change here for number of samples per class
sampling_dict = {
	"dark": 1,
	"tan1": 1,
	"tan2": 1,
	"int1": 1,
	"int2": 1,
	"lt1": 1,
	"lt2": 1,
	"very_lt": 1
}

all_predictions = {key: [] for key in sampling_dict}

image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f)) and (f.lower().endswith('.jpeg') or f.lower().endswith('.jpg'))]

for filename in image_paths:
    ita = get_ita(image=Image.open(filename))
    kinyanjui_type = get_kinyanjui_type(ita)
    try:
    	all_predictions[kinyanjui_type].append(filename)
    except:
    	print("Error")

sampled_image_paths = {}

for key in sampling_dict:
	if len(all_predictions[key])!=0:
		sampled_image_paths[key] = np.random.choice(np.array(all_predictions[key]), sampling_dict[key], replace=False)
	else:
		print("No predictions for {0}".format(key))

print(sampled_image_paths)