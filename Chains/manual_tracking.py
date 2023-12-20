"""Enable the manual tracking of chains."""

from typing import List
import os
import json

import matplotlib.pyplot as plt

folder = "/Users/sintes/Desktop/NASGuillaume/Chains/Chains 13.7%/2023-10-31_11h39m35s"
c = 13.7
fps = 30
lim_1 = 4200
lim_2 = 6490

folder_list: List[str] = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
folder_list.sort()
coords = []

for i in range(lim_1, lim_2, fps):
    im_name = folder_list[i]
    im = plt.imread(im_name)

    plt.figure()
    plt.imshow(im)
    coords.append(plt.ginput()[0])
    plt.close()
length = int(input("Length of the chain ?"))
res = {
    "folder": folder,
    "concentration": c,
    "fps": fps,
    "length": length,
    "lim1": lim_1,
    "lim1": lim_2,
    "coords": coords
}
f = open(f"/Users/sintes/Desktop/NASGuillaume/Chains/Manual_tracks/{folder.split('/')[-1]}_{lim_1}.txt", "w") 
f.write(json.dumps(res, separators=(",", ":"), indent=4))
f.close()