# -*- coding: utf-8 -*-

import numpy as np

G_FREQS = np.linspace(0,125,251)
G_NCHAN = 8
G_HEADER = 'Channel\tFrequency'


tsv = open('.\\emb_mappings.tsv','w')

tsv.write(G_HEADER + "\n")
dims=0
for freq in G_FREQS:
    for ch in range(G_NCHAN):
        dims +=1
        tsv.write(str(ch) + "\t" + str(int(freq)) + "\n")

for i in range(4):
    tsv.write("-1\t-1\n")

print("Dimensions: ", dims + 4)
tsv.flush()
tsv.close()