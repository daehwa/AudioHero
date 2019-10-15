import numpy as np
import csv

# File names
danger_labels = 'class_labels_indices_danger'+'.csv'
filename = 'eval_segments'
outfilename = filename+'_danger'

# File I/O
f_target = open(filename+'.csv', 'r', encoding='utf-8')
f_label = open(danger_labels, 'r', encoding='utf-8')
f_output = open(outfilename+'.csv', 'w', encoding='utf-8', newline='')
rdr_target = csv.reader(f_target)
rdr_label = csv.reader(f_label)
wr = csv.writer(f_output, quoting=csv.QUOTE_NONE, quotechar='')

# Load labels in list
labels = []
for line in rdr_label:
    if(line[2] == 'mid'):
        continue
    labels.append(line[2])
f_label.close()

cnt = 0
for line in rdr_target:
    if(cnt < 4):
        cnt = cnt+1
        wr.writerow(line)
        continue
    video_label = line[3:len(line)]
    video_label[0] = video_label[0].replace('\"','')
    video_label[0] = video_label[0].replace(' ','')
    video_label[len(video_label)-1] = video_label[0].replace('\"','')
    for label in video_label:
        if label in labels:
            result = line[0:3] + video_label
            wr.writerow(line)
            break
f_target.close()
f_output.close()