----
## DataSet
We use [AudioSet](https://research.google.com/audioset/), a dataset of over 2 million human-labeled 10-second YouTube video soundtracks. We extracted only dangerous sounds on Audioset. Please check [dataset](https://github.com/daehwa/AudioHero/tree/master/dataset) folder in this repository.

AudioSet provides links of youtube videos, their labels, and range of video in 10 seconds. You need to download it using any crawling code for AudioSet. We get help from this [repository](https://github.com/unixpickle/audioset).

We extracted dangerous sounds in Audioset. The dangerous sounds are indicated in [class\_label\_indices\_danger.csv](https://github.com/daehwa/AudioHero/blob/master/dataset/class_labels_indices_danger.csv). We did crawling dangerous sounds and you can download it [here](https://kaistackr-my.sharepoint.com/:f:/g/personal/daehwakim_kaist_ac_kr/EixWvOm0X25BrYqb8GRCPhUBKWoy22LUa3KtV3sjtnNScg?e=KcuEkp).