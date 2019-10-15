## How to use
1. Place the code [pre-processing.py](https://github.com/daehwa/AudioHero/blob/master/pre-processing/Pre-Processing/Code/pre-processing.py) to the same folder with raw data files.
2. run the code. Your code automaticaly finds wav files and convert it. Also, it skipped the already processed files.
    ```
    python pre-processing.py
    ```

## Output File
| Type      | Value |
| :---        |    :----:   |
| Format      | .wav       |
|   Sample rate   |   16kHz   |
|    Channel    |   1 (mono) |
| Bit Depth     |   PCM_16  |


## Output Example on Screen
```
Processing with 0hipL0SkXMo_40.000
already exist.

Processing with 2Pq9c0tFt1g_30.000
already exist.

Processing with _2MXlOXq9k8_340.000
-----Source File-----
Sample rate: 22050
Channels: 2
Subtype: PCM_16
---------------------
silent parts: []
-----Output File-----
Sample rate: 16000
Channels: 1
Subtype: PCM_16
---------------------
```
