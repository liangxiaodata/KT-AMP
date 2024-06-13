# KT-AMP
KT-AMP: Enhancing Antimicrobial Peptide Functions Prediction through Knowledge Transfer on Protein Language Model
# Requirements
* python == 3.7
* pytorch == 1.11.0
* Numpy == 1.21.6
* transformers == 4.27.3

# Files:
1.data:<br>
train and test dataset of  antimicrobial dataset:<br>
amp_train.tsv<br>
amp_test.tsv<br>

train and test dataset of  antibacterial dataset:<br>
AB_train.tsv<br>
AB_test.tsv<br>

train and test dataset of  antiviral dataset:<br>
AV_train.tsv<br>
AV_test.tsv<br>

train and test dataset of  antifungal dataset:<br>
AF_train.tsv<br>
AF_test.tsv<br>

2.code:<br>
phase 1 training and testing<br>
main_amp.py<br>

phase 2 training and testing<br>
main_other.py<br>

# Use:
Using a pre-trained model: Download the model files from [prot_t5_xl_half_uniref50-enc model on Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/tree/main) and place them in a folder named /prot_t5_xl_half_uniref50-enc at the same level of main_amp.py and main_other.py.


```bash
python main_amp.py
python main_other.py
```


# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Xiao Liang at liangxiaomath@csu.edu.cn




