# cs230

Our project is Speech Emotion Recognition through Deep Learning with Japanese Dataset

Original Japanese dataset is in filefolder 'Dataset'

Original English dataset is in filefolder 'Dataset_RAVDESS'

Initial attempt in split dataset is in 'train_valid_test_split+normalization.ipynb'

Japanese dataset is processed in 'wav_to_MFCC.ipynb'. .wav file is transferd into MFCC data and stored in filefolder 'MFCCsData' and 'MFCCsData2'. 'MFCCsData' contains audio data with muted head and tail cutted. 'MFCCsData2' contains audio data without cutting

English dataset is processed in 'wav_to_MFCC_RAVDESS.ipynb'. .wav file is transferd into MFCC data and stored in filefolder 'MFCCsData_RAVDESS' and 'MFCCsData_RAVDESS2.  'MFCCsData_RAVDESS' contains audio data with muted head and tail cutted. 'MFCCsData_RAVDESS2' contains audio data without cutting

Further normalization and spliting is operated in 'dataset.py'

'CNN_milestone.ipynb' is the milestone result of our CNN model

'DNN_final.ipynb' contains our baseline DNN model

'CNN_final.ipynb' contains our CNN model trained on Japanese dataset

'CNN_pretrain.ipynb' contains pretrained MobileNetV2 model trained on Japanese dataset and fine-tuning



