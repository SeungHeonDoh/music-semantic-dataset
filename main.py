import os
import pandas as pd
from preprocessing.msd_preprocessor import MSD_processor
from preprocessing.gtzan_preprocessor import GTZAN_processor
from preprocessing.mtat_preprocessor import MTAT_processor
from preprocessing.fma_preprocessor import FMA_processor
from preprocessing.openmic_preprocessor import OPENMIC_processor
from preprocessing.jamendo_preprocessor import JAMENDO_processor
from preprocessing.kvt_preprocessor import KVT_processor
from preprocessing.deezer_preprocessor import DEEZER_processor
from preprocessing.emo_preprocessor import EMO_processor
from preprocessing.constants import DATASET

def main():
    MSD_processor(msd_path= os.path.join(DATASET, 'msd'))
    GTZAN_processor(gtzan_path=os.path.join(DATASET, 'gtzan'))
    MTAT_processor(mtat_path=os.path.join(DATASET, 'mtat'))
    FMA_processor(fma_path=os.path.join(DATASET, 'fma'))
    OPENMIC_processor(openmic_path=os.path.join(DATASET, 'openmic'))
    JAMENDO_processor(jamendo_path=os.path.join(DATASET, 'jamendo'))
    EMO_processor(emo_path=os.path.join(DATASET, 'emo'))
    KVT_processor(kvt_path=os.path.join(DATASET, 'kvt'))
    DEEZER_processor(deezer_path= os.path.join(DATASET, 'deezer'))

if __name__ == '__main__':
    main()