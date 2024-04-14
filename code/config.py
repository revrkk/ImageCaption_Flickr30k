import os
import torch


class Config(object):

    def __init__(self) -> None:

        self.DEVICE = torch.device("mps")
        #self.DEVICE = torch.device("cpu")

        self.BATCH = 32
        self.EPOCHS = 10

        self.VOCAB_FILE = 'vocabulary.txt'
        self.VOCAB_SIZE = 5000

        self.NUM_LAYER = 1
        self.IMAGE_EMB_DIM = 512
        self.WORD_EMB_DIM = 512
        self.HIDDEN_DIM = 1024
        self.LR = 0.001

        self.EMBEDDING_WEIGHT_FILE = 'code/checkpoints/embeddings-32B-1024H-1L-e5.pt'
        self.ENCODER_WEIGHT_FILE = 'code/checkpoints/encoder-32B-1024H-1L-e5.pt'
        self.DECODER_WEIGHT_FILE = 'code/checkpoints/decoder-32B-1024H-1L-e5.pt'

        self.FLICKER_30K_DATASET_FOLDER = os.path.join(os.path.expanduser('~'),'Projects', 'ai', 'dataset', 'flicker30k')

        self.ROOT = os.path.join(os.path.expanduser('~'), 'Projects', 'ai', 'src', 'ImageCaption_Flickr30k')
