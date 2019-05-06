
from nltk.translate.bleu_score import sentence_bleu
import argparse

def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str, default='./test_captions/captions.txt')
    Argparser.add_argument('--candidate', type=str, default='./test_captions/res.txt')

    args = Argparser.parse_args()
    return args

args = argparser()

reference = open('./test_captions/captions.txt', 'r').readlines()
candidate = open('./test_captions/res.txt', 'r').readlines()

if len(reference) != len(candidate):
    raise ValueError('The number of sentences in both files do not match.')

score = 0

for i in range(len(reference)):
    ref_idmapimg = reference[i].strip().split(':')
    can_idmapimg = candidate[i].strip().split(':')
    score += sentence_bleu(ref_idmapimg[1].split(',')[0].strip().strip("[").strip("]}").strip('"').split(), can_idmapimg[1].strip().strip('<start>').strip(' <end>').replace('<unk>', '').split())
score /= len(reference)
print("The bleu score is: "+str(score))