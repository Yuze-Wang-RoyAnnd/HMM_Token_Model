import argparse  
import hmm
import os
from tqdm import tqdm

def load_subdir(path):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as fh:
            data.append(list(fh.read()))
    return data

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--pos_hmm', default=None, help='Path to the positive class hmm.')
    parser.add_argument('--neg_hmm', default=None, help='Path to the negative class hmm.')
    parser.add_argument('--datapath', default=None, help='Path to the test data.')

    args = parser.parse_args()

    # Load HMMs 
    pos_hmm = hmm.HiddenMarkovModel.loadModel(args.pos_hmm)
    neg_hmm = hmm.HiddenMarkovModel.loadModel(args.neg_hmm)

    correct = 0
    total = 0

    # test samples from positive datapath    
    samples = load_subdir(os.path.join(args.datapath, 'pos'))
    for sample in tqdm(samples):
        score1, _, _, = pos_hmm.layer.LL(sample)
        score2, _, _, = neg_hmm.layer.LL(sample)
        if score1 > score2:
            correct += 1
        total += 1
            
    # test samples from negative datapath
    samples = load_subdir(os.path.join(args.datapath, 'neg'))
    for sample in tqdm(samples):
        score1, _, _, = pos_hmm.layer.LL(sample)
        score2, _, _, = neg_hmm.layer.LL(sample)
        if score1 < score2:
            correct += 1
        total += 1
        
    # report accuracy  (no need for F1 on balanced data)
    print("%d/%d correct; accuracy %f"%(correct, total, correct/total))

    
if __name__ == '__main__':
    main()