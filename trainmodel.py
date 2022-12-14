import argparse
import hmm
import os
import random
import testmodel
from tqdm import tqdm

def makeTokenModel(size, observable) ->hmm.HiddenMarkovModel:
    states = []
    for i in range(size):
        states.append("S" + str(i))
    model = hmm.HiddenMarkovModel.initialize(states, observable)
    return model

def load_mem(path):
    p_observable_token = []
    p_observed_token = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        filename = os.fsencode(file)
        with open(os.path.join(directory, filename)) as f:
            lines = f.readline()
            p_observed_token.append(list(lines))
            #iter over line 
            for character in list(lines):
                if character in p_observable_token :
                    pass
                else:
                    p_observable_token.append(character)
        f.close()
    return p_observable_token, p_observed_token

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--dev_path', default=None, help='Path to development (i.e., testing) data.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--max_iters', type=int,required=False, help='The maximum number of EM iterations, if nothing entered, it will iterate until tolerance is met')
    parser.add_argument('--model_out', default=None, help='Filename to save the final model.')
    parser.add_argument('--hidden_states', type=int, default=10, help='The number of hidden states to use. (default 10)')
    parser.add_argument('--sample_size', type =int, required = False, help='amount of training data will be used')
    parser.add_argument('--tolerance', type=float, help='convergence tolernace')
    args = parser.parse_args()
    
    print("loading data into memory")
    observable, observed = load_mem(args.train_path)
    testing = testmodel.load_subdir(args.dev_path)

    
    
    if args.sample_size:
        observed = random.sample(observed, args.sample_size)
    
    
    model_pos = makeTokenModel(args.hidden_states, observable)
    
    print('initialize log likelihood on testing')
    '''
    score = 0
    for sequence in tqdm(testing):
        s, _, _, = model_pos.layer.LL(sequence)
        score += s


    print("modelScore: {}".format(score/len(testing)))
    '''
    if args.max_iters:
        model_pos.train(observed, args.max_iters, args.tolerance)
    else:
        model_pos.train(observed, args.tolerance)
    
    model_pos.saveModel(args.model_out + ".npy")

    
if __name__ == '__main__':
    main()

