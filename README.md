# Build Instruction 
This folder includes 3 python file :hmm.py, testmodel.py, trainmodel.py
it also include two pre-trained model that you can load for testing. All are trained with all 1000 samples with 30 epochs and 10 hidden_states, these are NOT trained until convergence so it may suffer accuracy issue
hmm
    - includes utility function to build a hmm model. user dont need to call it in console

testmodel
    - loads two pretrained hmms and test it on dataset
    - pos_hmm is path to positive hmm model
    - neg_hmm is path to negative hmm model
    - datapath is path to test data

trainmodel
    - train a hmm model and save the model
    - dev_path is path to testing file (aclImdbNorm/test/pos)
    - train_path is path to training file (aclImdbNorm/train/pos)
    - max_iters is the maximum amount of iteration model have to go through, if not supply it will train until tolerance is met
    - model_out is path to save the model
    - hidden_states is number of hidden states model will have
    - sample_size is the amount of sample will be used to training, if not supplied it will use all sample
    - tolerance is the convergance tolerance a.k.a the change of update value

test
    - This python file recreate the experiment to produce the image I upload with folder

# Finding

![Finding](answer_1.PNG)

The image above show the log-likelihood for model trained on positive dataset. it shows hidden states (2 - 14) on the x axis and log-likelihood on the y axis.However, there are several problems I encountered as I ran through the testing dataset.
First is that model will face a floating point percision issue for one states model. particularly the pi and transition matrix will result a probability slightly over 1 after the update has been made. I think this is due to the scaling and alpha beta calculation, to solve such an issue I have restricted the creation of 1 state hmm model.
Second problem is the model will always enter a local minimum after some epochs. When model enters the said local minimum, it will likeliy to die without reaching the true minimum if update tolerance is somewhat large. For this parictular graph, the model is trained on 100 samples and tolerance of 0.01, it produced a relatively good result but still resulted in some model to terminate early.
As the amount of hidden states within the model increases, the log likelihood for the said model also increases. But this increase in hidden states is not linearly proportional to the increase in accuracy, it is rather logirithmicly proportional. Therefore to produce the best accuracy with a relatively good perfromance, it is recommened to keep the amount of hidden states between the range of 10 - 12 for the best accuracy. 

12743/25000 correct; accuracy 0.509720 10 states 1000 samples training 30 epoch
12884/25000 correct; accuracy 0.515360 3 states 1000 samples training 30 epoch
The accuracy is not according to expecation since all the data are train using small sample sizes, this could also due to model not trained until convergence.
# Disclaimer

The model exhibit floating percision issue after a minimum has been reached. If the model continues training after the update is too small (that is if the pre-set tolerance is too small), it will trigger the pre-set floating point safe guard which kills the model training. If such scenario happens, please re-run the model with a higher tolerance. 