import matplotlib.pyplot as plt
import hmm
import random
from trainmodel import makeTokenModel, load_mem
print("loadData")
observable, observed = load_mem("aclImdbNorm/train/pos")

observed = random.sample(observed, 100)
x_axi = []
y_axi = []
for i in range(8, 12, 4):
    print('training for model states {}'.format(i))
    model = makeTokenModel(i, observable)
    model.train(observed, tol=0.01)
    y_axi.append(len(model.score_history))
    x_axi.append(i)
    
fig, ax = plt.subplots()
ax.plot(x_axi, y_axi)
ax.set_xlabel('Number of hidden States')
ax.set_ylabel('iteration')
ax.set_title('Iterations over hidden states with training tolerance of 0.01')
plt.savefig('answer1')