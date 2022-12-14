import numpy as np
from tqdm import tqdm
# matrix = [numberOfObservables * probabilityVector]
# supply matrix with Observation: ProbVector
class ProbabilityMatrix:
    def __init__(self, probabilityVector: dict):
        
        assert len(probabilityVector) > 1,"The numebr of input probability vector must be greater than one."
        assert len(probabilityVector.keys()) == len(set(probabilityVector.keys())),"All observables must be unique."

        self.states      = sorted(probabilityVector)
        self.observables = probabilityVector[self.states[0]].states
        self.values      = np.stack([probabilityVector[x].values for x in self.states])
        
    @classmethod
    def initialize(cls, states: list, observables: list):
        rand = [np.random.dirichlet(np.ones(len(observables))) for state in states] 
        aggr = [dict(zip(observables, rand[i])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def update_to(cls, array: np.ndarray, states: list, observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) for x in array]
        return cls(dict(zip(states, p_vecs)))
        
        
    #safeguard
    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)
        
# P  = [p0, p1, ... , p(N-1)]
class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()
        
        assert len(states) == len(probs), "The probabilities must match the states."
        assert abs(sum(probs) - 1.0) < 1e-12, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), "Probabilities must be numbers from [0, 1] interval."
        
        self.states = sorted(probabilities)
        self.values = [probabilities[state] for state in states]
        
    @classmethod
    def initialize(cls, states: list):
        rand = np.random.dirichlet(np.ones(len(states)))
        return cls(dict(zip(states, rand)))
    
    
    @classmethod
    def update_to(cls, array: np.ndarray, state: list):
        return cls(dict(zip(state, list(array))))


    #safeguards
    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])


    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

#N is number of observables
#M is number of states

#Transmission Matrix T
#       state A, State B, State C, .... State N
#State A 
#State B
#State C
# ...
#State N
# .      Sum 1 .  Sum 1

class HiddenMarkovChain:
    def __init__(self, T:ProbabilityMatrix, E:ProbabilityMatrix, pi:ProbabilityVector):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
        self.scaling = [] #[M]
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)
    
    @classmethod
    def loadlayer(cls, this_T, this_E, this_pi, states:list, observables:list):
        T = ProbabilityMatrix.update_to(this_T, states, states)
        E = ProbabilityMatrix.update_to(this_E, states, observables)
        pi = ProbabilityVector.update_to(this_pi, states)
        return cls(T, E, pi)

    
class HiddenMarkovLayer(HiddenMarkovChain):
    #forward algorithm
    #scaling adjusted
    def _alphas(self, observations: list) -> np.ndarray:
        #alpha =  [M * N]
        #states -> states .... -> states, where the probabilty is passed forward. 
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        #scaling
        self.scaling.append(1 / alphas[0, :].sum())
        alphas[0, :] *= self.scaling[0]
        
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) * self.E[observations[t]].T
            self.scaling.append(1 / alphas[t, :].sum())
            alphas[t, :] *= self.scaling[t]
            
        return alphas
    #backward
    #total state i <- total state i+1 <- ... <- total state N
    #_beta = [M * N]
    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1 * self.scaling[-1]
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
            betas[t, :] *= self.scaling[t]
        return betas
    
    #return xi for given markov chain and observation
    def _xis(self, alphas, betas, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        xis = np.zeros((L - 1, N, N))
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            xis[t, :, :] = P1 * P2
        return xis
    
    #return the log likelihood of given sample
    def LL(self, observation:list, train=False) -> float:
        alpha = self._alphas(observation)
        cur = np.asarray(self.scaling)
        score = -(np.log(cur).sum())
        if not train:
            self.scaling = []
        return score, cur, alpha
    
    
class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []
        
    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    #multi obsrevation assume conidtional indipendence
    def update(self, observations : np.ndarray) ->float:
        xis, gammas, scores = [], [], []
        #print("calculating Alpha Beta and Xi\n")
        for observation in tqdm(observations):
            score, cur, alpha = self.layer.LL(observation, True)
            beta = self.layer._betas(observation)
            xi = self.layer._xis(alpha, beta, observation)
            scores.append(score)
            xis.append(xi)
            gammas.append(alpha * beta / cur.reshape(-1, 1))
            #clear alpha
            self.layer.scaling = []
        
        #pi_hat
        #print("Adjusting for Pi\n")
        new_pi = np.zeros(len(self.layer.states))
        for gamma in gammas:
            new_pi += gamma[0]
        new_pi /= len(observations)
        #a_hat
        s1 = np.zeros((len(self.layer.states), len(self.layer.states)))
        s2 = np.zeros(len(self.layer.states))
        #print("Adjusting for Transition Probability\n")
        for xi, gamma in zip(xis, gammas):
            s1 += xi.sum(axis=0)
            s2 += gamma[:-1].sum(axis=0)
        new_T = s1/s2.reshape(-1, 1)
        
        obs_idx = [[self.layer.observables.index(x) for x in observation] for observation in observations]
        #print("Adjusting for Emission Probability\n")
        s3 = np.zeros((len(self.layer.observables), len(self.layer.states)))
        s4 = np.zeros(len(self.layer.states))
        for i, gamma in enumerate(gammas):
            for k in range(len(obs_idx[i])):
                s3[obs_idx[i][k]] += gamma[k]
            s4 += gamma.sum(axis=0)
        new_E = s3.T/s4.reshape(-1,1)

        self.layer.pi = ProbabilityVector.update_to(new_pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.update_to(new_T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.update_to(new_E, self.layer.states, self.layer.observables)
        
        return sum(scores)/len(observations)

    def saveModel(self, path:str):
        with open(path, 'wb') as f:
            np.save(f, self.layer.states)
            np.save(f, self.layer.observables)
            np.save(f, self.layer.T.values)
            np.save(f, self.layer.E.values)
            np.save(f, self.layer.pi.values)
        print("Model Saved!")
    
    @classmethod
    def loadModel(cls, path:str):
        with open(path, 'rb') as f:
            states = np.load(f)
            observables = np.load(f)
            T = np.load(f)
            E = np.load(f)
            pi = np.load(f)
            hmmlayer = HiddenMarkovLayer.loadlayer(T, E, pi, states, observables)
            f.close()
            return cls(hmmlayer)
        
    #single observation
    
    def train(self, observations, iteration=None, tol=None):
        self._score_init = 0
        self.score_history = []
        early_stopping = isinstance(tol, (int, float))
        iterate = isinstance(iteration, int)
        
        epoch = 0
        while(1):
            score = self.update(observations)
            print("Training... epoch = {}, score = {}.".format(epoch, score))
            if early_stopping and abs(self._score_init - score) < tol:
                print("tolerance met")
                break
            if iterate and epoch >= iteration:
                print('iteration met')
                break
            epoch +=1
            self._score_init = score
            self.score_history.append(score)

