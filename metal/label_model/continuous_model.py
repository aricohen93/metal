from collections import Counter
from functools import partial
from itertools import chain, product

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from scipy.stats import norm 
from torch.utils.data import DataLoader

from scipy.special import logsumexp

import torch.nn.functional as F 

from metal.classifier import Classifier
from metal.label_model.graph_utils import get_clique_tree
from metal.label_model.lm_defaults import lm_default_config
from metal.utils import MetalDataset, recursive_merge_dicts


class ContinuousModel(Classifier):
    """A LabelModel...TBD

    Args:
        k: (int) the cardinality of the classifier
    """

    # This class variable is explained in the Classifier class
    implements_l2 = True

    def __init__(self, k=2, **kwargs):
        config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(k, config)
        
        if config["device"] != "cpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def _check_L(self, L):
        """Run some basic checks on L."""
        if issparse(L):
            L = L.todense()


    def _generate_cov(self, L):
        self.cov = torch.cov(L.T)
        self.var = torch.diag(self.cov)

    def _generate_mean(self, L):
        self.mean = torch.mean(L, dim=0)


    def _generate_third_moment(self, L):
        self.third_mom = torch.mean(L**3, dim=0)



    def _init_params(self):
        """Initialize the learned params

        - \mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:

            self.mu[i*self.k + j, y] = P(\lambda_i = j | Y = y)

        and similarly for higher-order cliques.
        - Z is the inverse form version of \mu.
        """
        self.mu0 = nn.Parameter(self.mean * (1 - np.random.random(self.m)))
        self.mu1 = nn.Parameter(self.mean* (1 + np.random.random(self.m)))

        self.sigma0 = nn.Parameter(self.var * np.random.random(self.m))
        self.sigma1 = nn.Parameter(self.var * np.random.random(self.m))



    def get_conditional_probs(self, source=None):
        """Returns the full conditional probabilities table as a numpy array,
        where row i*(k+1) + ly is the conditional probabilities of source i
        emmiting label ly (including abstains 0), conditioned on different
        values of Y, i.e.:

            c_probs[i*(k+1) + ly, y] = P(\lambda_i = ly | Y = y)

        Note that this simply involves inferring the kth row by law of total
        probability and adding in to mu.

        If `source` is not None, returns only the corresponding block.
        """
        c_probs = np.zeros((self.m * (self.k + 1), self.k))
        mu = self.mu.detach().clone().numpy()

        for i in range(self.m):
            # si = self.c_data[(i,)]['start_index']
            # ei = self.c_data[(i,)]['end_index']
            # mu_i = mu[si:ei, :]
            mu_i = mu[i * self.k : (i + 1) * self.k, :]
            c_probs[i * (self.k + 1) + 1 : (i + 1) * (self.k + 1), :] = np.clip(mu_i, 0.01, 0.99)

            # The 0th row (corresponding to abstains) is the difference between
            # the sums of the other rows and one, by law of total prob
            c_probs[i * (self.k + 1), :] = np.clip(1 - mu_i.sum(axis=0), 0.0001, 0.9999)

        if source is not None:
            return c_probs[source * (self.k + 1) : (source + 1) * (self.k + 1)]
        else:
            return c_probs

    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)

        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        #self._set_constants(L)


        mu0 = self.mu0.detach().clone().numpy()
        mu1 = self.mu1.detach().clone().numpy()

        sigma0 = self.sigma0.detach().clone().numpy()
        sigma1 = self.sigma1.detach().clone().numpy()


        # Example data (replace with actual values)
        # Precompute constants for efficiency
        log_const_0 = -0.5 * np.sum(np.log(2 * np.pi * sigma0**2))
        log_const_1 = -0.5 * np.sum(np.log(2 * np.pi * sigma1**2))

        # Compute the log-PDF for Gaussian 0
        diff_0 = L - mu0  # Broadcasted subtraction
        log_pdf_0 = log_const_0 - 0.5 * np.sum((diff_0**2) / (sigma0**2), axis=1)

        # Compute the log-PDF for Gaussian 1
        diff_1 = L - mu1  # Broadcasted subtraction
        log_pdf_1 = log_const_1 - 0.5 * np.sum((diff_1**2) / (sigma1**2), axis=1)

        # Combine into a single (1000, 2) matrix
        log_pdf_matrix = np.stack([log_pdf_0, log_pdf_1], axis=1)

        log_numerator = log_pdf_matrix + np.log(self.p)

        Z = logsumexp(log_numerator, axis=1, keepdims=True)

        log_posterior_probs = log_numerator - Z 

        posterior_probs = np.exp(log_posterior_probs)

        return posterior_probs
        


    def loss_mu(self, *args):

        loss_1 = torch.norm(self.mean - self.mu0 * self.p[0] - self.mu1 * self.p[1]) ** 2 # expectation constraint 
        loss_2 = torch.norm(self.var - (self.sigma0 * self.p[0] + self.sigma1 * self.p[1] + self.p[0]*self.p[1]*(self.mu1-self.mu0)**2)) ** 2


        mu_diff = self.mu1 - self.mu0 
        outer_product = torch.ger(mu_diff, mu_diff)

        loss_3 = torch.norm(self.cov - self.p[0] * self.p[1] * outer_product) ** 2


        loss_6 = torch.norm(self.third_mom - (3*self.mu0*self.sigma0 - self.mu0**3)*self.p[0] - (3*self.mu1*self.sigma1 - self.mu1**3)*self.p[1]) ** 2

        loss_4 = 0.001 * torch.sum(F.relu(self.mu0 - self.mu1)) # mu1 > mu0
        loss_5 = 10*torch.sum(F.relu(-self.sigma0)) + 10*torch.sum(F.relu(-self.sigma1))

        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5 #+ loss_6


    def _set_class_balance(self, class_balance, Y_dev):
        """Set a prior for the class balance

        In order of preference:
        1) Use user-provided class_balance
        2) Estimate balance from Y_dev
        3) Assume uniform class distribution
        """
        if class_balance is not None:
            self.p = np.array(class_balance)
        elif Y_dev is not None:
            class_counts = Counter({c: 0 for c in range(1, self.k + 1)})
            class_counts.update(Y_dev)
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
            self.p = sorted_counts / sum(sorted_counts)

            if 0 in self.p:
                self.p = np.clip(self.p, 0.01, 0.99)
        else:
            self.p = (1 / self.k) * np.ones(self.k)
        self.P = torch.diag(torch.from_numpy(self.p)).float()
        
        self.P = self.P.to(self.device)


    def _set_constants(self, L):
        self.n, self.m = L.shape
        self.t = 1


    def train_model(
        self,
        L_train,
        Y_dev=None,
        class_balance=None,
        log_writer=None,
        **kwargs,
    ):
        """Train the model (i.e. estimate mu) in one of two ways, depending on
        whether source dependencies are provided or not:

        Args:
            L_train: An [n,m] scipy.sparse matrix with values in {0,1,...,k}
                corresponding to labels from supervision sources on the
                training set
            Y_dev: Target labels for the dev set, for estimating class_balance
            deps: (list of tuples) known dependencies between supervision
                sources. If not provided, sources are assumed to be independent.
                TODO: add automatic dependency-learning code
            class_balance: (np.array) each class's percentage of the population

        (1) No dependencies (conditionally independent sources): Estimate mu
        subject to constraints:
            (1a) O_{B(i,j)} - (mu P mu.T)_{B(i,j)} = 0, for i != j, where B(i,j)
                is the block of entries corresponding to sources i,j
            (1b) np.sum( mu P, 1 ) = diag(O)

        (2) Source dependencies:
            - First, estimate Z subject to the inverse form
            constraint:
                (2a) O_\Omega + (ZZ.T)_\Omega = 0, \Omega is the deps mask
            - Then, compute Q = mu P mu.T
            - Finally, estimate mu subject to mu P mu.T = Q and (1b)
        """
        self.config = recursive_merge_dicts(self.config, kwargs, misses="ignore")

        # TODO: Implement logging for label model?
        if log_writer is not None:
            raise NotImplementedError("Logging for LabelModel.")


        self._set_class_balance(class_balance, Y_dev)
        self._set_constants(L_train)
        self._check_L(L_train)
        
        # Whether to take the simple conditionally independent approach, or the
        # "inverse form" approach for handling dependencies
        # This flag allows us to eg test the latter even with no deps present

        # Creating this faux dataset is necessary for now because the LabelModel
        # loss functions do not accept inputs, but Classifer._train_model()
        # expects training data to feed to the loss functions.
        dataset = MetalDataset([0], [0])
        train_loader = DataLoader(dataset)

        L_train = torch.from_numpy(L_train)

        # Compute O and initialize params
        if self.config["verbose"]:
            print("Computing observable statistics...")
        self._generate_mean(L_train)
        self._generate_cov(L_train)
        self._generate_third_moment(L_train)
        self._init_params()


        print(f"Initial params: {self.mu0}, {self.mu1}, {self.sigma0}, {self.sigma1}")

        # Estimate \mu
        if self.config["verbose"]:
            print("Estimating \mu...")
        self._train_model(train_loader, self.loss_mu)


