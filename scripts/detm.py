import logging
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import nn

logger = logging.getLogger("detm")

class DETM(torch.nn.Module):
    '''
    Overall generative process for DETM:
    1. Draw initial topic embedding alpha^0_k \in N(0, I)
    2. Draw initial topic proportion mean eta_0 \in N (0, I)
    3. For time step t = 1, . . . , T:
        (a) Draw topic embeddings alpha^t_k \in N (alpha^{t-1}_k, gamma^2I) 
            for k = 1, . . . , K
        (b) Draw topic proportion means eta_t \in N (eta_{t-1},a^2I)
    4. For each document d:
        (a) Draw topic proportions theta_d \in LN (eta_td , a^2I).
        (b) For each word n in the document:
            i. Draw topic assignment z_dn \in Cat(theta_d ).
            ii. Draw word w_dn \in Cat(softmax(rho^T alpha^{td}_{z_dn})).
    '''
    def __init__(
        self,
        args,
        id2token,
        min_time,
        embeddings=None,
        # device="cpu",
        # adapt_embeddings=False,
    ):
        super(DETM, self).__init__()
        self.min_time = min_time
        self.device = args.device
        self.window_size = args.window_size
        ## define hyperparameters
        self.num_topics = args.num_topics
        self.num_times = args.num_times
        self.vocab_size = args.vocab_size # V: number of volabulary
        self.t_hidden_size = args.t_hidden_size # window number?
        self.eta_hidden_size = args.eta_hidden_size
        # "Let rho be an L * V matrix containing
        # "L-dimensional embeddings of the words in the vocabulary, 
        # "such that each column rho_v in R^L corresponds 
        # "to the embedding representation of the vth term."
        self.rho_size = args.rho_size # this or the next one is L?
        self.emsize = args.emb_size # I am guessing this is L? L-dimensional embedding?
        # I think for the topic model to work, we need to have rho_size == emb_size == L (embedding size)
        self.enc_drop = args.enc_drop
        # "eta_t is a latent variable that controls the prior
        # "mean over the topic proportions at time t."
        self.eta_nlayers = args.eta_nlayers
        self.eta_dropout = args.eta_dropout
        self.t_drop = torch.nn.Dropout(args.enc_drop)
        self.delta = args.delta # topic proportion distribution noise?
        self.train_embeddings = args.train_embeddings
        # the softmax at the end?
        self.theta_act = self.get_activation(args.theta_act)

        ## define the word embedding matrix \rho
        if args.train_embeddings:
            self.rho = torch.nn.Linear(self.rho_size, self.vocab_size, bias=False)
        else:
            # self.rho becomes a copy of embeddings, which are the pre-trained word embeddings
            num_embeddings, emsize = embeddings.size()
            # I am supposing num_embeddings == self.vocab_size, 
            #                emsize == self.emsize?
            rho = torch.nn.Embedding(num_embeddings, emsize) 
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(self.device)

        # conversion between id and token, 
        # should not be influential for the training process
        self.id2token = id2token

        # alpha is the topic embedding
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(
            torch.randn(self.num_topics, self.num_times, self.rho_size)
        )
        self.logsigma_q_alpha = nn.Parameter(
            torch.randn(self.num_topics, self.num_times, self.rho_size)
        )

        # theta is the derived topic distribution for each document via logistical normal distribution
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
            # why vocab_size + num_topics? are we putting the vocab and the topic embedding together?
            nn.Linear(self.vocab_size + self.num_topics, self.t_hidden_size),
            self.theta_act,
            nn.Linear(self.t_hidden_size, self.t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(
            self.t_hidden_size, self.num_topics, bias=True
        )

        # eta is the topic proportion mean for each time
        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)
        # KEY: eta uses LSTM
        self.q_eta = nn.LSTM(
            self.eta_hidden_size,
            self.eta_hidden_size,
            self.eta_nlayers,
            dropout=self.eta_dropout,
        )
        self.mu_q_eta = nn.Linear(
            self.eta_hidden_size + self.num_topics, self.num_topics, bias=True
        )
        self.logsigma_q_eta = nn.Linear(
            self.eta_hidden_size + self.num_topics, self.num_topics, bias=True
        )

    def get_activation(self, act):
        if act == "tanh":
            act = nn.Tanh()
        elif act == "relu":
            act = nn.ReLU()
        elif act == "softplus":
            act = nn.Softplus()
        elif act == "rrelu":
            act = nn.RReLU()
        elif act == "leakyrelu":
            act = nn.LeakyReLU()
        elif act == "elu":
            act = nn.ELU()
        elif act == "selu":
            act = nn.SELU()
        elif act == "glu":
            act = nn.GLU()
        else:
            print("Defaulting to tanh activations...")
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) )."""
        # formula used: 
        # L(v) = E_q [log p(D, theta, eta, alpha) - log q_v (theta, eta, alpha)]
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu) ** 2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(
                1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1
            )
        return kl

    def get_alpha(self):  ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(
            self.device
        ) # alpha size: T * K * L
        kl_alpha = []

        alphas[0] = self.reparameterize(
            self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :]
        )

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        kl_0 = self.get_kl(
            self.mu_q_alpha[:, 0, :],
            self.logsigma_q_alpha[:, 0, :],
            p_mu_0,
            logsigma_p_0,
        )
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(
                self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]
            )

            p_mu_t = alphas[t - 1]
            logsigma_p_t = torch.log(
                self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device)
            )
            kl_t = self.get_kl(
                self.mu_q_alpha[:, t, :],
                self.logsigma_q_alpha[:, t, :],
                p_mu_t,
                logsigma_p_t,
            )
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_eta(self, rnn_inp, is_train=True):  ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
        inp_t = torch.cat(
            [
                output[0],
                torch.zeros(self.num_topics, device=self.device),
            ],
            dim=0,
        )
        mu_t = self.mu_q_eta(inp_t)
        logsigma_t = self.logsigma_q_eta(inp_t) if is_train else 0
        etas[0] = self.reparameterize(mu_t, logsigma_t) if is_train else mu_t

        kl_eta = []
        if is_train:
            kl_eta.append(
                self.get_kl(
                    mu_t, 
                    logsigma_t, 
                    p_mu=torch.zeros(self.num_topics, device=self.device), 
                    p_logsigma=torch.zeros(self.num_topics, device=self.device)
                )
            )

        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t - 1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t) if is_train else 0
            etas[t] = self.reparameterize(mu_t, logsigma_t) if is_train else mu_t

            if is_train:
                kl_eta.append(
                    self.get_kl(
                        mu_t, 
                        logsigma_t, 
                        p_mu=etas[t - 1],
                        p_logsigma=torch.log(
                    self.delta
                    * torch.ones(self.num_topics, device=self.device)
                )))
    
        return (etas, torch.stack(kl_eta).sum()) if is_train else (etas, None)

    def get_theta(self, eta_td, normalized_bows, is_train=True):  ## amortized inference
        """Returns the topic proportions."""
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0 and is_train:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta) if is_train else mu_theta
        theta = F.softmax(z, dim=-1)
        if is_train:
            kl_theta = self.get_kl(
                mu_theta,
                logsigma_theta,
                eta_td,
                torch.zeros(self.num_topics, device=self.device),
            )
        return (theta, kl_theta) if is_train else (theta, None)

    def get_beta(self, alpha):
        """Returns the topic matrix \beta of shape K x V"""
        if self.train_embeddings:
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta

    def get_nll(self, theta, beta, bows, is_train=True):
        theta = theta.unsqueeze(1)  
        lik = torch.bmm(theta, beta).squeeze(1) 
        # TODO: I think conceptually they should be the same but I put it there
        # if is_train:
        #     theta = theta.unsqueeze(1)  
        #     lik = torch.bmm(theta, beta).squeeze(1)
        # else:
        #     lik = (theta.unsqueeze(2) * beta).sum(1) 

        loglik = torch.log(lik + 1e-6) 
        nll = -loglik * bows 
        nll = nll.sum(-1) 
        return nll

    def get_lik(self, theta, beta):
        lik = theta.unsqueeze(2) * beta
        return lik

    def forward(self, bows, normalized_bows, times, rnn_inp, 
                num_docs=None, is_train=True, get_lik = False):
        
        if is_train:
            bsz = normalized_bows.size(0)
            coeff = num_docs / bsz
            alpha, kl_alpha = self.get_alpha()

        else:
            alpha = self.mu_q_alpha
            kl_alpha = None

        # get eta
        eta, kl_eta = self.get_eta(rnn_inp, is_train=is_train)
        eta_td = eta[times.type("torch.LongTensor")]

        # get theta
        theta, kl_theta = self.get_theta(eta_td, normalized_bows, is_train=is_train)
        kl_theta = kl_theta.sum() * coeff if is_train else None

        # get beta
        if is_train:

            beta = self.get_beta(alpha)
            beta = beta[times.type("torch.LongTensor")]
        else:
            beta = self.get_beta(alpha[:, times.type("torch.LongTensor"), :])
            beta = beta.permute(1, 0, 2) 
        
        if get_lik:
            lik = self.get_lik(theta, beta)
            return lik

        nll, lik = self.get_nll(theta, beta, bows, is_train=is_train)

        if is_train:
            nll = nll.sum() * coeff
            nelbo = nll + kl_alpha + kl_eta + kl_theta
            loss = None
        else:
            nelbo = None
            sums = bows.sum(dim=1, keepdim=True)
            sums[sums == 0] = 1e-6
            loss = nll / sums.squeeze()
            loss = loss.mean().item()

        self.update_stats(nelbo, loss, nll, kl_alpha, kl_eta, kl_theta, is_train=is_train)

        return nelbo

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta."""
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

    def start_epoch(self):

        self.train_acc_nelbo = 0
        self.train_acc_nll = 0
        self.train_acc_kl_alpha_loss = 0
        self.train_acc_kl_eta_loss = 0
        self.train_acc_kl_theta_loss = 0
        self.train_cnt = 0

        self.eval_acc_loss = 0
        self.eval_cnt = 0
            
    def update_stats(self, nelbo, loss, nll, kl_alpha, kl_eta, kl_theta, is_train=True):

        if is_train:
            self.train_acc_nelbo += torch.sum(nelbo).item()
            self.train_acc_nll += torch.sum(nll).item()
            self.train_acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            self.train_acc_kl_eta_loss += torch.sum(kl_eta).item()
            self.train_acc_kl_theta_loss += torch.sum(kl_theta).item()
            self.train_cnt += 1
        
        else:
            self.eval_acc_loss += loss
            self.eval_cnt += 1
    
    def log_stats(self, epoch_num, lr, logger):

        eval_ppl = math.exp(self.eval_acc_loss / self.eval_cnt)

        logger.info(
        "Epoch {}: LR: {}, KL_theta: {}, KL_eta: {}, KL_alpha: {}, Rec_loss: {}, NELBO: {}, PPL: {}".format(
                epoch_num, lr,
                round(self.train_acc_kl_theta_loss / self.train_cnt, 2),
                round(self.train_acc_kl_eta_loss / self.train_cnt, 2),
                round(self.train_acc_kl_alpha_loss / self.train_cnt, 2),
                round(self.train_acc_nll / self.train_cnt, 2),
                round(self.train_acc_nelbo / self.train_cnt, 2),
                round(eval_ppl, 1)
            )
        )

        return eval_ppl