"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import argparse

import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange

import config
from dcopy import MNISTM
from models import Net
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb

import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def onehot_encoding(k,d):
    rtn = np.zeros(d)
    rtn[k] = 1.0
    return rtn

def main(args):
    clf_model = Net().to(device)
    clf_model.load_state_dict(torch.load(args.MODEL_FILE))
    
    feature_extractor = clf_model.feature_extractor
    discriminator = clf_model.classifier

    critic = nn.Sequential(
        nn.Linear(330, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,
                               shuffle=True, num_workers=0, pin_memory=True)
    
    target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch, drop_last=True,
                               shuffle=True, num_workers=0, pin_memory=True)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    clf_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            (source_x, source_y), (target_x, target_y) = next(batch_iterator)
            # Train critic
            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            source_y = source_y.to(device)
            
            with torch.no_grad():
                h_s = feature_extractor(source_x).data.view(source_x.shape[0], -1)
                h_t = feature_extractor(target_x).data.view(target_x.shape[0], -1)
                d = 10 #numberoflabels0,1,2,3...9
                M = 10 # M is the parameter standing for how much we want to let observation get involved.

                suffix_s = [M*onehot_encoding(source_y[_i].numpy(),d) for _i in range(len(source_x))]
                suffix_s = torch.tensor(suffix_s,dtype=torch.float32)

                h_s_suf = torch.cat((h_s,suffix_s),1)

                suffix_t = []
                for _i in range(len(target_x)):
                    if target_y[_i] == -1:
                        suffix_t.append( np.zeros(d))
                    else:
                        suffix_t.append(M*onehot_encoding(target_y[_i].numpy(),d))
                suffix_t = torch.tensor(suffix_t,dtype=torch.float32)

                h_t_suf = torch.cat((h_t,suffix_t),1)



            for _ in range(args.k_critic):
                gp = gradient_penalty(critic, h_t_suf, h_t_suf)

                critic_s = critic(h_t_suf)
                critic_t = critic(h_t_suf)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma*gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()

            # Train classifier
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                target_features = feature_extractor(target_x).view(target_x.shape[0], -1)

                source_preds = discriminator(source_features)
                clf_loss = clf_criterion(source_preds, source_y)


                suffix = [np.zeros(d) for _i in range(len(source_features))]
                suffix = torch.tensor(suffix,dtype=torch.float32)
                source_features_suf = torch.cat((source_features,suffix),1)
                target_features_suf = torch.cat((target_features,suffix),1)

                wasserstein_distance = critic(source_features_suf).mean() - critic(target_features_suf).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')
        torch.save(clf_model.state_dict(), 'trained_models/wdgrlcopy.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using WDGRL')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--iterations', type=int, default=500)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--k-critic', type=int, default=5)
    arg_parser.add_argument('--k-clf', type=int, default=1)
    arg_parser.add_argument('--gamma', type=float, default=10)
    arg_parser.add_argument('--wd-clf', type=float, default=1)
    args = arg_parser.parse_args()
    main(args)
