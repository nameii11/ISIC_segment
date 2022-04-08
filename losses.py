import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from skimage import filters
from scipy.ndimage.morphology import distance_transform_edt
from lovaszSoftmax import lovasz_hinge

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
    def forward(self, input, target):#, input_edge, edge

        #edge_loss = F.binary_cross_entropy(input_edge, edge)
        eps = 1e-6
        # bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        # num = target.size(0)
        # input = input.view(num, -1)
        # target = target.view(num, -1)
        # intersection = (input * target)
        # dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        # dice = 1 - dice.sum() / num
        intersection = input * target
        numerator = (input - intersection).sum() + (target - intersection).sum()
        denominator = input.sum() + target.sum()
        CEL = numerator / (denominator + eps)
        tversky = tversky_loss(input, target)
        #_, pred2 = (torch.max(input, 1))
        loss = tversky+ CEL #dice#F.binary_cross_entropy(input, target)### #lovasz_hinge(input, target)# + 0.5*EPE(edge, gt_edge)### + cel
        return loss

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[-1], y[0])
        #for i in range(1, len(x)):
        #    if weights[i] != 0:
        #        l += weights[i] * self.loss(x[i], y[0])
        return l


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        #print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

def tversky_loss(inputs, targets, beta=0.7, weights=None):
    batch_size = targets.size(0)
    loss = 0.0

    for i in range(batch_size):
        prob = inputs[i]
        ref = targets[i]

        alpha = 1.0-beta

        tp = (ref*prob).sum()
        fp = ((1-ref)*prob).sum()
        fn = (ref*(1-prob)).sum()
        tversky = tp/(tp + alpha*fp + beta*fn)
        loss = loss + (1-tversky)
    return loss/batch_size

def EPE(predicted_edge, gt_edge, sparse=False, mean=True):
    EPE_map = torch.norm(gt_edge-predicted_edge,2,1)
    if sparse:
        EPE_map = EPE_map[gt_edge != 0]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()


def getEdge(batch):
    edgeslist=[]
    for kk in range(batch.size(0)):
        x=batch[kk]
        # print(x.size())
        x=x.cpu().data.numpy()
        if len(x.shape)>2:
            x=np.transpose(x,(1,2,0))
            x=rgb2gray(x)
        edges = filters.sobel(x)
        edgeslist.append(edges)
    edgeslist=np.array(edgeslist)
    edgeslist=torch.Tensor(edgeslist).cuda()
    edgeslist=torch.autograd.Variable(edgeslist)
    return  edgeslist


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss