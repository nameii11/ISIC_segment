import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score as jac
from skimage import filters


def mean_iou(y_pred_in, y_true_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0

def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.7
    target_ = target > 0.7
    output_ = np.squeeze(output_)
    target_ = np.squeeze(target_)
    cm = confusion_matrix(target_.ravel(), output_.ravel()).ravel()
    #print(cm)
    # tn, fp, fn, tp = cm[0], cm[1], cm[2], cm[3]
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    smooth = 1e-5

    #m1 = output.ravel()  # Flatten
    #m2 = target.ravel()  # Flatten
    #intersection = (m1 * m2).sum()

    ja = (intersection + smooth) / (union + smooth)
    return ja




def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = torch.sigmoid(SR) > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum(((SR== 1) & (GT==1)))
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC


def dice_coeff(pred, target):
    smooth = 1e-5
    num = pred.size(0)
    pred = torch.sigmoid(pred).data.cpu().numpy()
    target = target.data.cpu().numpy()
    m1 = pred.ravel()  # Flatten
    m2 = target.ravel()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# def accuracy(output, target):
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     # output = (np.round(output)).astype('int')
#     target = target.view(-1).data.cpu().numpy()
#     output = output > 0.5
#     target = target > 0.5
#     # target = (np.round(target)).astype('int')
#     tp = int(((output != 0) * (target != 0)).sum())
#     fp = int(((output != 0) * (target == 0)).sum())
#     tn = int(((output == 0) * (target == 0)).sum())
#     fn = int(((output == 0) * (target != 0)).sum())
#
#     # output = np.squeeze(output_)
#     # target = np.squeeze(target_)
#     # (output == target).sum()
#
#     return float((tp + tn) / (tp + fp + tn + fn))
#     return (output == target).sum() / len(output)
def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)
def specificity(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    output = output > 0.6
    target = target > 0.6
    # target = (np.round(target)).astype('int')
    tp = int(((output != 0) * (target != 0)).sum())
    fp = int(((output != 0) * (target == 0)).sum())
    tn = int(((output == 0) * (target == 0)).sum())
    fn = int(((output == 0) * (target != 0)).sum())

    # output = np.squeeze(output_)
    # target = np.squeeze(target_)
    # (output == target).sum()

    return float((tn) / (fp + tn))

def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output = output
    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)


def getEdge(batch):
    edgeslist=[]
    for kk in range(batch.size(0)):
        x=batch[kk]
        # print(x.size())
        x=x.cpu().data.numpy()
        if len(x.shape)>2:
            x=np.transpose(x,(1,2,0))
            #x=rgb2gray(x)
        edges = filters.sobel(x)
        edgeslist.append(edges)
    edgeslist=np.array(edgeslist)
    edgeslist=torch.Tensor(edgeslist).cuda()
    edgeslist=torch.autograd.Variable(edgeslist)
    return  edgeslist