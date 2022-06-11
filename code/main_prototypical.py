from data_loaders import *
from models import *
from losses import *
from utils import *
from tsne import *

from scipy.io import savemat
import numpy as np
import torch
import sklearn
import imutils
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from scipy.io import savemat

# set random seed for all gpus
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---- INPUTS

dir_dataset = '../data_mdnorm_allfreqs/'
items = ['accel_izq', 'accel_der']
repetitions = 10
input_shape = (2, 256, 320)
n = 4  # queries
m = 4  # support
n_blocks = 2
epochs = 200 * int(n/4)
pre_training_ae_epochs = 0
learning_rate = 5*1e-3
distance_train = 'l2'
distance_inference = 'l2'
n_folds = 4
n_classes = 2
deterministic = False
projection = True
detach = False
pretrained = False
l2_norm = True
contrastive_loss = True
prototypical_inference = True

# ---- DATA SETS AND MODELS

# Set data generator
dataset_cruces = Dataset(dir_dataset=dir_dataset + 'cruces/', items=items, input_shape=input_shape)
dataset_cruces.label_cruces_adif()  # Labels from adif-ineco method

# ---- LOO VALIDATION
accuracy_repetitions = []
precision_repetitions = []
recall_repetitions = []
cm_repetitions = []
f1_repetitions = []

for i_repetition in np.arange(0, repetitions):

    X_all_cruces = dataset_cruces.X
    labels_all_cruces = dataset_cruces.labels
    preds_all = []
    refs_all = []
    idx_samples = np.arange(0, dataset_cruces.labels.shape[0])
    random.seed(seed)
    random.shuffle(idx_samples)
    step = round(dataset_cruces.labels.shape[0] / n_folds)

    for iFold in np.arange(0, n_folds):
        print('Fold ' + str(iFold+1) + '/' + str(n_folds))

        # Set cross validation sampls
        idx_test = list(idx_samples[iFold*step:(iFold+1)*step])
        idx_train = list(idx_samples[0:iFold*step]) + list(idx_samples[(iFold+1)*step:])

        labels_cruces_train = labels_all_cruces[idx_train]
        X_cruces_train = X_all_cruces[idx_train]
        X_cruces_test = X_all_cruces[idx_test]

        # Set train generator
        dataset_cruces_fold = dataset_cruces
        dataset_cruces_fold.indexes = idx_train
        data_generator_main = GeneratorFSL(dataset_cruces_fold, n=n, m=m, shuffle=True, classes=[0, 1])

        # Set model architectures
        E = Resnet(2, n_blocks=n_blocks, pretrained=pretrained)
        if not detach:
            params = list(E.parameters())
        else:
            params = []

        if deterministic:
            Classifier = torch.nn.Sequential(torch.nn.Linear(E.nfeats, n_classes))
            Classifier.cuda()
            params += list(Classifier.parameters())

        # Projection module over feature space
        if projection:
            Proj = torch.nn.Sequential(torch.nn.Linear(E.nfeats, E.nfeats // 4),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(E.nfeats // 4, E.nfeats // 4))
        else:
            Proj = torch.nn.Sequential()
        params += list(Proj.parameters())

        # Set Losses
        Lcontrastive = SupConLoss().cuda()

        # Set optimizer
        opt = torch.optim.Adam(params, lr=learning_rate)

        # ---- TRAINING

        # Move modules to gpu
        E.cuda()
        Proj.cuda()

        # Iterate over epochs
        for i_epoch in np.arange(0, epochs):
            # Init epoch losses
            lr_epoch = 0
            lclust_epoch = 0
            lapproach_epoch = 0
            Lcont_epoch = 0

            # Iterate over batches
            for i_iteration, (Xs, Xq, Y) in enumerate(data_generator_main):
                opt.zero_grad()  # Clear gradients

                # Move inputs to tensors
                Xs = torch.tensor(Xs).cuda().float()
                Xq = torch.tensor(Xq).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # Init losses
                L = torch.tensor(0.).cuda()

                # Forward query samples
                zq = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(Xq)[0]))
                zq = Proj(zq)
                if l2_norm:
                    zq = torch.nn.functional.normalize(zq, dim=1)

                if deterministic:
                    # Forward through classifier
                    logits = Classifier(zq)

                else:
                    # Forward support samples
                    E.eval()
                    x_ = Xs.view(Xs.shape[0]*Xs.shape[1], Xs.shape[2], Xs.shape[3], Xs.shape[4])
                    zs = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(x_)[0]))
                    zs = Proj(zs)
                    if l2_norm:
                        zs = torch.nn.functional.normalize(zs, dim=1)
                    zs = zs.view(Xs.shape[0], Xs.shape[1], zs.shape[-1])
                    E.train()

                    # Calculate centroids
                    C = torch.mean(zs, dim=1)

                    # Get softmax predictions via distances
                    x1 = zq.unsqueeze(1).repeat(1, 2, 1)  # Embeddings
                    x2 = C.unsqueeze(0).repeat(x1.shape[0], 1, 1)  # Cluster centers

                    if distance_train == 'l2':
                        logits = - torch.sum(torch.square(x1 - x2), dim=2)
                    elif distance_train == 'cosine':
                        logits = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

                # Get predicted probabilities
                p = torch.softmax(logits, dim=1)

                # Calculate losses and include in overall losses
                if not contrastive_loss:
                    Lclust_iteration = torch.nn.BCELoss()(p, Y)
                else:
                    Lclust_iteration = Lcontrastive(zq.unsqueeze(1), Y[:, 1])
                L += Lclust_iteration

                # Backward and weights update
                L.backward()  # Backward
                opt.step()  # Update weights
                opt.zero_grad()  # Clear gradients

                # Update epoch's losses
                if i_epoch >= pre_training_ae_epochs:
                    Lclust_iteration = Lclust_iteration.cpu().detach().numpy()
                    lclust_epoch += Lclust_iteration / len(data_generator_main)

                # Display training information per iteration
                info = "[INFO] Epoch {}/{}  -- Step {}/{}: ".format(
                    i_epoch + 1, epochs, i_iteration + 1, len(data_generator_main))
                if i_epoch >= pre_training_ae_epochs:
                    info += ", L_classification={:.6f}".format(Lclust_iteration)
                print(info, end='\r')

            # Display training information per epoch
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: ".format(
                i_epoch + 1, epochs, len(data_generator_main), len(data_generator_main))
            if i_epoch >= pre_training_ae_epochs:
                info += ", L_classification={:.6f}".format(lclust_epoch)
            print(info, end='\n')

            if i_epoch >= pre_training_ae_epochs:
                E.eval()

                # Take support samples and calculate centroids
                centroids = []
                for iCluster in [0, 1]:
                    xx = X_cruces_train[np.array(list(np.argwhere(labels_cruces_train == iCluster)))[:, 0]]
                    xx = torch.tensor(xx).cuda().float()
                    zz = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(xx)[0]))
                    if not contrastive_loss:
                        zz = Proj(zz)
                        if l2_norm:
                            zz = torch.nn.functional.normalize(zz, dim=1)

                    centroids.append(zz.mean(0).unsqueeze(0))

                z_support = torch.cat(centroids)

                x_cruce = torch.tensor(X_cruces_test).cuda().float()

                z_cruce, F = E(x_cruce)
                z_cruce = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(z_cruce))

                if not contrastive_loss:
                    z_cruce = Proj(z_cruce)
                    if l2_norm:
                        z_cruce = torch.nn.functional.normalize(z_cruce, dim=1)

                # Get softmax predictions via distances
                x1 = z_cruce.unsqueeze(1).repeat(1, 2, 1)  # Embeddings
                x2 = z_support.unsqueeze(0).repeat(x1.shape[0], 1, 1)  # Cluster centers

                if not prototypical_inference:
                    # Forward through classifier
                    logits = Classifier(z_cruce)
                else:
                    if distance_inference == 'l2':
                        logits = - torch.sum(torch.square(x1 - x2), dim=2)
                    elif distance_inference == 'cosine':
                        logits = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

                p = torch.softmax(logits, dim=1)

                '''

                cams = grad_cam(F[-1], torch.sum(logits[:, 0]), normalization='tanh_min_max', avg_grads=False, norm_grads=False)
                #cams = cams - torch.min(cams)
                #cams = cams / (1e-3 + torch.max(cams))

                # Restore original shape
                cams = torch.nn.functional.interpolate(cams.unsqueeze(0), size=(input_shape[1], input_shape[2]),
                                                       mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()
                x_cruce_plot = x_cruce[:, 1, :, :].detach().cpu().numpy()

                for i in [3]:
                    m_i = np.uint8(cams[i, :, :] * 255)
                    heatmap_m = cv2.applyColorMap(m_i, cv2.COLORMAP_JET)
                    # Move grayscale image to three channels
                    xh = cv2.cvtColor(np.uint8(np.squeeze(x_cruce_plot[i, :, :]) * 255), cv2.COLOR_GRAY2RGB)
                    # Combine original image and masks
                    fin_mask = cv2.addWeighted(xh, 0.7, heatmap_m, 0.3, 0)
                    plt.imshow(fin_mask)
                    plt.show()

                '''

                # Get predictions and embeddings
                preds = np.argmax(p.detach().cpu().numpy(), 1)
                refs = labels_all_cruces[idx_test]

                # Calculate metrics
                acc = sklearn.metrics.accuracy_score(refs, preds)
                f1 = sklearn.metrics.f1_score(refs, preds, average='macro')

                print("Accuracy={:.6f} ; F1-Score={:.6f}".format(acc, f1, end='\n'))
                E.train()

        preds_all.extend(list(preds))
        refs_all.extend(list(refs))

    preds_all = np.array(preds_all)
    refs_all = np.array(refs_all)

    # Confussion matrix
    cm = confusion_matrix(refs_all, preds_all, labels=[0, 1])

    # Calculate metrics
    acc = sklearn.metrics.accuracy_score(refs_all, preds_all)
    precision, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(refs_all, preds_all)
    precision = precision[1]
    recall = recall[1]
    f1 = 2 * (precision * recall) / ((precision + recall) + 1e-3)

    print('-'*40)
    print('-'*40)
    print('Repetition metrics: ')
    print("Accuracy={:.6f} ; Precision={:.6f} ; Recall={:.6f} ; F1={:.6f}".format(acc, precision, recall, f1, end='\n'))
    print('-'*40)
    print('-'*40)

    savemat('predicted_clustering_LOO.mat', {'preds': preds_all, 'samples': list(np.array(dataset_cruces.files)[idx_samples.astype(int)])})

    accuracy_repetitions.append(acc)
    precision_repetitions.append(precision)
    recall_repetitions.append(recall)
    f1_repetitions.append(f1)
    cm_repetitions.append(cm)

acc_avg = np.mean(accuracy_repetitions)
acc_std = np.std(accuracy_repetitions)
precision_avg = np.mean(precision_repetitions)
precision_std = np.std(precision_repetitions)
recall_avg = np.mean(recall_repetitions)
recall_std = np.std(recall_repetitions)
f1_avg = np.mean(f1_repetitions)
f1_std = np.std(f1_repetitions)
cm = np.mean(np.array(cm_repetitions), axis=0)

print('-' * 40)
print('-' * 40)
print('Overall Repetitions metrics: ')
print(cm)
print("Accuracy={:.6f}({:.6f}) ; Precision={:.6f}({:.6f})  ; Recall={:.6f}({:.6f}) ; f1-score={:.6f}({:.6f})".format(
      acc_avg, acc_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std,  end='\n'))
print('-' * 40)
print('-' * 40)

savemat('proposed_good_metrics_fc_8.mat', {'accuracy': accuracy_repetitions, 'f1': f1_repetitions})