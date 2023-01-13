# Implementation of paper: Unsupervised Deep Learning-based clustering for Human Activity Recognition
# Authors: Hamza Amrani, Daniela Micucci, Paolo Napoletano
# Department of Informatics, Systems and Communication
# University of Milano - Bicocca , Milan, Italy
# Email: h.amrani@campus.unimib.it, daniela.micucci@unimib.it, paolo.napoletano@unimib.it

import os
import time
import torch
import argparse
import numpy as np
import random
import time

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import convolutional_rnn

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.cuda.set_device(0)
torch.cuda.empty_cache()

nmi = normalized_mutual_info_score
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    ind = np.concatenate((row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)), axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class Encoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, is_bidirectional, dropout, dropout_rnn
    ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        self.num_dir = 2 if self.is_bidirectional else 1

        self.rnn = convolutional_rnn.Conv1dGRU(in_channels=input_dim,
                                   out_channels=hidden_dim,
                                   kernel_size=1,
                                   num_layers=num_layers,
                                   bidirectional=is_bidirectional,
                                   dropout=0.2,
                                   batch_first=False)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x= torch.unsqueeze(x, 3)

        outputs, hidden = self.rnn(x)

        if self.is_bidirectional:
            fwd = outputs[-1, :, : self.hidden_dim]
            bwd = outputs[0, :, self.hidden_dim :]
            output = torch.cat((fwd, bwd), dim=1)
            output = torch.squeeze(output)
        else:
            output = outputs[-1]

        return output

class DecoderRec(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, is_bidirectional, dropout, dropout_rnn
    ):
        super(DecoderRec, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        self.num_dir = 2 if self.is_bidirectional else 1

        self.rnn_dec = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=is_bidirectional,
            dropout=dropout_rnn,
        )
        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(self.num_dir * hidden_dim, input_dim)

    def forward(self, x_t, h_t):
        x_t = x_t.unsqueeze(0)
        o_t, hidden = self.rnn_dec(x_t, h_t)
        o_t = self.dense(o_t.squeeze(0))
        return o_t, hidden

class DecoderFut(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, is_bidirectional, dropout, dropout_rnn
    ):
        super(DecoderFut, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        self.num_dir = 2 if self.is_bidirectional else 1

        self.rnn_dec = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=is_bidirectional,
            dropout=dropout_rnn,
        )
        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(self.num_dir * hidden_dim, input_dim)

    def forward(self, x_t, h_t):
        x_t = x_t.unsqueeze(0)
        o_t, hidden = self.rnn_dec(x_t, h_t)
        o_t = self.dense(o_t.squeeze(0))
        return o_t, hidden

class AE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        emb_dim,
        enc_num_layers,
        enc_is_bidirectional,
        dec_num_layers,
        dec_is_bidirectional,
        dropout,
        dropout_rnn,
    ):
        super(AE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        self.enc_num_layers = enc_num_layers
        self.enc_is_bidirectional = enc_is_bidirectional
        self.enc_num_dir = 2 if self.enc_is_bidirectional else 1

        self.dec_num_layers = dec_num_layers
        self.dec_is_bidirectional = dec_is_bidirectional
        self.dec_num_dir = 2 if self.dec_is_bidirectional else 1

        # encoder
        enc_h_dim, dec_h_dim = self.hidden_dim, self.hidden_dim
        self.encoder = Encoder(
            input_dim,
            enc_h_dim,
            self.enc_num_layers,
            self.enc_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
        )

        self.fc1 = nn.Linear(self.enc_num_dir * enc_h_dim, self.emb_dim)
        self.bn1 = nn.BatchNorm1d(self.emb_dim)

        self.fc2 = nn.Linear(self.emb_dim, dec_h_dim)
        self.bn2 = nn.BatchNorm1d(dec_h_dim)

        self.fc3 = nn.Linear(self.enc_num_dir * enc_h_dim, dec_h_dim)
        self.bn3 = nn.BatchNorm1d(dec_h_dim)

        # decoders
        self.decoder_rec = DecoderRec(
            input_dim,
            dec_h_dim,
            self.dec_num_layers,
            self.dec_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
        )

        self.decoder_fut = DecoderFut(
            input_dim,
            dec_h_dim,
            self.dec_num_layers,
            self.dec_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
        )

    def forward(self, x, input_rec, input_fut, forcing=0.0):
        bs, max_len_rec, input_dim = input_rec.shape
        _, max_len_fut, _ = input_fut.shape

        enc_output = self.encoder(x)

        z = torch.tanh(self.bn1(self.fc1(enc_output)))
        hidden = torch.tanh(self.bn2(self.fc2(z)))

        outputs_rec = torch.zeros(max_len_rec, bs, input_dim).cuda()
        x_t_rec = torch.zeros(bs, input_dim).cuda()
        hidden_rec = hidden.unsqueeze(0).repeat(2, 1, 1)

        outputs_fut = torch.zeros(max_len_fut, bs, input_dim).cuda()
        x_t_fut = torch.zeros(bs, input_dim).cuda()
        hidden_fut = hidden.unsqueeze(0).repeat(2, 1, 1)

        for t in range(0, max_len_rec):
            o_t_rec, hidden_rec = self.decoder_rec(x_t_rec, hidden_rec)
            outputs_rec[t] = o_t_rec
            teacher_force = random.random() < forcing
            x_t_rec = input_rec[:, t] if teacher_force else o_t_rec
        x_rec = outputs_rec.permute((1, 0, 2))

        for t in range(0, max_len_fut):
            o_t_fut, hidden_fut = self.decoder_fut(x_t_fut, hidden_fut)
            outputs_fut[t] = o_t_fut
            teacher_force = random.random() < forcing
            x_t_fut = input_fut[:, t] if teacher_force else o_t_fut
        x_fut = outputs_fut.permute((1, 0, 2))

        return x_rec, x_fut, z

class Cluster(nn.Module):
    def __init__(self, n_clusters, emb_dim, alpha):
        super(Cluster, self).__init__()
        self.alpha = alpha
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, emb_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, z):
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), dim=2)
            / self.alpha
        )
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

class DISC_model(nn.Module):
    def __init__(
        self,
        n_clusters,
        input_dim,
        hidden_dim,
        emb_dim,
        enc_num_layers=2,
        enc_is_bidirectional=True,
        dec_num_layers=2,
        dec_is_bidirectional=False,
        dropout=0,
        dropout_rnn=0,
        alpha=1.0,
    ):
        super(DISC_model, self).__init__()

        self.emb_dim = emb_dim
        self.autoencoder = AE(
            input_dim,
            hidden_dim,
            emb_dim,
            enc_num_layers,
            enc_is_bidirectional,
            dec_num_layers,
            dec_is_bidirectional,
            dropout,
            dropout_rnn,
        )
        self.cluster = Cluster(n_clusters, self.emb_dim, alpha)
        self.pretrainMode = True

    def setPretrain(self,mode):
        self.pretrainMode = mode

    def updateClusterCenter(self, cc):
        self.cluster.data = torch.from_numpy(cc)

    def forward(self, x, input_rec, input_fut, forcing=0.0):
        x_rec, x_fut, z = self.autoencoder(x, input_rec, input_fut, forcing)

        if self.pretrainMode == False:
            q = self.cluster(z)
            return x_rec, x_fut, q, z
        return x_rec, x_fut, z

    def encode(self, x, input_rec, input_fut):
        return self.autoencoder(x, input_rec, input_fut)[-1]


class DeepInertialSensoryClustering:
    def __init__(self, n_clusters=6, input_dim=9, hidden_dim=256, embedding_dim=64, gamma=0.1, alpha=1.0):
        self.n_clusters=n_clusters
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def target_distribution(q):
        weight = (q ** 2  ) / q.sum(0)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)

    def logAccuracy(self,pred,label):
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
          % (cluster_acc(label, pred), nmi(label, pred)))

    @staticmethod
    def kld(q,p):
        return torch.sum(p*torch.log(p/q),dim=-1)

    @staticmethod
    def cross_entropy(q,p):
        return torch.sum(torch.sum(p*torch.log(1/(q+1e-7)),dim=-1))
    @staticmethod
    def depict_q(p):
        q1 = p / torch.sqrt(torch.sum(p,dim=0))
        qik = q1 / q1.sum()
        return qik

    @staticmethod
    def distincePerClusterCenter(dist):
        totalDist =torch.sum(torch.sum(dist, dim=0)/(torch.max(dist) * dist.size(1)))
        return totalDist

    def validateOnCompleteTestData(self,test_loader,model,cluster_method='kmeans'):
        to_eval = np.array([a
                            for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(test_loader)
                            for a in model(x.cuda(), input_rec.cuda(), input_fut.cuda(), 0.0)[-1].cpu().data.numpy()])

        true_labels = np.array([a
                        for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(test_loader)
                        for a in target.cpu().numpy() ])
        true_labels = true_labels.reshape((true_labels.shape[0]))

        if cluster_method=='kmeans':
            km = KMeans(n_clusters=len(np.unique(true_labels)), n_init=20)
            y_pred = km.fit_predict(to_eval)
        else:
            ac = AgglomerativeClustering(n_clusters=len(np.unique(true_labels)), affinity='euclidean', linkage='complete')
            y_pred = ac.fit_predict(to_eval)

        print(' '*8 + '|==>  nmi: %.4f,  acc: %.4f  <==|'
                    % (nmi(true_labels, y_pred), cluster_acc(true_labels, y_pred)))

    def pretrain(self,dataset,batch_size, epochs, pretrain_weights, cluster_method='kmeans'):
        print('START: DISC AutoEncoder pre-training...')

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                                shuffle=False)
        disc_ae = DISC_model(self.n_clusters, self.input_dim, self.hidden_dim, self.embedding_dim).cuda() #auto encoder
        print(disc_ae)

        if pretrain_weights is not None:
            disc_ae.load_state_dict(torch.load(pretrain_weights))
            print('Model found and loaded')
            with torch.no_grad():
                print('Evaluation on AutoEncoding Space')
                self.validateOnCompleteTestData(train_loader,disc_ae,cluster_method)
            return

        optimizer = torch.optim.Adam(disc_ae.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70)
        start_time = time.time()

        print('\nPre-training...')
        disc_ae.train()
        for epoch in range(epochs):
            scheduler.step()
            start_time_epoch = time.time()
            running_loss = 0.00
            running_loss_rec = 0.00
            running_loss_fut = 0.00
            for batch_idx, (
                            x,
                            input_rec,
                            input_fut,
                            target,
                            target_rec,
                            target_fut,
                            idx,
                        ) in enumerate(train_loader):

                x = x.cuda()
                target_rec = target_rec.cuda()
                input_rec = input_rec.cuda()
                target_fut = target_fut.cuda()
                input_fut = input_fut.cuda()

                # forward
                x_rec, x_fut, z = disc_ae(x, input_rec, input_fut, 0.0)
                loss_rec = F.mse_loss(x_rec, target_rec)
                loss_fut = F.mse_loss(x_fut, target_fut)
                loss = loss_rec + loss_fut

                # backward
                loss.backward()

                # weights update
                optimizer.step()
                optimizer.zero_grad()

                running_loss_rec += loss_rec
                running_loss_fut += loss_fut
                running_loss += loss.item()

            # ===================log========================
            len_loader = len(train_loader)
            print('epoch [{}/{}], lr: {}, \tloss: {:0.6f}, loss_rec: {:0.6f}, loss_fut: {:0.6f}, time: {:.2f}s'
                        .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'],
                                running_loss/len_loader, running_loss_rec/len_loader, running_loss_fut/len_loader,
                                time.time() - start_time_epoch))

            # ===================evaluation during training========================
            if epoch%20==0:
                with torch.no_grad():
                   self.validateOnCompleteTestData(train_loader,disc_ae,cluster_method)

        print('\nPre-training time: ',time.time() - start_time, 's')

        # ===================evaluation========================
        print('\nEvaluation on AutoEncoding Space')
        with torch.no_grad():
            for i in range(10):
                self.validateOnCompleteTestData(train_loader,disc_ae,cluster_method)

        print('\nSaving model...')
        torch.save(disc_ae.state_dict(), "ae_pretrain_train.pth")
        print('END: DISC AutoEncoder pre-training...')

    def fit(self, dataset,batch_size, tol=0.001, cluster_init='kmeans'):
        print('START: DISC Clustering ...')

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                                shuffle=False)
        print("Step 1: load pretrained model")
        model = DISC_model(self.n_clusters, self.input_dim, self.hidden_dim, self.embedding_dim).cuda()
        model.setPretrain(False)
        model.load_state_dict(torch.load("h_mh_ae_pretrain_train.pth"))

        print ("Step 2: initialize cluster centers")
        with torch.no_grad():
            encoder_pred = np.array([a
                                for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(train_loader)
                                for a in model.encode(x.cuda(), input_rec.cuda(), input_fut.cuda()).cpu().data.numpy()])

            true_labels = np.array([a
                            for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(train_loader)
                            for a in target.cpu().numpy() ])
        true_labels = true_labels.reshape((true_labels.shape[0]))

        if cluster_init == 'kmeans':
            print('Initializing cluster centers with k-means.')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, verbose=0)
            y_pred = kmeans.fit_predict(encoder_pred)
            cluster_centers = [kmeans.cluster_centers_]
        elif cluster_init == 'ward':
            print('Initializing cluster centers with AC-Ward.')
            ac = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')
            y_pred = ac.fit_predict(encoder_pred)

            nc = NearestCentroid()
            nc.fit(encoder_pred, y_pred)
            cluster_centers = [nc.centroids_]

        y_pred_last = np.copy(y_pred)
        model.updateClusterCenter(np.array(cluster_centers))

        print("Step 3: deep clustering")
        ct = time.time()
        ite=0
        epochs = 200

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for epoch in range(epochs):
            with torch.no_grad():
                q = np.array([a
                        for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(train_loader)
                        for a in model(x.cuda(), input_rec.cuda(), input_fut.cuda())[2].cpu().data.numpy()])
            y_pred = q.argmax(1)

            # evaluate the clustering performance
            print(' '*8 + '|==>  nmi: %.4f,  acc: %.4f  <==|'
                % (nmi(true_labels, y_pred), cluster_acc(true_labels, y_pred)))

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break

            running_loss = 0.00
            running_loss_rec = 0.00
            running_loss_fut = 0.00
            running_loss_c = 0.00

            for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(train_loader):
                # train on batch
                x = x.cuda()
                target_rec = target_rec.cuda()
                input_rec = input_rec.cuda()
                target_fut = target_fut.cuda()
                input_fut = input_fut.cuda()

                optimizer.zero_grad()
                # forward
                x_rec, x_fut, q, z  = model(x, input_rec, input_fut)
                p = self.target_distribution(q)
                loss_c = self.kld(q,p).mean()
                loss_rec = F.mse_loss(x_rec, target_rec)
                loss_fut = F.mse_loss(x_fut, target_fut)
                loss_ae = loss_rec + loss_fut
                loss = self.gamma*loss_c + loss_ae

                # backward
                loss.backward()
                # weights update
                optimizer.step()

                running_loss_rec += loss_rec
                running_loss_fut += loss_fut
                running_loss_c += loss_c
                running_loss += loss.item()

                ite+=1
            len_loader = len(train_loader)
            print('epoch {}, loss: {:.4f}, loss_c: {:.5f}, loss_rec: {:.4f}, loss_fut: {:.4f} -- delta label: {:.4f}, tol: 0.001'
                                    .format(epoch, running_loss/len_loader, self.gamma*running_loss_c/len_loader,
                                            running_loss_rec/len_loader, running_loss_fut/len_loader, delta_label))

        print('\nSaving model...')
        if cluster_init == 'kmeans':
            torch.save(model.state_dict(), "disc_train_kmeans.pth")
        else:
            torch.save(model.state_dict(), "disc_train_ward.pth")

        print('END: DISC AutoEncoder pre-training...')

    def run_on_test(self, dataset,batch_size, cluster_init='kmeans'):
        print('Running on Test!')
        model = DISC_model(self.n_clusters, self.input_dim, self.hidden_dim, self.embedding_dim).cuda()
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                                shuffle=False)
        # Autoencoding space
        model.setPretrain(True)
        model.load_state_dict(torch.load("h_mh_ae_pretrain_train.pth"))
        with torch.no_grad():
            print('Evaluation on AutoEncoding Space')
            if cluster_init=='kmeans':
                for i in range(10):
                    self.validateOnCompleteTestData(test_loader,model,cluster_init)
            else:
                self.validateOnCompleteTestData(test_loader,model,cluster_init)

        # End-to-end Deep Clustering
        model.setPretrain(False)
        if cluster_init=='kmeans':
            model.load_state_dict(torch.load("disc_train_kmeans.pth"))
        else:
            model.load_state_dict(torch.load("disc_train_ward.pth"))
        with torch.no_grad():
            true_labels = np.array([a
                            for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(test_loader)
                            for a in target.cpu().numpy() ])
            q = np.array([a
                        for batch_idx, (x,input_rec,input_fut,target,target_rec,target_fut,idx,) in enumerate(test_loader)
                        for a in model(x.cuda(), input_rec.cuda(), input_fut.cuda())[2].cpu().data.numpy()])
        y_pred = q.argmax(1)
        true_labels = true_labels.reshape((true_labels.shape[0]))

        print('Evaluation on End-to-end Deep Clustering')
        print(' '*8 + '|==>  nmi: %.4f,  acc: %.4f  <==|'
            % (nmi(true_labels, y_pred), cluster_acc(true_labels, y_pred)))


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--train_epochs', default=2, type=int)
    args = parser.parse_args()
    batch_size = args.batch_size

    from load_UCI_dataset import load_UCI_dataset, prepare_for_DSC
    print("Loading dataset...")

    x, x_test, y, y_test, labels = load_UCI_dataset(flatten=False)
    x, x_inverted, x_future, x_test, x_test_inverted, x_test_future = prepare_for_DSC(x, x_test)

    print('data shape: ', x.shape)

    X_train = []
    for i in range(x.shape[0]):
        tmp = [x[i,:,:], x_inverted[i,:,:], x_future[i,:,:]]
        X_train.append(tmp)
    X_train = np.array(X_train)

    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.Tensor(y)   

    tensor_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    train_loader = torch.utils.data.DataLoader(
                    dataset=tensor_dataset,
                    num_workers=4, 
                    pin_memory=True,
                    batch_size=batch_size,
                    drop_last=True,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=tensor_dataset,
                    num_workers=4, 
                    pin_memory=True,
                    batch_size=batch_size,
                    drop_last=True,
                    shuffle=False)

    print("Deep Inertial Sensory Clustering")
    
    disc = DeepInertialSensoryClustering(num_classes=6, seq_len=64, n_features=9, gru_hidden_dim=256, embedding_dim=64,gamma=0.1)
    disc.pretrain(train_loader, test_loader, args.pretrain_epochs)
    disc.train(train_loader, test_loader, args.train_epochs)
