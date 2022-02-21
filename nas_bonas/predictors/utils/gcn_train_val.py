import torch
import numpy as np
from scipy.stats import spearmanr
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, optimizer, loss, train_loader, epoch):
    logging.info("training gcn ... ")
    total_loss_train = 0
    count = 0
    total_difference = 0
    predicted = []
    ground_truth = []
    model.train()

    #iss = []
    #batches=[]
    for i_batch, sample_batched in enumerate(train_loader): # bisher wird nur get_item ausgeführt
        #iss.append(i_batch)
        #batches.append(sample_batched)
        adjs_cnn, features_cnn, adjs_rhn, features_rhn, accuracys = sample_batched['adjacency_matrix_cnn'], sample_batched['operations_cnn'], sample_batched['adjacency_matrix_rhn'], sample_batched['operations_rhn'], \
                                    sample_batched['accuracy'].view(-1, 1)
                                  
        adjs_cnn, features_cnn, adjs_rhn, features_rhn, accuracys = adjs_cnn.to(device), features_cnn.to(device), adjs_rhn.to(device), features_rhn.to(device), accuracys.to(device) # .cuda()
#        if i_batch == 1:
        print('shape_gcn_tensors')
        print(adjs_cnn.shape)
        optimizer.zero_grad()
        
        outputs = model(features_cnn, adjs_cnn, features_rhn, adjs_rhn) # feat_cnn, adj_cnn, feat_rhn, adj_rhn,
        loss_train = loss(outputs, accuracys)
        loss_train.backward()
        optimizer.step()
        count += 1
        difference = torch.mean(torch.abs(outputs - accuracys), 0) # y_pred-y_true
        total_difference += difference.item()
        total_loss_train += loss_train.item()
        vx = outputs.cpu().detach().numpy().flatten() # y_preds
        vy = accuracys.cpu().detach().numpy().flatten() # y_trues
        predicted.append(vx)
        ground_truth.append(vy)
    predicted = np.hstack(predicted)
    ground_truth = np.hstack(ground_truth)
    corr, p = spearmanr(predicted, ground_truth)
    logging.info("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
        total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "corr:{:.6f}".format(
        corr))


def validate(model, loss, validation_loader, logging=None):
    loss_val = 0
    overall_difference = 0
    count = 0
    predicted = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(validation_loader):
           
                                        
            adjs_cnn, features_cnn, adjs_rhn, features_rhn, accuracys = sample_batched['adjacency_matrix_cnn'], sample_batched['operations_cnn'], sample_batched['adjacency_matrix_rhn'], sample_batched['operations_rhn'], \
                                    sample_batched['accuracy'].view(-1, 1)
                                    
            adjs_cnn, features_cnn, adjs_rhn, features_rhn, accuracys = adjs_cnn.to(device), features_cnn.to(device), adjs_rhn.to(device), features_rhn.to(device), accuracys.to(device)# .cuda()
                                        
            outputs = model(features_cnn, adjs_cnn, features_rhn, adjs_rhn)
            
            loss_train = loss(outputs, accuracys)
            count += 1
            difference = torch.mean(torch.abs(outputs - accuracys), 0)
            overall_difference += difference.item()
            loss_val += loss_train.item()
            vx = outputs.cpu().detach().numpy().flatten()
            vy = accuracys.cpu().detach().numpy().flatten()
            predicted.append(vx)
            ground_truth.append(vy)
        predicted = np.hstack(predicted)
        ground_truth = np.hstack(ground_truth)
        corr, p = spearmanr(predicted, ground_truth)
    logging.info("test result " + " loss= {:.6f}".format(loss_val / count) + " abs_error:{:.6f}".format(
        overall_difference / count) + " corr:{:.6f}".format(corr))
    return corr, overall_difference / count, loss_val / count
