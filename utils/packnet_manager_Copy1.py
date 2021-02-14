import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#import visdom
from tqdm import tqdm
from torch.autograd import Variable
from . import Metric, classification_accuracy
from .packnet_prune import SparsePruner
from .metrics import fv_evaluate
from packnet_models import AngleLoss
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from .fast_gradient_sign_untargeted import FastGradientSignUntargeted
from .PGD_attack import LinfPGDAttack

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader, adv_train, attack, attack_val):
        self.args  = args
        self.model = model.cuda()
        #print('attack on ',next(model.parameters()).device)
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        self.pruner = SparsePruner(self.model, masks, self.args, None, None, self.inference_dataset_idx)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.adv_train = adv_train
        self.attack = attack
        self.attack_val = attack_val

        if args.dataset == 'face_verification':
            self.criterion = AngleLoss()
        elif args.dataset == 'emotion':
            class_counts = torch.from_numpy(np.array([74874, 134415, 25459, 14090, 6378, 3803, 24882]).astype(np.float32))
            class_weights = (torch.sum(class_counts) - class_counts) / class_counts
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
        else:
            self.criterion = nn.CrossEntropyLoss()
        return
        
    def train(self, optimizers, epoch_idx, curr_lrs, adv_train=True):
        #viz = visdom.Visdom(server="http://140.117.80.90", port=8097, env='CPG_imagenet-o')
        #viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
                ,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100
                 ,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
                116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
                161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199]
        #task_id = "imagenet-o"
        # Set model to training mode
        self.model.train()

        train_loss     = Metric('train_loss')
        train_accuracy1 = Metric('train_accuracy1')
        adv_train_accuracy1 = Metric('adv_train_accuracy1')
        #std_train_accuracy1 = Metric('std_train_accuracy5')
        #train_accuracy5 = Metric('train_accuracy5')
        #adv_train_accuracy5 = Metric('adv_train_accuracy5')
        #std_train_accuracy5 = Metric('std_train_accuracy5')

        with tqdm(total=len(self.train_loader),
                  desc='Train Epoch #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                #if self.args.cuda:
                data, target = data.cpu(), target.cuda()
                    #print(target.float())
                #adversarial_training
                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, target)
                    #adv_data = self.attack.perturb(data, target, 'mean',True)
                    #adv_data = self.attack.perturb(data, target)
                    #output = self.model(adv_data, _eval=False)
                    
                    adv_output = self.model(adv_data)
                else:
                    #output = self.model(data, _eval=False)
                    output = self.model(data)

                optimizers.zero_grad()
                _, predicted = torch.max(adv_output, 1)
                #adv_acc1, adv_acc5 = classification_accuracy(output, target, topk=(1, 5))                        
                #_, predicted = torch.max(output, 1)
                #acc1, acc5 = classification_accuracy(output, target, topk=(1, 5))
                #print('Predicted: ', predicted)
#######################################################################################################
                ## adversarial training
                #if adv_train:
                #        with torch.no_grad():
                #            #stand_output = self.model(data, _eval=True)
                #            stand_output = self.model(data)
                #        _, predicted = torch.max(stand_output, 1)
#
                #        # print(pred)
                #        #std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                #        std_acc1, std_acc5 = classification_accuracy(stand_output, target, topk=(1, 5))
#
                #        _, predicted = torch.max(output, 1)
                #        # print(pred)
                #        #adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                #        adv_acc1, adv_acc5 = classification_accuracy(output, target, topk=(1, 5))
#
                #else:
                #        
                #        adv_data = self.attack.perturb(data, target, 'mean', False)
#
                #        with torch.no_grad():
                #            #adv_output = self.model(adv_data, _eval=True)
                #            adv_output = self.model(adv_data)
                #        _, predicted = torch.max(adv_output, 1)
                #        # print(label)
                #        # print(pred)
                #        #adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                #        adv_acc1, adv_acc5 = classification_accuracy(adv_output, target, topk=(1, 5))
#
                #        _, predicted = torch.max(output, 1)
                #        # print(pred)
                #        #std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                #        std_acc1, std_acc5 = classification_accuracy(output, target, topk=(1, 5))
 #############################################################################################################   

                num = data.size(0)
                if self.args.dataset != 'face_verification':
                    adv_train_accuracy1.update(classification_accuracy(adv_output, target), num)
                    #train_accuracy1.update(acc1[0], data.size(0))
                    #train_accuracy5.update(acc5[0], data.size(0))
                    #std_train_accuracy1.update(std_acc1[0], data.size(0))
                    #adv_train_accuracy1.update(adv_acc1[0], data.size(0))
                    #std_train_accuracy5.update(std_acc5[0], data.size(0))
                    #adv_train_accuracy5.update(adv_acc5[0], data.size(0))
                    

                loss = self.criterion(adv_output, target)
                train_loss.update(loss, num)
                loss.backward()

                # Set fixed param grads to 0.
                self.pruner.do_weight_decay_and_make_grads_zero()

                # Gradient is applied across all ranks
                optimizers.step()
                

                    
                #calculate f1-score
                fscore = f1_score(target.cpu(),predicted.cpu() ,average='macro')
                #print confusion_matrix(暫時因為memory內存問題暫時不畫)
                #cn_matrix = confusion_matrix(
                #        y_true=target.cpu(),
                #        y_pred=predicted.cpu(),
                #        labels=label,
                #        normalize="true",
                #        )
                #plt.rcParams.update({'font.size': 1.8})
                #plt.rcParams.update({'figure.dpi': 300})
                #ConfusionMatrixDisplay(cn_matrix,label).plot(
                #        include_values=False, xticks_rotation="vertical"
                #        )
                #plt.rcParams.update({'figure.dpi': 100})
                #plt.rcParams.update({'font.size': 5})
                #plt.title("train_confusion_matrix")
                ##plt.figure(figsize=(10,5)) 
                #plt.tight_layout()
                #plt.savefig('train_confusion_matrix.png')
                #plt.close()
                #a =np.array(fscore)
                #print('loss:', a)
                #print('fscore:', fscore)
                

                # Set pruned weights to 0.
                self.pruner.make_pruned_zero()

                # Do forward-backward.
                #output = self.model(data)       

                t.set_postfix({'loss': train_loss.avg.item(),
                               #'std_accuracy@1': '{:.2f}'.format(std_train_accuracy1.avg.item()),
                               'adv_accuracy@1': '{:.2f}'.format(100. * adv_train_accuracy1.avg.item()),
                               'f1 score(macro)':fscore,
                               'lr': curr_lrs[0],
                               'sparsity': self.pruner.calculate_sparsity()})
                t.update(1)
            #visualize train result
            #viz.line(Y=np.array(fscore), X=np.arange(epoch_idx+1),win=fscore,opts=dict( title = 'imagenet-o: train_f1 score(micro)'),update='append')
            #viz.line(y=train_accuracy1.item(), X=np.arange(epoch_idx+1), win=line,
                #opts=dict( title = 'imagenet-o: train accuracy'))
            print(classification_report(target.cpu(),predicted.cpu()))
            #viz.line(np.array(train_loss.avg.item()), np.arange(epoch_idx+1), win='train_loss',opts=dict( title = 'imagenet-o: train loss'))
            
        return adv_train_accuracy1.avg.item()

    #{{{ Evaluate classification
    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        #val_accuracy1 = Metric('val_accuracy1')
        #val_accuracy5 = Metric('val_accuracy5')
        adv_val_accuracy1 = Metric('adv_val_accuracy1')
        #std_val_accuracy1 = Metric('std_val_accuracy5')
        #train_accuracy5 = Metric('val_accuracy5')
        #adv_val_accuracy5 = Metric('adv_val_accuracy5')
        #std_val_accuracy5 = Metric('std_val_accuracy5')
        label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
                ,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100
                 ,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
                116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
                161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199]

        with tqdm(total=len(self.val_loader),
                  desc='Validate Epoch  #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    #if self.args.cuda:
                    data, target = data.cpu(), target.cuda()

                    #output = self.model(data)
                    #_, predicted = torch.max(output, 1)
                    #std_acc1, std_acc5 = classification_accuracy(output, target, topk=(1, 5))
                    
                    ####adv_test###############################################
                    #if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack_val.perturb(data, target)
                        #adv_data = self.attack_val.perturb(data, target, 'mean',False)
                    ##adv_output = model(adv_data, _eval=True)
                    adv_output = self.model(adv_data)
                    #output = self.model(data)
                    _, adv_pred = torch.max(adv_output, 1)
                    #_, adv_pred = torch.max(output, 1)
                    ##adv_acc = evaluate(adv_pred.cpu().numpy(), target.cpu().numpy(), 'sum')
                    #adv_val_acc1, adv_val_acc5 = classification_accuracy(adv_output, target, topk=(1, 5))
                    ##total_adv_acc += adv_acc1
                    #else:
                        #total_adv_acc = -num
                    ##########################################################
                    num = data.size(0)
                    #val_loss.update(self.criterion(output, target), num)
                    val_loss.update(self.criterion(adv_output, target), num)
                    adv_val_accuracy1.update(classification_accuracy(adv_output, target), num)
                    #adv_val_accuracy1.update(classification_accuracy(output, target), num)
                    #acc1, acc5 = classification_accuracy(output, target, topk=(1, 5))
                    
                    #_, predicted = torch.max(output, 1)
                    # print(pred)
                    #std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                    #std_val_acc1, std_val_acc5 = classification_accuracy(output, target, topk=(1, 5))
                    #update accuracy
                    #val_accuracy1.update(acc1[0], data.size(0))
                    #val_accuracy5.update(acc5[0], data.size(0))
                    #std_val_accuracy1.update(std_acc1[0], data.size(0))
                    #adv_val_accuracy1.update(adv_val_acc1[0], data.size(0))
                    #std_val_accuracy5.update(std_acc5[0], data.size(0))
                    #adv_val_accuracy5.update(adv_val_acc5[0], data.size(0))
                    #calculate f1_score
                    fscore = f1_score(target.cpu(),adv_pred.cpu() ,average='macro')
                    #print confusion_matrix(暫時因為memory內存問題暫時不畫)
                    #cn_matrix = confusion_matrix(
                    #        y_true=target.cpu(),
                    #        y_pred=adv_pred.cpu(),
                    #        labels=label,
                    #        normalize="true",
                    #        )
                    #plt.rcParams.update({'font.size': 1.8})
                    #plt.rcParams.update({'figure.dpi': 300})
                    #ConfusionMatrixDisplay(cn_matrix,label).plot(
                    #        include_values=False, xticks_rotation="vertical"
                    #        )
                    #plt.rcParams.update({'figure.dpi': 100})
                    #plt.rcParams.update({'font.size': 5})
                    #plt.title("val_confusion_matrix")
                    ##plt.figure(figsize=(10,5)) 
                    #plt.tight_layout()
                    #plt.savefig('val_confusion_matrix.png')
                    #plt.close()
                        

                    t.set_postfix({'loss': val_loss.avg.item(),
                                   #'std_accuracy@1': '{:.2f}'.format(std_val_accuracy1.avg.item()),
                                   'adv_accuracy@1': '{:.2f}'.format(100. * adv_val_accuracy1.avg.item()),
                                   'f1 score(macro)':fscore,
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                   'zero ratio': self.pruner.calculate_zero_ratio()})
                    t.update(1)
            #visualize validation result
            #viz.line(np.array(val_accuracy1), X=np.arange(epoch_idx+1), win='vacc',opts=dict(title =  'imagenet-o: val accuracy'))
            #viz.line(np.array(val_loss), X=np.arange(epoch_idx+1), win='vloss',opts=dict(title =  'imagenet-o: val loss'))
            print(classification_report(target.cpu(),adv_pred.cpu()))
            #viz.line(np.array(fscore), X=np.arange(epoch_idx+1), win='valfscore',opts=dict( title = 'imagenet-o: val_f1 score(micro)'))
        return adv_val_accuracy1.avg.item()
    #}}}

    #{{{ Evaluate LFW
    def evalLFW(self, epoch_idx):
        distance_metric = True
        subtract_mean   = False
        self.model.eval() # switch to evaluate mode
        labels, embedding_list_a, embedding_list_b = [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader),
                      desc='Validate Epoch  #{}: '.format(epoch_idx + 1),
                      ascii=True) as t:
                for batch_idx, (data_a, data_p, label) in enumerate(self.val_loader):
                    data_a, data_p = data_a.cuda(), data_p.cuda()
                    data_a, data_p, label = Variable(data_a, volatile=True), \
                                            Variable(data_p, volatile=True), Variable(label)
                    # ==== compute output ====
                    out_a = self.model.module.forward_to_embeddings(data_a)
                    out_p = self.model.module.forward_to_embeddings(data_p)
                    # do L2 normalization for features
                    if not distance_metric:
                        out_a = F.normalize(out_a, p=2, dim=1)
                        out_p = F.normalize(out_p, p=2, dim=1)
                    out_a = out_a.data.cpu().numpy()
                    out_p = out_p.data.cpu().numpy()

                    embedding_list_a.append(out_a)
                    embedding_list_b.append(out_p)
                    # ========================
                    labels.append(label.data.cpu().numpy())
                    t.update(1)

        labels = np.array([sublabel for label in labels for sublabel in label])
        embedding_list_a = np.array([item for embedding in embedding_list_a for item in embedding])
        embedding_list_b = np.array([item for embedding in embedding_list_b for item in embedding])
        tpr, fpr, accuracy, val, val_std, far = fv_evaluate(embedding_list_a, embedding_list_b, labels,
                                                distance_metric=distance_metric, subtract_mean=subtract_mean)
        print('Test set: Accuracy: {:.5f}+-{:.5f}'.format(np.mean(accuracy),np.std(accuracy)))
        return np.mean(accuracy)
    #}}}

    def one_shot_prune(self, one_shot_prune_perc):
        self.pruner.one_shot_prune(one_shot_prune_perc)
        return

    def save_checkpoint(self, optimizers, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.module.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    self.shared_layer_info[self.args.dataset][
                        'conv_bias'][name] = module.bias
            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_mean'][name] = module.running_mean
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_var'][name] = module.running_var
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_weight'][name] = module.weight
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_bias'][name] = module.bias
            elif isinstance(module, nn.Linear) and 'features' in name:
                self.shared_layer_info[self.args.dataset]['fc_bias'][name] = module.bias
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[self.args.dataset][
                    'prelu_layer_weight'][name] = module.weight

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info,
            # 'optimizer_network_state_dict': optimizers[0].state_dict(),
        }

        torch.save(checkpoint, filepath)
        return

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            #checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            for name, param in state_dict.items():
                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                else:
                    curr_model_state_dict[name].copy_(param)
        return

    def load_checkpoint_for_inference(self, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                else:
                    curr_model_state_dict[name].copy_(param)

            # load the batch norm params and bias in convolution in correspond to curr dataset
            for name, module in self.model.module.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[self.args.dataset]['conv_bias'][name]
                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_var'][name]
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'bn_layer_weight'][name]
                    module.bias = self.shared_layer_info[self.args.dataset][
                        'bn_layer_bias'][name]
                elif isinstance(module, nn.Linear) and 'features' in name:
                    module.bias = self.shared_layer_info[self.args.dataset]['fc_bias'][name]
                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'prelu_layer_weight'][name]
