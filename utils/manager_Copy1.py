import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb
from torch.autograd import Variable
from . import Metric, classification_accuracy
from .prune import SparsePruner
from .metrics import fv_evaluate
import models.layers as nl
from models import AngleLoss
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from .fast_gradient_sign_untargeted import FastGradientSignUntargeted
from .PGD_attack import LinfPGDAttack

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader, begin_prune_step, end_prune_step, adv_train, attack, attack_val):
        self.args = args
        self.model = model.cuda()
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        self.pruner = SparsePruner(self.model, masks, self.args, begin_prune_step, end_prune_step, self.inference_dataset_idx)

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

    def train(self, optimizers, epoch_idx, curr_lrs, curr_prune_step, adv_train=True):
        # Set model to training mode
        self.model.train()

        train_loss     = Metric('train_loss')
        adv_train_accuracy1 = Metric('adv_train_accuracy1')
        #adv_train_accuracy5 = Metric('adv_train_accuracy5')

        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                #if self.args.cuda:
                data, target = data.cpu(), target.cuda()
               #adversarial_training
                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    #adv_data = self.attack.perturb(data, target, 'mean', True)
                    adv_data = self.attack.perturb(data, target)
                    #output = self.model(adv_data, _eval=False)
                    
                    adv_output = self.model(adv_data)
                else:
                    #output = self.model(data, _eval=False)
                    output = self.model(data)

                optimizers.zero_grad()
                _, predicted = torch.max(adv_output, 1)
                #adv_acc1, adv_acc5 = classification_accuracy(output, target, topk=(1, 5))
                num = data.size(0)
                if self.args.dataset != 'face_verification':
                    #train_accuracy1.update(acc1[0], data.size(0))
                    #train_accuracy5.update(acc5[0], data.size(0))
                    #std_train_accuracy1.update(std_acc1[0], data.size(0))
                    adv_train_accuracy1.update(classification_accuracy(adv_output, target), num)
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

                # Set pruned weights to 0.
                if self.args.mode == 'prune':
                    self.pruner.gradually_prune(curr_prune_step)
                    curr_prune_step += 1

                if self.inference_dataset_idx == 1:
                    t.set_postfix({'loss': train_loss.avg.item(),
                                   'adv_accuracy@1': '{:.2f}'.format(100. *adv_train_accuracy1.avg.item()),
                                   #'adv_accuracy@5': '{:.2f}'.format(adv_train_accuracy5.avg.item()),
                                   'f1 score(macro)':fscore,
                                   'lr': curr_lrs[0],
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'network_width_mpl': self.args.network_width_multiplier})
                else:
                    t.set_postfix({'loss': train_loss.avg.item(),
                                   'adv_accuracy@1': '{:.2f}'.format(100. *adv_train_accuracy1.avg.item()),
                                   #'adv_accuracy@5': '{:.2f}'.format(adv_train_accuracy5.avg.item()),
                                   'f1 score(macro)':fscore,
                                   'lr': curr_lrs[0],
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'network_width_mpl': self.args.network_width_multiplier})
                t.update(1)

        summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                   'adv_accuracy@1': '{:.2f}'.format(100. *adv_train_accuracy1.avg.item()),
                   'f1 score(macro)':fscore,
                   'lr': curr_lrs[0],
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'network_width_mpl': self.args.network_width_multiplier}

        if self.args.log_path:
            logging.info(('In train()-> Train Ep. #{} '.format(epoch_idx + 1)
                         + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return adv_train_accuracy1.avg.item(), curr_prune_step

    #{{{ Evaluate classification
    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        adv_val_accuracy1 = Metric('adv_val_accuracy1')
        #adv_val_accuracy5 = Metric('adv_val_accuracy5')
        

        with tqdm(total=len(self.val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    #f self.args.cuda:
                    data, target = data.cpu(), target.cuda()
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, target, 'mean', False)
                        #adv_data = self.attack.perturb(data, target)
                    ##adv_output = model(adv_data, _eval=True)
                    adv_output = self.model(adv_data)
                    _, adv_pred = torch.max(adv_output, 1)
                    ##adv_acc = evaluate(adv_pred.cpu().numpy(), target.cpu().numpy(), 'sum')
                    #adv_val_acc1, adv_val_acc5 = classification_accuracy(adv_output, target, topk=(1, 5))

                    
                    num = data.size(0)
                    val_loss.update(self.criterion(adv_output, target), num)
                    adv_val_accuracy1.update(classification_accuracy(adv_output, target), num)
                    #adv_val_accuracy5.update(adv_val_acc5[0], data.size(0))
                    #calculate f1_score
                    fscore = f1_score(target.cpu(),adv_pred.cpu() ,average='macro')

                    if self.inference_dataset_idx == 1:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'adv_accuracy@1': '{:.2f}'.format(100. *adv_val_accuracy1.avg.item()),
                                       #'adv_accuracy@5': '{:.2f}'.format(adv_val_accuracy5.avg.item()),
                                       'f1 score(macro)':fscore,
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    else:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'adv_accuracy@1': '{:.2f}'.format(100. *adv_val_accuracy1.avg.item()),
                                       #'adv_accuracy@5': '{:.2f}'.format(adv_val_accuracy5.avg.item()),
                                       'f1 score(macro)':fscore,
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'shared_ratio': self.pruner.calculate_shared_part_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. *adv_val_accuracy1.avg.item()),
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                   'zero ratio': '{:.3f}'.format(self.pruner.calculate_zero_ratio()),
                   'mpl': self.args.network_width_multiplier}
        if self.inference_dataset_idx != 1:
            summary['shared_ratio'] = '{:.3f}'.format(self.pruner.calculate_shared_part_ratio())

        if self.args.log_path:
            logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
                         + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return adv_val_accuracy1.avg.item()
    #}}}

    #{{{ Evaluate LFW
    def evalLFW(self, epoch_idx):
        distance_metric = True
        subtract_mean   = False
        self.pruner.apply_mask()
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
        print('In evalLFW(): Test set: Accuracy: {:.5f}+-{:.5f}'.format(np.mean(accuracy),np.std(accuracy)))
        logging.info(('In evalLFW()-> Validate Epoch #{} '.format(epoch_idx + 1)
                     + 'Test set: Accuracy: {:.5f}+-{:.5f}, '.format(np.mean(accuracy),np.std(accuracy))
                     + 'task_ratio: {:.2f}'.format(self.pruner.calculate_curr_task_ratio())))
        return np.mean(accuracy)
    #}}}

    def save_checkpoint(self, optimizers, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if module.bias is not None:
                    self.shared_layer_info[self.args.dataset][
                        'bias'][name] = module.bias
                if module.piggymask is not None:
                    self.shared_layer_info[self.args.dataset][
                        'piggymask'][name] = module.piggymask
            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_mean'][name] = module.running_mean
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_var'][name] = module.running_var
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_weight'][name] = module.weight
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_bias'][name] = module.bias
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[self.args.dataset][
                    'prelu_layer_weight'][name] = module.weight

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info
        }
        torch.save(checkpoint, filepath)
        return

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if ('piggymask' in name or name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                    # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name][:param.size(0), :param.size(1), :, :].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
                    curr_model_state_dict[name][:param.size(0)].copy_(param)
                elif 'classifiers' in name:
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                else:
                    try:
                        curr_model_state_dict[name].copy_(param)
                    except:
                        pdb.set_trace()
                        print("There is some corner case that we haven't tackled")
        return

    def load_checkpoint_only_for_evaluate(self, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if 'piggymask' in name: # we load piggymask value in main.py
                    continue

                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue

                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name].copy_(
                            param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1), :, :])

                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name].copy_(
                            param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1)])

                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0)])

                else:
                    curr_model_state_dict[name].copy_(param)

            # load the batch norm params and bias in convolution in correspond to curr dataset
            for name, module in self.model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[self.args.dataset]['bias'][name]

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_var'][name]
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'bn_layer_weight'][name]
                    module.bias = self.shared_layer_info[self.args.dataset][
                        'bn_layer_bias'][name]

                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'prelu_layer_weight'][name]
        return
