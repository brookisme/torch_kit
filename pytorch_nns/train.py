import time
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import pytorch_nns.helpers as h
#
# CONFIG
#
MAX_DOTS=25
DOT='.'
FLOAT_TMPL='{:>.5f}'
BATCH_OUT_TMPL="[epoch: {}, step: {}/{}] loss={} ({}) {}"
EPOCH_OUT_TMPL="[epoch: {}, step: {}/{}] loss={} {}"
INPT_KEY='input'
TARG_KEY='target'


#
# HELPERS
#
def print_batch_end(epoch,index,steps,loss_str):
    out_str=EPOCH_OUT_TMPL.format(
        epoch+1,
        index+1,
        steps,
        loss_str,
        DOT*MAX_DOTS)
    print(out_str,flush=True)



#
# METHODS 
#
def fit(
        model,
        dataloader,
        criterion,
        optimizer,
        device=None,
        nb_epochs=1,
        noise_reducer=None,
        inputs_key=INPT_KEY,
        targets_key=TARG_KEY,
        output_processor=None,
        output_processor_kwargs={}):
    r""" fit model
    WIP: 
        This is an early pass based on 
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=dataloader#train-the-network
    """
    model.train()
    steps=len(dataloader)
    print_freq=max(steps*noise_reducer//MAX_DOTS,1)
    h.print_line()
    nb_dots=1
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        epoch_loss=0.0
        print_epoch=(not (noise_reducer and (epoch%noise_reducer)))
        for index, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs=batch[inputs_key].float()
            targets=batch[targets_key].float()
            if device:
                inputs=inputs.to(device)
                targets=targets.to(device) 
            outputs=model(inputs)
            if output_processor:
                outputs=output_processor(outputs,**output_processor_kwargs)
            loss=criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            batch_loss=loss.item()
            epoch_loss+=batch_loss
            batch_loss_str=FLOAT_TMPL.format(batch_loss)
            loss_str=FLOAT_TMPL.format(epoch_loss)
            nb_dots+=index//print_freq
            out_str=BATCH_OUT_TMPL.format(
                epoch+1,
                index+1,
                steps,
                loss_str,
                batch_loss_str,
                DOT*nb_dots)
            print(out_str,end="\r",flush=True)
        if print_epoch:
            nb_dots=1
            print_batch_end(epoch,index,steps,loss_str)
    if not print_epoch:
        print_batch_end(epoch,index,steps,loss_str)
    h.print_line()
    print('COMPLETE: loss={:.5f}'.format(epoch_loss))
    model.train(mode=False)

"""
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""


def run(model, train_loader, criterion, optimizer,device=None, nb_epochs=2, noise_reducer=1):
    best_prec1=0
    for epoch in range(0,nb_epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, noise_reducer)

        # evaluate on validation set
        prec1 = validate(train_loader, model, criterion, device, noise_reducer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('-'*70)
        print("best[{}]: ".format(epoch),is_best,best_prec1)
        print('-'*70)


def train(train_loader, model, criterion, optimizer, epoch, device, noise_reducer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, b in enumerate(train_loader):
        input=b['input'].float()
        target=b['target'].float()
        # measure data loading time
        data_time.update(time.time() - end)

        if device is not None:
            input=input.to(device)
            target=target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        report_tmpl="Epoch: [{0}][{1}/{2}],  Time {batch_time.val:.3f} ({batch_time.avg:.3f}),  Data {data_time.val:.3f} ({data_time.avg:.3f}),  Loss {loss.val:.4f} ({loss.avg:.4f}),  Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
        if i % noise_reducer == 0:
            report=report_tmpl.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1)
            print(report,end="\r",flush=True)
    print(report,flush=True)


def validate(val_loader, model, criterion, device, noise_reducer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, b in enumerate(val_loader):
            input=b['input'].float()
            target=b['target'].float()
            if device is not None:
                input=input.to(device)
                target=target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            report_tmpl="Test: [{0}/{1}],  Time {batch_time.val:.3f} ({batch_time.avg:.3f}),  Loss {loss.val:.4f} ({loss.avg:.4f}),  Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
            if i % noise_reducer == 0:
                report=report_tmpl.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1)
                print(report,end="\r",flush=True)
        print(report,flush=True)
        print('='*70)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        o=torch.argmax(output,dim=1)
        t=torch.argmax(target,dim=1)
        batch_size = target.size(0)
        size=target.size(-1)
        eqval=o.eq(t)
        acc=eqval.sum()/(size**2)
        return acc,acc*10
        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return res

