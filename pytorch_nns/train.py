import torch.cuda
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#
# CONFIG
#
MAX_DOTS=50
DOT='.'
FLOAT_TMPL='{:>.5f}'
BATCH_OUT_TMPL="[epoch: {}, step: {}/{}] loss={} ({}) {}"
EPOCH_OUT_TMPL="[epoch: {}, step: {}/{}] loss={} {}"
INPT_KEY='input'
TARG_KEY='target'



#
# HELPERS
#
def print_line(char='-',length=75):
    print(char*length)


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
    steps=len(dataloader)
    print_freq=max(steps*noise_reducer//MAX_DOTS,1)
    print_line()
    nb_dots=1
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        epoch_loss=0.0
        print_epoch=(not (noise_reducer and (epoch%noise_reducer)))
        for index, batch in enumerate(dataloader):
            inputs=batch[inputs_key].float()
            targets=batch[targets_key].float()
            optimizer.zero_grad()
            outputs=model(inputs)
            if output_processor:
                outputs=output_processor(outputs,**output_processor_kwargs)
            loss=criterion(outputs, targets)
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
    print_line()
    print('COMPLETE: loss={:.5f}'.format(epoch_loss))
