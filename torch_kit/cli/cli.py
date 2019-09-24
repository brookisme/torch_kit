from __future__ import print_function
import os,sys
sys.path.append('..')
import re
import yaml
import click
from . import train
from . import config as c
from pprint import pprint
#
# CONFIG
#
IS_DEV=c.get('is_dev')
DRY_RUN=c.get('dry_run')
RUN_KEY=c.get('run_key')
DEV_HELP='<bool> reduce the amount of data for a quick test run'
DRY_RUN_HELP='<bool> load data and model, loop through runs but skip training'
NOISE_REDUCER=None
POWEROFF=True
POWEROFF_WAIT=30
PRINT_SUMMARY=True
ARG_KWARGS_SETTINGS={
    'ignore_unknown_options': True,
    'allow_extra_args': True
}
TRAIN_HELP='train your model'
SCORE_HELP='produce scores for your model'
NOISE_REDUCER_HELP='print every N lines'
POWEROFF_HELP='poweroff after training'
POWEROFF_WAIT_HELP='number of minutes after training to wait before poweroff'
PRINT_SUMMARY_HELP='print model summary'
""" THOUGHTS

torch_kit train dlv3p  --- uses dlv3p.py and dlv3p.yaml (== dlv3p.run or just dlv3p)
torch_kit train dlv3p foo --- uses dlv3p.py and foo.yaml (== foo.run)
torch_kit train dlv3p foo.bar --- uses dlv3p.py and foo.yaml (foo.bar instead of foo.run)

* MODEL
    - model(** model_name.model)
* DATASETS/LOADERS
    - datasets(** model_name.datasets)

"""


#
# CLI INTERFACE
#
@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj={}


@click.command(
    help=TRAIN_HELP,
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.argument('module',type=str)
@click.option(
    '--dev',
    help=DEV_HELP,
    default=IS_DEV,
    type=bool)
@click.option(
    '--dry_run',
    help=DRY_RUN_HELP,
    default=DRY_RUN,
    type=bool)
@click.option(
    '--noise_reducer',
    help=NOISE_REDUCER_HELP,
    default=NOISE_REDUCER,
    type=int)
@click.option(
    '--poweroff',
    help=POWEROFF_HELP,
    default=POWEROFF,
    type=bool)
@click.option(
    '--poweroff_wait',
    help=POWEROFF_WAIT_HELP,
    default=POWEROFF_WAIT,
    type=int)
@click.option(
    '--summary',
    help=PRINT_SUMMARY_HELP,
    default=PRINT_SUMMARY,
    type=bool)
@click.pass_context
def train_model(ctx,
        module,
        dev,
        dry_run,
        noise_reducer,
        poweroff,
        poweroff_wait,
        summary):
    args,kwargs=_args_kwargs(ctx.args)
    if args:
        config=args[0]
    else:
        config=module
    train_configs=_get_training_configs(config)
    for cfig in train_configs:
        tm=train.TrainManager(module,cfig)
        tm.run(
            dev=dev,
            dry_run=dry_run,
            noise_reducer=noise_reducer,
            poweroff=poweroff,
            poweroff_wait=poweroff_wait,
            print_summary=summary)


@click.command(
    help=SCORE_HELP,
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.pass_context
def score(ctx):
    print('score')




#
# HELPERS
#
def _get_training_configs(dot_path):
    parts=dot_path.split('.')
    fname=parts[0]
    cfig=yaml.safe_load(open(f'{parts[0]}.yaml'))
    if not parts[1:]:
        parts.append(RUN_KEY)
    for p in parts[1:]:
        cfig=cfig[p]
    if not isinstance(cfig,list):
        cfig=[cfig]
    return cfig


def _args_kwargs(ctx_args):
    args=[]
    kwargs={}
    for a in ctx_args:
        if re.search('=',a):
            k,v=a.split('=')
            kwargs[k]=v
        else:
            args.append(a)
    return args,kwargs


#
# MAIN
#
cli.add_command(train_model,name='train')
cli.add_command(score)
if __name__ == "__main__":
    cli()