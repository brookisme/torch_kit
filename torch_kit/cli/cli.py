from __future__ import print_function
import os,sys
sys.path.append('..')
import re
import click
from . import train
from . import config as c
#
# CONFIG
#
IS_DEV=c.get('is_dev')
DRY_RUN=c.get('dry_run')
DEV_HELP='<bool> reduce the amount of data for a quick test run'
DRY_RUN_HELP='<bool> load data and model, loop through runs but skip training'
ARG_KWARGS_SETTINGS={
    'ignore_unknown_options': True,
    'allow_extra_args': True
}
TRAIN_HELP='train your model'
SCORE_HELP='produce scores for your model'




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
@click.argument('method',type=str)
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
@click.pass_context
def train(ctx,method,dev,dry_run):
    print('train',method,dev,dry_run)


@click.command(
    help=SCORE_HELP,
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.pass_context
def score(ctx):
    print('score')


#
# MAIN
#
cli.add_command(train)
cli.add_command(score)
if __name__ == "__main__":
    cli()