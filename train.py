import os
from configs import cfg, args

from utils.log_util import Logger
from data.create_dataset import create_dataloader
from nets.network import Network
from train.optimizer import get_optimizer
from train.trainer import Trainer


def main():
    log = Logger()
    log.print_config()

    model = Network()
    optimizer = get_optimizer(model)
    trainer = Trainer(model, optimizer)
    train_loader = create_dataloader('train')

    # estimate start epoch
    epoch = trainer.iter // len(train_loader) + 1
    while True:
        if trainer.iter > cfg.train.maxiter:
            break        
        trainer.train(epoch=epoch, train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()

if __name__ == '__main__':
    cfg['subject'] = args.subject
    cfg.logdir = os.path.join('experiments', cfg.category, cfg.task, cfg.subject, cfg.experiment)
    main()
