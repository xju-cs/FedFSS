import argparse
import collections
import time
import glob
import warnings
import datetime
from torch.utils.data import DataLoader

from utils import *
from datasets import BaseDataset
from config.config import Config

save_root = 'exp-r100/fedprox'
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
## record time
train_start_time = datetime.datetime.now()
print('start time:{}'.format(train_start_time))

parser = argparse.ArgumentParser(description="AL_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_cpu', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.97)
parser.add_argument("--loss_type", type=str, default='am')

## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)

## hyperparameters
parser.add_argument('--mu',type=float,default=0.001)
## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

## import config
cfg = Config()
config = cfg.generate_config(num_clients=4,fed_mode= 'hete',client_datasets=['vox1','vox1','vox1','vox1'])
args.client_train_path = config['client_train_path']
args.client_train_lists = config['client_train_lists']
args.client_test_path = config['client_test_path']
args.client_test_lists = config['client_test_lists']
args.server_test_path = config['server_test_path']
args.server_test_lists = config['server_test_lists']
args.num_clients = config['num_clients']
args.round = config['round']
args.exp_name = config['exp_name']
args.fed_mode = config['fed_mode']
args.test_step = config['round']

args.exp_name='{}_{}'.format(args.exp_name,args.mu)
args.save_path = os.path.join(save_root, args.exp_name)
args = init_args(args)

logger = get_logger(args)
logger.log("args: {}\n".format(args))

cmd = 'cp ./train_fedProx.py {}'.format(args.save_path)
os.system(cmd)
logger.log('copy train_fedProx.py to {}'.format(args.save_path))


def train(model, client_index, epoch, loader,server_para,mu):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']
    top1, loss_value, num = 0, 0, 1e-7

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        data, labels = batch
        progress.set_description("Train {}".format(client_index))
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        fed_prox_reg=model.compute_fedprox_regularizer(server_para)
        nloss = loss.mean()
        nloss+=(mu/2) * fed_prox_reg

        model.zero_grad()
        nloss.backward()
        model.optim.step()

        num += 1
        top1 += acc
        loss_value += nloss.detach().cpu().numpy()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss='{:.4}'.format(loss_value / num),
            acc='{:.4}'.format(100 * top1 / num),
            prox='{:.4}'.format(fed_prox_reg)
        )
    progress.close()

    return loss_value / num, lr, 100 * top1 / num


def main(args):
    # Build clients
    client_train_datasets = []
    client_train_loaders = []
    client_models = []
    client_evaluators = []
    for i in range(args.num_clients):
        logger.log("Building Client {}...".format(i+1))
        client_train_datasets.append(BaseDataset(args.client_train_lists[i], args.client_train_path[i], args.num_frames))
        client_train_loaders.append(DataLoader(client_train_datasets[i], batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.n_cpu,
                                               drop_last=True))
        args.n_class = client_train_datasets[i].get_speaker_number()
        client_models.append(Model(**vars(args)))
        client_evaluators.append(Evaluator(args.client_test_lists[i], args.client_test_path[i]))
    args.weights = compute_client_avg_weight(client_train_datasets)
    logger.log("args.weights: {}".format(args.weights))

    # Build server
    server_model = Model(**vars(args))
    global_para=server_model.upload_parameters(mode=args.fed_mode)
    server_evaluators = []
    for server_test_list, server_test_path in zip(args.server_test_lists, args.server_test_path):
        server_evaluators.append(Evaluator(server_test_list, server_test_path))

    EERs = [100]
    epoch = 1

    if not os.path.exists(args.score_save_path):
        with open(args.score_save_path, "w") as f:
            f.write("Time,Epoch,Client,Trials,LR,Loss,Acc,EER,minDCF(0.01),minDCF(0.001),bestEER\n")
            f.close()
    score_file = open(args.score_save_path, "a+")

    while epoch <= args.max_epoch:
        logger.log("\nEpoch {}:".format(epoch))
        epoch_start_time = datetime.datetime.now()
        logger.log(epoch_start_time)

        # Training for each client
        for i in range(args.num_clients):
            # Update client with server model
            if epoch % args.round == 1:
                client_models[i].copy_weights(server_model, mode=args.fed_mode)
                logger.log("Updated Client {} with Server".format(i + 1))

            # Training
            loss, lr, acc = train(client_models[i], i+1, epoch, client_train_loaders[i], global_para, args.mu)

            # Save Model
            client_models[i].save_parameters(args.model_save_path + "/model_%04d_%d.model" % (epoch, i + 1))

            # Evaluation for client
            EER, minDCF01, minDCF001 = None, None, None
            if epoch % args.test_step == 0:
                print("Client {} Evaluation on {}".format(i+1, args.client_test_lists[i]))
                EER, minDCF01, minDCF001 = client_evaluators[i].eval(client_models[i])
                EERs.append(EER)
                logger.log(
                    "{} ACC:{:.4f}, EER:{:.4f}, minDCF:{:.4f}, bestEER:{:.4f}".format(datetime.datetime.now(), acc, EER,
                                                                                      minDCF01, min(EERs)),
                    time.strftime("%Y-%m-%d %H:%M:%S"))
            score_file.write(
                "{},{},{},{},{},{},{},{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i + 1,
                                                            args.client_test_lists[i], lr, loss, acc, EER,
                                                            minDCF01, minDCF001, min(EERs)))
            score_file.flush()

        # Update server with client model
        if epoch % args.round == 0:
            server_model.update_parameters(client_models, args.weights, mode=args.fed_mode)
            logger.log("Updated Server with Clients")

            # Evaluation for server
            for j in range(len(server_evaluators)):
                logger.log("Server Evaluation on {}".format(args.server_test_lists[j]))
                EER, minDCF01, minDCF001 = server_evaluators[j].eval(server_model)
                EERs.append(EER)
                logger.log(
                    "{} EER:{:.4f}, minDCF:{:.4f}, bestEER:{:.4f}".format(datetime.datetime.now(), EER, minDCF01,
                                                                          min(EERs)),
                    time.strftime("%Y-%m-%d %H:%M:%S"))
                score_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, "server",
                                                                args.server_test_lists[j], {}, {}, {}, EER,
                                                                minDCF01, minDCF001, min(EERs)))
                score_file.flush()
        global_para=server_model.upload_parameters(mode=args.fed_mode)
        epoch += 1
        ## record time
        epoch_end_time = datetime.datetime.now()
        logger.log('this epoch time:{}\n'.format(epoch_end_time - epoch_start_time))
    train_end_time = datetime.datetime.now()
    logger.log('total train time:{}\n'.format(train_end_time - train_start_time))


if __name__ == '__main__':
    main(args)
