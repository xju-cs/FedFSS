import argparse
import collections
import time
import glob
import warnings
import datetime
from torch.utils.data import DataLoader

from utils import *
from datasets import BaseDataset

save_root = 'exp-r100/standard'

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
parser.add_argument("--dataset", type=str, default='vox1')
parser.add_argument("--test_step", type=int, default=5)
parser.add_argument("--train_list", type=str, default='train_list')
parser.add_argument("--train_path", type=str, default='train_path')
parser.add_argument("--test_list", type=str, default='test_list')
parser.add_argument("--test_path", type=str, default='test_path')

## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args.save_path = os.path.join(save_root, '{}_{}'.format(args.dataset, args.loss_type))
args = init_args(args)

def train(model, epoch, loader):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1, loss_value, num = 0, 0, 1e-7

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        data, labels = batch
        progress.set_description("train")
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        nloss = loss.mean()

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
        )
    progress.close()

    return loss_value / num, lr, 100 * top1 / num


def main(args):
    print(args.train_path)
    train_dataset = BaseDataset(args.train_list, args.train_path, args.num_frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_cpu,
                              drop_last=True)
    args.n_class = train_dataset.get_speaker_number()
    evaluator = Evaluator(args.test_list, args.test_path)
    EERs = [100]
    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        with open(args.score_save_path, "r") as f:
            lines = f.readlines()
            if 'final' in lines[-1] or 'final' in lines[-2]:
                print('this experiment is already finished!')
                print(lines[-1])
                return
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        model = Model(**vars(args))
        model.load_parameters(modelfiles[-1])
    ## Otherwise, system will train from scratch
    else:
        epoch = 1
        model = Model(**vars(args))
    if not os.path.exists(args.score_save_path):
        with open(args.score_save_path, "w") as f:
            f.write("Time,Epoch,LR,Loss,Acc,EER,minDCF(0.01),minDCF(0.001),bestEER\n")
            f.close()
    score_file = open(args.score_save_path, "a+")

    while epoch <= args.max_epoch:
        print("\nEpoch {}:".format(epoch))
        epoch_start_time = datetime.datetime.now()
        print(epoch_start_time)

        # Training
        loss, lr, acc = train(model, epoch, train_loader)

        # Save Model
        if epoch >= args.max_epoch-10:
            model.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)

        # Evaluation every [test_step] epochs
        EER, minDCF01, minDCF001 = None, None, None
        if epoch % args.test_step == 0:
            EER, minDCF01, minDCF001 = evaluator.eval(model)
            EERs.append(EER)
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "ACC %2.2f%%, EER %2.3f%%, bestEER %2.3f%%" % (acc, EER, min(EERs)))
        score_file.write(
            "{},{},{},{},{},{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, lr, loss, acc, EER,
                                                  minDCF01, minDCF001, min(EERs)))
        score_file.flush()
        epoch += 1
        ## record time
        epoch_end_time = datetime.datetime.now()
        print('this epoch time:{}\n'.format(epoch_end_time - epoch_start_time))

    models = []
    for i in range(5):
        model_path = os.path.join(args.model_save_path, "model_%04d.model" % (args.max_epoch - i))
        model = Model(**vars(args))
        model.load_parameters(model_path)
        print("{} is loaded!".format(model_path))
        models.append(model)
    ensemble_model = Model(**vars(args))
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)
    ensemble_model.load_state_dict(fed_state_dict)

    EER, minDCF01, minDCF001 = evaluator.eval(ensemble_model)
    EERs.append(EER)
    print(time.strftime("%Y-%m-%d %H:%M:%S"),
          "final, EER:%2.3f%%, minDCF(0.01):%2.3f, minDCF(0.001):%2.3f, bestEER %2.3f%%" % (
              EER, minDCF01, minDCF001, min(EERs)))
    score_file.write(
        "{},{},,,,{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), "final", EER, minDCF01, minDCF001,
                                        min(EERs)))
    score_file.flush()
    # record time
    train_end_time = datetime.datetime.now()
    print('total train time:{}\n'.format(train_end_time - train_start_time))


if __name__ == '__main__':
    main(args)
