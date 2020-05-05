# Importing libs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import sys
import os
import json

# Adding models directory to systems path
cwd = os.getcwd()
sys.path.append(cwd + '/../model')
sys.path.append(cwd + '/../utils')
sys.path.append(cwd + '/..')

# Importing from custom files
from ae import AE
from losses import cauchy_loss, comp_loss
from utils import make_one_hot
from slack_bot import send_slack_notif

# Reading Hyperparameters
with open("./hyperparams_ae.json", "r") as read_file:
    hyperparams = json.load(read_file)

K = hyperparams["K"]
q_lambda = hyperparams["q_lambda"]
train_stages = hyperparams["train_stages"]
batch_size = hyperparams["batch_size"]

saved_model_name = hyperparams["saved_model_name"]
train_data_folder = hyperparams["train_data_folder"]
val_data_folder = hyperparams["val_data_folder"]
cauchy_loss_weight = hyperparams["cauchy_loss_weight"]

# Primary device for computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(train_data_folder) == 0:
    print("Set data folder path")
    assert len(train_data_folder)

# Train, val transform
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(56),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.Resize(224),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Loop for all different stages
for trn_stage_no, train_stage in enumerate(train_stages):
    print("Started with train stage-{}".format(trn_stage_no + 1))
    send_slack_notif("Started with train stage-{}".format(trn_stage_no + 1))

    tensorboard_folder = '../runs/ae_hash/ae_stage{}'.format(trn_stage_no + 1)
    writer = SummaryWriter(tensorboard_folder)

    trainset = torchvision.datasets.ImageFolder(train_data_folder, transform=train_transform)
    valset = torchvision.datasets.ImageFolder(val_data_folder, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    model = AE(K=K).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    if train_stage["use_weight"]:
        model.load_state_dict(torch.load(saved_model_name + '_stage1.pt'), strict=False)

    # Adding layer parameters for different (10x faster than pretrained) learning rate
    fast_learning_layers = ['hashed_layer.{}'.format(ii) for ii in [0, 2, 4, 6]]
    fast_learning_layers = ['module.' + s + sb for s in fast_learning_layers for sb in ['.weight', '.bias']]

    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in fast_learning_layers, model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in fast_learning_layers, model.named_parameters()))))
    assert len(params) == len(fast_learning_layers)

    # Initializing losses and adding loss params to optimizer with higher lr
    criterion1 = comp_loss()
    criterion2 = cauchy_loss(K=K, q_lambda=q_lambda)
    params.extend(list(criterion1.parameters()))
    params.extend(list(criterion2.parameters()))

    # Initializing optimizer and scheduler
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': train_stage["lr_cauchy"]}], lr=train_stage["lr"], weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=train_stage["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams["scheduler_step"], gamma=hyperparams["scheduler_gamma"], last_epoch=-1)

    train_iter_count = 0
    val_iter_count = 0
    val_iter100_count = 0
    best_loss = 100000.0

    # Train loop
    for epoch in range(train_stage["num_epoch"]):
        print("Epoch {} of {}:".format(epoch + 1, train_stage["num_epoch"]))
        send_slack_notif("Epoch {} of {} Started!".format(epoch + 1, train_stage["num_epoch"]))

        for i, param_group in enumerate(optimizer.param_groups):
            print("Current LR: {} of {}th group".format(param_group['lr'], i))

        train_mse = 0.0
        train_msssim = 0.0
        train_hash = 0.0
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc="Batches") as pbar:
            for i, (data) in enumerate(train_loader):
                model.train()

                img, label = data
                label = label.to(device)
                label = make_one_hot(label)
                img = img.to(device)

                _, output, hashed_layer = model(img)

                if (i % 100 == 0 and epoch == 0) or (i % 500 == 0 and epoch > 0):

                    # PATH ASSUMES ONLY 1 STAGE
                    save_image(torch.cat((img, output)), "../results/ae_hash/images/train_check/{}_{}.jpg".format(epoch, i), nrow=batch_size)

                loss, mse, msssim = criterion1(output, img)
                loss_hash = criterion2(hashed_layer, label)

                if torch.isnan(loss_hash).any():
                    torch.save(model.state_dict(), "nan_aya_wo_model_weights.pt")
                    torch.save(img, "nan_dene_wala_img_batch.pt")
                    assert not torch.isnan(loss_hash).any()

                train_hash += loss_hash.cpu().item()
                train_mse += mse.cpu().item()
                train_msssim += msssim.cpu().item()
                train_loss += loss.cpu().item()

                for key, val in {'train_iter_loss': loss, 'train_iter_msssim': msssim, 'train_iter_hash': loss_hash, 'train_iter_mse': mse}.items():
                    writer.add_scalar(key, val.cpu().item(), train_iter_count)

                train_iter_count += 1
                total_loss = loss + loss_hash * cauchy_loss_weight
                total_loss = loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Val loop
                if i == len(train_loader) - 1:
                    val_loss = 0.0
                    val_msssim = 0.0
                    val_mse = 0.0
                    val_hash = 0.0
                    with torch.no_grad():
                        model.eval()
                        with tqdm(total=len(val_loader), desc="Val Batches at {}th batch".format(i)) as vbar:
                            for j, (data) in enumerate(val_loader):
                                img, label = data
                                label = label.to(device)
                                label = make_one_hot(label)
                                img = img.to(device)
                                _, output, hashed_layer = model(img)

                                if j % 50 == 0:

                                    # FILENAME ASSUMES ONLY ONE STAGE
                                    file_name = '../results/ae_hash/images/val_check/{}_{}.jpg'.format(epoch, j)
                                    save_image(torch.cat((img, output)), file_name, nrow=batch_size)

                                loss, mse, msssim = criterion1(output, img)
                                loss_hash = criterion2(hashed_layer, label)

                                val_loss += loss_hash.cpu().item() * cauchy_loss_weight
                                val_loss += loss.cpu().item()
                                val_msssim += msssim.cpu().item()
                                val_mse += mse.cpu().item()
                                val_hash += loss_hash.cpu().item()

                                for key, val in {'validation_iter_loss': loss, 'validation_iter_msssim': msssim, 'validation_iter_hash': loss_hash, 'validation_iter_mse': mse}.items():
                                    writer.add_scalar(key, val.cpu().item(), val_iter_count)

                                val_iter_count += 1
                                vbar.update(1)

                        for key, val in {'validation_100_loss': val_loss, 'validation_100_msssim': val_msssim, 'validation_100_hash': val_hash, 'validation_100_mse': mse}.items():
                            writer.add_scalar(key, val * batch_size / len(valset), val_iter100_count)

                        val_iter100_count += 1
                        if val_loss / len(valset) < best_loss:
                            torch.save(model.state_dict(), saved_model_name + '_hash_stage_mean{}.pt'.format(trn_stage_no + 1))
                            best_loss = val_loss / len(valset)

                pbar.update(1)

        sched.step()
        print("Train mse: {}, MSSSIM: {}, hash: {}, total loss: {}".format(train_mse, train_msssim, train_hash, train_loss))
        send_slack_notif("Train mse: {}, MSSSIM: {}, hash: {}, total loss: {} at the end of epoch {}".format(train_mse, train_msssim, train_hash, train_loss, epoch))
        for key, val in {'train_epoch_mse': train_mse, 'train_epoch_msssim': train_msssim, 'train_epoch_hash': train_hash, 'train_epoch_loss': train_loss}.items():
            writer.add_scalar(key, val * batch_size / len(trainset), epoch)
