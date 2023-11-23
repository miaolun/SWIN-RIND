import os

# import torch
# import torchvision
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from dataloaders.dataloader import Mydataset
from torch.utils.data import DataLoader
from option.my_options import myNet_Options
from model.swin_rind import MyNet
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.edge_loss2 import *
from utils.loss import SegmentationLosses
from utils.saver import Saver
from utils.summaries import TensorboardSummary
import scipy.io as sio


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        print(self.saver.experiment_dir)
        self.output_dir = os.path.join(self.saver.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Define Dataloader
        self.train_dataset = Mydataset(root_path=self.args.data_path, split='trainval', crop_size=self.args.crop_size)
        self.test_dataset = Mydataset(root_path=self.args.data_path, split='test', crop_size=self.args.crop_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.workers)

        # Define network
        self.model = MyNet()
        # if self.args.swin_transformer:
        #     # state_dict = self.model.state_dict()
        #     self.model = torch.nn.DataParallel(self.model)
        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)

            checkpoint = torch.load(r'.\model_zoo\your_checkpoint.ckpt')
            pretrained_dict = checkpoint['model']
            state_dict = self.model.state_dict()
            # state_dict.update(pretrained_dict)
            load_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
            state_dict.update(load_dict)
            self.model.load_state_dict(state_dict, strict=False)
            if load_dict:
                print('chekcpoint loaded!')
            else:
                print('No parameters')



        # Define Criterion
        self.criterion = AttentionLoss2()
        self.criterion2 = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='focal')
        self.criterion3 = DiceLoss()

        # loss weight initialize
        self.para_k = nn.Parameter(torch.tensor([1.0], device=0), requires_grad=True)
        self.para_t = nn.Parameter(torch.tensor([1.0], device=0), requires_grad=True)
        self.para_q = nn.Parameter(torch.tensor([1.0], device=0), requires_grad=True)
        self.para_h = nn.Parameter(torch.tensor([1.0], device=0), requires_grad=True)

        # Define Optimizer
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}, {'params': self.para_k},
                                          {'params': self.para_t},{'params': self.para_q},
                                          {'params': self.para_h}],
                                         lr=self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, (image, target) in enumerate(tbar):
            if self.args.cuda:
                image, target = image.cuda(), target.cuda() # (b,3,w,h) (b,5,w,h)
            # loss calculation
            out_depth, out_normal, out_reflectance, out_illumination, _ = self.model(image)

            depth_gt = torch.unsqueeze(target[:, 1, :, :], dim=1)
            normal_gt = torch.unsqueeze(target[:, 2, :, :], dim=1)
            reflectance_gt = torch.unsqueeze(target[:, 3, :, :], dim=1)
            illumination_gt = torch.unsqueeze(target[:, 4, :, :], dim=1)

            loss_attn_depth = self.criterion(out_depth, depth_gt)
            loss_attn_normal = self.criterion(out_normal, normal_gt)
            loss_attn_reflectance = self.criterion(out_reflectance, reflectance_gt)
            loss_attn_illumination = self.criterion(out_illumination, illumination_gt)

            loss_dice_depth = self.criterion3(out_depth, depth_gt)
            loss_dice_normal = self.criterion3(out_normal, normal_gt)
            loss_dice_reflectance = self.criterion3(out_reflectance, reflectance_gt)
            loss_dice_illumination = self.criterion3(out_illumination, illumination_gt)

            k = self.para_k
            t = self.para_t
            q = self.para_q
            h = self.para_h

            loss_1 = (1.0 / torch.square(k)) * loss_attn_depth + \
                     (1.0 / torch.square(t)) * loss_attn_normal + \
                     (1.0 / torch.square(q)) * loss_attn_reflectance + \
                     (1.0 / torch.square(h)) * loss_attn_illumination
            loss_2 = 1000.0 * ((1.0 - loss_dice_depth) + (1.0 - loss_dice_normal) + (1.0 - loss_dice_reflectance) + (1.0 - loss_dice_illumination))
            chain_value = torch.log(k*t*q*h)

            total_loss = loss_1 + loss_2 + chain_value

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_loss += total_loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', total_loss.item(), i + num_img_tr * epoch)


        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            if (epoch + 1) % 10 == 0:
                is_best = False
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)

    def test(self, epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.depth_output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1), 'depth/mat')
        if not os.path.exists(self.depth_output_dir):
            os.makedirs(self.depth_output_dir)
        self.normal_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1), 'normal/mat')
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        self.reflectance_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1), 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir):
            os.makedirs(self.reflectance_output_dir)
        self.illumination_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1), 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir):
            os.makedirs(self.illumination_output_dir)

        self.viz_output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1), 'img')
        if not os.path.exists(self.viz_output_dir):
            os.makedirs(self.viz_output_dir)
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                # resize imgs
                _, _, H, W = image.shape
                resize_img = F.interpolate(image, size=(H - 1, W - 1), mode='nearest')
                out_depth, out_normal, out_reflectance, out_illumination, viz = self.model(resize_img)
                out_depth = F.interpolate(out_depth, size=image.size()[2:], mode='bilinear', align_corners=True)
                out_normal = F.interpolate(out_normal, size=image.size()[2:], mode='bilinear', align_corners=True)
                out_reflectance = F.interpolate(out_reflectance, size=image.size()[2:], mode='bilinear', align_corners=True)
                out_illumination = F.interpolate(out_illumination, size=image.size()[2:], mode='bilinear', align_corners=True)
            # mat file save
            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)), {'result': illumination_pred})
            # viz information
            # _, _, H, W = viz[0].size()
            # img = torch.zeros((len(viz), 1, H, W))
            # # for j in range(len(viz)):
            # #     img[j, 0, :, :] = viz[j][0, 0, :, :]
            # viz_1 = torch.cat([viz[2][0, 0, :, :], viz[3][0, 0, :, :]], 1)
            # viz_2 = torch.cat([viz[1][0, 0, :, :], viz[0][0, 0, :, :]], 1)
            # img = torch.cat([viz_1, viz_2], 0)
            # torchvision.utils.save_image(img, os.path.join(self.viz_output_dir, "{}.jpg".format(name)))

def main():
    options = myNet_Options()
    args = options.parse()
    #args.cuda = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # args.lr = 1e-5
    print(args)

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        # trainer.test(epoch)
        trainer.training(epoch)
        # if (epoch+1)%10==0:
        if (epoch + 1) % 10 == 0:
            trainer.test(epoch)

if __name__ == "__main__":
    main()
