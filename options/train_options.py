import argparse
import os
import os.path as osp
from pathlib import Path


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--source", type=str, default='gta5',help="source dataset : gta5 or synthia")
        parser.add_argument("--target", type=str, default='cityscapes',help="target dataset : cityscapes")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--data-dir", type=str, default='/path/to/dataset/source', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list", type=str, default='/path/to/dataset/source_list', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data-dir-target", type=str, default='/path/to/dataset/target', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='/path/to/dataset/target_list', help="Path to the file listing the images in the target dataset.")
        # for self-training, keep it by now
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")

        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial weight of the segmentation network")
        parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument("--num-clusters", type=int, default=50, help="K, mentioned in the paper") # patch paper

        # TODO: could try to fool also for source input?!
        parser.add_argument("--lambda-adv-fake", type=float, default=0.001,
                            help="lambda of adv loss while training G.")
        parser.add_argument("--lambda-adv-real", type=float, default=1,
                            help="lambda of adv loss while training D.")
        parser.add_argument("--lambda-p", type=float, default=1,
                            help="lambda of Ld mentioned in the paper") # patch paper
        parser.add_argument("--lambda-adv-fake-p", type=float, default=0.001,
                            help="lambda of adv loss while training G & H.") # patch paper
        parser.add_argument("--lambda-adv-real-p", type=float, default=1,
                            help="lambda of adv loss while training Dp, the discriminator of patch-representation.") # patch paper

        # TODO
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--learning-rate-D", type=float, default=1e-4, help="initial learning rate for discriminator.")
        parser.add_argument("--learning-rate-Dp", type=float, default=1e-4, help="initial learning rate for patch discriminator.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss. Used in SGD in DeepLab")

        # TODO
        parser.add_argument("--num-steps", type=int, default=250000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=120000, help="Number of training steps for early stopping.")


        parser.add_argument("--restore-from", type=str, default=None, help="Where to restore model parameters from.")
        parser.add_argument("--save-checkpoint-every", type=int, default=10000, help="Save checkpoint every often.")
        parser.add_argument("--checkpoint-dir", type=str, default='./checkpoints/', help="Where to save snapshots of the model.")
        parser.add_argument("--create-snapshot-folder", type=str, default='snapshot',
                            help="name of created folder for storing model pth and argument txt, the folder is under args.checkpoint_dir")
        parser.add_argument("--add-tb-image-every", type=int, default=100, help="Add images to tensorboard record every often.")
        parser.add_argument("--tb-create-exp-folder", type=str, default='experiment', help="name of tensorboard-created-experiment folder")
        parser.add_argument("--print-freq", type=int, default=100, help="print loss and time fequency.")


        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")

        
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        snapshot_dir = os.path.join(args.checkpoint_dir, args.create_snapshot_folder)
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        file_name = osp.join(snapshot_dir, 'args.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
        