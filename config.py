import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="cafecup")
    parser.add_argument("dataset_model_name", type=str, default="cafecup")
    parser.add_argument("--model_ckpt", type=str, default="None")
    parser.add_argument("ckpt_name", type=str, default="None")
    parser.add_argument("--debug", action='store_false')
    parser.add_argument("--train_batchsize", type=int, default=8) # 8
    parser.add_argument("--valid_batchsize", type=int, default=1) # 8
    parser.add_argument("--test_batchsize", type=int, default=1) # 8
    parser.add_argument("--train_epoch", type=int, default=50) # 50
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default=0, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    return parser