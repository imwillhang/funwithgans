import argparse

from wgan import build_and_train

parser = argparse.ArgumentParser(description='hyperparams', add_help=False)
parser.add_argument('--epochs', type=int, nargs = '?', default=150)
parser.add_argument('--batch_sz', type=int, nargs = '?', default=64)
parser.add_argument('--dropout', type=float, nargs = '?', default= 0.5)
parser.add_argument('--lr', type=float, nargs = '?', default= 1e-3)
parser.add_argument('--max_grad_norm', type=float, nargs = '?', default= 5.0)
parser.add_argument('--reg', type=float, nargs = '?', default= 0.01)
parser.add_argument('--clip_grad', type=float, nargs = '?', default=5.0)
parser.add_argument('--z_dim', type=int, nargs = '?', default=64)
parser.add_argument('--d_train', type=int, nargs = '?', default=2)
parser.add_argument('--mode', type=int, nargs = '?', default=0)
parser.add_argument('--K', type=float, nargs = '?', default=0.01)
parser.add_argument('--loss_func', type=str, nargs = '?', default='mse')


if __name__ == '__main__':
	config = parser.parse_args()
	build_and_train(config)