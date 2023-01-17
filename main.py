import argparse
from trainer import Trainer

if __name__=='__main__':
    ## Parser 
    parser = argparse.ArgumentParser(description="Train the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model", default='unet', type=str, dest="model")
    parser.add_argument("--loss", default='BCE', type=str, dest="loss")
    parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
    parser.add_argument("--batch_size", default=10, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
    
    parser.add_argument("--data_dir", default="../data/brain_mri", type=str, dest="data_dir")
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./logs", type=str, dest="log_dir")
    parser.add_argument("--result_dir", default="./results", type=str, dest="result_dir")
    
    parser.add_argument("--mode", default="train", type=str, dest="mode")
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
    
    args = parser.parse_args()
    mode = args.mode
    trainer = Trainer(args.model, args.loss, args.data_dir, args.lr, args.batch_size, args.num_epoch, args.ckpt_dir, args.log_dir, args.result_dir, args.train_continue)
    
    if mode == 'train':
        print('mode: ', mode)
        trainer.train()