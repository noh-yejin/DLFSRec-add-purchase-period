import os
import torch
import argparse
import numpy as np

from models import DLFSRecModel
from trainers import DLFSRecTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_dataloader, get_rating_matrix, get_data_dic, \
    get_feats_vec
import time 
# import IPython; IPython.embed(colors="Linux"); exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/Grocery_and_Gourmet_Food/", type=str)
    parser.add_argument("--data_name", default="Grocery_and_Gourmet_Food", type=str)
    parser.add_argument("--output_dir", default="outputs/", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    
    # model args
    parser.add_argument("--model_name", default="DLFS-Rec", type=str)
    parser.add_argument("--hidden_size", default=32, type=int, help="hidden size of model")#128
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")#4
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--attribute_hidden_size", default=32, type=int, help="hidden size of model") # ->128
    parser.add_argument("--initializer_range", default=0.02, type=float)

    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate of adam")
    parser.add_argument("--train_batch_size", default=256, type=int, help="number of train batch_size")
    parser.add_argument("--eval_batch_size", default=512, type=int, help="number of eval batch_size")
    parser.add_argument("--epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true",default=False)  # CPU 사용 여부를 결정하는 옵션
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--full_sort", action="store_true")
    parser.add_argument("--patience", default=10, type=int,
                        help="how long to wait after last time validation loss improved")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float,
                        help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument("--fusion_type", default="add", type=str)
    parser.add_argument("--side_info_fused", action="store_false", help="frlp")
    parser.add_argument("--period_add_like_cxt", default=False, type=bool, help="period information add in side-information like context information")
    parser.add_argument("--e_commerce", default=False, type=bool, help="for e_commerce_dataset")
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument("--grocery", default=False, type=bool, help="for grocery_dataset")

    args = parser.parse_args()
    print(args)

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    # environment setting
    set_seed(args.seed)
    check_path(args.output_dir)
    
    if not args.no_cuda:  # CUDA를 사용하지 않는 경우에만 환경 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    args.cuda_condition = torch.cuda.is_available() 
    #  log file
    cur_time = get_local_time()
    args_str = f'{args.model_name}-{args.data_name}-{cur_time}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    data_dic = get_data_dic(args)


    args.item_size = data_dic['n_items']  # 0 ~ max_item
    args.feature_size = data_dic['feature_size']


    args.items_feature = get_feats_vec(data_dic['items_feat'], data_dic)

    if args.cuda_condition:
        args.items_feature = args.items_feature.cuda()

    if args.full_sort:
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(data_dic)

    train_dataloader, eval_dataloader, test_dataloader = get_dataloader(args, data_dic)

    model = DLFSRecModel(args=args) 
    trainer = DLFSRecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=args.full_sort)
    else:
        start_time = time.time()
        # save model path
        args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience,
                                       verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        end_time = time.time()
        execution_time = end_time - start_time

        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)

        print("---------------Sample 99 results---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=args.full_sort)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
        f.write(f"To run Epoch:{args.epochs} , It took {hours} hours, {minutes} minutes, {seconds} seconds\n")


# python main.py --model_name="baseline" --epochs=100 --patience=40 --fusion_type="concat" --hidden_size=64 
# python main.py --model_name="add period" --epochs=100 --patience=40 --fusion_type="concat" --period_add_like_cxt=True --hidden_size=64 
# python main.py --model_name="baseline_e_commerce" --epochs=100 --patience=40 --fusion_type="concat"--hidden_size=64 --data_dir='./data/e_commerce_cosmetic/' --data_name='e_commerce_cosmetic' --e_commerce=True --use_multi_gpu
# python main.py --model_name="add period_e_commerce" --epochs=100 --patience=40 --fusion_type="concat"--hidden_size=64 --data_dir='./data/e_commerce_cosmetic/' --data_name='e_commerce_cosmetic' --e_commerce=True --use_multi_gpu --period_add_like_cxt=True

main()
