"""
scripts for experiments of different loss functions
from https://github.com/Roche/BalancedLossNLP
"""

train_num = len(train_loader)
if args.loss_function == 'BCE':
    loss_function = ResampleLoss(reweight_func=None, loss_weight=1.0,
                             focal=dict(focal=False, alpha=0.5, gamma=2),
                             logit_reg=dict(),
                             class_freq=freq_file, train_num=train_num)

if args.loss_function == 'FL':
    loss_function = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(),
                                class_freq=freq_file, train_num=train_num)
        
if args.loss_function == 'CBloss': #CB
    loss_function = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(),
                                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                class_freq=freq_file, train_num=train_num) 
        
if args.loss_function == 'R-BCE-Focal': # R-FL
    loss_function = ResampleLoss(reweight_func='rebalance', loss_weight=1.0, 
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(),
                                map_param=dict(alpha=0.1, beta=10.0, gamma=args.gamma), 
                                class_freq=freq_file, train_num=train_num)

if args.loss_function == 'NTR-Focal': # NTR-FL
    loss_function = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=args.neg_scale),
                                class_freq=freq_file, train_num=train_num)
        
if args.loss_function == 'DBloss-noFocal': # DB-0FL
    loss_function = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                                focal=dict(focal=False, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=args.neg_scale),
                                map_param=dict(alpha=0.1, beta=10.0, gamma=args.gamma), 
                                class_freq=freq_file, train_num=train_num)

if args.loss_function == 'CBloss-ntr': # CB-NTR
    loss_function = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=args.neg_scale),
                                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                class_freq=freq_file, train_num=train_num)
        
if args.loss_function == 'DBloss': # DB
    loss_function = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                focal=dict(focal=True, alpha=0.5, gamma=2),
                                logit_reg=dict(init_bias=0.05, neg_scale=args.neg_scale),
                                map_param=dict(alpha=0.1, beta=10.0, gamma=args.gamma), 
                                class_freq=freq_file, train_num=train_num)