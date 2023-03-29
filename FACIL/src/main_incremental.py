import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce

import utils
from torch.utils import data
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var
import torch.utils.data

from approach.moles import *
from datasets.memory_dataset import MemoryDataset

def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')
    parser.add_argument('--moles', action='store_true',
                        help='if model will be poisoned by moles')
    parser.add_argument('--safe', action='store_true',
                        help='to avoid using attacked/confounding combos when the combo would better perform flipped')
    parser.add_argument('--no_double', action='store_true',
                        help='to avoid attacking same class twice')
    parser.add_argument('--rho', default=0.5, type=float,
                        help='provide threshold rho value')
    parser.add_argument('--test', action='store_true',
                        help='to write logs to test file')


    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    from networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))


    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, total_set, taskcla, classes, mapping, mapping_id, transform = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)
    if 'cifar100' in args.datasets or 'cifar100_icarl' in args.datasets:
        class_order = dataset_config['cifar100_icarl']['class_order']
    if 'imagenet_subset' in args.datasets:
        class_order = dataset_config['imagenet_subset']['class_order']

    #sample mole indices
    # mole_indices = [19274, 42307, 39076, 14118, 39386, 23024, 35896, 55359, 47591, 438, 4544, 16331, 36290, 26676, 33227, 32053, 39478, 41704, 38778, 51388, 7772, 5046, 31503, 9623, 14620, 12742, 29741, 57431, 12453, 36941, 59048, 30544, 2465, 50410, 13134, 37567, 25462, 58942, 32673, 48955, 30084, 5042, 5587, 11114, 12718, 47039, 8022, 24850, 16207, 5201, 35098, 32595, 8756, 38755, 59938, 14407, 46322, 11231, 56683, 37903, 45030, 3514, 44869, 41023, 16975, 45670, 15974, 26134, 11814, 58687, 59213, 18021, 56472, 55102, 5259, 45150, 39011, 34904, 46470, 52011, 32145, 54981, 14521, 8204, 47301, 50965, 1995, 56798, 2241, 33381, 46881, 32575, 31416, 18006, 19323, 45270, 13095, 44768, 31212, 34832, 10109, 17387, 25716, 26693, 44871, 18506, 21289, 23914, 42709, 12869, 21985, 47783, 20420, 56186, 42437, 52791, 43246, 16803, 30952, 38838]
    # mole_data = {'x': [], 'y': []}
    # for i, image, label in zip(range(len(total_set)), total_set.data, total_set.targets):
    #     if i in mole_indices:
    #         #mole_data['x'].append(image)
    #         mole_data['x'].append(np.asarray(image))
    #         mole_data['y'].append(mapping_id[label])
    # mole_set = MemoryDataset(mole_data, transform, mole_indices)
    # mole_loader = data.DataLoader(mole_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    # num_row = 5
    # num_col = 5
    # fig, axes = plt.subplots(num_row, num_col, figsize=(0.75 * num_col, 0.75 * num_row))
    # for i in range(num_row * num_col):
    #     ax = axes[i // num_col, i % num_col]
    #     ax.imshow(mole_data['x'][i])
    #     ax.set_title(mole_data['y'][i], fontsize=8)
    #     ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=False, labelleft=False,
    #                    labelbottom=False, labelright=False)
    #     ax.set_yticks([])
    #     ax.set_xticks([])
    #     ax.set_yticklabels([])
    #     ax.set_xticklabels([])
    # plt.tight_layout()
    # plt.savefig('moles.png', bbox_inches='tight')  # was plt.show()
    # print(mapping)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print(taskcla)
    if args.moles:
        if 'cifar100' in args.datasets:
            table = np.load('./probabilitymatrix_cifar-100.npy', allow_pickle=True)
        if 'imagenet_subset' in args.datasets:
            table = np.load('./probabilitymatrix_imagenet_subset.npy', allow_pickle=True)
        moles = moleRecruitment(table, table.shape[1])
    id_attacked, attacked_confounding = {}, {}

    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    seen, prev_attacked = [], []
    file = '/{}_moles{}_seed{}_rho{}.txt'.format(args.datasets[0], args.moles, args.seed, args.rho)
    if args.test:
        f = open('logs/test.txt', 'w', newline='')
    else:
        f = open('logs/' + str(args.approach) + file, 'w', newline='')
    writer = csv.writer(f)
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        writer.writerow(['*' * 108])
        writer.writerow(['Task {:2d}'.format(t)])
        writer.writerow(['*' * 108])
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        task_classes = [classes[c] for c in class_order[len(seen):len(seen)+taskcla[t][1]]]

        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        if t and args.moles:
            combos, prev_attacked = selectMultiCombo(moles, prev_attacked, args.rho, seen, task_classes, classes, args.safe, args.no_double)
            print(combos)
            if (combos):
                mole_indices = moleSetMulti(moles, combos, table, task_classes, classes, t, len(taskcla), args.batch_size)
                mole_data = {'x': [], 'y': []}
                for i, image, label in zip(range(len(total_set)), total_set.data, total_set.targets):
                    if i in mole_indices:
                        mole_data['x'].append(np.asarray(image))
                        mole_data['y'].append(mapping_id[label])
                mole_set = MemoryDataset(mole_data, transform, mole_indices)
                mole_loader = data.DataLoader(mole_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=args.pin_memory)
                appr.nepochs = 1
                appr.train(t, mole_loader, val_loader[t])
                appr.nepochs = args.nepochs

                for percentile, attacked, confounding in combos:
                    if attacked in attacked_confounding.keys():
                        attacked_confounding[attacked].append(confounding)
                    else:
                        attacked_confounding[attacked] = [confounding]

        for c in task_classes:
            seen.append(c)
        print('-' * 108)

        # Test
        appr.confusion_matrix = np.zeros([len(classes), len(classes)], int)
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])
        attacked_stats, confounding_list = [], []
        for i, r in enumerate(appr.confusion_matrix[:taskcla[t][1] * (t+1)]):
            writer.writerow(['{0:15s} - {1:.1f}|'.format(mapping[i], r[i] / np.sum(r) * 100)])
            print('{0:15s} - {1:.1f}|'.format(mapping[i], r[i] / np.sum(r) * 100))
            if mapping[i] in attacked_confounding.keys():
                attacked_stats.append('{0:15s} - {1:.1f}'.format(mapping[i], r[i] / np.sum(r) * 100))
                confounding_list.append(attacked_confounding[mapping[i]])
        if t and args.moles:
            writer.writerow(["ATTACKED CLASSES"])
            print("ATTACKED CLASSES")
            for i, j in zip(attacked_stats, confounding_list):
                writer.writerow([str(i) + '     confounding = ' + str(j) + '|'])
                print(i, '     confounding = ', j, "|")

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(acc_tag, name="acc_tag", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.log_result(forg_tag, name="forg_tag", step=t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

    # Print Summary
    f.close()
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
