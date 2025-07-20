from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory, OTBDataset, UAVDataset, LaSOTDataset, \
    VOTDataset, NFSDataset, VOTLTDataset, UAV10Dataset, DTBDataset
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
    EAOBenchmark, F1Benchmark
from pysot.tracker.base_tracker import SiameseTracker
import torch.nn.functional as F

import optuna
import logging

class SiamHSTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamHSTracker, self).__init__()

        self.score_size = cfg.TRAIN.OUTPUT_SIZE
        self.anchor_num = 1
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.model = model
        self.model.eval()

    def generate_anchor(self, mapp):
        def dcon(x):
            x[np.where(x <= -1)] = -0.99
            x[np.where(x >= 1)] = 0.99
            return (np.log(1 + x) - np.log(1 - x)) / 2

        size = cfg.TRAIN.OUTPUT_SIZE
        x = np.tile((cfg.ANCHOR.STRIDE * (np.linspace(0, size - 1, size)) + 63) - cfg.TRAIN.SEARCH_SIZE // 2,
                    size).reshape(-1)
        y = np.tile(
            (cfg.ANCHOR.STRIDE * (np.linspace(0, size - 1, size)) + 63).reshape(-1, 1) - cfg.TRAIN.SEARCH_SIZE // 2,
            size).reshape(-1)
        shap = (dcon(mapp[0].cpu().detach().numpy())) * (cfg.TRAIN.SEARCH_SIZE // 2)
        xx = np.int16(np.tile(np.linspace(0, size - 1, size), size).reshape(-1))
        yy = np.int16(np.tile(np.linspace(0, size - 1, size).reshape(-1, 1), size).reshape(-1))
        w = shap[0, yy, xx] + shap[1, yy, xx]
        h = shap[2, yy, xx] + shap[3, yy, xx]
        x = x - shap[0, yy, xx] + w / 2
        y = y - shap[2, yy, xx] + h / 2

        anchor = np.zeros((size ** 2, 4))

        anchor[:, 0] = x
        anchor[:, 1] = y
        anchor[:, 2] = w
        anchor[:, 3] = h
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.image = img

        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])

        self.size = np.array([bbox[2], bbox[3]])
        self.firstbbox = np.concatenate((self.center_pos, self.size))
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scaleaa = s_z

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow_im(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.template = z_crop

        self.model.template(z_crop)

    def con(self, x):
        return x * (cfg.TRAIN.SEARCH_SIZE // 2)

    def track(self, img, iter):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scaleaa
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z

        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow_im(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        pred_bbox = self.generate_anchor(outputs['loc']).transpose()

        score2 = self._convert_score(outputs['cls1']) * cfg.TRACK.w1
        score3 = (outputs['cls2']).view(-1).cpu().detach().numpy() * cfg.TRACK.w2
        score = (score2 + score3) / 2

        def change(r):
            return np.maximum(r, 1. / (r + 1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / (self.size[1] + 1e-5)) /
                     (pred_bbox[2, :] / (pred_bbox[3, :] + 1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        self.good_crop = self.get_subwindow_im(img, self.center_pos,
                                               cfg.TRACK.EXEMPLAR_SIZE,
                                               s_z, self.channel_average)
        # if best_score > 0.9:
        #     self.good_crop = self.get_subwindow_im(img, self.center_pos,
        #                                            cfg.TRACK.EXEMPLAR_SIZE,
        #                                            s_z, self.channel_average)
        #
        # if iter > 1 and best_score < 0.7:
        self.model.templete_update(self.good_crop)
        return {
            'bbox': bbox,
            'best_score': best_score,
        }



def eval(dataset, tracker_name):
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    # root = os.path.join(root, dataset)
    tracker_dir = "./"
    trackers = [tracker_name]
    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'DTB70' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV123_10fps' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV123' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'NFS' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao
    elif 'VOT2018-LT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                              show_video_level=False)

    return 0


# fitness function
def objective(trial):
    # different params
    cfg.TRACK.WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.250, 0.450)
    cfg.TRACK.PENALTY_K = trial.suggest_uniform('penalty_k', 0.000, 0.300)
    cfg.TRACK.LR = trial.suggest_uniform('scale_lr', 0.100, 0.500)
    # cfg.TRACK.WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.45, 0.48)
    # cfg.TRACK.PENALTY_K = trial.suggest_uniform('penalty_k', 0.07, 0.11)
    # cfg.TRACK.LR = trial.suggest_uniform('scale_lr', 0.43, 0.49)

    # rebuild tracker
    tracker = SiamHSTracker(model)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    tracker_name = os.path.join('tune_results', args.dataset, model_name, model_name + \
                                '_wi-{:.3f}'.format(cfg.TRACK.WINDOW_INFLUENCE) + \
                                '_pk-{:.3f}'.format(cfg.TRACK.PENALTY_K) + \
                                '_lr-{:.3f}'.format(cfg.TRACK.LR))
    total_lost = 0
    flag = False
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img, idx)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(tracker_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
            if v_idx + 1 < 16 and total_lost > 11:
                flag = True
                break
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        eao = eval(dataset=dataset_eval, tracker_name=tracker_name)
        if flag:
            eao = -1
            info = "very bad skipp!!!"
        else:
            info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, EAO: {:1.3f}, LN: {:d}".format(
                model_name, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.PENALTY_K, cfg.TRACK.LR, eao, total_lost)
        logging.getLogger().info(info)
        print(info)
        return eao
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)

                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img, idx)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))
        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
            model_name, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.PENALTY_K, cfg.TRACK.LR, auc)
        logging.getLogger().info(info)
        print(info)
        return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tuning for SiamBAN')
    parser.add_argument('--dataset', default='DTB70', type=str, help='dataset')
    parser.add_argument('--config', default='../experiments/my_temporal_alexnet/config.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='snapshot/checkpoint_e11.pth', type=str,
                        help='snapshot of models to eval')
    parser.add_argument("--gpu_id", default="1", type=str, help="gpu id")

    args = parser.parse_args()

    torch.set_num_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join('D:/test_set', args.dataset)  ##DTB70, VOT
    # dataset_root = os.path.join('D:/test_set', args.dataset, 'data_seq') ##UAV123, UAV10fps
    # dataset_root = os.path.join('D:/test_set', args.dataset, 'test') ###got,

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # Eval dataset
    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         'D:/test_set'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset_eval = OTBDataset(args.dataset, root)
    elif 'LaSOT' == args.dataset:
        dataset_eval = LaSOTDataset(args.dataset, root)
    elif 'NFS' in args.dataset:
        dataset_eval = NFSDataset(args.dataset, root)
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset_eval = VOTDataset(args.dataset, root)
    elif 'VOT2018-LT' == args.dataset:
        dataset_eval = VOTLTDataset(args.dataset, root)
    elif 'DTB70' == args.dataset:
        dataset_eval = DTBDataset(args.dataset, root)
    elif 'UAV123_10fps' == args.dataset:
        dataset_eval = UAV10Dataset(args.dataset, root + '/data_seq')
    elif 'UAV123' == args.dataset:
        dataset_eval = UAVDataset(args.dataset, root + '/data_seq')

    tune_result = os.path.join('tune_results', args.dataset)
    if not os.path.isdir(tune_result):
        os.makedirs(tune_result)
    log_path = os.path.join(tune_result, (args.snapshot).split('/')[-1].split('.')[0] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


