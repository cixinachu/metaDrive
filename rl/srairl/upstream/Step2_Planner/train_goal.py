import copy
import torch.nn as nn
from utils.arrays import *
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
from pathlib import Path
import os
from datetime import datetime  # 补充 datetime 用于日志命名
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import time
import hashlib

import pickle
import sys
import types
import pandas as pd
if "pandas.core.indexes.numeric" not in sys.modules:
    _numeric = types.ModuleType("pandas.core.indexes.numeric")
    _numeric.Int64Index = pd.Index
    _numeric.UInt64Index = pd.Index
    _numeric.Float64Index = pd.Index
    sys.modules[_numeric.__name__] = _numeric
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


def cycle(dl):
    while True:
        for data in dl:
            yield data


def trajectory_process(trajectory, decision_time):
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, horizon=32, use_padding=False, if_test=False):
        self.horizon = horizon
        self.use_padding = use_padding
        self.max_path_length = 151

        with open(data_path, "rb") as input_file:
            data = pickle.load(input_file)

        self.record_trajectory = []
        self.record_goal = []
        self.if_test = False

        if if_test:
            data = data[150:]
            self.if_test = True
        else:
            data = data[:150]

        for pair in data:
            end_frame = pair['end_frame'].values[0]
            decision_frame = pair['decision_frame'].values[0]
            lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']].iloc[20:]
            fv = pair[['frame', 'id_x', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']].iloc[20:]
            nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']].iloc[20:]
            olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']].iloc[20:]

            lcv = trajectory_process(lcv, decision_frame)
            fv = trajectory_process(fv, decision_frame)
            nlv = trajectory_process(nlv, decision_frame)
            olv = trajectory_process(olv, decision_frame)

            base_y = copy.deepcopy(lcv['y'].values[0])
            base_x = copy.deepcopy(lcv['x'].values[0])

            lcv['y'] -= base_y
            lcv['x'] -= base_x
            fv['y'] -= base_y
            fv['x'] -= base_x
            nlv['y'] -= base_y
            nlv['x'] -= base_x
            olv['y'] -= base_y
            olv['x'] -= base_x
            self.record_trajectory.append([lcv.iloc[:20], fv.iloc[:20], nlv.iloc[:20], olv.iloc[:20]])
            self.record_goal.append(lcv.iloc[100] - lcv.iloc[0])

        if not if_test:
            all_x = []
            all_v = []
            for idx in range(150):
                sub = self.record_goal[idx][['x', 'v_x']].values
                all_x.append(sub[0])
                all_v.append(sub[1])

            self.max_x, self.min_x = np.max(all_x), np.min(all_x)
            self.max_v, self.min_v = np.max(all_v), np.min(all_v)

    def __len__(self):
        return len(self.record_trajectory)

    def __getitem__(self, idx, eps=1e-4):
        lcv = self.record_trajectory[idx][0][['y', 'x', 'v_x', 'v_y']].values
        fv = self.record_trajectory[idx][1][['y', 'x', 'v_x', 'v_y']].values
        nlv = self.record_trajectory[idx][2][['y', 'x', 'v_x', 'v_y']].values
        olv = self.record_trajectory[idx][3][['y', 'x', 'v_x', 'v_y']].values
        goal = self.record_goal[idx][['x', 'v_x']].values  # delta x and delta v

        lcv[:, 0] /= 50
        lcv[:, 1] /= 500
        lcv[:, 2] = (lcv[:, 2] + 40) / 80
        lcv[:, 3] = (lcv[:, 3] + 40) / 80

        fv[:, 0] /= 50
        fv[:, 1] /= 500
        fv[:, 2] = (fv[:, 2] + 40) / 80
        fv[:, 3] = (fv[:, 3] + 40) / 80

        nlv[:, 0] /= 50
        nlv[:, 1] /= 500
        nlv[:, 2] = (nlv[:, 2] + 40) / 80
        nlv[:, 3] = (nlv[:, 3] + 40) / 80

        olv[:, 0] /= 50
        olv[:, 1] /= 500
        olv[:, 2] = (olv[:, 2] + 40) / 80
        olv[:, 3] = (olv[:, 3] + 40) / 80

        # goal[0] /= 500
        # goal[1] = (goal[1] + 40) / 80

        return [lcv, fv, nlv, olv, goal]


class GoalPredictor(torch.nn.Module):
    def __init__(self, context_dim, input_size=2):
        super(GoalPredictor, self).__init__()
        self.context_dim = context_dim

        self.lcv_gru = nn.GRU(input_size=16, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.fv_gru = nn.GRU(input_size=4, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(context_dim, context_dim),
                                 nn.ReLU(),
                                 nn.Linear(context_dim, 2),
                                 nn.Sigmoid())

    def forward(self, lcv, fv, nlv, olv):
        x = torch.cat([lcv, fv, nlv, olv], dim=-1)
        _, hidden_lcv = self.lcv_gru(x)

        pred = self.mlp(hidden_lcv[0])
        return pred


def evaluation(model, loader, bound_min, bound_max):
    model.eval()
    for batch in loader:
        b_lcv = batch[0].float().to(bound_min.device)
        b_fv = batch[1].float().to(bound_min.device)
        b_nlv = batch[2].float().to(bound_min.device)
        b_olv = batch[3].float().to(bound_min.device)
        b_goal = batch[4].float().to(bound_min.device)

        pred = model(b_lcv, b_fv, b_nlv, b_olv)

        pred = pred * (bound_max - bound_min) + bound_min

        x_rmse = (nn.functional.mse_loss(pred[:, 0], b_goal[:, 0], reduction='mean')).sqrt()
        v_rmse = (nn.functional.mse_loss(pred[:, 1], b_goal[:, 1], reduction='mean')).sqrt()

    return x_rmse.item(), v_rmse.item()


def main(args):
    device = args.device
    data_path = args.data_path
    train_dataset = SequenceDataset(data_path, horizon=100)
    test_dataset = SequenceDataset(data_path, horizon=100, if_test=True)

    # 基准路径与日志/模型目录
    base_dir = Path(__file__).resolve().parent
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.output_dir) if args.output_dir else base_dir / "runs" / "planner_goal" / f"train_{time_str}"
    writer = SummaryWriter(log_dir=str(log_dir))  # 统一记录所有指标
    print(f">>> TensorBoard logging to: {log_dir}")
    results_dir = log_dir / "results"  # 模型保存目录位于当前日志下
    os.makedirs(results_dir, exist_ok=True)

    bound_min = torch.tensor([train_dataset.min_x, train_dataset.min_v]).to(device).float()
    bound_max = torch.tensor([train_dataset.max_x, train_dataset.max_v]).to(device).float()

    model = GoalPredictor(128).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                             num_workers=0, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                              num_workers=0, shuffle=False, pin_memory=True)

    best_rmse = 1e6  # 旧逻辑：只在达到最好指标时保存（恢复该逻辑）
    best_v_rmse = float('inf')
    metrics_started = time.monotonic()
    for epoch in range(args.epochs):
        if args.studio_dir and (Path(args.studio_dir) / 'STOP').exists():
            break
        epoch_loss = []
        for batch in dataloader:
            model.train()
            b_lcv = batch[0].float().to(device)
            b_fv = batch[1].float().to(device)
            b_nlv = batch[2].float().to(device)
            b_olv = batch[3].float().to(device)
            b_goal = batch[4].float().to(device)

            b_goal = (b_goal - bound_min) / (bound_max - bound_min)
            pred = model(b_lcv, b_fv, b_nlv, b_olv)

            loss = nn.functional.mse_loss(pred, b_goal)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())

        test_x, test_v = evaluation(model, test_loader, bound_min, bound_max)

        print('Epoch {}, Training loss {:.4}, test x {:.4}, test v {:.4}'.format(epoch,
                                                                                np.mean(epoch_loss), test_x, test_v))

        # 记录到 TensorBoard，便于可视化
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        writer.add_scalar('test/x_rmse', test_x, epoch)
        writer.add_scalar('test/v_rmse', test_v, epoch)

        # 仅在指标提升时保存最佳模型（恢复旧逻辑）；否则不覆盖
        if test_x < best_rmse:
            best_rmse = test_x
            best_v_rmse = test_v
            model_path = results_dir / "goal_model.pth"
            torch.save({
                'state_dict': model.state_dict(),
                'bound_min': bound_min.detach().cpu(),
                'bound_max': bound_max.detach().cpu(),
                'architecture': {'context_dim': 128, 'input_size': 2},
                'metadata': {'obs_len': 20, 'horizon': 100, 'dt': 0.1,
                             'pred_len': 100, 'normalization_version': 'planner-goal-v1',
                             'dataset_revision': hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest()},
            }, model_path)
            print(f'Best model saved to {model_path} (x_rmse={test_x:.4f})')
        if args.studio_dir:
            studio_dir=Path(args.studio_dir)
            values={'train/loss':float(np.mean(epoch_loss)),'validation/goal_x_rmse':float(test_x),
                    'validation/goal_speed_rmse':float(test_v)}
            with (studio_dir/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'step':epoch+1,'elapsed':time.monotonic()-metrics_started,
                                         'values':values},allow_nan=False)+'\n')
            status={'state':'running','kind':'train','step':epoch+1,'total':args.epochs,
                    'episodes':0,'device':str(device),'phase':'planner_goal'}
            tmp=studio_dir/'status.json.tmp';tmp.write_text(json.dumps(status),encoding='utf-8');tmp.replace(studio_dir/'status.json')

    writer.close()  # 训练结束关闭日志
    if args.studio_dir:
        checkpoint=results_dir/'goal_model.pth'
        if not checkpoint.is_file():raise RuntimeError('goal predictor did not produce a validation-selected checkpoint')
        summary={'phase':'planner_goal','best_validation_x_rmse':best_rmse,
                 'best_validation_speed_rmse':best_v_rmse,
                 'checkpoint':str(checkpoint.relative_to(Path(args.output_dir).resolve()))}
        tmp=Path(args.studio_dir)/'planner_goal_summary.json.tmp'
        tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(tmp.with_suffix(''))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--data_path',default=str(Path(__file__).resolve().parents[4]/'rl/srairl/upstream/data/sample_data.pkl'))
    parser.add_argument('--output_dir',default='')
    parser.add_argument('--studio_dir',default='')
    parser.add_argument('--epochs',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    main(args)
