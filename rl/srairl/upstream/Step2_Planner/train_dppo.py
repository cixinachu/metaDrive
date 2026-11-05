import torch
import numpy as np
from model.temporal import TemporalUnet
from model.diffusion import GaussianDiffusion
from utils.training import Trainer
from utils.dataset import SequenceDataset
from utils.arrays import *
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
from pathlib import Path
import os
from datetime import datetime
import warnings
import hashlib
warnings.filterwarnings('ignore')
import argparse
import json
import time
from utils.arrays import set_device


def main(args):
    device = args.device
    set_device(device)
    data_path = args.data_path
    train_dataset = SequenceDataset(data_path, horizon=100)

    # 统一基准路径为当前脚本所在目录，避免从不同工作目录运行时找不到日志/模型保存位置
    base_dir = Path(__file__).resolve().parent

    model = TemporalUnet(horizon=100,
                         transition_dim=4+2,
                         dim=64,
                         cond_dim=4,
                         dim_mults=(1, 2, 4, 8),
                         attention=False).to(device)

    diffusion = GaussianDiffusion(model,
                                  horizon=100,
                                  observation_dim=4,
                                  action_dim=2,
                                  n_timesteps=20,
                                  loss_type='l2',
                                  clip_denoised=False,
                                  predict_epsilon=False,
                                  ## loss weighting
                                  action_weight=10,
                                  loss_weights=None,
                                  loss_discount=1).to(device)

    # 确保模型保存目录存在，避免 torch.save 找不到父目录
    results_dir = Path(args.output_dir) if args.output_dir else base_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 提前创建 TensorBoard writer，便于传入 Trainer 内部日志（避免下方未定义）
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.studio_dir) if args.studio_dir else base_dir / "runs" / "planner_dppo" / f"train_{time_str}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f">>> TensorBoard logging to: {log_dir}")

    metrics_started = time.monotonic()
    progress_file = Path(args.studio_dir)/'metrics.jsonl' if args.studio_dir else None
    def progress(row):
        studio_dir=Path(args.studio_dir)
        values={'train/loss':float(row['loss']),'train/loss_bc':float(row['loss_bc'])}
        if row.get('loss_dppo') is not None:values['train/loss_dppo']=float(row['loss_dppo'])
        with progress_file.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'step':int(row['step']),'elapsed':time.monotonic()-metrics_started,
                                     'values':values},allow_nan=False)+'\n')
        status={'state':'running','kind':'train','step':int(row['step']),'total':args.steps,
                'episodes':0,'device':device,'phase':'planner_dppo'}
        tmp=studio_dir/'status.json.tmp';tmp.write_text(json.dumps(status),encoding='utf-8');tmp.replace(studio_dir/'status.json')
        return not (studio_dir/'STOP').exists()

    trainer = Trainer(diffusion,
                      dataset=train_dataset,
                      train_batch_size=args.batch_size,
                      train_lr=2e-4,
                      gradient_accumulate_every=2,
                      ema_decay=0.995,
                      sample_freq=20000,
                      save_freq=4000,
                      label_freq=int(1E6 // 5),
                      save_parallel=False,
                      results_folder=str(results_dir),
                      bucket=None,
                      n_reference=8,
                      writer=writer,  # 传入 TensorBoard Writer，训练内部日志与当前 log_dir 对齐
                      save_last_only=True,

                      # ---- DPPO over denoising (no Step1 reward) ----
                      use_dppo=True,
                      dppo_coef=0.02,      # 先小一点，别抢 BC
                      ppo_clip=0.2,
                      dppo_gamma=0.99,
                      dppo_every=10,
                      progress_callback=progress if args.studio_dir else None,
                      checkpoint_metadata={
                          'obs_len': 20, 'horizon': 100, 'pred_len': 100, 'dt': 0.1,
                          'transition_dim': 6,
                          'n_diffusion_steps': 20,
                          'dataset_revision': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                          'normalization_version': 'planner-goal-v1',
                      })

    # Main Loop
    print(f'Training planner for {args.steps} update steps')
    trainer.train(n_train_steps=args.steps)

    writer.close()
    if args.studio_dir:
        checkpoint=results_dir/'state_final.pt'
        if not checkpoint.is_file():raise RuntimeError('planner training did not save a final checkpoint')
        summary={'phase':'planner_dppo','steps':trainer.step,
                 'checkpoint':str(checkpoint.relative_to(Path(args.output_dir).resolve()))}
        tmp=Path(args.studio_dir)/'planner_dppo_summary.json.tmp'
        tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(tmp.with_suffix(''))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--data_path',default='')
    parser.add_argument('--output_dir',default='')
    parser.add_argument('--studio_dir',default='')
    parser.add_argument('--steps',type=int,default=1_000_000)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    if not args.data_path:raise ValueError('--data_path is required; provide a native MetaDrive demonstration archive')
    if not args.output_dir:args.output_dir=str(Path(__file__).resolve().parent/'results')
    main(args)

