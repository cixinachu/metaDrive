"""Fixed NGSIM experiment membership and reproducible multi-clip training resets."""
from pathlib import Path
import hashlib
import json
import gymnasium as gym
import numpy as np
from datasets.importer import ROOT,load_clip,digest
from studio.config import clean_request,environment_config
from env.safe_metadrive_env import SafeMetaDriveEnv

DEFAULT_EXPERIMENT=ROOT/'datasets/experiments/ngsim-comparison-v3/selected.json'


def members(split, path=DEFAULT_EXPERIMENT):
    experiment=json.loads(Path(path).read_text(encoding='utf-8'))
    experiment['membership_manifest_sha256']=digest(path)
    if experiment.get('membership_sha256'):
        ids=[c['clip_id'] for c in experiment['clips']]
        actual=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if actual!=experiment['membership_sha256']:
            raise ValueError('Experiment membership list fingerprint mismatch')
    entries=[c for c in experiment['clips'] if c['split']==split]
    if not entries:raise ValueError(f'No clips for {split}')
    for c in entries:
        _,m=load_clip(c['clip_id'])
        expected_config=c.get('selection_config_sha256',experiment['config_sha256'])
        if (m['experiment_split']!=split or m['selection_config_sha256']!=expected_config
                or m['clip_sha256']!=c.get('clip_sha256',m['clip_sha256'])
                or m['event_class']!=c['event_class']
                or m.get('recording_group')!=c.get('recording_cluster',m.get('recording_group'))
                or m['source_sha256']['trajectories.csv']!=c.get('source_sha256',experiment.get('source_sha256'))):
            raise ValueError('Experiment membership/fingerprint mismatch')
    if split=='validation' and experiment.get('validation_selection'):
        chosen=set(experiment['validation_selection'])
        experiment['validation_selection_entries']=[c for c in entries if c['clip_id'] in chosen]
        if len(chosen)!=len(experiment['validation_selection_entries']):
            raise ValueError('Validation selection has missing or duplicate clip IDs')
    return entries,experiment


class NGSIMCollectionEnv(gym.Env):
    """Train with balanced class/clip sampling; evaluate each listed clip once.

    Evaluation advances through the fixed list exactly once; no repeated-seed samples.
    """
    def __init__(self, settings, split='train', path=DEFAULT_EXPERIMENT, clip_ids=None):
        super().__init__()
        self.entries,self.experiment=members(split,path)
        if clip_ids is not None:
            requested=list(clip_ids)
            by_id={e['clip_id']:e for e in self.entries}
            if len(requested)!=len(set(requested)) or any(x not in by_id for x in requested):
                raise ValueError('Evaluation clip IDs must be unique members of the requested split')
            self.entries=[by_id[x] for x in requested]
            if not self.entries:raise ValueError('Empty evaluation clip list')
        self.settings=dict(settings);self.split=split;self.rng=np.random.default_rng(settings['seed']);self.cursor=0
        self.train_classes=sorted({e['event_class'] for e in self.entries}) if split=='train' else []
        self.train_by_class={k:[e for e in self.entries if e['event_class']==k] for k in self.train_classes}
        self.env=None;self.current=None
        self._make(self.entries[0])
        self.observation_space=self.env.observation_space;self.action_space=self.env.action_space

    def _make(self, entry):
        if self.env is not None:self.env.close()
        raw=dict(self.settings,data_source='ngsim',dataset_id=entry['clip_id'],episodes=1,num_scenarios=1,seed_start=1000 if self.split=='train' else 0)
        c=clean_request(raw,'train' if self.split=='train' else 'eval')
        self.env=SafeMetaDriveEnv(research_config=environment_config(c,'train' if self.split=='train' else 'eval'))
        self.current=entry

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        if self.split=='train':
            event_class=self.train_classes[int(self.rng.integers(len(self.train_classes)))]
            class_entries=self.train_by_class[event_class]
            entry=class_entries[int(self.rng.integers(len(class_entries)))]
        else:
            if self.cursor>=len(self.entries):raise RuntimeError('Evaluation collection exhausted; do not count repeats as new samples')
            entry=self.entries[self.cursor];self.cursor+=1
        self._make(entry)
        obs,info=self.env.reset(seed=1000 if self.split=='train' else 0)
        info['experiment_clip']=self.current['clip_id'];return obs,info

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info.update(experiment_clip=self.current['clip_id'],experiment_split=self.split,event_class=self.current['event_class'],recording=self.current['recording'])
        return obs,reward,terminated,truncated,info

    def close(self):
        if self.env is not None:self.env.close();self.env=None
