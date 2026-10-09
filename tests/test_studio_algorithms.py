"""Numerical and serialization checks, independent of MetaDrive rendering."""
import tempfile
import unittest
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.logger import configure
from rl.constrained_sac import (SACLagrangian,WCSAC,WCSACIQN,CostReplayBuffer,gaussian_cvar,
                                quantile_huber_loss,gaussian_targets)
from studio.config import clean_request
from studio.worker import empirical_cvar


class TinyCostEnv(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return np.zeros(3,dtype=np.float32),{}
    def step(self,a):
        self.steps+=1
        return np.full(3,.1,dtype=np.float32),float(a[0]),False,self.steps>=8,{'cost':float((a[0]+1)/2)}


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_gaussian_cvar_and_targets(self):
        m=torch.tensor([[2.]]);v=torch.tensor([[4.]])
        self.assertAlmostEqual(gaussian_cvar(m,v,.5).item(),2+2*np.sqrt(2/np.pi),places=5)
        self.assertEqual(gaussian_cvar(m,v,1).item(),2)
        tm,tv=gaussian_targets(torch.tensor([[1.]]),torch.tensor([[.5]]),m,v,torch.tensor([[2.]]))
        self.assertAlmostEqual(tm.item(),2)
        self.assertAlmostEqual(tv.item(),1)
        tm,tv=gaussian_targets(torch.tensor([[1.]]),torch.tensor([[0.]]),m,v,m)
        self.assertEqual(tm.item(),1);self.assertAlmostEqual(tv.item(),1e-6)

    def test_quantile_loss_targets_and_action_gradient(self):
        pred=torch.tensor([[0.,0.]],requires_grad=True);tau=torch.tensor([[.25,.75]])
        loss=quantile_huber_loss(pred,torch.ones_like(pred),tau)
        self.assertAlmostEqual(loss.item(),.25)
        loss.backward();self.assertTrue((pred.grad<0).all())
        for cls in (SACLagrangian,WCSAC,WCSACIQN):
            model=cls('MlpPolicy',TinyCostEnv(),device='cpu',buffer_size=128,safety_width=16,n_quantiles=8,policy_kwargs={'net_arch':[16,16]})
            a=torch.zeros((4,1),requires_grad=True)
            risk=model.risk(torch.ones((4,3)),a);risk.sum().backward()
            self.assertIsNotNone(a.grad);self.assertGreater(a.grad.abs().sum().item(),0)

    def test_cost_replay_timeout_alignment(self):
        e=TinyCostEnv();b=CostReplayBuffer(100,e.observation_space,e.action_space,device='cpu',n_envs=2)
        obs=np.zeros((2,3),dtype=np.float32)
        b.add(obs,obs,np.zeros((2,1)),np.array([1.,2.]),np.array([True,True]),[{'cost':.2,'TimeLimit.truncated':True},{'cost':.8}])
        batch=b._get_samples(np.zeros(128,dtype=int))
        for reward,cost,done in zip(batch.rewards,batch.costs,batch.dones):
            self.assertAlmostEqual(cost.item(),.2 if reward.item()==1 else .8,places=6)
            self.assertEqual(done.item(),0 if reward.item()==1 else 1)

    def test_train_save_reload_and_resume(self):
        for cls in (SACLagrangian,WCSAC,WCSACIQN):
            with self.subTest(algorithm=cls.__name__), tempfile.TemporaryDirectory() as d:
                model=cls('MlpPolicy',TinyCostEnv(),device='cpu',buffer_size=128,learning_starts=8,batch_size=8,
                    safety_width=16,n_quantiles=8,policy_kwargs={'net_arch':[16,16]},cost_limit=0,seed=7)
                before={k:v.clone() for k,v in model.safety_critic.state_dict().items()};dual=model.log_multiplier.item()
                model.set_logger(configure(d,[]));model.learn(32)
                self.assertGreater(model.log_multiplier.item(),dual)
                self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.safety_critic.state_dict().items()))
                self.assertTrue(all(np.isfinite(v) for v in model.logger.name_to_value.values()))
                obs=np.zeros(3,dtype=np.float32);action=model.predict(obs,deterministic=True)[0]
                model.save(d+'/model.zip')
                loaded=cls.load(d+'/model.zip',env=TinyCostEnv(),device='cpu')
                np.testing.assert_array_equal(action,loaded.predict(obs,deterministic=True)[0])
                self.assertEqual(model.log_multiplier.item(),loaded.log_multiplier.item())
                for k,v in model.safety_critic.state_dict().items():torch.testing.assert_close(v,loaded.safety_critic.state_dict()[k])
                loaded.set_logger(configure(d,[]));loaded.learn(16)

    def test_config_split_and_independent_eval(self):
        with self.assertRaises(ValueError):clean_request({'seed_start':0},'train')
        with self.assertRaises(ValueError):clean_request({'seed_start':1000},'eval')
        with self.assertRaises(ValueError):clean_request({'seed_start':0,'num_scenarios':1,'episodes':2},'eval')
        with self.assertRaises(ValueError):clean_request({'timesteps':5,'learning_starts':10},'train')
        self.assertEqual(clean_request({'seed_start':1000},'preview')['seed_start'],1000)
        self.assertAlmostEqual(empirical_cvar([1,2,3,4],.375),(4+.5*3)/1.5)


if __name__=='__main__':unittest.main()
