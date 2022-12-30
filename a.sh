#rd
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Ant-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Reacher-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Swimmer-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Hopper-v2 --seed 1 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Ant-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Reacher-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Swimmer-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Hopper-v2 --seed 2 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Ant-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Reacher-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Swimmer-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Hopper-v2 --seed 3 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Ant-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Reacher-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Swimmer-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Hopper-v2 --seed 4 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Ant-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Reacher-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Swimmer-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=0 nohup poetry run python cleanrl/ppo_rd.py --total-timesteps 4000000  --env-id Hopper-v2 --seed 5 --track --wandb-project-name test30 &

#baseline
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 1 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 2 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 3 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 4 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=1 nohup poetry run python cleanrl/ppo_baseline.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 5 --track --wandb-project-name test30 &

#ES

CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 1 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 2 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 3 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 4 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=2 nohup poetry run python cleanrl/ppo_policy_es.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 5 --track --wandb-project-name test30 &

#old-approx

CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 1 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 1 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 2 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 2 --track --wandb-project-name test30 &


CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 3 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 3 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 4 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 4 --track --wandb-project-name test30 &

CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Ant-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id HalfCheetah-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Reacher-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Swimmer-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id InvertedDoublePendulum-v2 --seed 5 --track --wandb-project-name test30 &
CUDA_VISIBLE_DEVICES=3 nohup poetry run python cleanrl/ppo_old.py --target-kl 0.03 --total-timesteps 4000000  --env-id Hopper-v2 --seed 5 --track --wandb-project-name test30 &

