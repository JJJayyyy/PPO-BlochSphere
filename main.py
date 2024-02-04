import argparse
import gym
import numpy as np
import csv

import torch
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

from models import Policy, Value, ActorCritic
from replay_memory import Memory
from running_state import ZFilter

from bloch_sys_env import Continuous_BlochSphere



torch.set_default_tensor_type('torch.DoubleTensor')
PI = torch.DoubleTensor([3.1415926])

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="bloch", metavar='G',
                    choices=["MountainCarContinuous-v0", "bloch"],
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='batch size (default: 5000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--result_file', type=str, default="blochsphere_2",
                    help='file to store the result')
args = parser.parse_args()
print(f'args : {args}')

if args.env_name == "MountainCarContinuous-v0":
    env = gym.make(args.env_name)
elif args.env_name == "bloch":
    env = Continuous_BlochSphere()
else:
    raise Exception(f"env {args.env_name} is not acceptable")


num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# env.seed(args.seed)
# obs, info = env.reset(seed=0)
torch.manual_seed(args.seed)

results_file = open(f'{args.result_file}.csv', 'w', newline='')
results_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
results_writer.writerow(['Average reward'])

if args.use_joint_pol_val:
    print("ActorCritic")
    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=0.001)
else:
    print("Policy")
    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    opt_value = optim.Adam(value_net.parameters(), lr=0.001)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_actor_critic(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std, v = ac_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params_actor_critic(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    action_means, action_log_stds, action_stds, values = ac_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    action_var = Variable(actions)
    # compute probs from actions above
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old, values_old = ac_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    ac_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_ac.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()

    vf_loss1 = (values - targets).pow(2.)
    vpredclipped = values_old + torch.clamp(values - values_old, -args.clip_epsilon, args.clip_epsilon)
    vf_loss2 = (vpredclipped - targets).pow(2.)
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    total_loss = policy_surr + vf_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm(ac_net.parameters(), 40)
    opt_ac.step()


def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    opt_value.zero_grad()
    value_loss = (values - targets).pow(2.).mean()
    value_loss.backward()
    opt_value.step()

    action_var = Variable(actions)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old = policy_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    policy_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_policy.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    opt_policy.step()

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []
avg_rewards = []
step_avg_rewards = []

# for i_episode in tqdm(range(200), desc="Training Progress"):
for i_episode in range(200):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    distance_list = []
    while num_steps < args.batch_size:
        state, _ = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000):
            if args.use_joint_pol_val:
                action = select_action_actor_critic(state)
            else:
                action = select_action(state)

            action = action.data[0].numpy() 
            next_state, reward, term, trunc, info = env.step(action)
            # print(next_state, reward, action)
            distance_list.append(info)
            done = term or trunc
            reward_sum += reward
            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    if i_episode % 20 == 0:
        plt.figure()
        plt.plot(distance_list)
        plt.title('Angle distance changes per Step')
        plt.xlabel('step')
        plt.ylabel('distance')
        plt.savefig(f"dist_{i_episode}.png")

    reward_batch /= num_episodes
    avg_rewards.append(reward_batch)
    step_avg_rewards.append(reward_batch/num_steps)

    batch = memory.sample()
    if args.use_joint_pol_val:
        update_params_actor_critic(batch)
    else:
        update_params(batch)

    if reward_batch > 50:
       args.render = True

    if i_episode % args.log_interval == 0:
        print('Episode {:<5}Last reward: {:<10.2f}Average reward: {:<10.2f}steps: {}'.format(i_episode, reward_sum, reward_batch, num_steps))
        results_writer.writerow([reward_batch])

plt.figure()
plt.plot(avg_rewards)
plt.title('Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig("avg_rewards.png")

plt.figure()
plt.plot(step_avg_rewards)
plt.title('step Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average step Reward')
plt.savefig("avg_step_rewards.png")