import os
import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims, A_fc1_dims,
                 A_fc2_dims, batch_size, n_agents, agent_name, noise):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.noise = noise
        self.local_critic_loss = []
        self.local_actor_loss = []
        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='actor', agent_label=agent_name)
        self.critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='critic', agent_label=agent_name)

        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='target_actor', agent_label=agent_name)

        self.target_critic = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='target_critic', agent_label=agent_name)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float64).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        #print('check this variable for convergence!!! : ', mu)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=self.number_actions),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def local_learn(self, global_loss, state, action, reward_l, state_, terminal):

        states = state
        states_ = state_
        actions = action
        rewards = reward_l
        done = terminal
        self.global_loss = global_loss

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.local_critic_loss.append(critic_loss.detach().cpu().numpy())

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        self.actor.train()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        print_actor_loss = actor_loss.detach().cpu().numpy()
        print_actor_loss = print_actor_loss.reshape(1,-1)
        print("actor loss(without plus global loss) : ", print_actor_loss[0][:6])
        actor_loss = T.mean(actor_loss) + T.mean(self.global_loss)
        self.local_actor_loss.append(actor_loss.detach().cpu().numpy())
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)