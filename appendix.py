"""
Deep-RL in gym using Keras - appendix

Additional methods to tinker with. Currently includes a training loop that evaluates 
model performance (in terms of reward) during training. The training loop also has a 
"""

def averagePerformanceProgress(model, games = 10, timesteps=timesteps, scenario = 'CartPole-v0',
                              output = 0):
    """
    Mini, efficient testing loop to use while training. Quickly finds the average reward over
    a certain amount of games and prints the output if needed.
    """
    env = gym.make(scenario)
    total_reward = 0.
    for game in range(games):
        observation = env.reset()
        for t in range(timesteps):
            state = np.array(observation, ndmin=2)
            action = np.argmax(model.predict(state)[0])
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    if output == 1:
        print("Average reward over {} games: {}".format(games, total_reward / games))
    return total_reward / games

def train_with_progress(model, epochs, output = 1, set_size = 10, 
                        early_termination_threshold = None, timesteps = timesteps, render = 0):

    """
    Train while reporting progress from averagePerformanceProgress(). Replcaes the training loop from the tutorial.
    Early termination threshold is used to halt training when training error reaches a certain threshold (not covered in tutorial).
    """
    
    reward_hist = []
    
    
    loss_over_set = 0.
    frames_over_set = 0
    for i_episode in range(epochs):
        
        observation = env.reset()
        loss = 0.
        total_reward = 0
        for t in range(timesteps):
            if i_episode % set_size == set_size - 1 and render == 1:
                env.render()
            state_0 = np.array(observation, ndmin=2)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state_0)[0])
                
            observation, reward, done, info = env.step(action)
            state_1 = np.array(observation, ndmin=2)
            total_reward += reward
            
            replayMemory.remember([state_0, action, reward, state_1], done)
            
            
            inputs, targets = replayMemory.get_batch(model, max_batch_size=batch_size)
  
            
            batch_loss = model.train_on_batch(inputs, targets)
            loss += batch_loss
            
            frames_over_set += 1
            loss_over_set += batch_loss
            
            if done:
                reward_hist.append(total_reward)
                break
                
        
        if i_episode % set_size == set_size - 1:
            avgPerformance = averagePerformanceProgress(model, games = 10, timesteps=timesteps, scenario = 'CartPole-v0',
                              output = 0)
            avg_loss_over_set = loss_over_set/frames_over_set
            loss_over_set = 0.
            frames_over_set = 0
            if output > 0:
                print("Episode finished after {} timesteps".format(t+1))
                print("Epoch {:03d}/{:03d} | Average Loss per frame over set: {:.4f} | Current average reward of model: {}".format(
                    i_episode + 1, epochs, avg_loss_over_set, avgPerformance))
            
            
            if early_termination_threshold is not None:
                if avg_loss_over_set < early_termination_threshold:
                    print("Training terminated early after {} epochs with {} average loss over the past set.".format(i_episode + 1, avg_loss_over_set))
                    env.reset()
                    env.close()
                    return
                
            
            
        
    env.reset()
    env.close()