#!/usr/bin/env python
# coding: utf-8

# #### Notebook for running React experiments

# In[1]:


import sys, os
sys.path.append('..')
root  = '../root/'


# In[4]:


import os
os.environ['OPENAI_API_KEY']='sk'


# In[5]:


import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy


# #### Load the HotpotQA Sample

# In[6]:


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)


# #### Define the Reflexion Strategy

# In[7]:


print(ReflexionStrategy.__doc__)


# In[8]:


strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION


# #### Initialize a React Agent for each question

# In[9]:


agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for _, row in hotpot.iterrows()]


# #### Run `n` trials

# In[10]:


n = 5
trial = 0
log = ''


# In[12]:


for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy = strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')


# #### Save the result log

# In[ ]:


with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))

