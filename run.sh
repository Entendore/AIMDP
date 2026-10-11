#!/usr/bin/env python3
"""
Chatbot Project Setup Script
Generates all necessary files for the Markov Chain and MDP chatbot project.
"""

import os
import sys

def create_file(filename, content):
    """Helper to create a file with content."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created {filename}")

def setup_project():
    """Generate all project files."""
    
    print("="*60)
    print("Setting up Chatbot Project")
    print("="*60)
    
    # 1. Main chatbot code
    chatbot_code = '''import re
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# ==========================================
# 1. Dataset Loading
# ==========================================
def load_corpus(filename="dataset.txt"):
    """Load corpus from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

class Tokenizer:
    @staticmethod
    def encode(text, order):
        text = text.lower()
        tokens = re.findall(r"\\b[\\w\']+\\b|[.,!?;]", text)
        return ["<START>"] * order + tokens + ["<END>"]

    @staticmethod
    def decode(tokens):
        words = [t for t in tokens if t not in ("<START>", "<END>")]
        text = " ".join(words)
        text = re.sub(r"\\s+([.,!?;])", r"\\1", text)
        if text:
            text = text[0].upper() + text[1:]
        return text

# ==========================================
# 2. Markov Chain Bot
# ==========================================
class MarkovBot:
    def __init__(self, corpus, order=2, rep_penalty=0.1):
        self.order = order
        self.rep_penalty = rep_penalty
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)
        self._train(corpus)

    def _train(self, corpus):
        for line in corpus:
            tokens = Tokenizer.encode(line, self.order)
            for i in range(len(tokens) - 1):
                state = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.transitions[state][next_token] += 1
                self.unigrams[next_token] += 1

    def _find_best_seed(self, prompt_tokens):
        """Finds the longest matching suffix of the prompt in the training data."""
        for i in range(len(prompt_tokens), 0, -1):
            state = tuple(prompt_tokens[-i:])
            if state in self.transitions:
                return state
        return tuple(["<START>"] * self.order)

    def _sample(self, counts, temperature=1.0):
        words = list(counts.keys())
        weights = [counts[w] ** (1.0 / temperature) for w in words]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(words, weights=probs, k=1)[0]

    def respond(self, prompt="", temperature=1.0, max_len=30):
        tokens = Tokenizer.encode(prompt, 0)
        state = self._find_best_seed(tokens)
        output = list(state)

        for _ in range(max_len):
            next_token = None
            for i in range(len(state), 0, -1):
                sub_state = state[-i:]
                if sub_state in self.transitions:
                    counts = dict(self.transitions[sub_state])
                    for word in output[-3:]:
                        if word in counts:
                            counts[word] *= self.rep_penalty
                    next_token = self._sample(counts, temperature)
                    break
            
            if next_token is None:
                next_token = self._sample(self.unigrams, temperature)

            if next_token == "<END>":
                break
            output.append(next_token)
            state = tuple(output[-self.order:])

        return Tokenizer.decode(output)

# ==========================================
# 3. MDP Bot
# ==========================================
class MDPBot:
    def __init__(self, corpus, order=2, gamma=0.95, step_cost=0.2, theta=1e-4):
        self.order = order
        self.gamma = gamma
        self.step_cost = step_cost
        self.transitions = defaultdict(lambda: defaultdict(float))
        self._train(corpus)
        self.policy, self.values = self._asynchronous_value_iteration(theta)

    def _task_reward(self, word):
        positive = {"good", "great", "happy", "love", "wonderful", "nice", 
                    "fun", "joyful", "best", "excellent", "beautiful", "perfect"}
        return 2.0 if word in positive else 0.0

    def _train(self, corpus):
        for line in corpus:
            tokens = Tokenizer.encode(line, self.order)
            for i in range(len(tokens) - 1):
                state = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.transitions[state][next_token] += 1
        
        for s in self.transitions:
            total = sum(self.transitions[s].values())
            for w in self.transitions[s]:
                self.transitions[s][w] /= total

    def _asynchronous_value_iteration(self, theta):
        states = list(self.transitions.keys())
        V = defaultdict(float)
        policy = {}
        
        while True:
            max_delta = 0.0
            random.shuffle(states)
            
            for s in states:
                v_old = V[s]
                best_val = -float(\'inf\')
                best_a = None
                
                for a in self.transitions[s]:
                    fluency_r = math.log(self.transitions[s][a])
                    task_r = self._task_reward(a)
                    
                    if a == "<END>":
                        v_next = 0.0
                        task_r += 3.0
                        cost = 0.0
                    else:
                        s_next = s[1:] + (a,)
                        v_next = V.get(s_next, 0.0)
                        cost = self.step_cost
                        
                    val = fluency_r + task_r - cost + self.gamma * v_next
                    
                    if val > best_val:
                        best_val = val
                        best_a = a
                        
                V[s] = best_val
                policy[s] = best_a
                max_delta = max(max_delta, abs(v_old - V[s]))
                
            if max_delta < theta:
                break
        return policy, V

    def respond(self, prompt="", max_len=30):
        tokens = Tokenizer.encode(prompt, 0)
        state = tuple(tokens[-self.order:])
        output = list(state)

        for _ in range(max_len):
            s = tuple(output[-self.order:])
            
            if s in self.policy:
                a = self.policy[s]
            else:
                found = False
                for i in range(self.order - 1, 0, -1):
                    sub_s = s[-i:]
                    if sub_s in self.transitions:
                        a = max(self.transitions[sub_s], key=self.transitions[sub_s].get)
                        found = True
                        break
                if not found:
                    break
                    
            if a == "<END>":
                break
            output.append(a)
            
        return Tokenizer.decode(output)

# ==========================================
# 4. Interactive Arena
# ==========================================
def main():
    random.seed(42)
    print("Loading corpus...")
    corpus = load_corpus()
    print("Training models...")
    markov = MarkovBot(corpus, order=2, rep_penalty=0.1)
    mdp = MDPBot(corpus, order=2, gamma=0.95, step_cost=0.2)
    print("Ready!\\n")

    print("="*60)
    print(" Welcome to the Advanced Chatbot Arena")
    print("="*60)
    print(" Commands:")
    print("   /m <prompt>  : Markov (with repetition penalty)")
    print("   /d <prompt>  : MDP (optimized for positivity & brevity)")
    print("   /b <prompt>  : Compare Both side-by-side")
    print("   /temp <val>  : Set Markov temperature (default 1.0)")
    print("   /quit        : Exit")
    print("="*60)

    temp = 1.0
    while True:
        try:
            user_input = input("\\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input or user_input.lower() == "/quit":
            break
            
        if user_input.startswith("/temp "):
            try:
                temp = float(user_input.split()[1])
                print(f"  -> Markov temperature set to {temp}")
            except ValueError:
                print("  -> Invalid number.")
            continue
            
        prompt = ""
        cmd = "/b"
        if user_input.startswith("/m "): cmd, prompt = "/m", user_input[3:]
        elif user_input.startswith("/d "): cmd, prompt = "/d", user_input[3:]
        elif user_input.startswith("/b "): cmd, prompt = "/b", user_input[3:]
        else: prompt = user_input

        print("-" * 60)
        if cmd in ("/m", "/b"):
            print(f"[Markov] {markov.respond(prompt, temperature=temp)}")
        if cmd in ("/d", "/b"):
            print(f"[MDP]    {mdp.respond(prompt)}")
        print("-" * 60)

if __name__ == "__main__":
    main()
'''
    
    # 2. Dataset
    dataset = """Hello there, how are you doing today?
I am doing great, thank you for asking!
What is your favorite thing to do on the weekend?
I love reading science fiction books and going for long hikes.
That sounds wonderful. Do you have a favorite author?
Yes, I really enjoy the works of Isaac Asimov and Arthur C. Clarke.
Classic sci-fi is the best. What about movies?
I am a big fan of Blade Runner and 2001 A Space Odyssey.
Those are masterpieces. Do you prefer dogs or cats?
I am definitely a dog person. They are so loyal and energetic.
I agree. My golden retriever loves to play fetch in the park.
That is so cute. How long have you had him?
We adopted him three years ago. He was a rescue.
Rescuing animals is a noble thing to do. What do you do for work?
I am a software engineer. I build machine learning models.
That is fascinating. Is artificial intelligence going to take over the world?
Haha, no. AI is just a tool to help humans solve complex problems.
That makes sense. What is the most challenging part of your job?
Debugging code and cleaning up messy data take up most of my time.
I can imagine. Data is never as clean as you want it to be.
Exactly. But when the model finally works, it is very rewarding.
I bet. Do you like to cook when you are not working?
I try to. I am experimenting with Italian cuisine right now.
Oh, I love pasta. What is your favorite dish to make?
I am perfecting my homemade carbonara recipe.
Carbonara is delicious. Do you use guanciale or pancetta?
Traditionalists say guanciale, but pancetta is easier to find.
I agree. What are your plans for the upcoming holiday?
I am planning a trip to the mountains to go skiing.
Skiing is a great workout. Which mountain are you visiting?
We are going to Aspen. The slopes there are beautiful.
Have a wonderful trip! Stay safe on the slopes.
Thank you! I will. Have a great rest of your week!
You too. Goodbye for now!
Goodbye! Talk to you soon.
The weather is supposed to be sunny and bright tomorrow.
I hope so. I want to go for a walk in the park.
Walking in the park is a great way to relax and clear your mind.
It really is. Nature has a way of putting things in perspective.
I completely agree. Do you listen to podcasts while you walk?
Sometimes. I like listening to history and science podcasts.
Those are very educational. What is your favorite one?
I really enjoy Radiolab and Hardcore History.
Hardcore History is incredibly detailed. It takes commitment to listen!
Yes, some episodes are over four hours long. But they are worth it.
I might have to check it out. Thanks for the recommendation.
You are very welcome. Let me know what you think of it.
I will. Well, it was great chatting with you today.
It was wonderful talking to you as well. Have a fantastic day!
You too! Goodbye!"""
    
    # 3. README
    readme = """# Markov Chain vs MDP Chatbot

An advanced chatbot comparison project demonstrating two different approaches to text generation:
1. **Markov Chain Bot**: Probabilistic n-gram model with repetition penalty and temperature control
2. **MDP Bot**: Reinforcement learning approach using Value Iteration with reward shaping

## Features

### Markov Chain Bot
- N-gram backoff for robust generation
- Repetition penalty to avoid loops
- Temperature control for creativity adjustment
- Fuzzy context seeding from user prompts

### MDP Bot
- Asynchronous Value Iteration (Gauss-Seidel)
- Reward shaping for positivity and brevity
- Step cost to encourage concise responses
- Terminal rewards for proper sentence endings

## Installation

No external dependencies required! Uses only Python standard library.

```bash
# Just run the setup script
python setup_project.py

# Then run the chatbot
python chatbot.py