from torch import optim

EMPTY = 0
X = 1
O = -1

def new_state():
    return [EMPTY]*9  # liste de 9 zéros

def available_actions(s):
    return [i for i,v in enumerate(s) if v == EMPTY]

def apply_action(s, a, player):
    assert s[a] == EMPTY, "Action illégale"
    s2 = s.copy()
    s2[a] = player
    return s2

def check_winner(s):
    lines = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6),
    ]
    for i,j,k in lines:
        if s[i] != EMPTY and s[i] == s[j] == s[k]:
            return s[i]        # +1 (X) ou -1 (O)
    if EMPTY not in s:
        return 0              # nul
    return None               # partie non terminée

def pretty_print(s):
    m = {EMPTY:'.', X:'X', O:'O'}
    for r in range(0,9,3):
        print(m[s[r]], m[s[r+1]], m[s[r+2]])
    print()


# plateau vide
s = new_state()
assert len(s) == 9 and all(v == 0 for v in s)
assert available_actions(s) == list(range(9))
assert check_winner(s) is None

# X joue au centre
s1 = apply_action(s, 4, X)
assert s1[4] == X and s[4] == EMPTY  # immutabilité OK

# victoire horizontale
s2 = [X, X, X, 0,0,0, 0,0,0]
assert check_winner(s2) == X

# nul
s3 = [X,O,X, X,O,O, O,X,X]
assert check_winner(s3) == 0

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

def encode_state(s):
    return torch.tensor(s, dtype=torch.float32, device=device)


import torch.nn as nn


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.layers(x)

model = QNet().to(device)

s = new_state()           # plateau vide
x = encode_state(s)       # tensor [9]
q_values = model(x)       # passe dans réseau

print("State  :", x)
print("Q-values:", q_values)
print("Shape Q:", q_values.shape)

import random
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, player, model, eps=0.2):
        self.player = player  # X = +1 ou O = -1
        self.model = model
        self.eps = eps        # exploration

    def select_action(self, state):
        actions = available_actions(state)

        # Exploration
        if random.random() < self.eps:
            return random.choice(actions)

        # Exploitation : utiliser le réseau
        x = encode_state(state).unsqueeze(0)  # shape [1,9]
        with torch.no_grad():
            q_values = self.model(x)[0]       # shape [9]

        # On met une énorme pénalité sur les coups illégaux
        illegal = [i for i in range(9) if i not in actions]
        q_values[illegal] = -9999.0

        return q_values.argmax().item()

    def train_step(self, s, a, r, s_next, done, gamma=0.99):
        # Encode
        x = encode_state(s).unsqueeze(0)  # [1,9]
        x_next = encode_state(s_next).unsqueeze(0)  # [1,9]

        # Q actuel
        q_values = self.model(x)[0]  # [9]
        q_value = q_values[a]  # scalaire

        # Q target
        with torch.no_grad():
            if done:
                target = torch.tensor(r, device=device)
            else:
                next_q = self.model(x_next)[0].max()
                target = torch.tensor(r, device=device) + gamma * next_q

        # Loss = (Q - target)^2
        loss = F.mse_loss(q_value, target)

        # Backprop
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


model = QNet().to(device)
agent = DQNAgent(player=X, model=model, eps=0.0)  # pas d’exploration

s = new_state()
action = agent.select_action(s)
print("Action choisie sur plateau vide :", action)
assert action in range(9)


model = QNet().to(device)
agent = DQNAgent(player=X, model=model, eps=0.3)

s = new_state()
a = 0  # l'IA joue en haut gauche
s2 = apply_action(s, a, X)

# Simulation : on décide que c'est une victoire de X ➜ r = +1
loss1 = agent.train_step(s, a, +1.0, s2, done=True)
print("Loss :", loss1)

# Après update ➜ Q-value doit monter
with torch.no_grad():
    q_after = agent.model(encode_state(s).unsqueeze(0))[0][a]
print("Q-value mise à jour :", q_after.item())

print("Étape 4 OK si la Q-value est > 0")


def train_agent(agent, episodes=20000):
    for ep in range(episodes):
        s = new_state()
        p = agent.player
        done = False

        while not done:
            # IA joue
            a = agent.select_action(s)
            s_next = apply_action(s, a, p)
            outcome = check_winner(s_next)

            if outcome is not None:
                # récompense
                r = +1.0 if outcome == p else (0.5 if outcome == 0 else -1.0)
                agent.train_step(s, a, r, s_next, done=True)
                done = True
                break

            # adversaire random
            a2 = random.choice(available_actions(s_next))
            s2 = apply_action(s_next, a2, -p)
            outcome = check_winner(s2)

            if outcome is not None:
                r = +1.0 if outcome == p else (0.5 if outcome == 0 else -1.0)
                agent.train_step(s, a, r, s2, done=True)
                done = True
            else:
                # pas de récompense immédiate
                agent.train_step(s, a, 0.0, s2, done=False)

            s = s2

        # décroissance epsilon
        agent.eps *= 0.995

        if ep % 500 == 0:
            print(f"Episode {ep}/{episodes}, eps={agent.eps:.3f}")

model = QNet().to(device)
agent = DQNAgent(player=X, model=model, eps=0.3)

train_agent(agent, episodes=3000)

print("Entraînement terminé ✅")



def play_vs_agent(agent):
    s = new_state()
    player = X  # l'IA commence

    print("\nTu es O. L'IA est X.\n")
    pretty_print(s)

    while True:
        if player == X:
            # IA choisit un coup
            a = agent.select_action(s)
            s = apply_action(s, a, X)
            print("L'IA joue :", a)
        else:
            # Humain
            actions = available_actions(s)
            print("Cases disponibles :", actions)
            while True:
                try:
                    a = int(input("Choisis une case (0-8) : "))
                    if a in actions: break
                    print("Coup illégal !")
                except:
                    print("Entre un nombre valide.")
            s = apply_action(s, a, O)

        pretty_print(s)
        outcome = check_winner(s)
        if outcome is not None:
            if outcome == X:
                print("🤖 L'IA a gagné !")
            elif outcome == O:
                print("🎉 Tu as gagné ! Bravo !")
            else:
                print("🤝 Match nul")
            break

        player = -player  # change de joueur


play_vs_agent(agent)