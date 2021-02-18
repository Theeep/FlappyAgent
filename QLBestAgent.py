import random

class QLBestAgent:
    
    q = {}
    episode = []
    epsilon = 0.1
    learning_rate = 0.1
    discounting = 0
    n = 0

    # If we haven't encountered this state before we set it up with an arbitrary value
    # This is a parameter that can be tuned
    initial_q_val = 0

    def __init__(self, _learning_rate=1, _discounting=1, _epsilon=0.1):
        self.learning_rate = _learning_rate
        self.discounting = _discounting
        self.epsilon = _epsilon
        return
    
    def reward_values(self):
        """ returns the reward values used for training
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def discretize_state(self, state):
        """
            This function is to simplify the state,
            removing redundant variables in the environment,
            and to discretize some variables that are a hassle

            for instance next_pipe_top_y is equal to next_pipe_bottom_y + 100
            so one is redundant, same holds for next_next_pipe_top_y
        """
        dist_from_pipes = state["next_pipe_dist_to_player"]
        discretized_dist_from_pipes = "far" if dist_from_pipes > 110 else "close"

        y_top = state["next_pipe_top_y"]
        y_bottom = state["next_pipe_bottom_y"]
        y_bird = state["player_y"]
        discretized_bird_in_hallway = 0
        if y_bird >= y_top and y_bird <= y_bottom:
            discretized_bird_in_hallway = 0
        elif y_bird < y_top:
            discretized_bird_in_hallway = "above"
        else:
            discretized_bird_in_hallway = "below"
        
        vel_chunks = 18/8
        discretized_bird_velocity = state["player_vel"]//vel_chunks

        if discretized_bird_in_hallway == 0 and discretized_dist_from_pipes == "close":
            y_bird_into_hallway = y_bird - y_top
            hallway_size = y_bottom - y_top
            hallway_size_chunks = hallway_size / 10
            discretized_bird_in_hallway = y_bird_into_hallway//hallway_size_chunks


        return (
            discretized_dist_from_pipes,
            discretized_bird_in_hallway,
            discretized_bird_velocity,
        )

    def get_sa_value(self, s, a):

        if (s, a) not in self.q:
            modified_initial = self.initial_q_val
            if s[0] == "far" and (s[1] == "below" or s[1] == "above"):
                modified_initial -= 1
            elif s[1] == "below" and a == 1:
                modified_initial -= 2
            elif s[1] == "above" and a == 0:
                modified_initial -= 2
            self.q[(s, a)] = modified_initial
        return self.q[(s, a)]


    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # increasing N by 1 in order to decrease learning rate after 100000 frames
        self.n += 1

        if self.n == 100000:
            self.n = 0
            self.learning_rate = self.learning_rate/2

        prev_state = self.discretize_state(s1)
        post_state = self.discretize_state(s2)
        if end:
            max_post_action_q = 0
        else:
            max_post_action_q = self.get_sa_value(post_state, self.policy(s2))
        prev_state_prev_q = self.get_sa_value(prev_state, a)
        self.q[(prev_state,a)] = prev_state_prev_q + self.learning_rate*(r + self.discounting*max_post_action_q - prev_state_prev_q)
        
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        discrete_state = self.discretize_state(state)

        # return random if epsilon check
        chance = random.random()
        if chance <= self.epsilon:
            return random.randint(0,1)
        
        flap = (self.initial_q_val,0) if (discrete_state,0) not in self.q else (self.q[(discrete_state,0)],0)
        noop = (self.initial_q_val,1) if (discrete_state,1) not in self.q else (self.q[(discrete_state,1)],1)
        return max([flap,noop], key=lambda x:x[0])[1]

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        discrete_state = self.discretize_state(state)

        flap = (self.initial_q_val,0) if (discrete_state,0) not in self.q else (self.q[(discrete_state,0)],0)
        noop = (self.initial_q_val,1) if (discrete_state,1) not in self.q else (self.q[(discrete_state,1)],1)
        return max([flap,noop], key=lambda x:x[0])[1] if flap[0] != noop[0] else random.randint(0,1)
