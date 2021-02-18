import random
import numpy as np
import math
import sys

class LinearApproxAttempt:
    
    q = {}
    #returns = {}
    episode = []
    epsilon = 0.1
    discounting = 0

    n = {}


    # If we haven't encountered this state before we set it up with an arbitrary value
    # This is a parameter that can be tuned
    initial_q_val = 2

    # linear function approximation values
    theta_flap = np.array([0.0, 0.0, 0.0, 0.0])
    theta_noop = np.array([0.0, 0.0, 0.0, 0.0])

    def __init__(self, _learning_rate=1, _discounting=1, _epsilon=0.1, _state_action_pair="default"):
        self.learning_rate = _learning_rate
        self.discounting = _discounting
        self.epsilon = _epsilon
        self.state_action_pair = _state_action_pair
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

        if self.state_action_pair == "default":

            y_chunks = 512/15
            mid_pipes = (state["next_pipe_top_y"]+50)//y_chunks
            bird_chunk = state["player_y"]//y_chunks
            x_chunks = 288/15 # 288 - bird_pos
            vel_chunks = 18/15
            return (
                    state["player_vel"]//vel_chunks,
                    #state["player_y"]//y_chunks,
                    state["next_pipe_dist_to_player"]//x_chunks,
                    #mid_pipes,
                    mid_pipes-bird_chunk,
            )

        if self.state_action_pair == "linear_approximation":

            return (
                    state["player_vel"],
                    state["player_y"],
                    state["next_pipe_dist_to_player"],
                    state["next_pipe_top_y"],
                    #mid_pipes-bird_chunk,
            )
    
    def get_phi(self, s):
        return np.array(s)

    def get_sa_value(self, s, a):

        if self.state_action_pair == "default":

            if (s, a) not in self.q:
                # We give actions that are in the direction of the middle of the pipes an increased value
                increased_init = self.initial_q_val
                if s[2] < 0 and a == 0:
                    increased_init += 1
                elif s[2] > 0 and  a==1:
                    increased_init += 1
                self.q[(s, a)] = increased_init

            return self.q[(s, a)]

        if self.state_action_pair == "linear_approximation":

            for i in self.theta_flap:
                if math.isnan(i):
                    print("WE GOT A NAN in flap")
                    sys.exit(0)
                    break

            for i in self.theta_noop:
                if math.isnan(i):
                    print("WE GOT A NAN in noop")
                    sys.exit(0)
                    break

            if a == 0:
                return self.theta_flap.dot(self.get_phi(s))
            else:
                return self.theta_noop.dot(self.get_phi(s))

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        prev_state = self.discretize_state(s1)
        post_state = self.discretize_state(s2)
        if end:
            if (post_state,0) not in self.q:
                self.q[(post_state,0)] = 0 
            if (post_state,1) not in self.q:
                self.q[(post_state,1)] = 0 

        if self.state_action_pair == "default":
        
            max_post_action_q = self.get_sa_value(post_state, self.policy(s2))
            prev_state_prev_q = self.get_sa_value(prev_state, a)
            self.q[(prev_state,a)] = prev_state_prev_q + self.learning_rate*(r + self.discounting*max_post_action_q - prev_state_prev_q)

        if self.state_action_pair == "linear_approximation":

            phi = self.get_phi(prev_state)
            s2_q = 0 if end else self.get_sa_value(post_state, a)
            s1_q = self.get_sa_value(prev_state, a)
            working_theta = None

            # flap
            if a == 0:
                #print("s2", self.get_phi(post_state))
                #print("s1", self.get_phi(prev_state))
                print("Theta-flap:",self.theta_flap)
                print("Theta-noop:",self.theta_noop)
                print("s2q s1q r", s2_q, s1_q, r)
                print("flap calc ", (self.learning_rate * (r + (self.discounting*s2_q) - s1_q)))
                self.theta_flap += ((self.learning_rate * (r + ((self.discounting*s2_q) - s1_q))) * phi)
            else:
                #print("s2", self.get_phi(post_state))
                #print("s1", self.get_phi(prev_state))
                print("Theta-flap:",self.theta_flap)
                print("Theta-noop:",self.theta_noop)
                print("s2q s1q r", s2_q, s1_q, r)
                print("noop calc ", (self.learning_rate * (r + (self.discounting*s2_q) - s1_q)))
                self.theta_noop += ((self.learning_rate * (r + ((self.discounting*s2_q) - s1_q))) * phi)

        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        discrete_state = self.discretize_state(state)

        if self.state_action_pair == "default":
        
            flap = (-1000,0) if (discrete_state,0) not in self.q else (self.q[(discrete_state,0)],0)
            noop = (-1000,1) if (discrete_state,1) not in self.q else (self.q[(discrete_state,1)],1)
            chance = random.random()
            return max([flap,noop], key=lambda x:x[0])[1] if chance > self.epsilon else random.randint(0,1)

        if self.state_action_pair == "linear_approximation":

            chance = random.random()
            if chance > self.epsilon:
                return random.randint(0,1)
            flap_value = self.get_sa_value(discrete_state, 0)
            noop_value = self.get_sa_value(discrete_state, 1)

            if flap_value == noop_value:
                return random.randint(0, 1)
            elif flap_value > noop_value:
                return 0
            else:
                return 1

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        discrete_state = self.discretize_state(state)

        if self.state_action_pair == "default":

            flap = (-1000,0) if (discrete_state,0) not in self.q else (self.q[(discrete_state,0)],0)
            noop = (-1000,1) if (discrete_state,1) not in self.q else (self.q[(discrete_state,1)],1)
            return max([flap,noop], key=lambda x:x[0])[1] if flap[0] != noop[0] else random.randint(0,1)

        if self.state_action_pair == "linear_approximation":

            flap_value = self.get_sa_value(discrete_state, 0)
            noop_value = self.get_sa_value(discrete_state, 1)

            if flap_value == noop_value:
                return random.randint(0, 1)
            elif flap_value > noop_value:
                return 0
            else:
                return 1