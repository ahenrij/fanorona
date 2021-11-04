from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from faronona.faronona_action import FarononaAction
from faronona.faronona_action import FarononaActionType
from faronona.faronona_state import FarononaState
import copy
import numpy as np
from .model import *



class AI(FarononaPlayer):

    name = "Blinders QL"

    TRAINING = True
    EPSILON = 0.6           # EXPLORATION PROBABILITY
    DISCOUNT_FACTOR = .9    # PERFERENCE FOR LATER REWARD 1 vs NOW 0, also called GAMMA

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value
        self.model = create_model()


    def play(self, state, remain_time):
        """Q Learning Agent."""
        action = None
        if np.random.uniform(0, 1) < self.EPSILON:
            action = self.play_random(state, remain_time)           # Exploratory phase, discover new good moves
        else:
            action = self.play_qtable_action(state, remain_time)    # Exploitation phase, based on QTable

        # Give feedback
        # Q_table[current_state, action] = 
        # (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
        next_state, _ = FarononaRules.act(state, action, self.position)
        reward = self.reward(state, next_state)

        return action


    def play_random(self, state, remain_time):
        #Retrieve a random action
        action = FarononaRules.random_play(state, self.position)
        #Extract departure and arrival of the piece
        return self.best_action_win(action)

    
    def reward(self, state: FarononaState, next_state: FarononaState):
        """Get the reward from a state to the next one."""
        next_score = next_state.get_player_info(self.position)['score']
        old_score = state.get_player_info(self.position)['score']
        return next_score - old_score

    
    def is_end_game_action(self, state: FarononaState, action: FarononaAction)->bool:
        """Return True is next action end the game"""
        action_copy = copy.deepcopy(action)
        state_copy = copy.deepcopy(state)
        next_state, _ = FarononaRules.act(state_copy, action_copy, self.position)
        # TODO: Handle when action lead to a state where opponent action end game
        return FarononaRules.is_end_game(next_state)


    def best_action_win(self, state: FarononaState, action: FarononaAction) -> FarononaAction:
        """Choose the best action between win by approach and by remote."""
        actionDict = action.get_action_as_dict()
        at = actionDict['action']['at']
        to = actionDict['action']['to']
        #check if it is a win move both for approach and remote
        if (FarononaRules.is_win_approach_move(at, to, state, self.position) is not None) and (FarononaRules.is_win_remote_move(at, to, state, self.position) is not None) and len(FarononaRules.is_win_approach_move(at, to, state, self.position)) != 0 and len(FarononaRules.is_win_remote_move(at, to, state, self.position)) != 0:
            # between win approach and win remoate, check which can let me gain the more adverse pieces
            if len(FarononaRules.is_win_approach_move(at, to, state, self.position)) < len(FarononaRules.is_win_remote_move(at, to, state, self.position)):
                action = FarononaAction(action_type=FarononaActionType.MOVE, win_by='REMOTE', at=at, to=to)
            else: 
                action = FarononaAction(action_type=FarononaActionType.MOVE, win_by='APPROACH', at=at, to=to)
        return action

    ######################
    #
    #   Model methods
    #
    ######################

    def train_model(self, boards, positions, Q):
        """Train model"""
        pass

    def predict_action(self, state: FarononaState):
        """Returns best action for current state from the Q-table."""
        board = transform_board(state.get_json_state()['board'])
        pos = self.position
        Q = self.model.predict([board, pos])
        model_action = np.argmax(Q)
        # TODO: Problem is there is not just 8 actions in our problem
        # There is need to take into account which piece to move too
        # This is solve with MCTS.
        return model_action


