from copy import deepcopy
from faronona.faronona_action import FarononaAction, FarononaActionType
from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from mcts import Node, Search


class AI(FarononaPlayer):

    name = "Blinders 2"

    # MCTS Parameters
    EPSILON = .1
    MAX_ROLLOUT_DEPTH = 10
    N_ITERATIONS = 12

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value
        
        # Variables
        self.n_plays = 0
        
        

    def play(self, state, remain_time):
        # manage simulation time
        self.n_plays += 1

        if self.n_plays > 15 or (self.n_plays > 5 and self.winner(state) != self.position):
            # MCTS Strategy
            return self.play_mcts(state, remain_time)
        else:
            # Random Play ?
            return self.play_random(state, remain_time)


    def play_mcts(self, state, remain_time):
        root = Node(self.position, state)
        search_tree = Search(root, max_rollout_depth=self.MAX_ROLLOUT_DEPTH)
        action = search_tree.best_action(n_iterations=self.N_ITERATIONS)
        return action


    def play_random(self, state, remain_time):
        #Retrieve a random action
        action = FarononaRules.random_play(state, self.position)
        #Extract departure and arrival of the piece
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

    def winner(self, state):
        score = state.score
        return max(score, key=score.get)