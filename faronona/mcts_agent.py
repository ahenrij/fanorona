from copy import deepcopy
from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from mcts import Node, Search


class AI(FarononaPlayer):

    name = "Blinders 1"

    EPSILON = .1
    MAX_ROLLOUT_DEPTH = 5
    N_ITERATIONS = 10

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value

        # MCTS Parameters
        

    def play(self, state, remain_time):
        # manage remaining time
        root = Node(self.position, state)
        search_tree = Search(root, max_rollout_depth=self.MAX_ROLLOUT_DEPTH)
        action = search_tree.best_action(n_iterations=self.N_ITERATIONS)
        return action

    
