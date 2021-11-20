from faronona.faronona_player import FarononaPlayer
from mcts import Node, Search


class AI(FarononaPlayer):

    name = "MCTS Player"

    # MCTS Parameters
    EPSILON = 0.1
    MAX_ROLLOUT_DEPTH = float('inf')
    N_ITERATIONS = 15

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value
        

    def play(self, state, remain_time):
        # TODO: Manage remaining time
        root = Node(self.position, state)
        search_tree = Search(root, max_rollout_depth=self.MAX_ROLLOUT_DEPTH)
        action = search_tree.best_action(n_iterations=self.N_ITERATIONS, epsilon=self.EPSILON)
        return action

    
