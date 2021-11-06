from copy import deepcopy
from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from mcts import Node, Search


class AI(FarononaPlayer):

    name = "Blinders"

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value

    def play(self, state, remain_time):
        # manage simulation time
        """
        print("MCTS Agent {}".format(self.position))
        print("Got {} actions for player {}".format(
            len(FarononaRules.get_player_actions(deepcopy(state), self.position)),
            self.position
            ))
        """
        root = Node(self.position, state)
        action = Search(root).best_action(n_simulations=10)
        return action

    
