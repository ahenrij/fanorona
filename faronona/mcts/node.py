"""MCTS Tree Node."""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np
from core.player import Color
from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from faronona.faronona_state import FarononaState
from faronona.faronona_action import FarononaAction, FarononaActionType


class Node:
    def __init__(self, agent: int, state: FarononaState, parent: Tuple = (None, None)) -> None:
        """Constructor of Tree Node.

        Args:
            agent (int): integer position of the IA Agent.
            state (FarononaState): Game state.
            parent (Optional[Tuple[FarononaAction, Node]]): Parent node and action played to reach it.
                                                            Defaults to None.
        """
        self.agent = agent
        self.state = deepcopy(state)
        self.parent: Optional[Tuple[FarononaAction, Node]] = parent
        self.children: List[Node] = []
        self._number_of_visits: int = 0
        self._results: Dict[int, int] = defaultdict(lambda: 0)
        self._untried_actions: List[FarononaAction] = self.untried_actions()

    def untried_actions(self):
        """Return all possible actions in current state."""
        self._untried_actions = self.get_possible_actions(self.state, self.current_player)
        return self._untried_actions

    @property
    def q(self) -> int:
        """Returns the reward from the results."""
        wins = self._results[self.agent]
        loses = self._results[-1 * self.agent]
        return wins - loses

    @property
    def n(self) -> int:
        """Returns number of time this node has been visited."""
        return self._number_of_visits

    def expand(self):
        """Expand the tree by playing an untried action."""
        action = self._untried_actions.pop()
        next_state, _ = self.move(self.state, action, self.current_player)
        child_node = Node(self.agent, next_state, parent=(action, self))
        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        """Is game finished ?"""
        if self.state.get_latest_player() is None:
            return False
        return FarononaRules.is_end_game(self.state)

    def rollout(self) -> int:
        """Simulate entire game randomly from this note state.

        Returns:
            int: Game result. 0 for tie, 1 for victory and -1 for loss.
        """
        current_state = deepcopy(self.state)
        current_player = self.current_player
        
        while not FarononaRules.is_end_game(current_state):
            possible_moves = self.get_possible_actions(current_state, current_player)
            action = self.rollout_policy_v2(current_state, possible_moves, current_player)
            current_state, _ = self.move(current_state, action, current_player)
            current_player = current_state.get_next_player()

        return current_state.score

    def rollout_policy(self, possible_moves: List[FarononaAction]) -> FarononaAction:
        """Rollout move selection policy, currently random."""
        return possible_moves[np.random.randint(0, len(possible_moves))]

    def rollout_policy_v2(self, state: FarononaState, possible_moves: List[FarononaAction], player: int) -> FarononaAction:
        """Rollout move selection policy, pick best action for next move."""
        actions_scores = {}
        for move in possible_moves:
            next_state, _ = self.move(deepcopy(state), move, player)
            actions_scores[move] = next_state.score[player]

        return max(actions_scores, key=actions_scores.get)

    def backpropagate(self, score):
        """Backrpropagation of rollout simulation."""
        self._number_of_visits += 1.
        self._results[-1] += score[-1] # value backed up.
        self._results[1] += score[1]
        _, parent = self.parent
        if parent:
            parent.backpropagate(score)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, epsilon=0.9):
        """Return child with the greater Upper Confidence Bounds.

        Args:
            epsilon (float, optional): The exploration factor. Defaults to 0.9.

        Returns:
            Node: The best child.
        """
        choices_weights = [(c.q / c.n) + epsilon * np.sqrt((2 * np.log(self.n) / c.n)) for c in self.children]
        return self.children[np.argmax(choices_weights)]


    #################
    #               #
    #     UTILS     #
    #               #
    #################

    def move(self, state: FarononaState, action: FarononaAction, player: int) -> FarononaState:
        _state = deepcopy(state)
        _state, done = FarononaRules.act(_state, action, player)
        FarononaRules.moment_player(_state, self.players)
        return _state, done
        
    @property
    def players(self):
        players = {}
        players[-1] = FarononaPlayer("-1", Color(-1))
        players[1] = FarononaPlayer("1", Color(1))
        return players

    @property
    def current_player(self):
        return self.state.get_next_player()

    def get_possible_actions(self, state: FarononaState, player: int) -> List[FarononaAction]:
        possible_actions = []
        actions = FarononaRules.get_player_actions(state, player)
        if actions is None:
            print("Debug")
        for action in actions:
            possible_actions.extend(self.get_win_strategy_actions(state, action, player))

        if len(possible_actions) == 0:
            print("Debug")
        #is_end_game = FarononaRules.is_end_game(self.state)
        # assert not is_end_game and len(possible_actions) > 0, "There should be some possible action."
        return possible_actions

    def get_win_strategy_actions(self, state: FarononaState, action: FarononaAction, player: int) -> List[FarononaAction]:
        """Given an action return the opposition win strategy's action if it is a win.
        None if opposite is not a win."""
        actions = []
        action_dict = action.get_action_as_dict()
        at, to = action_dict['action']['at'], action_dict['action']['to']

        #check if it is a win move for approach
        if (FarononaRules.is_win_approach_move(at, to, state, player) is not None) and len(FarononaRules.is_win_approach_move(at, to, state, player)) != 0:
            actions.append(FarononaAction(action_type=FarononaActionType.MOVE, win_by='APPROACH', at=at, to=to))
        
        #check if it is a win move for remote
        if (FarononaRules.is_win_remote_move(at, to, state, player) is not None)  and len(FarononaRules.is_win_remote_move(at, to, state, player)) != 0:
            actions.append(FarononaAction(action_type=FarononaActionType.MOVE, win_by='REMOTE', at=at, to=to))

        if not actions:
            actions.append(action)

        return actions
