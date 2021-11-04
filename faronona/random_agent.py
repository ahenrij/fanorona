from faronona.faronona_player import FarononaPlayer
from faronona.faronona_rules import FarononaRules
from faronona.faronona_action import FarononaAction
from faronona.faronona_action import FarononaActionType
from copy import deepcopy


class AI(FarononaPlayer):

    name = "War of Hearts"

    def __init__(self, color):
        super(AI, self).__init__(self.name, color)
        self.position = color.value

    def play(self, state, remain_time):
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

        if self.is_end_game_action(state, action):
            print("Game is over")
        
        return action

    
    def is_end_game_action(self, state, action: FarononaAction)->bool:
        """Return True is next action end the game"""
        action_copy = deepcopy(action)
        state_copy = deepcopy(state)
        next_state, _ = FarononaRules.act(state_copy, action_copy, self.position)
        # TODO: Handle when action lead to a state where opponent action end game
        return FarononaRules.is_end_game(next_state)