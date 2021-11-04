"""
Deep learning model that accept 
X = [x1, x2]
and fit and predict
y which a the Q values for a given state (x1, x2)

Note:
x1.shape = (9, 5)     board state
x2.shape = (1, )      player position
y.shape = (8, )       QValues for 8 actions

Created 11/02/2021 by Henri A.
"""

from typing import List, Tuple
from core import Color
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from faronona.faronona_action import FarononaAction


LEARNING_RATE = 1e-3
BOARD_STATE_SHAPE = (9, 5) # and position shape (1, )
ADDITIONAL_STATE_SHAPE = (1, )
INTERMEDIATE_OUTPUT_SHAPE = (12, )
ACTION_SHAPE = (8, )

#######################
#
#    Private methods
#
#######################

def create_model():
    """Create tf model that predict q_values from current board state and position"""
    model_board = _create_board_model()
    model_additional = _create_additional_model()
    combined_input= tf.keras.layers.concatenate([model_board.output, model_additional.output])

    # Create final network that output 8 values
    x = tf.keras.layers.Dense(INTERMEDIATE_OUTPUT_SHAPE, activation="relu")(combined_input)
    x = tf.keras.layers.Dense(ACTION_SHAPE, activation="linear")(x)
    model = Model(inputs=[model_board.input, model_additional.input], outputs=x)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model


#######################
#
#    Transformations
#
#######################
    
def transform_board(board: np.ndarray) -> np.ndarray:
    """Transform board into array[array] of integers"""
    board = np.where(board=='white', Color.white, board)
    board = np.where(board=='green', Color.green, board)
    board = np.where(board=='empty', Color.empty, board)
    return board


def transform_action(action: FarononaAction) -> int:
    """Transform vector movement into integers representation"""
    action_dict = action.get_action_as_dict()
    at = action_dict['action']['at']
    to = action_dict['action']['to']
    return ModelAction.get_action(at, to)



#######################
#
#    Private methods
#
#######################

def _create_board_model():
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(24, input_shape=BOARD_STATE_SHAPE, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(INTERMEDIATE_OUTPUT_SHAPE, activation='linear', kernel_initializer=init))
    return model


def _create_additional_model():
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(24, input_shape=ADDITIONAL_STATE_SHAPE, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(INTERMEDIATE_OUTPUT_SHAPE, activation='linear', kernel_initializer=init))
    return model



################################
#
#    Model Actions Utilties
#
################################

class ModelAction:
    """Enumeration of differents types of actions possible."""
    TOP             = 1
    TOP_RIGHT       = 2
    RIGHT           = 3
    BOTTOM_RIGHT    = 4
    BOTTOM          = 5
    BOTTOM_LEFT     = 6
    LEFT            = 7
    TOP_LEFT        = 8

    actions = {
        ( 0, 1): TOP,
        ( 1, 1): TOP_RIGHT,
        ( 1, 0): RIGHT,
        ( 1,-1): BOTTOM_RIGHT,
        ( 0,-1): BOTTOM,
        (-1,-1): BOTTOM_LEFT,
        (-1, 0): LEFT,
        (-1, 1): TOP_LEFT,
    }

    @classmethod
    def get_action(cls, at: Tuple[int, int], to: Tuple[int, int]) -> int:
        """Provided movement position and return integers representation of the action performed"""
        vector = tuple(np.subtract(to, at))
        return cls.actions[vector]

    @classmethod
    def get_vector(cls, action) -> Tuple[int, int]:
        """Given an integer representation of an action, return the corresponding vector."""
        return list(cls.actions.keys())[list(cls.actions.values()).index(action)]

    @classmethod
    def get_destination(cls, at: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Return the action 'to' coordinates, given action integer representation and initial position 'at'"""
        vector = ModelAction.get_vector(action)
        return tuple([sum(x) for x in zip(at, vector)])

    @classmethod
    def get_action(cls, model_action: int) -> FarononaAction:
        """Return """