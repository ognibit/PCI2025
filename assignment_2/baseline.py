"""
SETUP

pip install networkx
pip install matplotlib
"""

from dataclasses import dataclass
import math
import time
import random
from enum import Enum
import argparse
import os

from vi import Agent, Config, Simulation, HeadlessSimulation
from vi.util import count

import networkx as nx
import polars as pl
import pygame
from pygame.math import Vector2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Aggregation')

parser.add_argument('--seed', type=int, default=0,
                    help='Random Seed value, set to 0 for random seed (default: 0)')
parser.add_argument('--duration', type=int, default=120,
                    help='Simulation duration value in seconds (default: 120)')
parser.add_argument('--frame_limit', type=int, default=60,
                    help='fps limit, 0 uncaps (Fastest) (default: 60)')
parser.add_argument('--radius', type=int, default=30,
                    help='Radius value (default: 30)')
parser.add_argument('--repr_time_prey', type=int, default=500,
                    help='Prey reproduction timer value in ticks 60 = 1sec  (default: 500)')
parser.add_argument('--prey_amount', type=int, default=100,
                    help='Initial amount of prey (default: 100)')
parser.add_argument('--predator_amount', type=int, default=4,
                    help='Initial amount of predators (default: 4)')
parser.add_argument('--death_time_predator', type=int, default=300,
                    help='Predator death timer value in ticks (60 = 1sec) (default: 300)')
parser.add_argument('--repr_amount_predator', type=int, default=10,
                    help='Amount of prey eaten needed to reproduce  (default: 10)')
parser.add_argument('--eat_probability', type=float, default=0.01,
                    help='Probability to succesfully eat prey (default: 0.01)')
parser.add_argument('--tests', type=int,
                    help='Tests to be run and saved (Number of headless tests)')

# Parse arguments
args = parser.parse_args()

# Image Control
IMG_PREY=0
IMG_PREDATOR=1
images = ["images/triangle.png","images/red_triangle.png"]

@dataclass
class BaseConfig(Config):
    repr_time_prey: int = args.repr_time_prey
    death_time_predator: int = args.death_time_predator
    repr_amount_predator: int = args.repr_amount_predator
    eat_probability: float = args.eat_probability


class PreyBase(Agent[BaseConfig]):

    class State(Enum):
        ALIVE     = 1
        REPRODUCE = 2
        DEAD      = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state: State = self.State.ALIVE
        self.timer: int = random.randint(0, self.config.repr_time_prey) # ticks counter in state
        self.angle: float = 0.0

        self.moveSmooth: float = 0.90
        self.moveStd: float = 0.1
        self.change_image(IMG_PREY)
        self.change_image(IMG_PREY)
    # __init__

    def random_walk(self):

        self.angle = self.angle + random.gauss(0, self.moveStd)
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)

        speed: float = self.config.movement_speed
        vx = self.move[0] * self.moveSmooth + speed * dx
        vy = self.move[1] * self.moveSmooth + speed * dy

        self.move = Vector2(vx, vy)
        self.pos += self.move

    # random_walk

    def onAlive(self):
        assert self.state == self.State.ALIVE

        self.random_walk()

        newState: State = self.state

        if self.timer >= self.config.repr_time_prey:
            newState = self.State.REPRODUCE
            self.timer = 0 #Reset timer 

        return newState
    # onAlive

    def onReproduce(self):
        assert self.state == self.State.REPRODUCE
        self.reproduce() #Reproduces agent
        newState: State = self.State.ALIVE
        return newState
    # onReproduce

    def onDead(self):
        assert self.state == self.State.DEAD
        newState: State = self.state

        return newState

    # onDead

    def change_position(self):
        self.there_is_no_escape()

        oldState: State = self.state
        match self.state:
            case self.State.ALIVE:
                self.state = self.onAlive()
            case self.State.REPRODUCE:
                self.state = self.onReproduce()
            case self.State.DEAD:
                self.state = self.onDead()
            case _:
                raise RuntimeError("Prey: invalid state")


        self.timer += 1
    # change_position

# PreyBase

class PredatorBase(Agent[BaseConfig]):

    class State(Enum):
        ALIVE     = 1
        REPRODUCE = 2
        HUNT      = 3
        EAT       = 4
        DEAD      = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state: State = self.State.ALIVE
        self.timer: int = random.randint(0, 1) * 60 # ticks counter in state
        self.angle: float = 0.0

        self.moveSmooth: float = 0.90
        self.moveStd: float = 0.1

        self.eat_increment: int = 0

        self.change_image(IMG_PREDATOR)
    # __init__

    def random_walk(self):

        self.angle = self.angle + random.gauss(0, self.moveStd)
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)

        speed: float = self.config.movement_speed
        vx = self.move[0] * self.moveSmooth + speed * dx
        vy = self.move[1] * self.moveSmooth + speed * dy

        self.move = Vector2(vx, vy)
        self.pos += self.move

    # random_walk

    def onAlive(self):
        assert self.state == self.State.ALIVE

        self.random_walk()

        newState: State = self.state

        if self.timer >= self.config.death_time_predator:
            newState = self.State.DEAD
            self.timer = 0 #Reset timer

        #FIXME remove redundancy here or in hunt
        for agent in self.in_proximity_performance():
            if isinstance(agent, PreyBase):
                newState = self.State.HUNT

        return newState
    # onAlive

    def onReproduce(self):
        assert self.state == self.State.REPRODUCE
        newState: State = self.State.ALIVE

        self.reproduce() #Reproduces agent
        self.eat_increment = 0

        return newState
    # onReproduce

    def onHunt(self):
        assert self.state == self.State.HUNT
        newState: State = self.State.ALIVE

        for agent, _ in self.in_proximity_accuracy():
            if isinstance(agent, PreyBase):
                if (1 - self.config.eat_probability) < random.random():

                    agent.kill()
                    newState = self.State.EAT

        return newState
    # onHunt

    def onEat(self):
        assert self.state == self.State.EAT

        newState: State = self.State.ALIVE

        self.timer = 0
        self.eat_increment += 1

        if self.eat_increment >= self.config.repr_amount_predator:
            newState = self.State.REPRODUCE

        return newState
    # onEat 

    def onDead(self):
        assert self.state == self.State.DEAD
        newState: State = self.state

        #FIXME The kill method crashes the simulation when last agen is killed
        # Intended behaviour, possible issue?
        self.kill()

        return newState
    # onDead

    def change_position(self):
        self.there_is_no_escape()

        #oldState: State = self.state
        match self.state:
            case self.State.ALIVE:
                self.state = self.onAlive()
            case self.State.REPRODUCE:
                self.state = self.onReproduce()
            case self.State.HUNT:
                self.state = self.onHunt()
            case self.State.EAT:
                self.state = self.onEat()
            case self.State.DEAD:
                self.state = self.onDead()
            case _:
                raise RuntimeError("Predator: invalid state")

        self.timer += 1
    # change_position

# PredatorBase
def run_sim(conf, headless = True):
    if headless:
        return  (HeadlessSimulation(conf)
                .batch_spawn_agents(args.predator_amount, PredatorBase, images=images)
                .batch_spawn_agents(args.prey_amount, PreyBase, images=images)
                .run()
                .snapshots)
    else:
        return  (Simulation(conf)
                .batch_spawn_agents(args.predator_amount, PredatorBase, images=images)
                .batch_spawn_agents(args.prey_amount, PreyBase, images=images)
                .run()
                .snapshots)
    
def sample_data(df, rate = 60):
    #Samples data by rate parameter
    df_aggregate: list = []
    for i in range(0, df['frame'].max(), rate):
        df_aggregate.append(df.filter(df['frame'] == i))

    return pl.concat(df_aggregate)

# sample_data

def save_data(df, dir_name, sim_name):
    #Creates folder if it doesn't exist and save dataframe to parquet
    amount = args.tests if args.tests else 1
    fname = f"{sim_name}_Simulation_T{amount}_D{args.duration}"
    if not os.path.isdir(dir_name):

        try:
            os.mkdir(dir_name)
            print(f"Directory '{dir_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{dir_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    df.write_parquet(f"{dir_name}/{fname}.parquet")

# Start of main code block

WIDTH = Config().window.width
HEIGHT = Config().window.height

conf = BaseConfig(image_rotation=True,
                        fps_limit = args.frame_limit,
                        movement_speed=0.3,
                        radius=args.radius,
                        seed=args.seed if args.seed else None, # for repeatibility
                        duration=60 * args.duration)

if args.tests:

    df_collection = []
    print(f"Starting {args.tests} Headless Simulations!")
    for i in range(1, int(args.tests) + 1):

        print(f"Running simulation: {i}")
        if args.seed:
            conf.seed = args.seed + i
        dfHless = run_sim(conf, True)
        temp = sample_data(dfHless)
        df_collection.append(temp.with_columns(pl.lit(i).alias("sim_id")))

    df = pl.concat(df_collection)

else:

    dfHead = run_sim(conf, False)
    df = sample_data(dfHead)

save_data(df, "base_simulation", "BaseLine")
print("All Done!")
