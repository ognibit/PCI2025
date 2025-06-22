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
from vi.simulation import Shared

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

parser.add_argument('--prey_amount', type=int, default=60,
                    help='Initial amount of prey (default: 60)')

parser.add_argument('--repr_amount_prey', type=int, default = 2,
                    help="Amount of food needed for prey to pass into reprodcution (default: 2)")

parser.add_argument('--eat_interval_prey', type=int, default = 30,
                    help="Tick amount to pass for prey to eat (default: 30)")

parser.add_argument('--death_time_prey', type=int, default = 600,
                    help="Time for prey dying without food (default: 600)")

parser.add_argument('--prey_toxicity_creation', type=int, default = 1,
                    help="Amount of toxicity created when prey eats (default: 1)")

parser.add_argument('--predator_amount', type=int, default=15,
                    help='Initial amount of predators (default: 15)')

parser.add_argument('--death_time_predator', type=int, default=600,
                    help='Predator death timer value in ticks (60 = 1sec) (default: 600)')

parser.add_argument('--repr_amount_predator', type=int, default=6,
                    help='Amount of prey eaten needed to reproduce  (default: 6)')

parser.add_argument('--eat_probability', type=float, default=0.15,
                    help='Probability to succesfully eat prey (default: 0.15)')

parser.add_argument('--food_start', type=int, default=20,
                    help='Amount of food the simulation is initialized with (default: 20)')

parser.add_argument('--food_interval', type=int, default=180,
                    help='Interval for increasing food (default: 180)')

parser.add_argument('--food_amount', type=int, default=50,
                    help='Food added per food interval (default: 50)')

parser.add_argument('--toxic_start', type=int, default=100,
                    help='Amount of toxicity the simulation starts with (default: 100)')

parser.add_argument('--toxicity_red_interval', type=int, default=120,
                    help='Time for interval for reducing toxicity (default: 120)')

parser.add_argument('--toxicity_red_percent', type=float, default=0.1,
                    help='Percentage of toxicity reduced per interval (default: 0.01)')

parser.add_argument('--fatal_toxicity', type=int, default=1000,
                    help='Amount of toxicity for instant death (default: 1000)')

parser.add_argument('--tests', type=int,
                    help='Tests to be run and saved (Number of headless tests)')

parser.add_argument('--graph', type=bool, default=False,
                    help='Will create graphs for each simulation (Default False)')

# Parse arguments
args = parser.parse_args()



# Image Control
IMG_PREY=0
IMG_PREDATOR=1
images = ["images/triangle.png","images/red_triangle.png"]


@dataclass
class BaseConfig(Config):
    repr_amount_prey : int = args.repr_amount_prey
    eat_interval_prey: int = args.eat_interval_prey
    death_time_prey: int = args.death_time_prey
    death_time_predator: int = args.death_time_predator
    repr_amount_predator: int = args.repr_amount_predator
    eat_probability: float = args.eat_probability
    food_interval: int = args.food_interval
    food_amount: int = args.food_amount
    prey_toxicity_creation: int = args.prey_toxicity_creation
    toxicity_red_interval: int = args.toxicity_red_interval
    toxicity_red_percent: int = args.toxicity_red_percent
    fatal_toxicity: int = args.fatal_toxicity
    

class SharedManager(Agent[BaseConfig]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.food_timer: int = random.randint(0, self.config.food_interval)
        self.toxic_timer: int = 0

    def update(self):
        self.save_data("toxic", self.shared.toxicity_exists)
        self.save_data("food", self.shared.food_available)
    # update


    def change_position(self):
    # Increase food
        if self.food_timer >= self.config.food_interval:
            self.food_timer = 0
            self.shared.food_available += self.config.food_amount

        if self.toxic_timer >= self.config.toxicity_red_interval:
            self.toxic_timer = 0
            self.shared.toxicity_exists = int(
                self.shared.toxicity_exists *
                (1 - self.config.toxicity_red_percent)
            ) 

        self.food_timer += 1
        self.toxic_timer += 1

    # update
# FoodManager

class PreyFood(Agent[BaseConfig]):

    class State(Enum):
        ALIVE     = 1
        REPRODUCE = 2
        EAT       = 3
        DEAD      = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state: State = self.State.ALIVE
        self.timer: int = random.randint(0, self.config.death_time_prey) # ticks counter in state
        self.eat_timer: float = random.randint(0, self.config.eat_interval_prey)
        self.angle: float = 0.0
        self.moveSmooth: float = 0.90
        self.moveStd: float = 0.1
        self.eat_increment: int = 0
        self.change_image(IMG_PREY)
        self.change_image(IMG_PREY)
    # __init__

    def eat(self):
        result = False
        if self.shared.food_available >= 1:
            self.shared.food_available -= 1
            result = True
        return result

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

        reducedThreshold: int = (self.config.death_time_prey * 
                (1 - (self.shared.toxicity_exists/self.config.fatal_toxicity)))

        if self.eat_timer >= self.config.eat_interval_prey:
            newState = self.State.EAT

        if self.timer >= reducedThreshold:
            self.timer = 0 #Reset timer 
            newState = self.State.DEAD

        return newState
    # onAlive

    def onReproduce(self):
        assert self.state == self.State.REPRODUCE
        newState: State = self.State.ALIVE
        self.eat_increment = 0
        self.reproduce() #Reproduces agent
        return newState
    # onReproduce

    def onEat(self):
        assert self.state == self.State.EAT
        newState: State = self.State.ALIVE
        
        if self.eat():
            self.shared.toxicity_exists += 1
            self.eat_increment += 1
            self.timer = 0

        if self.eat_increment >= self.config.repr_amount_prey:
            newState: State = self.State.REPRODUCE

        return newState
    # onEat

    def onDead(self):
        assert self.state == self.State.DEAD
        newState: State = self.state
        self.kill()

        return newState

    # onDead

    def update(self):
        self.save_data("toxic", self.shared.toxicity_exists)
        self.save_data("food", self.shared.food_available)
    # update

    def change_position(self):
        self.there_is_no_escape()
        
        oldState: State = self.state
        match self.state:
            case self.State.ALIVE:
                self.state = self.onAlive()
            case self.State.REPRODUCE:
                self.state = self.onReproduce()
            case self.State.EAT:
                self.state = self.onEat()
            case self.State.DEAD:
                self.state = self.onDead()
            case _:
                raise RuntimeError("Prey: invalid state")


        self.timer += 1
        self.eat_timer += 1
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

        reducedThreshold: int = (self.config.death_time_prey * 
                (1 - (self.shared.toxicity_exists/self.config.fatal_toxicity)))

        if self.timer >= reducedThreshold:
            newState = self.State.DEAD
            self.timer = 0 #Reset timer

        for agent in self.in_proximity_performance():
            if isinstance(agent, PreyFood):
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
            if isinstance(agent, PreyFood):
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

    def update(self):
        self.save_data("toxic", self.shared.toxicity_exists)
        self.save_data("food", self.shared.food_available)
    # update

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
@dataclass
class BaseShared(Shared):
    food_available: int = args.food_start
    toxicity_exists: int = args.toxic_start

class NewSim(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prng_move = random.Random()
        prng_move.seed(self.config.seed)

        self.shared = BaseShared(prng_move=prng_move)

class NewSimHeadless(HeadlessSimulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prng_move = random.Random()
        prng_move.seed(self.config.seed)

        self.shared = BaseShared(prng_move=prng_move)

def run_sim(conf, headless = True):
    
    if headless:
        return  (NewSimHeadless(conf)
                .batch_spawn_agents(args.predator_amount, PredatorBase, images=images)
                .batch_spawn_agents(args.prey_amount, PreyFood, images=images)
                .spawn_agent(SharedManager, images=["images/food.png"])
                .run()
                .snapshots)
    else:
        return  (NewSim(conf)
                .batch_spawn_agents(args.predator_amount, PredatorBase, images=images)
                .batch_spawn_agents(args.prey_amount, PreyFood, images=images)
                .spawn_agent(SharedManager, images=["images/food.png"])
                .run()
                .snapshots)
    
def sample_data(df, rate = 60):
    #Samples data by rate parameter
    df_aggregate: list = []
    for i in range(0, df['frame'].max(), rate):
        df_aggregate.append(df.filter(df['frame'] == i))

    return pl.concat(df_aggregate)

# sample_data

def plot_population_graph(df, i = 0, dir_name = "plots_base"):
    """
    Plot and save a graph of prey and predator populations over time.
    """
    import numpy as np
    # First get toxicity and food levels (they should be constant per frame)
    env_data = (
        df.group_by("frame")
        .agg(
            pl.first("toxic").alias("toxic"),
            pl.first("food").alias("food")
        )
    )
    
    # Then get population counts
    pop_counts = (
        df.group_by(["frame", "image_index"])
        .agg(pl.len().alias("population"))
    )

    # Pivot the population data
    pop_pivot = pop_counts.pivot(
        values="population",
        index="frame",
        columns="image_index"
    ).fill_null(0)

    # Join with environment data
    plot_data = pop_pivot.join(env_data, on="frame")

    x_values = plot_data["frame"].to_numpy()
    prey_counts = plot_data["0"].to_numpy()
    predator_counts = plot_data["1"].to_numpy()
    toxicity = plot_data["toxic"].to_numpy()
    food = plot_data["food"].to_numpy()

    # Downsample for plotting
    max_points = 1000
    if len(x_values) > max_points:
        idx = np.linspace(0, len(x_values) - 1, max_points).astype(int)
        x_values = x_values[idx]
        prey_counts = prey_counts[idx]
        predator_counts = predator_counts[idx]
        toxicity = toxicity[idx]
        food = food[idx]

    # Sort by frame index to ensure proper line plot
    sort_idx = np.argsort(x_values)
    x_values = x_values[sort_idx]
    prey_counts = prey_counts[sort_idx]
    predator_counts = predator_counts[sort_idx]
    toxicity = toxicity[sort_idx]
    food = food[sort_idx]

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(x_values, prey_counts, label="Prey", color="blue", linewidth=2)
    plt.plot(x_values, predator_counts, label="Predator", color="red", linewidth=2)
    plt.plot(x_values, toxicity, label="Toxicity", color="purple", linewidth=1, alpha=0.7)
    #plt.plot(x_values, food, label="Food", color="green", linewidth=1, alpha=0.7)
    
    plt.xlabel("Time (Frame Index)")
    plt.ylabel("Count")
    plt.title(f"Population Dynamics (Simulation {i})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
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
    
    plot_filename = f"toxicSimulation_D{args.duration}_R{args.radius}_S{args.seed if args.seed else 'random'}A_{args.food_amount}SIM_ID{i}.png"
    plt.savefig(dir_name +"/"+ plot_filename)
    print(f"Population plot saved as {plot_filename}")
    
    return plot_filename


def save_data(df, dir_name, sim_name):
    #Creates folder if it doesn't exist and save dataframe to parquet
    amount = args.tests if args.tests else 1
    fname = f"{sim_name}_Simulation_T{amount}_D{args.duration}_S{args.seed if args.seed else 'random'}I{args.food_interval}"
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
        if args.graph:

            plot_population_graph(temp, i, dir_name = "plots_toxic")

        df_collection.append(temp.with_columns(pl.lit(i).alias("sim_id")))

    df = pl.concat(df_collection)

else:

    dfHead = run_sim(conf, False)
    df = sample_data(dfHead)
    if args.graph:

        plot_population_graph(df, dir_name = "plots_toxic")

save_data(df, "toxic_simulation", "toxicSystem")
print("All Done!")
