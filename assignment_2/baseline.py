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
parser.add_argument('--duration', type=int, default=300,
                    help='Simulation duration value in seconds (default: 60)')
parser.add_argument('--frame_limit', type=int, default=60,
                    help='fps limit, 0 uncaps (Fastest) (default: 60)')
parser.add_argument('--radius', type=int, default=30,
                    help='Radius value (default: 30)')
parser.add_argument('--repr_time_prey', type=int, default=160,
                    help='Prey reproduction timer value in ticks 60 = 1sec  (default: 320)')
parser.add_argument('--prey_amount', type=int, default=10,
                    help='Initial amount of prey (default: 10)')
parser.add_argument('--predator_amount', type=int, default=3,
                    help='Initial amount of predators (default: 3)')
parser.add_argument('--death_time_predator', type=int, default=640,
                    help='Predator death timer value in ticks (60 = 1sec) (default: 640)')
parser.add_argument('--repr_amount_predator', type=int, default=2,
                    help='Amount of prey eaten needed to reproduce  (default: 1)')
parser.add_argument('--eat_probability', type=float, default=1.0,
                    help='Probability to succesfully eat prey (default: 1.0)')
parser.add_argument('--tests', type=int,
                    help='DEPRECATED DONT USE (Number of headless tests)')

# Parse arguments
args = parser.parse_args()

img = pygame.image.load("images/site.png")
MIN_DIST:int = img.get_width()


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
        # FIXME Change wariables to be initialized
        super().__init__(*args, **kwargs)

        self.state: State = self.State.ALIVE
        self.timer: int = 1 #random.randint(0, 5) * 60 # ticks counter in state
        self.angle: float = 0.0

        self.moveSmooth: float = 0.90
        self.moveStd: float = 0.1
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
        #FIXME Add connection to dead and reproduce state
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
        for agent, _ in self.in_proximity_accuracy():
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

        # if oldState != self.state:
        #     # state changed, reset the timer
        #     self.timer = 0
        self.timer += 1
    # change_position

# PredatorBase
WIDTH = Config().window.width
HEIGHT = Config().window.height

# Snapshots analysis

def dist(a: tuple[int,int], b: tuple[int,int]):
    """
    Minimum distance in the wraparound plane
    """
    global WIDTH, HEIGHT
    W = WIDTH
    H = HEIGHT

    dx = abs(a[0] - b[0])
    dx = min(dx, W - dx)

    dy = abs(a[1] - b[1])
    dy = min(dy, H - dy)

    # euclidean distance
    return math.sqrt((dx*dx) + (dy*dy))
# dist


def frame_metrics(frame):
    """
    frame: a dataframe where 'frame' is constant.

    Return: number of clusters, ecs
    """
    global MIN_DIST
    N = len(frame) # number of agents

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(0, N):
        xi = frame.item(row=i, column="x")
        yi = frame.item(row=i, column="y")

        for j in range(i + 1, N):
            xj = frame.item(row=j, column="x")
            yj = frame.item(row=j, column="y")

            d: float = dist((xi,yi),(xj,yj))
            if d < MIN_DIST:
                G.add_edge(i,j)

    # Expected Cluster Size
    # for each agent (node) get the size of the cluster where it belongs
    sizes: list[int] = [0]*N
    clusters: int = 0
    for cluster in nx.connected_components(G):
        clusters += 1
        # cluster is a set
        for node in cluster:
            # every node must be in only one cluster
            assert sizes[node] == 0
            # squared to give more weight to big clusters
            sizes[node] = len(cluster) ** 2

    ecs: float = math.sqrt(sum(sizes) / N)

    return clusters, ecs
# frame_metrics

# Analysis

def collect_metrics(df, rate=60):
    """
    Create a dataframe with a row for every rate frames in df with the metrics
    """
    M = df["frame"].max()

    frames: list = []
    ecsList: list = []
    clustersList: list = []

    for i in range(0, M+1, rate):

        snap = df.filter(df["frame"] == i)
        clusters, ecs = frame_metrics(snap)

        frames.append(i)
        ecsList.append(ecs)
        clustersList.append(clusters)

    df = pl.DataFrame({
        "frame": frames,
        "ECS": ecsList,
        "Clusters": clustersList,
    })

    return df
# collect_metrics

def run_simulation(simulation):
    #FIXME add more configs
    """
    return metrics
    """
    global WIDTH, HEIGHT
    df = (
        simulation
        .batch_spawn_agents(args.predator_amount, PredatorBase, images=["images/red_triangle.png"])
        .batch_spawn_agents(args.prey_amount, PreyBase, images=["images/triangle.png"])
        .run()
        .snapshots
    )
    return collect_metrics(df)
# run_simulation
# FIXME Add more configs,
conf = BaseConfig(image_rotation=True,
                        fps_limit = args.frame_limit,
                        movement_speed=0.3,
                        radius=args.radius,
                        seed=args.seed if args.seed else None, # for repeatibility
                        duration=60 * args.duration)
if args.tests:
#########################################################3
#
#        Deprecated
#
#########################################################
    print("TEST MODE. Running {args.tests} simulations")
    simulation = HeadlessSimulation(conf)
    print(f"Simulation 001")
    metrics = run_simulation(simulation)
    metrics = metrics.rename({"ECS":"ESC_001",
                              "Clusters": "Clusters_001"})

    for i in range(1, int(args.tests)):
        simIdStr = f"{i+1:03d}"
        print(f"Simulation {simIdStr}")
        conf.seed = i*10 # otherwise it repeats always the same simulation
        simulation = HeadlessSimulation(conf)
        mtr = run_simulation(simulation)
        metrics = metrics.join(mtr, on="frame")
        metrics = metrics.rename({"ECS": f"ESC_{simIdStr}",
                                  "Clusters": f"Clusters_{simIdStr}"})
    # for

    fname: str = f"simulations_T{args.tests}_J{args.join_n}.parquet"
    metrics.write_parquet(fname)
    print(f"DONE. Saved on {fname}")
else:

    # Standalone, visual simulation
    simulation = Simulation(conf)
    run_simulation(simulation)

#########################################################3
#
#        Deprecated
#
#########################################################
#     fname = f"D{args.duration:.4f}_R{args.radius}_S{args.seed}"

#     # Plotting
#     plt.figure(figsize=(12, 8))

#     plt.subplot(2, 1, 1)
#     plt.plot(metrics["frame"], metrics["ECS"], 'b-', label='ECS')
#     plt.xlabel('Frame')
#     plt.ylabel('ECS')
#     plt.title(f'Expected Cluster Size')
#     plt.ylim(0, metrics["ECS"].max())
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(metrics["frame"], metrics["Clusters"], 'r-', label='Clusters')
#     plt.xlabel('Frame')
#     plt.ylabel('Clusters')
#     plt.title(f'Number of Clusters')
#     plt.ylim(0, metrics["Clusters"].max()+1)
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("plots/"+fname+".png", dpi=300, bbox_inches='tight')
# # if test
