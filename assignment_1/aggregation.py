"""
SETUP

pip install networkx
pip install matplotlib
"""

from dataclasses import dataclass
import math
import time
import sys
import random
from enum import Enum

import networkx as nx
import polars as pl
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import matplotlib.pyplot as plt

P_SEED: int = int(sys.argv[1])
P_RADIUS: int = int(sys.argv[2])
P_DURATION: int = int(sys.argv[3])

MIN_DIST = 10 # threshold

@dataclass
class CockroachConfig(Config):
    timer_leave: int = 10
    timer_join: int = 10

class Cockroach(Agent[CockroachConfig]):

    class State(Enum):
        WANDERING = 1
        JOIN      = 2
        STILL     = 3
        LEAVE     = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state: State = self.State.WANDERING
#        self.speed: float = 0.1
        self.angle: float = 0.0

        self.moveSmooth: float = 0.90
        self.moveStd: float = 0.1
    # __init__

    def onWandering(self):
        assert self.state == self.State.WANDERING

        # random walk
        self.angle = self.angle + random.gauss(0, self.moveStd)
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)

        speed: float = self.config.movement_speed
        vx = self.move[0] * self.moveSmooth + speed * dx
        vy = self.move[1] * self.moveSmooth + speed * dy

        self.move = Vector2(vx, vy)
        self.pos += self.move

        return self.State.WANDERING
    # onWandering

    def onJoin(self):
        assert self.state == self.State.JOIN
        return self.State.JOIN
    # onJoin

    def onStill(self):
        assert self.state == self.State.STILL
        return self.State.STILL
    # onStill

    def onLeave(self):
        assert self.state == self.State.LEAVE
        return self.State.LEAVE
    # onLeave

    def change_position(self):
        self.there_is_no_escape()

        match self.state:
            case self.State.WANDERING:
                self.state = self.onWandering()
            case self.State.JOIN:
                self.state = self.onJoin()
            case self.State.STILL:
                self.state = self.onStill()
            case self.State.LEAVE:
                self.state = self.onLeave()
            case _:
                raise RuntimeError("Cockroach: invalid state")
    # change_position

# Cockroach

df = (
    Simulation(
        CockroachConfig(image_rotation=True,
                        fps_limit = 25,
                        movement_speed=1,
                        radius=P_RADIUS,
                        seed=P_SEED, # for repeatibility
                        duration=60 * P_DURATION)
    )
    .batch_spawn_agents(20, Cockroach, images=["images/triangle.png"])
    .run()
    .snapshots
)

# Snapshots analysis


WIDTH = Config().window.width
HEIGHT = Config().window.height

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.6f} seconds")
        return result
    return wrapper
# timer_decorator

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
    Return: ecs
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
    for cluster in nx.connected_components(G):
        # cluster is a set
        for node in cluster:
            # every node must be in only one cluster
            assert sizes[node] == 0
            # squared to give more weight to big clusters
            sizes[node] = len(cluster) ** 2

    ecs: float = math.sqrt(sum(sizes) / N)

    return ecs
# frame_metrics

# Analysis

@timer_decorator
def collect_metrics(df, rate=120):
    """
    Create a dataframe with a row for every rate frames in df with the metrics
    """
    M = df["frame"].max()

    frames: list = []
    ecsList: list = []
    dispersions: list = []
    separations: list = []
    alignments: list = []

    for i in range(0, M+1, rate):

        snap = df.filter(df["frame"] == i)
        ecs = frame_metrics(snap)

        frames.append(i)
        ecsList.append(ecs)

    df = pl.DataFrame({
        "frame": frames,
        "ECS": ecsList,
    })

    return df
# collect_metrics

metrics = collect_metrics(df)
print(metrics)

P_SEED: int = int(sys.argv[1])
P_RADIUS: int = int(sys.argv[2])
P_DURATION: int = int(sys.argv[3])
fname = f"D{P_DURATION:.4f}_R{P_RADIUS}_S{P_SEED}"
#metrics.write_parquet("plots/"+fname+".parquet")


# Plotting
#title = f"A: {P_ALIGN:.4f},C: {P_COHESION:.4f},S: {P_SEPARATION:.4f}, R:{P_RADIUS}, D:{P_SEED}"
plt.figure(figsize=(12, 8))

# Plot all three metrics vs frame
plt.plot(metrics["frame"], metrics["ECS"], 'b-', label='ECS')
plt.xlabel('Frame')
plt.ylabel('ECS')
#plt.title(f'Dispersion ({title})')
plt.ylim(0, metrics["ECS"].max())
plt.grid(True)

plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("plots/"+fname+".png", dpi=300, bbox_inches='tight')
