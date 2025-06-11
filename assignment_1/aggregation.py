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

from vi import Agent, Config, Simulation
from vi.util import count

import networkx as nx
import polars as pl
from pygame.math import Vector2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Aggregation')

parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed value (default: 42)')
parser.add_argument('--duration', type=int, default=10,
                    help='Duration value (default: 10)')
parser.add_argument('--frame_limit', type=int, default=25,
                    help='fps limit, 0 uncaps (Fastest) (default: 25)')
parser.add_argument('--radius', type=int, default=20,
                    help='Radius value (default: 20)')
parser.add_argument('--timer_leave', type=int, default=60,
                    help='Timer leave value in ticks (default: 60)')
parser.add_argument('--timer_join', type=int, default=60,
                    help='Timer join value in ticks (default: 60)')
parser.add_argument('--join_n', type=int, default=4,
                    help='Neighbours to have 0.5 change of join (default: 4)')
parser.add_argument('--still_sample', type=int, default=60,
                    help='Ticks for the STILL frequency sampling (default: 60)')

# Parse arguments
args = parser.parse_args()

P_SEED: int = int(args.seed)
P_RADIUS: int = int(args.radius)
P_DURATION: int = int(args.duration)

MIN_DIST:int = 20 # threshold

# Sigmoid to generate a probability distribution based on the number of
# neighbours (x). The signmoid allows to not saturate the probabilities.
# The "shift" parameter allows to center the 50% of change on that value.
def sigmoid(x, shift=0):
    return 1 / (1 + math.exp(shift - x))

@dataclass
class CockroachConfig(Config):
    timer_leave: int = args.timer_leave
    timer_join: int = args.timer_join
    join_n: int = args.join_n
    still_sample: int = args.still_sample

class Cockroach(Agent[CockroachConfig]):

    class State(Enum):
        WANDERING = 1
        JOIN      = 2
        STILL     = 3
        LEAVE     = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state: State = self.State.WANDERING
        self.timer: int = 0 # ticks counter in state
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

    def onWandering(self):
        assert self.state == self.State.WANDERING

        self.random_walk()

        # calculate the probability of join using the neighbours
        newState: State = self.state
        if self.on_site():

            n:int = count(self.in_proximity_performance())
            p: float = random.random()

            if p <= sigmoid(n, shift=self.config.join_n):
                newState = self.State.JOIN

        return newState
    # onWandering

    def onJoin(self):
        assert self.state == self.State.JOIN

        #FIXME do we test that the agent remains into the site? I made it so
        #  that if he is not on the site that he keeps wandering so we don't have 
        # them standing outside the site as much
        self.random_walk()

        newState: State = self.state
        if self.timer >= self.config.timer_join:
            newState = self.State.STILL

        return newState
    # onJoin

    def onStill(self):
        assert self.state == self.State.STILL

        newState: State = self.state

        # every "still_sample" ticks, sample the probability to leave
        if self.timer % self.config.still_sample == 0:
            # calculate the probability of leave using the neighbours
            n:int = count(self.in_proximity_performance())
            p: float = random.random()

            # the "dual analog" of the join probability
            if p <= (1 - sigmoid(n, shift=self.config.join_n)):
                newState = self.State.LEAVE

        return newState
    # onStill

    def onLeave(self):
        assert self.state == self.State.LEAVE

        #FIXME do we test that the agent is exiting the site?
        self.random_walk()

        newState: State = self.state
        if self.timer >= self.config.timer_leave:
            newState = self.State.WANDERING

        return newState
    # onLeave

    def change_position(self):
        self.there_is_no_escape()

        oldState: State = self.state
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

        if oldState != self.state:
            # state changed, reset the timer
            self.timer = 0

        self.timer += 1
    # change_position

# Cockroach

WIDTH = Config().window.width
HEIGHT = Config().window.height

df = (
    Simulation(
        CockroachConfig(image_rotation=True,
                        fps_limit = args.frame_limit,
                        movement_speed=1,
                        radius=args.radius,
                        seed=args.seed, # for repeatibility
                        duration=60 * args.duration)
    )
    .spawn_site("images/site.png", x = WIDTH // 2, y= HEIGHT // 2) #Spawn site in the middle
    .batch_spawn_agents(20, Cockroach, images=["images/triangle.png"])
    .run()
    .snapshots
)

# Snapshots analysis



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
def collect_metrics(df, rate=60):
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

fname = f"D{args.duration:.4f}_R{args.radius}_S{args.seed}"
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
