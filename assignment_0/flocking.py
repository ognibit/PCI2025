from dataclasses import dataclass

from vi import Agent, Config, Simulation

from pygame.math import Vector2

import sys


P_ALIGN: float = float(sys.argv[1])
P_COHESION: float =  float(sys.argv[2])
P_SEPARATION: float =  float(sys.argv[3])
P_RADIUS: int = int(sys.argv[4])
P_SEED: int = int(sys.argv[5])


print("Align", P_ALIGN)
print("Cohesion", P_COHESION)
print("Separation", P_SEPARATION)
print("Radius", P_RADIUS)
print("Seed", P_SEED)


@dataclass
class FlockingConfig(Config):
    # TODO: Modify the weights and observe the change in behaviour.
    alignment_weight: float = P_ALIGN
    cohesion_weight: float =  P_COHESION
    separation_weight: float =  P_SEPARATION
    #MaxVelocity: float = 100 
    #FIXME mass, delta T, what values?
    mass: float = 10
    dt: float = 1


class FlockingAgent(Agent[FlockingConfig]):
    # By overriding `change_position`, the default behaviour is overwritten.
    # Without making changes, the agents won't move.
    def change_position(self):
        # Pac-Man-style teleport the agent to the other side of the screen
        # when it is outside of the playable area.
        self.there_is_no_escape()

        n: int = 0
        velAvg: Vector2 = Vector2((0,0))
        posAvg: Vector2 = Vector2((0,0))
        sepAvg: Vector2 = Vector2((0,0))

        for agent, dist in self.in_proximity_accuracy():
            velAvg += agent.move
            posAvg += agent.pos
            sepAvg += self.pos - agent.pos
            n += 1

        if (n > 0):
            velAvg = velAvg / n
            posAvg = posAvg / n
            sepAvg = sepAvg / n

            align: Vector2 = velAvg - self.move
            separation: Vector2 = sepAvg
            cohesion: Vector2 = (posAvg - self.pos) - self.move

            force: Vector2 =((self.config.alignment_weight * align) +
                            (self.config.separation_weight * separation) +
                            (self.config.cohesion_weight * cohesion))
            force /= self.config.mass

            self.move += force
            self.move = self.move.normalize()
            #FIXME: Continue working on speed cap
            #self.move = min(self.move.length(), self.config.MaxVelocity) * self.move.normalize()

        self.pos += self.move * self.config.dt #Fixed to align with pseudocode
# FlockingAgent

df = (
    Simulation(
        # TODO: Modify `movement_speed` and `radius` and observe the change in behaviour.
        FlockingConfig(image_rotation=True,
                       fps_limit = 300,
                       movement_speed=1,
                       radius=P_RADIUS,
                       seed=P_SEED, # for repeatibility
                       duration=60 * 60)
    )
    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png"])
    .run()
    .snapshots
)

# Snapshots analysis

import math
import time
import polars as pl

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

def as_unit_circle(x, y):
    """
    Convert 2D plane coordinate into position in the unit circle.
    The plane is intended as wraparound.

    Return: cos(angle(x)), sin(angle(x)), cos(angle(y)), sin(angle(y))
    """
    global WIDTH, HEIGHT
    W = WIDTH
    H = HEIGHT

    # as radians
    ax: float = (2*math.pi) * (x/W)
    ay: float = (2*math.pi) * (y/H)

    cx: float = math.cos(ax)
    sx: float = math.sin(ax)

    cy: float = math.cos(ay)
    sy: float = math.sin(ay)

    return cx, sx, cy, sy
# as_unit_circle

def frame_metrics(frame):
    """
    Calculate dispersion, separation and alignment costs for one frame.
    They are costs, therefore the lower the better (0 best).

    frame: a dataframe where 'frame' is constant.

    Return: (dispersion, separation, alignment)
    """
    global WIDTH, HEIGHT
    W = WIDTH
    H = HEIGHT
    N = len(frame) # number of agents

    # alignment: sample standard deviation (ddof = 1) 
    align: float = frame["angle"].std()

    # SEPARATION (collect also coordinate for the centroid)

    # separation = min(0, THRESHOLD - min(pairwise distances))
    MIN_DIST = 10 # threshold
    mindist: float = W*H # above the highest possible distance

    # centroid trigonometric coordinates
    cosx: float = 0.0
    sinx: float = 0.0
    cosy: float = 0.0
    siny: float = 0.0

    for i in range(0, N):
        xi = frame.item(row=i, column="x")
        yi = frame.item(row=i, column="y")

        cx, sx, cy, sy = as_unit_circle(xi, yi)
        cosx += cx
        sinx += sx
        cosy += cy
        siny += sy

        for j in range(i + 1, N):
            xj = frame.item(row=j, column="x")
            yj = frame.item(row=j, column="y")

            d: float = dist((xi,yi),(xj,yj))
            mindist = min(mindist, d)

    # penalty if two boids distance is between 0 and MIN_DIST
    # it's a min() therefore no average necessary
    sepa:float = max(0, MIN_DIST - mindist)

    # DISPERSION
    cosx /= N
    sinx /= N
    cosy /= N
    siny /= N

    centr_x:float = math.atan2(sinx, cosx) * W / (2*math.pi)
    centr_y:float = math.atan2(siny, cosy) * H / (2*math.pi)

    centroid: tuple[int,int] = (round(centr_x) % W, round(centr_y) % H)

    # dispersion as average distance from the centroid
    disp: float = 0.0

    for i in range(0, N):
        xi = frame.item(row=i, column="x")
        yi = frame.item(row=i, column="y")
        disp += dist((xi,yi), centroid)

    disp /= N

    return disp, sepa, align
# frame_metrics

# Analysis

@timer_decorator
def collect_metrics(df, rate=120):
    """
    Create a dataframe with a row for every rate frames in df with the metrics
    """
    M = df["frame"].max()

    frames: list = []
    dispersions: list = []
    separations: list = []
    alignments: list = []

    for i in range(0, M+1, rate):

        snap = df.filter(df["frame"] == i)
        dispersion, separation, alignment = frame_metrics(snap)

        frames.append(i)
        dispersions.append(dispersion)
        separations.append(separation)
        alignments.append(alignment)

    df = pl.DataFrame({
        "frame": frames,
        "dispersion": dispersions,
        "separation": separations,
        "alignment": alignments
    })

    return df
# collect_metrics

metrics = collect_metrics(df)
#print(metrics)
fname = f"A{P_ALIGN:.4f}_C{P_COHESION:.4f}_S{P_SEPARATION:.4f}_R{P_RADIUS}_D{P_SEED}"
metrics.write_parquet("plots/"+fname+".parquet")

import matplotlib.pyplot as plt

# Plotting
title = f"A: {P_ALIGN:.4f},C: {P_COHESION:.4f},S: {P_SEPARATION:.4f}, R:{P_RADIUS}, D:{P_SEED}"
plt.figure(figsize=(12, 8))

# Plot all three metrics vs frame
plt.subplot(2, 2, 1)
plt.plot(metrics["frame"], metrics["dispersion"], 'b-', label='Dispersion')
plt.xlabel('Frame')
plt.ylabel('Dispersion')
plt.title(f'Dispersion ({title})')
plt.ylim(0, metrics["dispersion"].max())
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(metrics["frame"], metrics["separation"], 'r-', label='Separation')
plt.xlabel('Frame')
plt.ylabel('Separation')
plt.title(f'Separation ({title})')
plt.ylim(0, metrics["separation"].max())
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(metrics["frame"], metrics["alignment"], 'g-', label='Alignment')
plt.xlabel('Frame')
plt.ylabel('Alignment')
plt.title(f'Alignment ({title})')
plt.ylim(0, metrics["alignment"].max())
plt.grid(True)

# All metrics on one plot
plt.subplot(2, 2, 4)
plt.plot(metrics["frame"], metrics["dispersion"], 'b-', label='Dispersion')
plt.plot(metrics["frame"], metrics["separation"], 'r-', label='Separation')
plt.plot(metrics["frame"], metrics["alignment"], 'g-', label='Alignment')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.title(f'All Metrics ({title})')
plt.legend()
plt.grid(True)

plt.tight_layout()
#plt.show()
plt.savefig("plots/"+fname+".png", dpi=300, bbox_inches='tight')
