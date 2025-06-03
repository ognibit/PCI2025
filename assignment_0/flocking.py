from dataclasses import dataclass

from vi import Agent, Config, Simulation, HeadlessSimulation

from pygame.math import Vector2


@dataclass
class FlockingConfig(Config):
    # TODO: Modify the weights and observe the change in behaviour.
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5


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

            #FIXME mass, delta T, what values?
            mass: float = 1
            dt: float = 1
            force: Vector2 =((self.config.alignment_weight * align) +
                            (self.config.separation_weight * separation) +
                            (self.config.cohesion_weight * cohesion))
            force /= mass

            self.move += force
            #FIXME add a speed cap
            self.move = self.move.normalize() * dt
        # if n > 0

        self.pos += self.move
# FlockingAgent

df = (
# FIXME experiments
    Simulation(
#    HeadlessSimulation(
        # TODO: Modify `movement_speed` and `radius` and observe the change in behaviour.
        FlockingConfig(image_rotation=True,
                       movement_speed=1,
                       radius=50,
                       seed=42, #FIXME repeatibility
                       duration=1000) # FIXME test
    )
    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png"])
    .run()
    .snapshots
)

# Analysis
import math
lastFrame = df["frame"].max()
lastRun = df["frame"] == lastFrame
snap = df.filter(lastRun)

# pairwise distances
n = len(snap) # must be the number of agents
centroid: Vector2 = Vector2((0,0))
mindist: float = 1000

# calculate both the sum of distances and the centroid
for i in range(0, len(snap)):
    xi = snap.item(row=i, column="x")
    yi = snap.item(row=i, column="y")
    centroid += Vector2((xi,yi))

    for j in range(i + 1, len(snap)):
        xj = snap.item(row=j, column="x")
        yj = snap.item(row=j, column="y")

        dist: float = math.sqrt((xi - xj)**2 + (yi - yj)**2)
        mindist = min(mindist, dist)

MIN_DIST = 10
# penalty if two boids distance is between 0 and MIN_DIST
cost_separation = max(0, MIN_DIST - mindist)

centroid /= n

cost_cohesion: float = 0.0
for i in range(0, len(snap)):
    xi = snap.item(row=i, column="x")
    yi = snap.item(row=i, column="y")
    cost_cohesion += math.sqrt((xi - centroid[0])**2 + (yi - centroid[1])**2)

cost_cohesion /= n

print("Centroid", centroid)
print("Cohesion", cost_cohesion)
print("Separation", cost_separation)
