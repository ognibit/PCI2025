from dataclasses import dataclass

from vi import Agent, Config, Simulation

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

(
    Simulation(
        # TODO: Modify `movement_speed` and `radius` and observe the change in behaviour.
        FlockingConfig(image_rotation=True,
                       movement_speed=1,
                       radius=50,
                       duration=10000)
    )
    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png"])
    .run()
)
