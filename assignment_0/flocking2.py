from dataclasses import dataclass
import math

from vi import Agent, Config, Simulation
from pygame.math import Vector2


@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.7 
    cohesion_weight: float = 0.3  
    separation_weight: float = 0.5  
    obstacle_avoidance_weight: float = 1.5


class FlockingAgent(Agent[FlockingConfig]):
    def change_position(self):
        self.there_is_no_escape()
        
        # Get all neighbors within the sensing radius
        nearby_agents = list(self.in_proximity_accuracy())

        if not nearby_agents:
            if self.move.length() > 0:
                self.move = self.move.normalize() * self.config.movement_speed * 0.8 

            self.pos += self.move
            return
            
        count = len(nearby_agents)
        
        # Initialize vectors for alignment, cohesion, and separation
        alignment_force = Vector2(0, 0)
        cohesion_center = Vector2(0, 0)
        separation_force = Vector2(0, 0)
        
        for agent, distance in nearby_agents:
            # Alignment: Add up all velocity vectors of neighbors
            alignment_force += agent.move
            
            # Cohesion: Add up all positions of neighbors
            cohesion_center += agent.pos            

            # Separation: Calculate repulsion vector 
            towards_self = self.pos - agent.pos

            # Scale separation force by distance (closer agents have stronger effect)
            if towards_self.length() > 0:
                # Use an exponential decay function for stronger close-range repulsion, such that clsoer ranges repulse more than further ones
                desired_min_distance = self.config.radius * 0.4  # Minimum desired separation distance
                repulsion_strength = math.exp(desired_min_distance / max(distance, 0.1)) - 1.0
                separation_force += towards_self.normalize() * repulsion_strength
        
        if count > 0:
            # Calcuate average forces
            alignment_force /= count
            cohesion_center /= count
            separation_force /= count
        
        
        # Normalize alignment force to get the heading direction
        if alignment_force.length() > 0:
            alignment_force = alignment_force.normalize()
            # Subtract current velocity to get steering force
            alignment_steering = alignment_force - self.move.normalize()
        else:
            alignment_steering = Vector2(0, 0)
            


        cohesion_steering = Vector2(0, 0)
        if count > 0:
            # Vector pointing to center of neighbors
            towards_center = cohesion_center - self.pos
            if towards_center.length() > 0:
                towards_center = towards_center.normalize()
                cohesion_steering = towards_center - self.move.normalize()   

        # Apply weights to the forces
        weighted_alignment = alignment_steering * self.config.alignment_weight
        weighted_cohesion = cohesion_steering * self.config.cohesion_weight
        weighted_separation = separation_force * self.config.separation_weight
        
        
        steering_force = weighted_alignment + weighted_cohesion + weighted_separation
        
        self.move += steering_force
        

        for dist, pos in self.obstacle_intersections():
            pos_vec = Vector2(pos)
            avoidance_vector = self.pos - pos_vec
            if avoidance_vector.length() > 0:
                avoidance_vector = avoidance_vector.normalize()
                self.move += avoidance_vector * self.config.obstacle_avoidance_weight

        # Limit speed to movement_speed
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed
            
        self.pos += self.move




config = Config()
x, y = config.window.as_tuple()


(
    Simulation(
        FlockingConfig(
            image_rotation=True,    
            movement_speed=1,     
            radius=75,              
            duration=10000,
            fps_limit=0       
        )
    )
    .batch_spawn_agents(80, FlockingAgent, images=["Assignment_0/images/triangle.png"])  
    # .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/triangle@200px.png", x // 2, y // 2)
    # .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/triangle@50px.png", x // 4, y // 4)
    .run()
)
