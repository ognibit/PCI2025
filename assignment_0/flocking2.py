from dataclasses import dataclass
import math

from vi import Agent, Config, Simulation
from pygame.math import Vector2


@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.3
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5  
    obstacle_avoidance_weight: float = 5


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

        # Enhanced obstacle avoidance to prevent phasing through blocks
        # Look ahead for potential collisions using a larger scale 
        future_collisions = list(self.obstacle_intersections(scale=1.5))
        # Check for immediate collisions with normal scale
        current_collisions = list(self.obstacle_intersections(scale=1.0))
        
        avoidance_vector = Vector2(0, 0)
        
        # Handle current collisions with higher priority
        for pos in current_collisions:
            pos_vec = Vector2(pos)
            away_vector = self.pos - pos_vec
            
            if away_vector.length() > 0:
                # Stronger avoidance for actual collisions
                avoidance_vector += away_vector.normalize() * 1.5
                # Add slight perpendicular component to help navigate around obstacles
                perp_vector = Vector2(-away_vector.y, away_vector.x).normalize()
                angle_with_move = self.move.angle_to(away_vector)
                # Choose the perpendicular direction that better matches current movement
                if abs(angle_with_move) < 90:
                    avoidance_vector += perp_vector * 0.5
                else:
                    avoidance_vector -= perp_vector * 0.5
        
        # Handle potential future collisions with lower priority
        if not current_collisions:
            for pos in future_collisions:
                pos_vec = Vector2(pos)
                away_vector = self.pos - pos_vec
                
                if away_vector.length() > 0:
                    # We're moving toward the obstacle if dot product is negative
                    heading_component = self.move.dot(away_vector.normalize())
                    if heading_component < 0:
                        avoidance_vector += away_vector.normalize()
        
        # Apply the avoidance vector with weight
        if avoidance_vector.length() > 0:
            self.move += avoidance_vector.normalize() * self.config.obstacle_avoidance_weight

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
            # fps_limit=0  # FPS for smoother animation   
        )
    )
    .batch_spawn_agents(80, FlockingAgent, images=["Assignment_0/images/triangle.png"])  
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 4, y // 4)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 2, y // 4)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", 3 * x // 4, y // 4)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 4, y // 2)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 2, y // 2)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", 3 * x // 4, y // 2)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 4, 3 * y // 4)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", x // 2, 3 * y // 4)
    .spawn_obstacle("/Users/janczarnecki/Desktop/pci/Assignment_0/images/Solid_red_small.png", 3 * x // 4, 3 * y // 4)
    .run()
)


"""
Obstacle Avoidance Logic Explained
The obstacle avoidance system uses a two-tier approach to prevent agents from colliding with or phasing through obstacles:

1. Collision Detection
Current Collisions: Uses obstacle_intersections(scale=1.0) to detect when agents are directly overlapping with obstacles
Future Collisions: Uses obstacle_intersections(scale=1.5) to detect potential collisions before they happen by checking a larger area around the agent
2. Priority-Based Response
Current Collisions (High Priority):

Applies a strong avoidance force (1.5x) directly away from the obstacle
Adds a perpendicular component to help the agent steer around the obstacle rather than bouncing back
The perpendicular direction is intelligently chosen based on the angle between the agent's movement and the avoidance vector
Future Collisions (Lower Priority):

Only considered if no current collisions exist
Uses dot product to determine if the agent is actually heading toward the obstacle
Only applies avoidance force when the agent is moving toward the obstacle
3. Force Application
The final avoidance vector is normalized and multiplied by the obstacle_avoidance_weight
This ensures consistent response strength regardless of how many obstacle points were detected
This comprehensive approach prevents phasing through objects by detecting collisions early and responding appropriately based on the specific collision scenario.

"""