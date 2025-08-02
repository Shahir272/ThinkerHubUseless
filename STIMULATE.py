import pygame
import random
import logging
from dataclasses import dataclass
import sys
from pygame.locals import *
import math

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pygame.init()

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60
GREEN, BLACK, GRAY, DARK_GRAY, WHITE, RED, BLUE, YELLOW, LIGHT_BLUE = (
    (153, 204, 0), (0, 0, 0), (128, 128, 128),
    (100, 100, 100), (255, 255, 255), (255, 0, 0),
    (0, 0, 255), (255, 255, 0), (173, 216, 230)
)
LANES = 5
LANE_HEIGHT = 40
LANE_SPACING = 15
LANE_VERTICAL_OFFSET = 400
LANE_Y_POSITIONS = [LANE_VERTICAL_OFFSET + i * (LANE_HEIGHT + LANE_SPACING) for i in range(LANES)]
# Add this with the other constants
PEDESTRIAN_COUNT = 3
PEDESTRIAN_CROSSING_HEIGHT = (LANE_Y_POSITIONS[-1] + LANE_HEIGHT) - LANE_Y_POSITIONS[0] + LANE_SPACING
PEDESTRIAN_CROSSING_Y_START = LANE_Y_POSITIONS[0] - LANE_SPACING

LANE_DIRECTIONS = ['right', 'left', 'right', 'left', 'right']

# Crosswalk constants
CROSSWALK_WIDTH = 100
CROSSWALK_Y_START = LANE_Y_POSITIONS[0] - LANE_SPACING
CROSSWALK_Y_END = LANE_Y_POSITIONS[-1] + LANE_HEIGHT + LANE_SPACING
CROSSWALK_X_POSITION = WIDTH / 2 - CROSSWALK_WIDTH / 2

# Speed control
SLOWDOWN_FACTOR = 0.2
SPEEDUP_FACTOR = 5.0
current_speed_mode = 1.0

# Button setup
SLOWDOWN_BUTTON_RECT = pygame.Rect(WIDTH - 250, 20, 120, 40)
SPEEDUP_BUTTON_RECT = pygame.Rect(WIDTH - 120, 20, 120, 40)

# Explosion constants
EXPLOSION_DURATION = 30
EXPLOSION_PARTICLES = 50

# Disintegration constants
DISINTEGRATION_PARTICLES = 100
DISINTEGRATION_LIFETIME = 60

# Rescue system
RESCUE_RADIUS = 60
PREDICTION_WINDOW = 2.0 # seconds

# --- New Shake Constants for Exponential Growth ---
BASE_SHAKE_INTENSITY = 5
SHAKE_GROWTH_FACTOR = 1.1 # Multiplier for exponential growth
MAX_SHAKE_INTENSITY = 30
BASE_SHAKE_DURATION = 30
SHAKE_DURATION_GROWTH_FACTOR = 5
MAX_SHAKE_DURATION = 120
shake_intensity = BASE_SHAKE_INTENSITY # Dynamic variable for current intensity

# Screen setup (to handle fullscreen properly)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Traffic Simulation with Rescue System")
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
SLOWDOWN_BUTTON_RECT = pygame.Rect(WIDTH - 250, 20, 120, 40)
SPEEDUP_BUTTON_RECT = pygame.Rect(WIDTH - 120, 20, 120, 40)

# Counters
cars_spawned = 0
crashes = 0
trains_spawned = 0
rescued_cars = 0
successful_rescues = 0
failed_rescues = 0
train_hit_count = 0
pedestrian_accidents = 0

# --- Automatic Respawn Timer Variables ---
respawn_pending = False
respawn_timer_start = 0
RESPAWN_DELAY = 3000 # 3 seconds in milliseconds

# --- Image Loading ---
try:
    car_image = pygame.image.load('car_top_view.png').convert_alpha()
    train_car_image = pygame.image.load('train_car_top_view.png').convert_alpha()
    pedestrian_image = pygame.image.load('image_2cf650.png').convert_alpha()
    rotated_car_image = pygame.transform.rotate(car_image, 90)
    rotated_train_car_image = pygame.transform.rotate(train_car_image, 90)
    counter_font = pygame.font.Font(None, 36)
except pygame.error as e:
    logging.error(f"Failed to load image: {e}")
    counter_font = pygame.font.SysFont('Consolas', 36)
    car_image = pygame.Surface((120, 120))
    train_car_image = pygame.Surface((120, 120))
    pedestrian_image = pygame.Surface((30, 30))
    car_image.fill(BLUE)
    train_car_image.fill(GRAY)
    pedestrian_image.fill(LIGHT_BLUE)
    rotated_car_image = pygame.transform.rotate(car_image, 90)
    rotated_train_car_image = pygame.transform.rotate(train_car_image, 90)

@dataclass
class Vehicle:
    x: int
    y: int
    speed: int
    direction: str
    lane: int
    width: int = 120
    height: int = 120
    original_image: pygame.Surface = None # Store the original image for color sampling

@dataclass
class Pedestrian:
    x: float
    y: float
    speed: int
    direction: str  # 'up' or 'down'
    size: int = 25
    is_crossing: bool = False
    crossing_progress: float = 0.0  # 0.0 to 1.0
    crossing_start_x: float = 0.0  # Where they started crossing
@dataclass
class Train:
    x: float
    y: float
    speed: int
    length: int = 4
    car_length: int = 120
    car_height: int = 50
    has_passed: bool = False
    is_entering: bool = False
    is_exiting: bool = False
    is_broken: bool = False

@dataclass
class Bogey:
    x: float
    y: float
    dx: float
    dy: float
    car_length: int
    car_height: int
    rotation: float = 0.0
    rotational_speed: float = 0.0
    lifetime: int = 200

@dataclass
class Explosion:
    x: int
    y: int
    particles: list
    duration: int = EXPLOSION_DURATION

# NEW: Dataclass for the car disintegration particles
@dataclass
class DisintegrationParticle:
    x: float
    y: float
    dx: float
    dy: float
    size: int
    color: tuple
    lifetime: int

# New dataclass for the reddish stain spot
@dataclass
class Stain:
    x: int
    y: int
    color: tuple
    lifetime: int = 120

@dataclass
class Cloud:
    x: float
    y: float
    speed: float
    image: pygame.Surface

# NEW: Dataclass to store car collision predictions
@dataclass
class CarPrediction:
    car: Vehicle
    will_collide: bool
    time_to_collision: float

cars = []
explosions = []
pedestrians = []
clouds = []
stains = []  # New list to hold stain objects
car_particles = []  # NEW: List to hold particles from disintegrated cars
train = Train(x=WIDTH / 2, y=HEIGHT + 200, speed=5)
train_passing = False
barrier_angle = 90
barrier_speed = 2
barrier_close_distance = 250
barrier_open_distance = 150
shake_duration = 0
shake_intensity = BASE_SHAKE_INTENSITY
car_predictions = []
bogies = []

def spawn_initial_cars(num_cars_per_lane=4):
    """Spawns an initial number of cars in each lane at the start of the simulation."""
    global cars_spawned, cars
    for lane in range(LANES):
        direction = LANE_DIRECTIONS[lane]
        y_pos = LANE_Y_POSITIONS[lane]
        for i in range(num_cars_per_lane):
            if direction == 'right':
                x_pos = random.randint(-400, -100) - (i * 150)
            else:
                x_pos = random.randint(WIDTH + 100, WIDTH + 400) + (i * 150)
            
            cars.append(Vehicle(
                x=x_pos,
                y=y_pos,
                speed=random.randint(2, 4),
                direction=direction,
                lane=lane,
                original_image=rotated_car_image if direction == 'left' else pygame.transform.flip(rotated_car_image, True, False)
            ))
            cars_spawned += 1

def spawn_pedestrian():
    """Spawns a new pedestrian at the side of the road."""
    global pedestrians
    
    # Choose which side to spawn on (left or right of road)
    if random.choice([True, False]):
        x = random.uniform(50, WIDTH//3)  # Left side
    else:
        x = random.uniform(WIDTH*2//3, WIDTH-50)  # Right side
    
    # Start at top or bottom of lanes
    if random.choice([True, False]):
        y = LANE_Y_POSITIONS[0] - 30  # Above top lane
        direction = 'down'
    else:
        y = LANE_Y_POSITIONS[-1] + LANE_HEIGHT + 30  # Below bottom lane
        direction = 'up'
    
    pedestrians.append(Pedestrian(
        x=x,
        y=y,
        speed=random.randint(1, 3),
        direction=direction,
        crossing_start_x=x  # Remember where they started
    ))

    
def respawn_train():
    """Resets the train to its initial state and position."""
    global train, train_passing, barrier_angle, train_hit_count, bogies, respawn_pending, shake_duration, shake_intensity
    
    train.x = WIDTH / 2
    train.y = HEIGHT + 200
    train.speed = 5
    train.has_passed = False
    train.is_entering = False
    train.is_exiting = False
    train.is_broken = False
    
    train_passing = False
    barrier_angle = 90
    train_hit_count = 0
    bogies = []
    respawn_pending = False
    shake_duration = 0
    shake_intensity = BASE_SHAKE_INTENSITY

def create_explosion(x, y):
    particles = []
    for _ in range(EXPLOSION_PARTICLES):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(1, 5)
        size = random.randint(2, 6)
        lifetime = random.randint(20, EXPLOSION_DURATION)
        particles.append({
            'x': x, 'y': y, 'dx': math.cos(angle) * speed,
            'dy': math.sin(angle) * speed, 'size': size,
            'color': (random.randint(200, 255), random.randint(100, 150), 0),
            'lifetime': lifetime
        })
    return Explosion(x=x, y=y, particles=particles)

# NEW: Function to create car disintegration particles
def create_disintegration(car):
    global car_particles
    car_rect = pygame.Rect(car.x, car.y, car.width, car.height)
    
    # Check for a valid image before sampling
    if car.original_image is None:
        logging.warning("Car image not found for disintegration, using default colors.")
        for _ in range(DISINTEGRATION_PARTICLES):
            car_particles.append(DisintegrationParticle(
                x=random.randint(car_rect.left, car_rect.right),
                y=random.randint(car_rect.top, car_rect.bottom),
                dx=random.uniform(-5, 5),
                dy=random.uniform(-5, 5),
                size=random.randint(2, 4),
                color=BLUE,
                lifetime=DISINTEGRATION_LIFETIME
            ))
    else:
        # Sample colors from the car image to create the particles
        for _ in range(DISINTEGRATION_PARTICLES):
            try:
                # Randomly select a pixel from the car image
                px = random.randint(0, car.width - 1)
                py = random.randint(0, car.height - 1)
                color = car.original_image.get_at((px, py))
            except (IndexError, pygame.error):
                # Fallback to a default color if sampling fails
                color = BLUE
            
            car_particles.append(DisintegrationParticle(
                x=random.randint(car_rect.left, car_rect.right),
                y=random.randint(car_rect.top, car_rect.bottom),
                dx=random.uniform(-5, 5),
                dy=random.uniform(-5, 5),
                size=random.randint(2, 4),
                color=color,
                lifetime=DISINTEGRATION_LIFETIME
            ))

def draw_explosions(surface):
    for explosion in explosions:
        for particle in explosion.particles:
            if particle['lifetime'] > 0:
                pygame.draw.circle(
                    surface, particle['color'],
                    (int(particle['x']), int(particle['y'])),
                    particle['size']
                )

# NEW: Function to draw the car particles
def draw_car_particles(surface):
    for particle in car_particles:
        if particle.lifetime > 0:
            # Fade the particles by adjusting alpha
            alpha = (particle.lifetime / DISINTEGRATION_LIFETIME) * 255
            color = (particle.color[0], particle.color[1], particle.color[2], int(alpha))
            s = pygame.Surface((particle.size, particle.size), pygame.SRCALPHA)
            s.fill(color)
            surface.blit(s, (particle.x, particle.y))

def update_explosions():
    global explosions
    explosions = [e for e in explosions if e.duration > 0]
    for explosion in explosions:
        explosion.duration -= 1
        for particle in explosion.particles:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            particle['dy'] += 0.1
            particle['lifetime'] -= 1

# NEW: Function to update the car particles
def update_car_particles():
    global car_particles
    car_particles = [p for p in car_particles if p.lifetime > 0]
    for particle in car_particles:
        particle.x += particle.dx * current_speed_mode
        particle.y += particle.dy * current_speed_mode
        particle.dy += 0.1 # Gravity effect
        particle.lifetime -= 1

# New function to draw the stains
def draw_stains(surface):
    """Draws the reddish stain spots on the road."""
    for stain in stains:
        if stain.lifetime > 0:
            pygame.draw.circle(surface, stain.color, (stain.x, stain.y), 15)

# New function to update the stains' lifetime
def update_stains():
    """Updates the lifetime of stains and removes them when expired."""
    global stains
    stains = [stain for stain in stains if stain.lifetime > 0]
    for stain in stains:
        stain.lifetime -= 1

def create_cloud_surface():
    """Creates a pygame.Surface with a cloud drawn on it."""
    width = random.randint(80, 200)
    height = random.randint(30, 60)
    cloud_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Draw a few overlapping ovals to make a cloud shape
    num_ovals = random.randint(4, 7)
    for _ in range(num_ovals):
        oval_w = random.randint(30, 70)
        oval_h = random.randint(20, 40)
        oval_x = random.randint(0, width - oval_w)
        oval_y = random.randint(0, height - oval_h)
        # Use a semi-transparent white for a softer look
        pygame.draw.ellipse(cloud_surface, (255, 255, 255, 180), (oval_x, oval_y, oval_w, oval_h))
        
    return cloud_surface

def spawn_cloud(on_screen=False):
    """Spawns a new cloud in the sky."""
    y = random.randint(5, int(HEIGHT * 0.2) - 70) # Stay within the skybox, minus cloud height
    speed = random.uniform(0.2, 0.6)
    cloud_surface = create_cloud_surface()
    
    x = random.randint(0, WIDTH) if on_screen else -cloud_surface.get_width()

    clouds.append(Cloud(x=x, y=y, speed=speed, image=cloud_surface))

def update_clouds():
    """Updates the position of clouds and spawns new ones."""
    global clouds
    for cloud in clouds:
        cloud.x += cloud.speed * current_speed_mode
    clouds = [cloud for cloud in clouds if cloud.x < WIDTH]
    if len(clouds) < 10 and random.random() < 0.01:
        spawn_cloud()

def draw_clouds(surface):
    """Draws the clouds in the sky."""
    for cloud in clouds:
        surface.blit(cloud.image, (cloud.x, cloud.y))

def draw_mountains(surface):
    """Draws a mountain range in the background."""
    sky_height = HEIGHT * 0.2
    
    # Far mountains (darker)
    mountain_color_1 = (90, 90, 100)
    points1 = [
        (0, sky_height),
        (WIDTH * 0.1, sky_height * 0.4),
        (WIDTH * 0.3, sky_height * 0.6),
        (WIDTH * 0.4, sky_height * 0.3),
        (WIDTH * 0.6, sky_height * 0.7),
        (WIDTH * 0.8, sky_height * 0.4),
        (WIDTH, sky_height * 0.6),
        (WIDTH, sky_height)
    ]
    pygame.draw.polygon(surface, mountain_color_1, points1)

    # Near mountains (lighter)
    mountain_color_2 = (120, 120, 130)
    points2 = [
        (0, sky_height),
        (WIDTH * 0.05, sky_height * 0.6),
        (WIDTH * 0.2, sky_height * 0.5),
        (WIDTH * 0.35, sky_height * 0.8),
        (WIDTH * 0.5, sky_height * 0.6),
        (WIDTH * 0.7, sky_height * 0.9),
        (WIDTH * 0.9, sky_height * 0.5),
        (WIDTH, sky_height)
    ]
    pygame.draw.polygon(surface, mountain_color_2, points2)

def draw_background(surface):
    surface.fill(GREEN)
    pygame.draw.rect(surface, LIGHT_BLUE, (0, 0, WIDTH, HEIGHT * 0.2))
    draw_mountains(surface)
    for y in LANE_Y_POSITIONS:
        pygame.draw.rect(surface, BLACK, (0, y, WIDTH, LANE_HEIGHT))
    
    pygame.draw.line(surface, DARK_GRAY, (WIDTH / 2 - 10, int(HEIGHT * 0.2)), (WIDTH / 2 - 10, HEIGHT), 5)
    pygame.draw.line(surface, DARK_GRAY, (WIDTH / 2 + 10, int(HEIGHT * 0.2)), (WIDTH / 2 + 10, HEIGHT), 5)
    for y in range(int(HEIGHT * 0.2), HEIGHT, 30):
        pygame.draw.rect(surface, BLACK, (WIDTH / 2 - 15, y, 30, 5))

def draw_controls(surface):
    slowdown_color = BLUE if current_speed_mode != SLOWDOWN_FACTOR else RED
    speedup_color = YELLOW if current_speed_mode != SPEEDUP_FACTOR else RED
    
    pygame.draw.rect(surface, slowdown_color, SLOWDOWN_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(surface, speedup_color, SPEEDUP_BUTTON_RECT, border_radius=10)
    
    font = pygame.font.SysFont('Arial', 20)
    slowdown_text = font.render("Slow Down", True, WHITE)
    surface.blit(slowdown_text, slowdown_text.get_rect(center=SLOWDOWN_BUTTON_RECT.center))
    speedup_text = font.render("Speed Up", True, BLACK)
    surface.blit(speedup_text, speedup_text.get_rect(center=SPEEDUP_BUTTON_RECT.center))

def draw_cars(surface):
    for car in cars:
        img = pygame.transform.scale(car.original_image, (car.height, car.width))
        if car.direction == 'left':
            img = pygame.transform.flip(img, True, False)
        
        for prediction in car_predictions:
            if prediction.car == car and prediction.will_collide:
                pygame.draw.rect(surface, RED, (car.x-2, car.y-2, car.width+4, car.height+4), 2)
                break
                
        surface.blit(img, (car.x, car.y))

def draw_pedestrians(surface):
    """Draws the pedestrians on the screen."""
    for person in pedestrians:
        img = pygame.transform.scale(pedestrian_image, (person.size, person.size))
        surface.blit(img, (person.x, person.y))

def draw_train(surface):
    rects = []
    if not train.is_broken:
        for i in range(train.length):
            offset_y = (i * (train.car_length + 20))
            car_x = train.x - train.car_height // 2
            car_y = train.y + offset_y - train.car_length // 2
            train_img = pygame.transform.scale(rotated_train_car_image, (train.car_height, train.car_length))
            rect = train_img.get_rect(center=(int(car_x + train.car_height // 2), int(car_y + train.car_length // 2)))
            rects.append(rect)
            surface.blit(train_img, rect)
    return rects

def draw_bogies(surface):
    """Draws the broken train cars (bogies) and returns their rects for collision detection."""
    bogey_rects = []
    for bogey in bogies:
        bogey_img = pygame.transform.scale(rotated_train_car_image, (bogey.car_height, bogey.car_length))
        rotated_img = pygame.transform.rotate(bogey_img, bogey.rotation)
        new_rect = rotated_img.get_rect(center=(int(bogey.x), int(bogey.y)))
        surface.blit(rotated_img, new_rect)
        bogey_rects.append(new_rect)
    return bogey_rects

def predict_collisions():
    global car_predictions
    car_predictions = []
    
    if train.is_broken:
        return
        
    crossing_x = WIDTH / 2
    crossing_y = HEIGHT / 2

    train_time_to_crossing = (train.y - crossing_y) / (train.speed * current_speed_mode * FPS) if (train.speed * current_speed_mode) > 0 else float('inf')

    for car in cars:
        will_collide = False
        time_to_collision = float('inf')
        
        if car.direction == 'right' and (car.x + car.width) < crossing_x:
            car_time_to_crossing = (crossing_x - (car.x + car.width)) / (car.speed * current_speed_mode * FPS)
        elif car.direction == 'left' and car.x > crossing_x:
            car_time_to_crossing = (car.x - crossing_x) / (car.speed * current_speed_mode * FPS)
        else:
            car_time_to_crossing = float('inf')

        if abs(car_time_to_crossing - train_time_to_crossing) < PREDICTION_WINDOW:
            if car_time_to_crossing > 0 and train_time_to_crossing > 0:
                will_collide = True
                time_to_collision = min(car_time_to_crossing, train_time_to_crossing)

        car_predictions.append(CarPrediction(
            car=car,
            will_collide=will_collide,
            time_to_collision=time_to_collision
        ))

def draw_counter(surface):
    counter_height = 80
    counter_width = 800
    counter_bg_rect = pygame.Rect((WIDTH - counter_width) // 2, HEIGHT - counter_height - 10,
                                  counter_width, counter_height)
    pygame.draw.rect(surface, BLACK, counter_bg_rect, border_radius=10)
    
    row1_y = counter_bg_rect.top + 15
    row2_y = counter_bg_rect.top + 45
    
    cars_text = counter_font.render(f"Cars: {cars_spawned}", True, WHITE)
    crashes_text = counter_font.render(f"Crashes: {crashes}", True, RED)
    trains_text = counter_font.render(f"Trains: {trains_spawned}", True, WHITE)
    
    rescued_text = counter_font.render(f"Rescued: {rescued_cars}", True, GREEN)
    failed_text = counter_font.render(f"Failed: {failed_rescues}", True, RED)

    train_hits_text = counter_font.render(f"Train Hits: {train_hit_count}", True, YELLOW)
    ped_accidents_text = counter_font.render(f"Pedestrian Accidents: {pedestrian_accidents}", True, YELLOW)
    
    surface.blit(cars_text, (counter_bg_rect.left + 20, row1_y))
    surface.blit(crashes_text, (counter_bg_rect.centerx - crashes_text.get_width()//2, row1_y))
    surface.blit(trains_text, (counter_bg_rect.right - trains_text.get_width() - 20, row1_y))
    
    surface.blit(rescued_text, (counter_bg_rect.left + 20, row2_y))
    surface.blit(train_hits_text, (counter_bg_rect.centerx - train_hits_text.get_width()//2, row2_y))
    surface.blit(ped_accidents_text, (counter_bg_rect.right - ped_accidents_text.get_width() - 20, row2_y))


def update_traffic():
    global cars, cars_spawned
    crossing_x_start = WIDTH / 2 - 50
    crossing_x_end = WIDTH / 2 + 50
    updated = []
    
    lane_counts = {lane: 0 for lane in range(LANES)}
    for car in cars:
        lane_counts[car.lane] += 1
    
    for car in cars:
        can_move = True
        if train_passing and barrier_angle > 45:
            if car.direction == 'right' and car.x + car.width > crossing_x_start and car.x < crossing_x_end:
                can_move = False
            elif car.direction == 'left' and car.x < crossing_x_end and car.x + car.width > crossing_x_start:
                can_move = False
        
        for other in cars:
            if other == car: continue
            if car.lane == other.lane:
                safe_distance = car.width * 1.5
                if car.direction == 'right' and other.x > car.x and other.x - car.x < safe_distance:
                    can_move = False
                elif car.direction == 'left' and other.x < car.x and car.x - other.x < safe_distance:
                    can_move = False
        
        if can_move:
            car.x += (car.speed if car.direction == 'right' else -car.speed) * current_speed_mode
        updated.append(car)
    cars = [car for car in updated if -200 < car.x < WIDTH + 200]
    
    if current_speed_mode == SPEEDUP_FACTOR:
        for lane in range(LANES):
            if lane_counts[lane] < 7 and random.random() < 0.05:
                if LANE_DIRECTIONS[lane] == 'right':
                    cars.append(Vehicle(
                        x=random.randint(-300, -100),
                        y=LANE_Y_POSITIONS[lane],
                        speed=random.randint(3, 5),
                        direction='right',
                        lane=lane,
                        original_image=rotated_car_image
                    ))
                else:
                    cars.append(Vehicle(
                        x=WIDTH + random.randint(100, 300),
                        y=LANE_Y_POSITIONS[lane],
                        speed=random.randint(3, 5),
                        direction='left',
                        lane=lane,
                        original_image=pygame.transform.flip(rotated_car_image, True, False)
                    ))
                cars_spawned += 1
    else:
        if random.randint(1, 100) == 1:
            available_right_lanes = [lane for lane in [0, 2, 4] if lane_counts[lane] < 5]
            if available_right_lanes:
                lane = random.choice(available_right_lanes)
                cars.append(Vehicle(
                    x=0, y=LANE_Y_POSITIONS[lane],
                    speed=random.randint(2, 4),
                    direction='right', lane=lane,
                    original_image=rotated_car_image
                ))
                cars_spawned += 1
        
        if random.randint(1, 100) == 1:
            available_left_lanes = [lane for lane in [1, 3] if lane_counts[lane] < 5]
            if available_left_lanes:
                lane = random.choice(available_left_lanes)
                cars.append(Vehicle(
                    x=WIDTH, y=LANE_Y_POSITIONS[lane],
                    speed=random.randint(2, 4),
                    direction='left', lane=lane,
                    original_image=pygame.transform.flip(rotated_car_image, True, False)
                ))
                cars_spawned += 1


def update_pedestrians():
    global pedestrians, pedestrian_accidents
    updated_pedestrians = []
    
    for person in pedestrians:
        # Move horizontally across all lanes
        if person.direction == 'right':
            person.x += person.speed * current_speed_mode
        else:
            person.x -= person.speed * current_speed_mode
        
        # Keep pedestrians that are still on screen
        if -50 < person.x < WIDTH + 50:
            updated_pedestrians.append(person)
    
    pedestrians = updated_pedestrians

    # Maintain exactly 3 pedestrians crossing at all times
    while len(pedestrians) < PEDESTRIAN_COUNT:
        spawn_pedestrian()
def update_train():
    global train, train_passing, trains_spawned, barrier_angle, train_hit_count
    
    if train.is_broken:
        return

    if train.y > HEIGHT + 150 and not train.has_passed:
        trains_spawned += 1
        train.has_passed = True

    train.y -= train.speed * current_speed_mode
    dist = abs(train.y - HEIGHT / 2)
    if dist < barrier_close_distance and not train_passing:
        train_passing = True
        train.is_entering = True
    
    tail_y = train.y + (train.length * (train.car_length + 20))
    if abs(tail_y - HEIGHT / 2) > barrier_open_distance and train_passing and train.is_entering:
        train.is_entering = False
        train.is_exiting = True
    
    if train.y < HEIGHT * 0.2:
        respawn_train()
    
    if train_passing and not train.is_exiting and barrier_angle < 90:
        barrier_angle = min(barrier_angle + barrier_speed * current_speed_mode, 90)
    elif not train_passing and barrier_angle > 0:
        barrier_angle = max(barrier_angle - barrier_speed * current_speed_mode, 0)

def update_bogies():
    global bogies
    updated_bogies = []
    for bogey in bogies:
        bogey.x += bogey.dx * current_speed_mode
        bogey.y += bogey.dy * current_speed_mode
        bogey.rotation += bogey.rotational_speed * current_speed_mode
        bogey.lifetime -= 1
        if bogey.lifetime > 0 and -200 < bogey.x < WIDTH + 200 and -200 < bogey.y < HEIGHT + 200:
            updated_bogies.append(bogey)
    bogies = updated_bogies

def check_for_collision(train_rects):
    global shake_duration, cars, explosions, crashes, train, bogies, train_hit_count, respawn_pending, respawn_timer_start, shake_intensity
    
    if train.is_broken:
        return

    cars_to_remove = []
    
    for car in cars:
        car_rect = pygame.Rect(car.x, car.y, car.width, car.height)
        for i, t_rect in enumerate(train_rects):
            if car_rect.colliderect(t_rect):
                logging.info(f"Collision detected between a car and train car {i+1}!")
                
                train_hit_count += 1
                
                if train_hit_count >= 3:
                    train.is_broken = True
                    respawn_pending = True
                    respawn_timer_start = pygame.time.get_ticks()

                    for j in range(train.length):
                        offset_y = (j * (train.car_length + 20))
                        bogey_x = train.x
                        bogey_y = train.y + offset_y
                        
                        angle = random.uniform(0, math.pi * 2)
                        speed = random.uniform(5, 10)
                        rot_speed = random.uniform(-10, 10)
                        
                        bogies.append(Bogey(
                            x=bogey_x,
                            y=bogey_y,
                            dx=math.cos(angle) * speed,
                            dy=math.sin(angle) * speed,
                            car_length=train.car_length,
                            car_height=train.car_height,
                            rotational_speed=rot_speed
                        ))
                
                # --- NEW LOGIC: Trigger disintegration and explosion ---
                create_disintegration(car)
                explosion_x = (car.x + car.width // 2 + t_rect.x + t_rect.width // 2) // 2
                explosion_y = (car.y + car.height // 2 + t_rect.y + t_rect.height // 2) // 2
                explosions.append(create_explosion(explosion_x, explosion_y))
                
                # --- NEW LOGIC: Exponentially increase shake intensity and duration ---
                crashes += 1
                shake_intensity = min(BASE_SHAKE_INTENSITY * (SHAKE_GROWTH_FACTOR ** crashes), MAX_SHAKE_INTENSITY)
                shake_duration = min(BASE_SHAKE_DURATION + (crashes * SHAKE_DURATION_GROWTH_FACTOR), MAX_SHAKE_DURATION)
                
                cars_to_remove.append(car)
                break
        if train.is_broken:
            break
    
    for car in cars_to_remove:
        if car in cars:
            cars.remove(car)

def check_for_pedestrian_collision():
    """
    Checks for collisions between pedestrians and cars/trains across all lanes.
    """
    global pedestrians, cars, train, pedestrian_accidents, stains, shake_duration, shake_intensity, crashes
    
    pedestrians_to_remove = []
    
    for person in pedestrians:
        person_rect = pygame.Rect(person.x, person.y, person.size, person.size)
        
        # Check collision with cars
        for car in cars:
            car_rect = pygame.Rect(car.x, car.y, car.width, car.height)
            if person_rect.colliderect(car_rect):
                logging.info("Pedestrian hit by a car!")
                pedestrians_to_remove.append(person)
                pedestrian_accidents += 1
                crashes += 1
                shake_intensity = min(BASE_SHAKE_INTENSITY * (SHAKE_GROWTH_FACTOR ** crashes), MAX_SHAKE_INTENSITY)
                shake_duration = min(BASE_SHAKE_DURATION + (crashes * SHAKE_DURATION_GROWTH_FACTOR), MAX_SHAKE_DURATION)
                stains.append(Stain(int(person.x), int(person.y), (200, 50, 50), 120))
                break
        
        # Check collision with train
        if not train.is_broken:
            train_car_width = train.car_height
            train_car_height = train.car_length
            
            for i in range(train.length):
                train_car_x = train.x - train_car_width // 2
                train_car_y = train.y + (i * (train_car_height + 20)) - train_car_height // 2
                train_rect = pygame.Rect(train_car_x, train_car_y, train_car_width, train_car_height)
                
                if person_rect.colliderect(train_rect):
                    logging.info("Pedestrian hit by the train!")
                    pedestrians_to_remove.append(person)
                    pedestrian_accidents += 1
                    crashes += 1
                    shake_intensity = min(BASE_SHAKE_INTENSITY * (SHAKE_GROWTH_FACTOR ** crashes), MAX_SHAKE_INTENSITY)
                    shake_duration = min(BASE_SHAKE_DURATION + (crashes * SHAKE_DURATION_GROWTH_FACTOR), MAX_SHAKE_DURATION)
                    stains.append(Stain(int(person.x), int(person.y), (200, 50, 50), 120))
                    break

    pedestrians[:] = [p for p in pedestrians if p not in pedestrians_to_remove]
def check_for_bogey_collision(bogey_rects):
    """
    Checks for collisions between the broken train bogies and cars,
    triggering an explosion and removing the colliding car and bogey.
    """
    global cars, bogies, crashes, explosions, shake_duration, shake_intensity
    
    cars_to_remove = []
    bogies_to_remove = []

    for car in cars:
        car_rect = pygame.Rect(car.x, car.y, car.width, car.height)
        
        for i, bogey_rect in enumerate(bogey_rects):
            if car_rect.colliderect(bogey_rect):
                logging.info(f"Collision detected between a car and a broken train bogey!")
                
                # --- NEW LOGIC: Trigger disintegration and explosion ---
                create_disintegration(car)
                explosion_x = (car.x + car.width // 2 + bogey_rect.x + bogey_rect.width // 2) // 2
                explosion_y = (car.y + car.height // 2 + bogey_rect.y + bogey_rect.height // 2) // 2
                explosions.append(create_explosion(explosion_x, explosion_y))
                
                cars_to_remove.append(car)
                bogies_to_remove.append(bogies[i])
                
                crashes += 1
                
                # --- NEW LOGIC: Exponentially increase shake intensity and duration ---
                shake_intensity = min(BASE_SHAKE_INTENSITY * (SHAKE_GROWTH_FACTOR ** crashes), MAX_SHAKE_INTENSITY)
                shake_duration = min(BASE_SHAKE_DURATION + (crashes * SHAKE_DURATION_GROWTH_FACTOR), MAX_SHAKE_DURATION)
                
                break
    
    cars[:] = [car for car in cars if car not in cars_to_remove]
    bogies[:] = [bogey for bogey in bogies if bogey not in bogies_to_remove]

def draw_level_crossing(surface):
    pygame.draw.circle(surface, DARK_GRAY, (int(WIDTH/2), int(HEIGHT/2)), 10)

# --- Main Loop ---
clock = pygame.time.Clock()
temp_surface = pygame.Surface((WIDTH, HEIGHT))
running = True

spawn_initial_cars()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                
                if SLOWDOWN_BUTTON_RECT.collidepoint(mouse_pos):
                    current_speed_mode = SLOWDOWN_FACTOR
                elif SPEEDUP_BUTTON_RECT.collidepoint(mouse_pos):
                    current_speed_mode = SPEEDUP_FACTOR
                else:
                    clicked_car = None
                    for car in reversed(cars):
                        car_rect = pygame.Rect(car.x, car.y, car.width, car.height)
                        if car_rect.collidepoint(mouse_pos):
                            clicked_car = car
                            break
                    
                    if clicked_car:
                        prediction = next((p for p in car_predictions if p.car == clicked_car), None)
                        
                        if prediction and prediction.will_collide:
                            successful_rescues += 1
                            rescued_cars += 1
                            if clicked_car in cars:
                                cars.remove(clicked_car)
                        else:
                            failed_rescues += 1
                    else:
                        current_speed_mode = 1.0

    if respawn_pending and pygame.time.get_ticks() - respawn_timer_start > RESPAWN_DELAY:
        respawn_train()

    update_traffic()
    update_pedestrians()
    update_train()
    update_bogies()
    predict_collisions()
    update_explosions()
    update_stains()
    update_car_particles() # NEW: Update the car particles
    
    draw_background(temp_surface)
    draw_stains(temp_surface)
    draw_cars(temp_surface)
    draw_pedestrians(temp_surface)
    draw_car_particles(temp_surface) # NEW: Draw the car particles
    train_rects = draw_train(temp_surface)
    bogey_rects = draw_bogies(temp_surface)
    draw_level_crossing(temp_surface)
    draw_explosions(temp_surface)
    draw_controls(temp_surface)
    draw_counter(temp_surface)
    
    check_for_collision(train_rects)
    check_for_bogey_collision(bogey_rects)
    check_for_pedestrian_collision()
    
    if shake_duration > 0:
        offset_x = random.randint(-int(shake_intensity), int(shake_intensity))
        offset_y = random.randint(-int(shake_intensity), int(shake_intensity))
        screen.blit(temp_surface, (offset_x, offset_y))
        shake_duration -= 1
    else:
        screen.blit(temp_surface, (0, 0))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()