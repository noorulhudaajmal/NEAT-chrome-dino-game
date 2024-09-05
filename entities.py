import random
from constants import GROUND_HEIGHT, SPEED, DINO_OFFSET, RUN_ANIMATION_TIME, WIDTH, CACTUS_MIN_DISTANCE


class Ground:
    def __init__(self, image):
        self.image = image
        self.width = self.image.get_width()
        self.rect1 = self.image.get_rect(midbottom=(0, GROUND_HEIGHT))
        self.rect2 = self.image.get_rect(midbottom=(self.width, GROUND_HEIGHT))
        self.current_x = 0

    def move(self):
        self.rect1.x -= SPEED
        self.rect2.x -= SPEED

        #if the first ground image moves off-screen, reset its position
        if self.rect1.right <= 0:
            self.rect1.left = self.rect2.right
            self.current_x = self.rect1.left

        #if the second ground image moves off-screen, reset its position
        if self.rect2.right <= 0:
            self.rect2.left = self.rect1.right
            self.current_x = self.rect2.left

    def draw(self, screen):
        screen.blit(self.image, self.rect1)
        screen.blit(self.image, self.rect2)


class Dino:
    def __init__(self, run_1_image, run_2_image, duck_1_image, duck_2_image, dead_image):
        self.run_1_image = run_1_image
        self.run_2_image = run_2_image
        self.duck_1_image = duck_1_image
        self.duck_2_image = duck_2_image
        self.dead_image = dead_image
        self.rect = self.run_1_image.get_rect(midbottom=(100, GROUND_HEIGHT - DINO_OFFSET))
        self.jump_speed = -20
        self.double_jump_speed = -25
        self.low_gravity = 0.8
        self.gravity = 1
        self.high_gravity = 0.8
        self.velocity = 0
        self.ducking = False
        self.dead = False
        self.run_time = 0
        self.score = 0

    def increment_score(self):
        self.score += 0.1

    def jump(self):
        if self.rect.bottom >= GROUND_HEIGHT - DINO_OFFSET:
            self.velocity = self.jump_speed
            self.gravity = self.high_gravity

    def double_jump(self):
        if self.rect.bottom >= GROUND_HEIGHT - DINO_OFFSET:
            self.velocity = self.double_jump_speed
            self.gravity = self.low_gravity

    def duck(self, ducking):
        self.ducking = ducking

    def move(self):
        if not self.dead:
            self.velocity += self.gravity
            self.rect.y += self.velocity
            if self.rect.bottom >= GROUND_HEIGHT - DINO_OFFSET:
                self.rect.bottom = GROUND_HEIGHT - DINO_OFFSET

    def die(self):
        self.dead = True

    def get_image(self):
        if self.dead:
            return self.dead_image
        elif self.ducking and self.rect.bottom == GROUND_HEIGHT - DINO_OFFSET:
            self.rect.bottom = GROUND_HEIGHT + 20
            self.run_time += 1
            if self.run_time // RUN_ANIMATION_TIME % 2 == 0:
                return self.duck_1_image
            else:
                return self.duck_2_image
            # return self.duck_image
        else:
            # Alternate imgs between run frames
            self.run_time += 1
            if self.run_time // RUN_ANIMATION_TIME % 2 == 0:
                return self.run_1_image
            else:
                return self.run_2_image

    def draw(self, screen):
        screen.blit(self.get_image(), self.rect)


class Obstacle:
    def __init__(self, images, obstacle_type, previous_obstacle=None, obstacles=None):
        self.images = images
        self.obstacle_type = obstacle_type
        self.image = random.choice(self.images)
        self.rect = self.image.get_rect()
        self.previous_obstacle = previous_obstacle
        self.obstacles = obstacles
        self.spawn_new_obstacle()

    def spawn_new_obstacle(self):
        self.image = random.choice(self.images)

        if self.obstacles:
            last_obstacle = max(self.obstacles, key=lambda obj: obj.rect.left)
            min_x = last_obstacle.rect.left + random.randint(CACTUS_MIN_DISTANCE, CACTUS_MIN_DISTANCE + 500)
            if self.obstacle_type == "cactus":
                self.rect = self.image.get_rect(midbottom=(random.randint(min_x, min_x + 200), GROUND_HEIGHT))
            elif self.obstacle_type == "bird":
                posY = random.choice([30, 100, 180])
                self.flying_level = 'Low' if posY==30 else 'Mid' if posY==100 else 'High'
                self.rect = self.image.get_rect(midbottom=(random.randint(min_x, min_x + 200), GROUND_HEIGHT - posY)) #[GROUND_HEIGHT - 30, GROUND_HEIGHT - 100, GROUND_HEIGHT - 180]

        else:
            if self.obstacle_type == "cactus":
                self.rect = self.image.get_rect(midbottom=(random.randint(WIDTH + 100, WIDTH + 300), GROUND_HEIGHT))
            elif self.obstacle_type == "bird":
                posY = random.choice([30, 100, 180])
                self.flying_level = 'Low' if posY==30 else 'Mid' if posY==100 else 'High'
                self.rect = self.image.get_rect(midbottom=(random.randint(WIDTH + 100, WIDTH + 300), GROUND_HEIGHT - posY))

    def move(self):
        self.rect.x -= SPEED
        if self.rect.right <= 0:
            self.spawn_new_obstacle()

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def collides_with(self, dino):
        if self.obstacle_type == "bird":
            if self.flying_level == "Mid" and dino.ducking:
                return False

        return self.rect.colliderect(dino.rect)


class Cactus(Obstacle):
    def __init__(self, images, previous_obstacle=None, obstacles=None):
        super().__init__(images, "cactus", previous_obstacle, obstacles)


class Bird(Obstacle):
    def __init__(self, images, previous_obstacle=None, obstacles=None):
        self.flap_counter = 0
        self.flying_level = None
        super().__init__(images, "bird", previous_obstacle, obstacles)

    def move(self):
        self.rect.x -= SPEED
        self.flap_counter += 1
        if self.flap_counter >= 10:  #alternating bird image for flapping
            self.flap_counter = 0
            self.image = self.images[0] if self.image == self.images[1] else self.images[1]

        if self.rect.right <= 0:
            self.spawn_new_obstacle()