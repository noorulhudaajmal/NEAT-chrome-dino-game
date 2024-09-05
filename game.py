import os
import pickle

import neat
import pygame
import random

from entities import Ground, Dino, Cactus, Bird
from constants import WIDTH, HEIGHT, WHITE, BLACK

# Initialize Pygame
pygame.init()

FONT = pygame.font.Font(None, 36)


class Game:
    clock_speed = 60

    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chrome Dino Game")

        # Load images
        self.ground_img, self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img, self.cactus_images, self.bird_image_1, self.bird_image_2 = self.load_images()

        self.ground = Ground(self.ground_img)
        self.dino = Dino(self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img)

        # Start with obstacles: Cacti and Bird
        self.obstacles = []
        previous_obstacle = None
        for _ in range(3):
            if random.random() <= 0.7:
                cactus = Cactus(self.cactus_images, previous_obstacle, self.obstacles)
                self.obstacles.append(cactus)
                previous_obstacle = cactus
            else:
                bird = Bird([self.bird_image_1, self.bird_image_2], previous_obstacle, self.obstacles)
                self.obstacles.append(bird)
                previous_obstacle = bird

        self.score = 0
        self.game_active = True
        self.clock = pygame.time.Clock()

        self.neural_net = self.load_best_genome("best_model.pkl")

    def load_best_genome(self, genome_file):
        """Load the best genome and create a neural network."""
        with open(genome_file, 'rb') as f:
            best_genome = pickle.load(f)

        config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        # neural network from the loaded genome
        neural_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        return neural_net

    def get_game_state(self, dino):
        nearest_obstacle = None
        for obstacle in self.obstacles:
            if obstacle.rect.right > dino.rect.left:
                nearest_obstacle = obstacle
                break

        if nearest_obstacle:
            return (
                dino.rect.bottom,  # Dino's Y position
                nearest_obstacle.rect.left,  # Obstacle X position
                nearest_obstacle.rect.bottom,  # Obstacle Y position
                nearest_obstacle.rect.width,  # Obstacle width
            )
        return (dino.rect.bottom, WIDTH, HEIGHT, 0)  # No obstacle

    def load_images(self):
        ground_image = pygame.image.load("assets/ground.png").convert_alpha()
        dino_run_1 = pygame.image.load("assets/dino_run1.png").convert_alpha()
        dino_run_2 = pygame.image.load("assets/dino_run2.png").convert_alpha()
        dino_duck_1 = pygame.image.load("assets/dino_duck1.png").convert_alpha()
        dino_duck_2 = pygame.image.load("assets/dino_duck2.png").convert_alpha()
        dino_dead = pygame.image.load("assets/dino_dead.png").convert_alpha()

        cactus_images = [
            pygame.image.load("assets/cactus_small.png").convert_alpha(),
            pygame.image.load("assets/cactus_big.png").convert_alpha(),
            pygame.image.load("assets/cactus_small_many.png").convert_alpha()
        ]

        bird_image_1 = pygame.image.load("assets/bird_1.png").convert_alpha()
        bird_image_2 = pygame.image.load("assets/bird_2.png").convert_alpha()

        return ground_image, dino_run_1, dino_run_2, dino_duck_1, dino_duck_2, dino_dead, cactus_images, bird_image_1, bird_image_2


    def display_score(self):
        score_text = pygame.font.Font(None, 36).render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_active = False
            if self.dino.dead and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
            elif not self.dino.dead:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.dino.jump()
                    if event.key == pygame.K_DOWN:
                        self.dino.duck(True)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.dino.duck(False)


    def handle_ai_events(self):
        game_state = self.get_game_state(self.dino)

        # Neural network processes the game state and outputs 3 values: jump, duck, do nothing
        action = self.neural_net.activate(game_state)

        if action[0] > 0.5:  # Jump
            self.dino.jump()
        elif action[1] > 0.5:  # Duck
            self.dino.duck(True)
        else:  # Do nothing or stop ducking
            self.dino.duck(False)

        #quitting the game using the pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_active = False

    def update(self):
        if not self.dino.dead:
            self.ground.move()
            self.dino.move()

            for obstacle in self.obstacles:
                obstacle.move()
                if obstacle.collides_with(self.dino):
                    self.dino.die()

            self.score += 0.1


    def draw(self):
        self.screen.fill(WHITE)
        self.ground.draw(self.screen)
        self.dino.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        self.display_score()
        if self.dino.dead:
            self.display_game_over_message()
        pygame.display.update()

    def display_game_over_message(self):
        game_over_text = pygame.font.Font(None, 48).render("Game Over! Press 'R' to Restart", True, BLACK)
        self.screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 50))

    def reset_game(self):
        self.dino = Dino(self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img)
        self.obstacles = []
        previous_obstacle = None
        for _ in range(3):
            if random.random() <= 0.7:
                cactus = Cactus(self.cactus_images, previous_obstacle, obstacles=self.obstacles)
                self.obstacles.append(cactus)
                previous_obstacle = cactus
            else:
                bird = Bird([self.bird_image_1, self.bird_image_2], previous_obstacle, obstacles=self.obstacles)
                self.obstacles.append(bird)
                previous_obstacle = bird
        self.score = 0

    def run(self):
        while self.game_active:
            self.handle_events()

            # Uncomment to let trained AI agent to play
            # self.handle_ai_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()

# Main function
def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()