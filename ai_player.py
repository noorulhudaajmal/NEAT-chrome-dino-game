import pickle
import time

import neat
import os
import pygame
import random

from entities import Ground, Dino, Cactus, Bird
from constants import WIDTH, HEIGHT, WHITE, BLACK



# Initialize Pygame
pygame.init()

FONT = pygame.font.Font(None, 36)


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chrome Dino Game")

        # Load images
        self.ground_img, self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img, self.cactus_images, self.bird_image_1, self.bird_image_2 = self.load_images()

        self.ground = Ground(self.ground_img)
        # self.dino = Dino(self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img)

        # Start with obstacles: Cacti and Bird
        self.obstacles = []
        previous_obstacle = None
        for _ in range(2):
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

    def load_images(self):
        ground_image = pygame.image.load("assets/ground.png").convert_alpha()
        dino_run_1 = pygame.image.load("assets/dino_run1.png").convert_alpha()
        dino_run_2 = pygame.image.load("assets/dino_run2.png").convert_alpha()
        dino_duck_1 = pygame.image.load("assets/dino_duck1.png").convert_alpha()
        dino_duck_2 = pygame.image.load("assets/dino_duck2.png").convert_alpha()
        dino_dead = pygame.image.load("assets/dino_dead.png").convert_alpha()

        # Load cactus images
        cactus_images = [
            pygame.image.load("assets/cactus_small.png").convert_alpha(),
            pygame.image.load("assets/cactus_big.png").convert_alpha(),
            pygame.image.load("assets/cactus_small_many.png").convert_alpha()
        ]

        bird_image_1 = pygame.image.load("assets/bird_1.png").convert_alpha()
        bird_image_2 = pygame.image.load("assets/bird_2.png").convert_alpha()

        return ground_image, dino_run_1, dino_run_2, dino_duck_1, dino_duck_2, dino_dead, cactus_images, bird_image_1, bird_image_2


    def display_score(self):
        score = 0
        for dino in self.dinos:
            if dino.score > score:
                score = dino.score
        score_text = pygame.font.Font(None, 36).render(f"Highest Score: {int(score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))


    def update(self):
        for dino in self.dinos:
            if not dino.dead:
                self.ground.move()
                dino.move()

                for obstacle in self.obstacles:
                    obstacle.move()
                    if obstacle.collides_with(dino):
                        dino.die()

                dino.increment_score()

    def draw(self):
        self.screen.fill(WHITE)
        self.ground.draw(self.screen)
        for dino in self.dinos:
            if not dino.dead:
                dino.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        self.display_score()
        pygame.display.update()


    def get_game_state(self, dino):
        # Return game state for NEAT AI to process
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


    def fitness_function(self, genomes, config):
        nets = []
        ge = []
        self.dinos = []
        generation = 0

        for genome_id, genome in genomes:
            genome.fitness = 0  #starting fitness at 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            self.dinos.append(Dino(self.dino_run_1_img, self.dino_run_2_img, self.dino_duck_1_img, self.dino_duck_2_img, self.dino_dead_img))
            ge.append(genome)

        run = True
        while run:
            self.update()
            self.draw()

            alive = len([dino for dino in self.dinos if not dino.dead])
            dead = len([dino for dino in self.dinos if dino.dead])
            max_fitness = max([genome.fitness for genome in ge])

            gen_text = pygame.font.Font(None, 36).render(f"Generation: {generation}", True, BLACK)
            alive_text = pygame.font.Font(None, 36).render(f"Alive: {alive}", True, BLACK)
            dead_text = pygame.font.Font(None, 36).render(f"Dead: {dead}", True, BLACK)
            fitness_text = pygame.font.Font(None, 36).render(f"Fitness: {max_fitness:.2f}", True, BLACK)

            self.screen.blit(gen_text, (10, 50))
            self.screen.blit(alive_text, (10, 90))
            self.screen.blit(dead_text, (10, 130))
            self.screen.blit(fitness_text, (10, 170))

            pygame.display.update()

            for i, dino in enumerate(self.dinos):
                #incrementing fitness based on score
                distance_traveled = dino.score
                ge[i].fitness += 0.2 * distance_traveled  # fitness proportionally to distance

                #Reward
                if not dino.dead:
                    ge[i].fitness += 1

                #penalty
                if dino.dead:
                    death_penalty = max(50 - dino.score, 0)
                    ge[i].fitness -= 2 + death_penalty * 0.1  #large penalty for early death
                    self.dinos.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    # continue
                else:
                    #retrieving game state for NEAT model and decide action
                    game_state = self.get_game_state(dino)
                    action = nets[i].activate(game_state)

                    #model outputs
                    if action[0] > 0.5:  # Jump
                        dino.jump()
                    elif action[1] > 0.5:  # Duck
                        dino.duck(True)
                    else:  #do nothing
                        dino.duck(False)

                    #reward
                    if dino.score >= 50:
                        ge[i].fitness += 50
                    if dino.score >= 100:
                        ge[i].fitness += 75
                    if dino.score >= 200:
                        ge[i].fitness += 100
                        self.dinos.pop(i)
                        nets.pop(i)
                        ge.pop(i)

            #all dinos are dead
            if len(self.dinos) == 0:
                run = False

            # FPS
            # self.clock.tick(60)

        generation += 1


    def run_neat(self, config_path):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self.fitness_function, 100)  #NEAT algorithm for 50 generations

        with open('best_model.pkl', 'wb') as f:
            pickle.dump(winner, f)
        print("Best genome saved to 'best_model.pkl'")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    game = Game()
    game.run_neat(config_path)
