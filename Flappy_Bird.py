import pygame
from pygame.locals import *
import random
import torch

class CrappyBirdGame:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.fps = 30

        # Screen dimensions
        self.screen_width = 551
        self.screen_height = 551
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Crappy Bird')

        # Font and colors
        self.font = pygame.font.SysFont('Bauhaus 93', 60)
        self.white = (255, 255, 255)

        # Images
        self.bg = pygame.image.load('assets/background.png')
        self.ground = pygame.image.load('assets/ground.png')
        self.button_img = pygame.image.load('assets/game_over.png')
        self.bird_imgs = [pygame.image.load(f'assets/bird{num}.png') for num in range(1, 4)]
        self.pipe_img = pygame.image.load('assets/pipe_bottom.png')

        # Game variables
        self.scroll_speed = 4
        self.game_over = False
        self.pipe_gap = 150
        self.pipe_dist_x = 150
        self.score = 0
        self.reward = 0
        self.pass_pipe = False
        self.penalty_applied = False
        self.rendering = True
        self.tick_limited = True
        self.flying = True 
        self.tal = 0 

        # Bird properties
        self.bird_index = 0
        self.bird_counter = 0
        self.bird_rect = self.bird_imgs[0].get_rect(center=(100, self.screen_height // 2))
        self.bird_vel = 0
        self.bird_clicked = False

        # Pipes
        self.pipes = []

        # Button
        self.button_rect = self.button_img.get_rect(center=(self.screen_width // 2, self.screen_height // 2))

        # Define the available actions
        self.actions = [0, 1]  # Do nothing, flap

    def draw_text(self, text, x, y):
        img = self.font.render(text, True, self.white)
        self.screen.blit(img, (x, y))

    def reset_game(self):
        self.pipes.clear()
        self.bird_rect.center = (100, self.screen_height // 2)
        self.bird_vel = 0
        self.flying = True
        self.game_over = False
        self.score = 0
        self.reward = 0
        self.penalty_applied = False

        return self.get_states()

    def get_states(self):
        distance, gap_center = self.get_distance_to_next_pipe()
        return self.bird_rect.y, distance, gap_center # Bird y, distance to next pipe, gap center
        
    def get_distance_to_next_pipe(self):
        for pipe in self.pipes:
            if pipe["rect"].centerx > self.bird_rect.centerx:
                distance = pipe["rect"].left - self.bird_rect.right
                gap_center = pipe["rect"].top - (self.pipe_gap / 2)
                return distance, gap_center
        return 0, 0  # No pipes in front

    def draw_pipes(self):
        for pipe in self.pipes:
            if pipe["pos"] == -1:
                self.screen.blit(self.pipe_img, pipe["rect"])
            else:
                flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
                self.screen.blit(flipped_pipe, pipe["rect"])

            # Draw a red circle at the gap height
            # if pipe["pos"] == -1:  # Bottom pipe
            #     gap_center = pipe["rect"].top - self.pipe_gap / 2
            #     pygame.draw.circle(self.screen, (255, 0, 0), (pipe["rect"].centerx, int(gap_center)), 10)

    def update_pipes(self):
        if not self.game_over:
            for pipe in self.pipes:
                pipe["rect"].x -= self.scroll_speed

            # Remove pipes that go off-screen
            self.pipes = [pipe for pipe in self.pipes if pipe["rect"].right > 0]

    def check_collision(self):
        #print(self.bird_rect.bottom)
        for pipe in self.pipes:
            if self.bird_rect.colliderect(pipe["rect"]):
                return True
        return self.bird_rect.bottom >= 520 or self.bird_rect.top <= 1

    def handle_bird_movement(self):
        if self.flying:
            self.bird_vel += 0.5
            if self.bird_vel > 8:
                self.bird_vel = 8
            if self.bird_rect.bottom < 520:
                self.bird_rect.y += int(self.bird_vel)
            if self.bird_rect.top < 0:
                self.bird_rect.top = 0

        if pygame.mouse.get_pressed()[0] == 1 and not self.bird_clicked:
            self.bird_vel = -10
            self.bird_clicked = True
        if pygame.mouse.get_pressed()[0] == 0:
            self.bird_clicked = False

        # Animate bird
        self.bird_counter += 1
        if self.bird_counter > 5:
            self.bird_counter = 0
            self.bird_index = (self.bird_index + 1) % len(self.bird_imgs)

    def spawn_pipes(self):
        if not self.pipes or self.pipes[-1]["rect"].left < self.pipe_dist_x:
            # Clamp pipe height to stay within screen limits
            pipe_height = random.randint(-150, 150)  # Adjust range as needed for balance
            btm_pipe = {"rect": self.pipe_img.get_rect(topleft=(self.screen_width, int(self.screen_height / 2) + pipe_height + self.pipe_gap / 2)), "pos": -1}
            top_pipe = {"rect": self.pipe_img.get_rect(bottomleft=(self.screen_width, int(self.screen_height / 2) + pipe_height - self.pipe_gap / 2)), "pos": 1}
            self.pipes.append(btm_pipe)
            self.pipes.append(top_pipe)

    def step(self, action):
        self.done = False
        self.reward = 0

        # Apply the action
        if action == 1 and not self.game_over:  # Flap
            self.bird_vel = -10
        elif action == 0 and not self.game_over:  # Do nothing
            pass

        # Update the environment
        if not self.game_over and self.flying:
            self.reward += 0.5  # Small reward for staying alive
            self.spawn_pipes()
            self.update_pipes()
            self.handle_bird_movement()


            # Check for collisions
            #print(self.check_collision())
            if self.check_collision():
                self.reward -= 50  # Penalty for dying
                self.game_over = True
                self.done = True

            # Update score and rewards
            for pipe in self.pipes:
                if not self.pass_pipe and pipe["rect"].right < self.bird_rect.left:
                    self.score += 1
                    self.reward += 10
                    self.pass_pipe = True

            if self.pass_pipe and self.pipes[0]["rect"].left > self.bird_rect.left:
                self.pass_pipe = False

        # Get the current state
        next_state = self.get_states()

        #     # Debugging statements
        # print(f"Action: {action}")
        # print(f"Next State: {next_state}")
        # print(f"Reward: {self.reward}")
        # print(f"Done: {self.done}")

        return next_state, self.reward, self.done
    
    def run(self):
        running = True
        while running:
            self.tal += 1
            print(self.tal)
            if self.game_over:
                raise Exception("Game over")
            
            if self.tick_limited:
                self.clock.tick(self.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and not self.flying and not self.game_over:
                    self.flying = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    if event.key == pygame.K_t:
                        self.tick_limited = not self.tick_limited
                    if event.key == pygame.K_f:
                        self.rendering = not self.rendering

            if not self.game_over and self.flying == True:
                self.reward += 0.5
                self.spawn_pipes()
                self.update_pipes()
                self.handle_bird_movement()

                
                if self.check_collision():
                    self.reward -= 50  # Apply penalty immediately when the bird dies
                    self.game_over = True

                print(self.get_states(), self.reward, self.game_over)
        
                # Update score and reward
                for pipe in self.pipes:
                    if not self.pass_pipe and pipe["rect"].right < self.bird_rect.left:
                        self.score += 1
                        self.reward += 10
                        self.pass_pipe = True

                if self.pass_pipe and self.pipes[0]["rect"].left > self.bird_rect.left:
                    self.pass_pipe = False


            if self.rendering:
                self.screen.blit(self.bg, (0, 0))
                self.draw_pipes()
                self.screen.blit(self.bird_imgs[self.bird_index], self.bird_rect)
                self.screen.blit(self.ground, (0, 520))
                self.draw_text(str(self.score), self.screen_width // 2, 30)

                if self.game_over:
                    self.screen.blit(self.button_img, self.button_rect.topleft)
                    
                    if pygame.mouse.get_pressed()[0] == 1 and self.button_rect.collidepoint(pygame.mouse.get_pos()):
                        self.reset_game()

                pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    game = CrappyBirdGame()
    game.run()


#fiks run
#fiks step
#ikke kumulativ reward
#fiks start problem
