"""
Test script for Humanoid Construction Environment
Includes pygame visualization for monitoring construction progress
"""

import os
import sys
import time
import json
import numpy as np
import gymnasium as gym
import pygame
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the construction environment
try:
    from humanoid_construction_env import HumanoidConstructionEnv
except ImportError:
    print("Installing humanoid_construction_env...")
    os.system("pip install -e .")
    from humanoid_construction_env import HumanoidConstructionEnv


class ConstructionVisualizer:
    """Pygame visualizer for construction environment metrics."""
    
    def __init__(self, width=1200, height=800):
        """Initialize pygame visualizer."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Humanoid Construction Environment - Test Visualization")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        
        # Data tracking
        self.rewards = []
        self.task_progress = []
        self.safety_violations = []
        self.blocks_placed = []
        self.max_history = 200
        
        # Construction site visualization
        self.site_width = 400
        self.site_height = 400
        self.site_x = width - self.site_width - 20
        self.site_y = 20
        
    def update(self, obs, reward, info, step):
        """Update visualization with new data."""
        # Update history
        self.rewards.append(reward)
        if len(self.rewards) > self.max_history:
            self.rewards.pop(0)
            
        self.task_progress.append(info.get('task_progress', 0))
        if len(self.task_progress) > self.max_history:
            self.task_progress.pop(0)
            
        self.safety_violations.append(info.get('safety_violations', 0))
        if len(self.safety_violations) > self.max_history:
            self.safety_violations.pop(0)
            
        self.blocks_placed.append(info.get('blocks_placed', 0))
        if len(self.blocks_placed) > self.max_history:
            self.blocks_placed.pop(0)
            
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw components
        self._draw_header(step, info)
        self._draw_metrics(info)
        self._draw_graphs()
        self._draw_construction_site(info)
        self._draw_task_panel(info)
        self._draw_safety_panel(info)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        
    def _draw_header(self, step, info):
        """Draw header information."""
        # Title
        title = self.font_large.render("Humanoid Construction Environment", True, self.WHITE)
        self.screen.blit(title, (20, 20))
        
        # Step counter
        step_text = self.font_medium.render(f"Step: {step}", True, self.CYAN)
        self.screen.blit(step_text, (20, 60))
        
        # Current task
        task = info.get('task', 'Unknown')
        task_text = self.font_medium.render(f"Task: {task.replace('_', ' ').title()}", True, self.YELLOW)
        self.screen.blit(task_text, (200, 60))
        
    def _draw_metrics(self, info):
        """Draw key metrics."""
        metrics_x = 20
        metrics_y = 100
        
        # Episode stats
        stats = info.get('episode_stats', {})
        
        # Total reward
        total_reward = stats.get('total_reward', 0)
        reward_color = self.GREEN if total_reward >= 0 else self.RED
        reward_text = self.font_medium.render(f"Total Reward: {total_reward:.2f}", True, reward_color)
        self.screen.blit(reward_text, (metrics_x, metrics_y))
        
        # Blocks placed
        blocks = stats.get('blocks_placed', 0)
        blocks_text = self.font_medium.render(f"Blocks Placed: {blocks}", True, self.BLUE)
        self.screen.blit(blocks_text, (metrics_x, metrics_y + 30))
        
        # Materials transported
        materials = stats.get('materials_transported', 0)
        materials_text = self.font_medium.render(f"Materials Transported: {materials}", True, self.PURPLE)
        self.screen.blit(materials_text, (metrics_x, metrics_y + 60))
        
        # Tasks completed
        tasks = stats.get('tasks_completed', 0)
        tasks_text = self.font_medium.render(f"Tasks Completed: {tasks}", True, self.GREEN)
        self.screen.blit(tasks_text, (metrics_x, metrics_y + 90))
        
    def _draw_graphs(self):
        """Draw performance graphs."""
        graph_x = 20
        graph_y = 250
        graph_width = 350
        graph_height = 150
        
        # Reward graph
        self._draw_graph(graph_x, graph_y, graph_width, graph_height,
                        self.rewards, "Instant Reward", self.GREEN, scale_auto=True)
        
        # Task progress graph
        self._draw_graph(graph_x + graph_width + 20, graph_y, graph_width, graph_height,
                        self.task_progress, "Task Progress", self.BLUE, scale_max=1.0)
        
        # Safety violations graph
        self._draw_graph(graph_x, graph_y + graph_height + 50, graph_width, graph_height,
                        self.safety_violations, "Safety Violations", self.RED, scale_max=10)
        
        # Blocks placed graph
        self._draw_graph(graph_x + graph_width + 20, graph_y + graph_height + 50, graph_width, graph_height,
                        self.blocks_placed, "Blocks Placed", self.PURPLE, scale_max=20)
        
    def _draw_graph(self, x, y, width, height, data, title, color, scale_auto=False, scale_max=None):
        """Draw a single graph."""
        # Background
        pygame.draw.rect(self.screen, self.GRAY, (x, y, width, height), 1)
        
        # Title
        title_text = self.font_small.render(title, True, self.WHITE)
        self.screen.blit(title_text, (x + 5, y - 20))
        
        if not data:
            return
            
        # Scale
        if scale_auto:
            data_min = min(data) if data else 0
            data_max = max(data) if data else 1
            data_range = max(data_max - data_min, 0.1)
        else:
            data_min = 0
            data_max = scale_max or 1
            data_range = data_max - data_min
            
        # Draw grid lines
        for i in range(5):
            grid_y = y + height - (i * height / 4)
            pygame.draw.line(self.screen, (50, 50, 50), (x, grid_y), (x + width, grid_y), 1)
            
        # Plot data
        if len(data) > 1:
            points = []
            for i, value in enumerate(data):
                px = x + (i * width / max(len(data) - 1, 1))
                py = y + height - ((value - data_min) / data_range * height)
                py = max(y, min(y + height, py))  # Clamp to graph bounds
                points.append((px, py))
                
            pygame.draw.lines(self.screen, color, False, points, 2)
            
            # Draw last value
            if data:
                last_value = data[-1]
                value_text = self.font_small.render(f"{last_value:.2f}", True, color)
                self.screen.blit(value_text, (x + width + 5, points[-1][1] - 8))
                
    def _draw_construction_site(self, info):
        """Draw bird's eye view of construction site."""
        # Background
        pygame.draw.rect(self.screen, self.GRAY, 
                        (self.site_x, self.site_y, self.site_width, self.site_height), 2)
        
        # Title
        title_text = self.font_medium.render("Construction Site View", True, self.WHITE)
        self.screen.blit(title_text, (self.site_x + 10, self.site_y - 25))
        
        # Foundation area
        foundation_x = self.site_x + self.site_width // 4
        foundation_y = self.site_y + self.site_height // 4
        foundation_w = self.site_width // 2
        foundation_h = self.site_height // 2
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (foundation_x, foundation_y, foundation_w, foundation_h), 2)
        
        # Draw placed blocks
        blocks_placed = info.get('blocks_placed', 0)
        for i in range(min(blocks_placed, 20)):
            block_x = foundation_x + (i % 4) * (foundation_w // 4)
            block_y = foundation_y + (i // 4) * (foundation_h // 5)
            pygame.draw.rect(self.screen, self.BLUE,
                           (block_x + 5, block_y + 5, foundation_w // 5, foundation_h // 6))
            
        # Draw crane position (simplified)
        crane_x = self.site_x + 50
        crane_y = self.site_y + self.site_height // 2
        pygame.draw.circle(self.screen, self.YELLOW, (crane_x, crane_y), 15)
        pygame.draw.line(self.screen, self.YELLOW, (crane_x, crane_y),
                        (crane_x + 100, crane_y - 50), 3)
        
        # Draw humanoid position (simplified)
        robot_x = self.site_x + self.site_width // 2
        robot_y = self.site_y + self.site_height // 2
        pygame.draw.circle(self.screen, self.GREEN, (robot_x, robot_y), 10)
        
        # Material storage area
        storage_x = self.site_x + self.site_width - 80
        storage_y = self.site_y + self.site_height - 80
        pygame.draw.rect(self.screen, self.ORANGE,
                        (storage_x, storage_y, 60, 60), 2)
        storage_text = self.font_small.render("Storage", True, self.ORANGE)
        self.screen.blit(storage_text, (storage_x + 5, storage_y + 25))
        
    def _draw_task_panel(self, info):
        """Draw task progress panel."""
        panel_x = self.site_x
        panel_y = self.site_y + self.site_height + 20
        panel_width = self.site_width
        panel_height = 120
        
        # Background
        pygame.draw.rect(self.screen, self.GRAY, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title_text = self.font_medium.render("Task Progress", True, self.WHITE)
        self.screen.blit(title_text, (panel_x + 10, panel_y + 5))
        
        # Progress bar
        progress = info.get('task_progress', 0)
        bar_x = panel_x + 20
        bar_y = panel_y + 40
        bar_width = panel_width - 40
        bar_height = 30
        
        pygame.draw.rect(self.screen, self.WHITE, 
                        (bar_x, bar_y, bar_width, bar_height), 2)
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.GREEN,
                           (bar_x + 2, bar_y + 2, fill_width - 4, bar_height - 4))
            
        # Progress text
        progress_text = self.font_medium.render(f"{progress * 100:.1f}%", True, self.WHITE)
        text_rect = progress_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.screen.blit(progress_text, text_rect)
        
        # Task details
        task = info.get('task', 'Unknown')
        details_text = self.font_small.render(f"Current: {task.replace('_', ' ').title()}", True, self.CYAN)
        self.screen.blit(details_text, (panel_x + 10, panel_y + 80))
        
    def _draw_safety_panel(self, info):
        """Draw safety status panel."""
        panel_x = self.site_x
        panel_y = self.site_y + self.site_height + 160
        panel_width = self.site_width
        panel_height = 100
        
        # Background
        violations = info.get('safety_violations', 0)
        border_color = self.RED if violations > 0 else self.GREEN
        pygame.draw.rect(self.screen, border_color,
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title_text = self.font_medium.render("Safety Status", True, self.WHITE)
        self.screen.blit(title_text, (panel_x + 10, panel_y + 5))
        
        # Safety indicators
        indicators = [
            ("Hard Hat", True, self.GREEN),
            ("Safety Zone", violations == 0, self.GREEN if violations == 0 else self.RED),
            ("Equipment Check", True, self.GREEN),
        ]
        
        for i, (name, status, color) in enumerate(indicators):
            y_pos = panel_y + 35 + i * 20
            
            # Status indicator
            pygame.draw.circle(self.screen, color if status else self.RED,
                             (panel_x + 20, y_pos), 5)
            
            # Label
            label_text = self.font_small.render(name, True, self.WHITE)
            self.screen.blit(label_text, (panel_x + 35, y_pos - 8))
            
        # Violations counter
        violations_text = self.font_small.render(f"Violations: {violations}", True, 
                                                self.RED if violations > 0 else self.GREEN)
        self.screen.blit(violations_text, (panel_x + panel_width - 100, panel_y + 70))
        
    def close(self):
        """Close the visualizer."""
        pygame.quit()


def test_construction_environment():
    """Test the humanoid construction environment."""
    print("\n" + "="*60)
    print("HUMANOID CONSTRUCTION ENVIRONMENT TEST")
    print("="*60)
    
    # Initialize environment
    try:
        env = gym.make('HumanoidConstruction-v0', render_mode='human')
    except:
        # If not registered, create directly
        env = HumanoidConstructionEnv(render_mode='human')
        
    # Initialize visualizer
    viz = ConstructionVisualizer()
    
    # Test parameters
    num_episodes = 3
    max_steps_per_episode = 500
    
    # Results tracking
    results = {
        'episodes': [],
        'avg_reward': 0,
        'max_reward': -float('inf'),
        'min_reward': float('inf'),
        'total_blocks': 0,
        'total_tasks_completed': 0,
        'total_violations': 0,
        'test_date': datetime.now().isoformat()
    }
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment
            obs, info = env.reset(seed=episode * 42)
            
            episode_reward = 0
            episode_data = {
                'episode': episode + 1,
                'steps': 0,
                'reward': 0,
                'task': info.get('task', 'Unknown'),
                'blocks_placed': 0,
                'safety_violations': 0,
                'task_completed': False
            }
            
            for step in range(max_steps_per_episode):
                # Check for pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                        
                # Take random action
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Update visualizer
                viz.update(obs, reward, info, step)
                
                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"  Step {step}: Reward={reward:.2f}, "
                          f"Task Progress={info['task_progress']:.2%}, "
                          f"Blocks={info['blocks_placed']}")
                    
                if terminated or truncated:
                    print(f"  Episode ended at step {step + 1}")
                    if info['task_progress'] >= 1.0:
                        print("  ✓ Task completed!")
                        episode_data['task_completed'] = True
                    break
                    
            # Update episode data
            episode_data['steps'] = step + 1
            episode_data['reward'] = episode_reward
            episode_data['blocks_placed'] = info.get('blocks_placed', 0)
            episode_data['safety_violations'] = info.get('safety_violations', 0)
            
            # Update results
            results['episodes'].append(episode_data)
            results['avg_reward'] += episode_reward
            results['max_reward'] = max(results['max_reward'], episode_reward)
            results['min_reward'] = min(results['min_reward'], episode_reward)
            results['total_blocks'] += episode_data['blocks_placed']
            results['total_violations'] += episode_data['safety_violations']
            if episode_data['task_completed']:
                results['total_tasks_completed'] += 1
                
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Blocks Placed: {episode_data['blocks_placed']}")
            print(f"  Safety Violations: {episode_data['safety_violations']}")
            print(f"  Task: {episode_data['task']}")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        
    finally:
        # Calculate final statistics
        results['avg_reward'] /= num_episodes
        
        # Print final results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Max Reward: {results['max_reward']:.2f}")
        print(f"Min Reward: {results['min_reward']:.2f}")
        print(f"Total Blocks Placed: {results['total_blocks']}")
        print(f"Tasks Completed: {results['total_tasks_completed']}/{num_episodes}")
        print(f"Total Safety Violations: {results['total_violations']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"construction_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Close environment and visualizer
        env.close()
        viz.close()
        
        print("\nTest completed successfully!")
        
        
def main():
    """Main entry point."""
    test_construction_environment()
    

if __name__ == "__main__":
    main()
