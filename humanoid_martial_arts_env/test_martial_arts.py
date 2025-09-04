"""
Test script for Humanoid Martial Arts Environment with pygame visualization

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

import sys
import os
import numpy as np
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pygame for visualization
import pygame
from pygame.locals import *

# Import the environment
from martial_arts_env import HumanoidMartialArtsEnv


class MartialArtsVisualizer:
    """Pygame-based visualizer for martial arts environment statistics."""
    
    def __init__(self, width=1200, height=800):
        """Initialize the visualizer."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Humanoid Martial Arts Environment - Training Monitor")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.ORANGE = (255, 165, 0)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 36)
        self.text_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Data storage
        self.reward_history = []
        self.technique_history = []
        self.combo_history = []
        self.balance_history = []
        self.power_history = []
        self.max_history_length = 200
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        
    def update(self, env_data):
        """Update visualization with new environment data."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                    
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Update histories
        if 'reward' in env_data:
            self.reward_history.append(env_data['reward'])
            if len(self.reward_history) > self.max_history_length:
                self.reward_history.pop(0)
                
        if 'techniques_count' in env_data:
            self.technique_history.append(env_data['techniques_count'])
            if len(self.technique_history) > self.max_history_length:
                self.technique_history.pop(0)
                
        if 'combo_length' in env_data:
            self.combo_history.append(env_data['combo_length'])
            if len(self.combo_history) > self.max_history_length:
                self.combo_history.pop(0)
                
        if 'balance_score' in env_data:
            self.balance_history.append(env_data['balance_score'])
            if len(self.balance_history) > self.max_history_length:
                self.balance_history.pop(0)
                
        if 'power_generated' in env_data:
            self.power_history.append(env_data['power_generated'])
            if len(self.power_history) > self.max_history_length:
                self.power_history.pop(0)
        
        # Draw title
        self._draw_title("Humanoid Martial Arts Training Monitor")
        
        # Draw main sections
        self._draw_current_stats(env_data)
        self._draw_reward_graph()
        self._draw_technique_panel(env_data)
        self._draw_performance_metrics(env_data)
        self._draw_combo_indicator(env_data)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        return True
        
    def _draw_title(self, title):
        """Draw the main title."""
        text = self.title_font.render(title, True, self.CYAN)
        text_rect = text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(text, text_rect)
        
        # Draw separator line
        pygame.draw.line(self.screen, self.CYAN, (50, 60), (self.width - 50, 60), 2)
        
    def _draw_current_stats(self, env_data):
        """Draw current environment statistics."""
        # Background panel
        panel_rect = pygame.Rect(50, 80, 350, 250)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title_text = self.header_font.render("Current Status", True, self.YELLOW)
        self.screen.blit(title_text, (70, 90))
        
        # Statistics
        y_offset = 130
        stats = [
            ("Step", env_data.get('step', 0)),
            ("Total Reward", f"{env_data.get('total_reward', 0):.2f}"),
            ("Techniques", env_data.get('techniques_count', 0)),
            ("Current Combo", env_data.get('combo_length', 0)),
            ("Stance Stability", f"{env_data.get('stance_stability', 0):.2f}s"),
            ("Robot Height", f"{env_data.get('robot_height', 0):.2f}m"),
        ]
        
        for label, value in stats:
            label_text = self.text_font.render(f"{label}:", True, self.WHITE)
            value_text = self.text_font.render(str(value), True, self.GREEN)
            self.screen.blit(label_text, (70, y_offset))
            self.screen.blit(value_text, (250, y_offset))
            y_offset += 30
            
    def _draw_reward_graph(self):
        """Draw reward history graph."""
        # Background panel
        panel_rect = pygame.Rect(420, 80, 350, 250)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title_text = self.header_font.render("Reward History", True, self.YELLOW)
        self.screen.blit(title_text, (440, 90))
        
        # Draw graph
        if len(self.reward_history) > 1:
            # Calculate scaling
            min_reward = min(self.reward_history)
            max_reward = max(self.reward_history)
            reward_range = max_reward - min_reward if max_reward != min_reward else 1
            
            # Draw axes
            graph_x = 440
            graph_y = 130
            graph_width = 310
            graph_height = 180
            
            pygame.draw.line(self.screen, self.GRAY, (graph_x, graph_y + graph_height), 
                           (graph_x + graph_width, graph_y + graph_height), 1)
            pygame.draw.line(self.screen, self.GRAY, (graph_x, graph_y), 
                           (graph_x, graph_y + graph_height), 1)
            
            # Plot points
            points = []
            for i, reward in enumerate(self.reward_history):
                x = graph_x + int(i * graph_width / len(self.reward_history))
                y = graph_y + graph_height - int((reward - min_reward) * graph_height / reward_range)
                points.append((x, y))
                
            # Draw lines between points
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
                
            # Draw min/max labels
            max_label = self.small_font.render(f"{max_reward:.1f}", True, self.WHITE)
            min_label = self.small_font.render(f"{min_reward:.1f}", True, self.WHITE)
            self.screen.blit(max_label, (graph_x - 40, graph_y))
            self.screen.blit(min_label, (graph_x - 40, graph_y + graph_height - 10))
            
    def _draw_technique_panel(self, env_data):
        """Draw technique execution panel."""
        # Background panel
        panel_rect = pygame.Rect(790, 80, 350, 250)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title_text = self.header_font.render("Technique Analysis", True, self.YELLOW)
        self.screen.blit(title_text, (810, 90))
        
        # Technique breakdown
        y_offset = 130
        techniques = env_data.get('technique_breakdown', {
            'punches': 0,
            'kicks': 0,
            'blocks': 0,
            'dodges': 0,
            'stances': 0
        })
        
        max_count = max(techniques.values()) if techniques.values() else 1
        
        for technique, count in techniques.items():
            # Label
            label_text = self.text_font.render(technique.capitalize(), True, self.WHITE)
            self.screen.blit(label_text, (810, y_offset))
            
            # Bar graph
            bar_width = int(150 * (count / max_count)) if max_count > 0 else 0
            bar_rect = pygame.Rect(920, y_offset, bar_width, 20)
            
            # Choose color based on technique
            colors = {
                'punches': self.RED,
                'kicks': self.ORANGE,
                'blocks': self.BLUE,
                'dodges': self.GREEN,
                'stances': self.CYAN
            }
            color = colors.get(technique, self.GRAY)
            
            pygame.draw.rect(self.screen, color, bar_rect)
            pygame.draw.rect(self.screen, self.WHITE, bar_rect, 1)
            
            # Count text
            count_text = self.small_font.render(str(count), True, self.WHITE)
            self.screen.blit(count_text, (1080, y_offset))
            
            y_offset += 35
            
    def _draw_performance_metrics(self, env_data):
        """Draw performance metrics panel."""
        # Background panel
        panel_rect = pygame.Rect(50, 350, 500, 200)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title_text = self.header_font.render("Performance Metrics", True, self.YELLOW)
        self.screen.blit(title_text, (70, 360))
        
        # Metrics
        y_offset = 400
        
        # Balance meter
        balance = env_data.get('balance_score', 0)
        self._draw_meter("Balance", balance, 70, y_offset, self.GREEN)
        
        # Power meter
        y_offset += 40
        power = env_data.get('power_generated', 0) / 100  # Normalize
        self._draw_meter("Power", power, 70, y_offset, self.RED)
        
        # Speed meter
        y_offset += 40
        speed = env_data.get('speed_score', 0)
        self._draw_meter("Speed", speed, 70, y_offset, self.YELLOW)
        
        # Accuracy meter
        y_offset += 40
        accuracy = env_data.get('technique_accuracy', 0)
        self._draw_meter("Accuracy", accuracy, 70, y_offset, self.CYAN)
        
    def _draw_meter(self, label, value, x, y, color):
        """Draw a horizontal meter."""
        # Label
        label_text = self.text_font.render(label, True, self.WHITE)
        self.screen.blit(label_text, (x, y))
        
        # Meter background
        meter_x = x + 100
        meter_width = 300
        meter_height = 20
        meter_rect = pygame.Rect(meter_x, y, meter_width, meter_height)
        pygame.draw.rect(self.screen, self.GRAY, meter_rect)
        
        # Meter fill
        fill_width = int(meter_width * min(1.0, max(0.0, value)))
        fill_rect = pygame.Rect(meter_x, y, fill_width, meter_height)
        pygame.draw.rect(self.screen, color, fill_rect)
        
        # Meter border
        pygame.draw.rect(self.screen, self.WHITE, meter_rect, 2)
        
        # Value text
        value_text = self.small_font.render(f"{value*100:.0f}%", True, self.WHITE)
        self.screen.blit(value_text, (meter_x + meter_width + 10, y))
        
    def _draw_combo_indicator(self, env_data):
        """Draw combo chain indicator."""
        # Background panel
        panel_rect = pygame.Rect(570, 350, 570, 200)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title_text = self.header_font.render("Combo Chain", True, self.YELLOW)
        self.screen.blit(title_text, (590, 360))
        
        # Combo visualization
        combo_chain = env_data.get('combo_chain', [])
        max_combo = env_data.get('max_combo', 0)
        
        # Draw combo circles
        y_offset = 420
        x_offset = 590
        
        for i in range(5):  # Show up to 5 combo slots
            circle_x = x_offset + i * 100
            circle_y = y_offset
            
            if i < len(combo_chain):
                # Active combo
                color = self._get_combo_color(i)
                pygame.draw.circle(self.screen, color, (circle_x, circle_y), 30)
                
                # Technique name
                technique = combo_chain[i][:3].upper() if i < len(combo_chain) else ""
                text = self.text_font.render(technique, True, self.BLACK)
                text_rect = text.get_rect(center=(circle_x, circle_y))
                self.screen.blit(text, text_rect)
            else:
                # Empty slot
                pygame.draw.circle(self.screen, self.GRAY, (circle_x, circle_y), 30, 2)
                
            # Draw connection lines
            if i < 4:
                line_start = (circle_x + 30, circle_y)
                line_end = (circle_x + 70, circle_y)
                pygame.draw.line(self.screen, self.GRAY, line_start, line_end, 2)
                
        # Combo stats
        y_offset = 480
        combo_text = f"Current: {len(combo_chain)} | Best: {max_combo}"
        text = self.text_font.render(combo_text, True, self.WHITE)
        self.screen.blit(text, (590, y_offset))
        
        # Combo multiplier
        if len(combo_chain) > 0:
            multiplier = 1 + len(combo_chain) * 0.5
            mult_text = self.header_font.render(f"{multiplier:.1f}x", True, self.ORANGE)
            self.screen.blit(mult_text, (1050, y_offset - 10))
            
    def _get_combo_color(self, index):
        """Get color for combo chain based on index."""
        colors = [self.GREEN, self.YELLOW, self.ORANGE, self.RED, self.MAGENTA]
        return colors[min(index, len(colors) - 1)]
        
    def close(self):
        """Close the visualizer."""
        pygame.quit()


def test_environment(num_episodes=20):
    """Test the Humanoid Martial Arts environment with visualization for multiple episodes."""
    
    print("=" * 80)
    print("HUMANOID MARTIAL ARTS ENVIRONMENT TEST")
    print("=" * 80)
    
    # Create environment with rendering
    print("\n1. Creating environment...")
    try:
        env = HumanoidMartialArtsEnv(render_mode="human")
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return
        
    # Check spaces
    print("\n2. Checking observation and action spaces...")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Create visualizer
    print("\n3. Creating pygame visualizer...")
    visualizer = MartialArtsVisualizer()
    
    all_results = []
    stop_requested = False

    for episode_idx in range(1, num_episodes + 1):
        print(f"\n{'='*40}\nEPISODE {episode_idx}/{num_episodes}\n{'='*40}")
        # Test reset
        print("\n4. Resetting environment for new episode...")
        try:
            obs, info = env.reset()
            print(f"✓ Environment reset successfully")
            print(f"   Initial observation shape: {obs.shape}")
            print(f"   Initial info: {info}")
        except Exception as e:
            print(f"✗ Failed to reset environment: {e}")
            env.close()
            visualizer.close()
            return

        # Run test episode
        print("\n5. Running test episode with random actions...")
        print("   (Press ESC to stop)")
        print("-" * 40)
        
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False
        
        # Episode statistics
        technique_breakdown = {
            'punches': 0,
            'kicks': 0,
            'blocks': 0,
            'dodges': 0,
            'stances': 0
        }
        
        max_combo = 0
        combo_chain = []
        
        # Performance tracking
        start_time = time.time()
        
        while not (terminated or truncated):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update statistics
            episode_reward += reward
            step_count += 1
            
            # Update technique counts (simplified detection)
            if step_count % 10 == 0:
                technique_type = np.random.choice(['punches', 'kicks', 'blocks', 'dodges', 'stances'])
                technique_breakdown[technique_type] += 1
                
                # Update combo
                if np.random.random() > 0.7:
                    combo_chain.append(technique_type)
                    if len(combo_chain) > 5:
                        combo_chain.pop(0)
                    max_combo = max(max_combo, len(combo_chain))
                else:
                    combo_chain = []
            
            # Prepare visualization data
            viz_data = {
                'step': step_count,
                'reward': reward,
                'total_reward': episode_reward,
                'techniques_count': sum(technique_breakdown.values()),
                'combo_length': len(combo_chain),
                'combo_chain': combo_chain,
                'max_combo': max_combo,
                'stance_stability': info.get('stance_stability', 0),
                'robot_height': obs[2] if len(obs) > 2 else 1.4,  # z-position
                'technique_breakdown': technique_breakdown,
                'balance_score': min(1.0, obs[2] / 1.75) if len(obs) > 2 else 0,
                'power_generated': np.random.random() * 100,  # Simulated
                'speed_score': np.random.random(),  # Simulated
                'technique_accuracy': np.random.random()  # Simulated
            }
            
            # Update visualization
            if not visualizer.update(viz_data):
                print("\nVisualization stopped by user")
                stop_requested = True
                break
                
            # Print periodic updates
            if step_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = step_count / elapsed_time
                print(f"Step {step_count}: Reward = {episode_reward:.2f}, "
                      f"Techniques = {sum(technique_breakdown.values())}, "
                      f"FPS = {fps:.1f}")
        # Episode complete
        elapsed_time = time.time() - start_time
        
        print("-" * 40)
        print("\n6. Episode Summary:")
        print(f"   Total steps: {step_count}")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Average reward per step: {episode_reward/step_count:.4f}")
        print(f"   Episode time: {elapsed_time:.2f} seconds")
        print(f"   Simulation FPS: {step_count/elapsed_time:.1f}")
        print(f"   Episode stats: {info.get('episode_stats', {})}")
        print(f"\n   Technique breakdown:")
        for technique, count in technique_breakdown.items():
            print(f"      {technique.capitalize()}: {count}")
        print(f"   Max combo length: {max_combo}")
        
        # Save test results
        print("\n7. Saving test results for this episode...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'HumanoidMartialArtsEnv',
            'episode': episode_idx,
            'total_steps': step_count,
            'total_reward': float(episode_reward),
            'average_reward': float(episode_reward / step_count) if step_count > 0 else 0.0,
            'episode_time': elapsed_time,
            'simulation_fps': step_count / elapsed_time if elapsed_time > 0 else 0.0,
            'technique_breakdown': technique_breakdown,
            'max_combo': max_combo,
            'episode_stats': info.get('episode_stats', {})
        }
        
        results_file = f"martial_arts_test_results_ep{episode_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}")
        all_results.append(results)
        
        if stop_requested:
            print("\nStopping further episodes due to user request.")
            break

    # Clean up
    print("\n8. Cleaning up...")
    visualizer.close()
    env.close()
    print("✓ Test completed successfully!")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    # Run the test for multiple episodes
    NUM_EPISODES = 20  # You can change this value as needed
    results = test_environment(num_episodes=NUM_EPISODES)
    
    # Print final message
    if results and len(results) > 0:
        print(f"\n✅ All {len(results)} episode(s) completed! The environment is working correctly.")
        print("You can now use this environment for training RL agents.")
    else:
        print("\n❌ Some tests failed or were interrupted. Please check the error messages above.")
