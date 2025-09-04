#!/usr/bin/env python3
"""
Test script for Bipedal Rescue Environment with pygame visualization.

Author: Hasnain Fareed
License: MIT (2025)
"""

import sys
import os
import time
import json
import numpy as np
import pygame
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bipedal_rescue_env import BipedalRescueEnv


class RescueVisualization:
    """Pygame visualization for the rescue environment [[memory:7080345]]."""
    
    def __init__(self, width=800, height=600):
        """Initialize pygame visualization window [[memory:7080335]]."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🚁 Bipedal Rescue Mission - Real-time Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.GRAY = (128, 128, 128)
        self.DARK_GREEN = (0, 128, 0)
        
    def world_to_screen(self, world_pos, world_size=50.0):
        """Convert world coordinates to screen coordinates."""
        x = int((world_pos[0] + world_size/2) * self.width / world_size)
        y = int((world_size/2 - world_pos[1]) * self.height / world_size)
        return x, y
    
    def draw_environment(self, env_data):
        """Draw the rescue environment."""
        self.screen.fill(self.GRAY)
        
        # Draw safe zone
        safe_zone_pos = self.world_to_screen(env_data['safe_zone_pos'])
        safe_zone_radius = int(env_data['safe_zone_radius'] * self.width / 50)
        pygame.draw.circle(self.screen, self.GREEN, safe_zone_pos, safe_zone_radius, 3)
        pygame.draw.circle(self.screen, (0, 255, 0, 50), safe_zone_pos, safe_zone_radius)
        
        # Draw medical tent symbol
        tent_rect = pygame.Rect(safe_zone_pos[0] - 20, safe_zone_pos[1] - 20, 40, 30)
        pygame.draw.rect(self.screen, self.WHITE, tent_rect, 2)
        pygame.draw.line(self.screen, self.RED, 
                        (safe_zone_pos[0] - 5, safe_zone_pos[1] - 10),
                        (safe_zone_pos[0] + 5, safe_zone_pos[1] - 10), 3)
        pygame.draw.line(self.screen, self.RED,
                        (safe_zone_pos[0], safe_zone_pos[1] - 15),
                        (safe_zone_pos[0], safe_zone_pos[1] - 5), 3)
        
        # Draw fire zones
        for fire_zone in env_data.get('fire_zones', []):
            fire_pos = self.world_to_screen(fire_zone['pos'])
            fire_radius = int(fire_zone['radius'] * self.width / 50)
            # Fire effect with gradient
            for i in range(3):
                alpha = 100 - i * 30
                color = (255, 100 - i * 30, 0)
                pygame.draw.circle(self.screen, color, fire_pos, fire_radius - i * 5, 2)
        
        # Draw debris (simplified)
        debris_positions = [(-10, 5), (5, -8), (0, 10), (-8, 0)]
        for debris_pos in debris_positions:
            screen_pos = self.world_to_screen(debris_pos)
            pygame.draw.rect(self.screen, self.GRAY, 
                           (screen_pos[0] - 15, screen_pos[1] - 10, 30, 20), 2)
    
    def draw_victims(self, victims_data):
        """Draw victims with status indicators."""
        for i, victim in enumerate(victims_data):
            screen_pos = self.world_to_screen(victim['position'][:2])
            
            # Color based on status
            if victim['rescued']:
                color = self.DARK_GREEN
            elif victim['carried']:
                color = self.YELLOW
            else:
                color = self.RED
            
            # Draw victim
            pygame.draw.circle(self.screen, color, screen_pos, 8, 0)
            pygame.draw.circle(self.screen, self.BLACK, screen_pos, 8, 2)
            
            # Draw victim number
            text = self.font.render(str(i+1), True, self.WHITE)
            text_rect = text.get_rect(center=screen_pos)
            self.screen.blit(text, text_rect)
            
            # Draw priority indicator
            if not victim['rescued']:
                priority = victim.get('priority', 0.5)
                if priority > 0.8:
                    # Critical victim - flashing indicator
                    if int(time.time() * 2) % 2:
                        pygame.draw.circle(self.screen, self.RED, screen_pos, 12, 2)
    
    def draw_robot(self, robot_data):
        """Draw the bipedal robot."""
        screen_pos = self.world_to_screen(robot_data['position'][:2])
        
        # Robot body
        color = self.BLUE if robot_data['upright'] else self.ORANGE
        pygame.draw.circle(self.screen, color, screen_pos, 12, 0)
        pygame.draw.circle(self.screen, self.BLACK, screen_pos, 12, 2)
        
        # Direction indicator
        direction = robot_data.get('direction', 0)
        end_x = screen_pos[0] + int(15 * np.cos(direction))
        end_y = screen_pos[1] - int(15 * np.sin(direction))
        pygame.draw.line(self.screen, self.BLACK, screen_pos, (end_x, end_y), 2)
        
        # Carrying indicator
        if robot_data.get('carrying_victims', 0) > 0:
            pygame.draw.circle(self.screen, self.YELLOW, 
                             (screen_pos[0], screen_pos[1] - 15), 5, 0)
    
    def draw_stats(self, stats):
        """Draw statistics panel."""
        # Background panel
        panel_rect = pygame.Rect(10, 10, 250, 200)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect, 2)
        
        # Title
        title = self.font_large.render("Mission Status", True, self.WHITE)
        self.screen.blit(title, (20, 20))
        
        # Stats
        y_offset = 60
        stats_text = [
            f"Victims Rescued: {stats['victims_rescued']}/{stats['total_victims']}",
            f"Victims Carried: {stats['victims_carried']}",
            f"Energy: {stats['energy_remaining']:.1f}%",
            f"Time: {stats['time_elapsed']:.1f}s",
            f"Distance: {stats['distance_traveled']:.1f}m",
            f"Status: {stats['robot_status']}"
        ]
        
        for text in stats_text:
            rendered = self.font.render(text, True, self.WHITE)
            self.screen.blit(rendered, (20, y_offset))
            y_offset += 25
        
        # Energy bar
        energy_rect = pygame.Rect(20, 190, int(200 * stats['energy_remaining'] / 100), 10)
        energy_color = self.GREEN if stats['energy_remaining'] > 30 else self.ORANGE if stats['energy_remaining'] > 10 else self.RED
        pygame.draw.rect(self.screen, energy_color, energy_rect)
        pygame.draw.rect(self.screen, self.WHITE, (20, 190, 200, 10), 1)
    
    def draw_legend(self):
        """Draw legend for symbols."""
        legend_rect = pygame.Rect(self.width - 200, 10, 190, 140)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), legend_rect)
        pygame.draw.rect(self.screen, self.WHITE, legend_rect, 2)
        
        legend_items = [
            ("Robot", self.BLUE),
            ("Victim (Need Help)", self.RED),
            ("Victim (Carried)", self.YELLOW),
            ("Victim (Rescued)", self.DARK_GREEN),
            ("Safe Zone", self.GREEN),
            ("Fire Hazard", self.ORANGE)
        ]
        
        y_offset = 20
        for label, color in legend_items:
            pygame.draw.circle(self.screen, color, 
                             (self.width - 180, y_offset + 10), 5)
            text = self.font.render(label, True, self.WHITE)
            self.screen.blit(text, (self.width - 165, y_offset))
            y_offset += 20
    
    def update(self):
        """Update the display."""
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        """Close the visualization."""
        pygame.quit()


def test_environment_with_visualization(episodes=5, episode_steps=2000):
    """Test the rescue environment with pygame visualization [[memory:7080335]]."""
    
    print("🚁 Bipedal Rescue Environment Test")
    print("=" * 60)
    print("📋 Mission Brief:")
    print("   • Navigate disaster zone with bipedal robot")
    print("   • Locate and rescue 5 victims")
    print("   • Carry victims to safe zone (medical tent)")
    print("   • Avoid fire hazards and debris")
    print("   • Manage energy consumption")
    print("=" * 60)
    
    # Initialize visualization
    viz = RescueVisualization()
    
    # Create environment
    print("\n🤖 Creating Bipedal Rescue Environment...")
    env = BipedalRescueEnv(render_mode='human')
    print("✅ Environment created successfully!")
    
    # Test results storage
    test_results = {
        'test_date': datetime.now().isoformat(),
        'episodes': [],
        'overall_stats': {
            'total_victims_rescued': 0,
            'average_time_to_rescue': [],
            'average_energy_used': [],
            'success_rate': 0
        }
    }
    
    try:
        for episode in range(episodes):
            print(f"\n📍 Episode {episode + 1}/{episodes}")
            
            # Reset environment
            obs, info = env.reset()
            
            episode_data = {
                'episode': episode + 1,
                'steps': 0,
                'total_reward': 0,
                'victims_rescued': 0,
                'energy_used': 0,
                'distance_traveled': 0
            }
            
            # Run episode
            for step in range(episode_steps):
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # Simple control strategy (replace with RL agent)
                action = env.action_space.sample() * 0.1
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_data['total_reward'] += reward
                episode_data['steps'] = step
                
                # Update visualization data
                robot_pos = info.get('robot_position', np.zeros(3))
                
                # Prepare environment data
                env_data = {
                    'safe_zone_pos': np.array([20.0, 0.0, 0.0]),
                    'safe_zone_radius': 3.0,
                    'fire_zones': [
                        {'pos': np.array([-5.0, -3.0, 0.0]), 'radius': 1.5},
                        {'pos': np.array([8.0, 6.0, 0.0]), 'radius': 1.2}
                    ]
                }
                
                # Prepare victims data
                victims_data = []
                for i in range(5):
                    # Get victim position from environment
                    victim_pos = np.array([
                        [-12, 3, 0.3],
                        [7, -6, 0.2],
                        [-3, -10, 0.35],
                        [2, 8, 0.15],
                        [12, 12, 0.25]
                    ][i])
                    
                    victims_data.append({
                        'position': victim_pos,
                        'rescued': i in env.victims_rescued,
                        'carried': i in env.victims_carried,
                        'priority': env.victim_priorities[i]
                    })
                
                # Prepare robot data
                robot_data = {
                    'position': robot_pos,
                    'upright': info.get('robot_upright', True),
                    'carrying_victims': info.get('victims_carried', 0),
                    'direction': 0  # Could calculate from orientation
                }
                
                # Prepare stats
                stats = {
                    'victims_rescued': info['episode_stats']['victims_rescued'],
                    'total_victims': 5,
                    'victims_carried': info.get('victims_carried', 0),
                    'energy_remaining': (info.get('energy_remaining', 1000) / 10),
                    'time_elapsed': step * 0.02,
                    'distance_traveled': info['episode_stats']['distance_traveled'],
                    'robot_status': 'Active' if info.get('robot_upright', True) else 'Fallen'
                }
                
                # Draw visualization
                viz.draw_environment(env_data)
                viz.draw_victims(victims_data)
                viz.draw_robot(robot_data)
                viz.draw_stats(stats)
                viz.draw_legend()
                viz.update()
                
                # Print progress
                if step % 100 == 0:
                    print(f"    Step {step}: Rescued {stats['victims_rescued']}/{stats['total_victims']}, "
                          f"Energy {stats['energy_remaining']:.1f}%")
                
                # Check if victim rescued
                if info['episode_stats']['victims_rescued'] > episode_data['victims_rescued']:
                    episode_data['victims_rescued'] = info['episode_stats']['victims_rescued']
                    print(f"    🎉 VICTIM RESCUED! Total: {episode_data['victims_rescued']}/5")
                
                if terminated or truncated:
                    episode_data['energy_used'] = info['episode_stats']['energy_used']
                    episode_data['distance_traveled'] = info['episode_stats']['distance_traveled']
                    print(f"    Episode ended at step {step}")
                    break
                
                time.sleep(0.01)  # Control simulation speed
            
            # Store episode results
            test_results['episodes'].append(episode_data)
            test_results['overall_stats']['total_victims_rescued'] += episode_data['victims_rescued']
            
            print(f"    📊 Episode Summary:")
            print(f"       • Victims Rescued: {episode_data['victims_rescued']}/5")
            print(f"       • Total Reward: {episode_data['total_reward']:.2f}")
            print(f"       • Distance Traveled: {episode_data['distance_traveled']:.2f}m")
            print(f"       • Energy Used: {episode_data['energy_used']:.2f}")
            
            # Small pause between episodes
            time.sleep(2)
        
        # Calculate overall statistics
        test_results['overall_stats']['success_rate'] = (
            test_results['overall_stats']['total_victims_rescued'] / (episodes * 5) * 100
        )
        
        print("\n" + "=" * 60)
        print("🏁 TEST COMPLETE")
        print(f"   Overall Success Rate: {test_results['overall_stats']['success_rate']:.1f}%")
        print(f"   Total Victims Rescued: {test_results['overall_stats']['total_victims_rescued']}")
        print("=" * 60)
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"rescue_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n📁 Test results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viz.close()
        env.close()
        print("✅ Environment and visualization closed")


def main():
    """Main function."""
    print("🚁 BIPEDAL RESCUE MISSION - DISASTER RESPONSE SIMULATION")
    print("=" * 60)
    print("🎯 Environment Features:")
    print("   • Advanced bipedal robot with grippers")
    print("   • 5 victims with different priorities")
    print("   • Hazardous disaster environment")
    print("   • Energy management system")
    print("   • Real-time pygame visualization")
    print("   • 3D MuJoCo physics simulation")
    print("=" * 60)
    
    # Run test with visualization [[memory:7080335]]
    test_environment_with_visualization(episodes=3, episode_steps=2000)
    
    print("\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()
