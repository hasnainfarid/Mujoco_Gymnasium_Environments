"""
Test script for Humanoid Dancing Environment with Pygame visualization

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

import numpy as np
import time
import json
from datetime import datetime
import pygame
import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dancing_env import HumanoidDancingEnv, register_env
    print("✅ Successfully imported HumanoidDancingEnv")
except ImportError as e:
    print(f"❌ Failed to import HumanoidDancingEnv: {e}")
    traceback.print_exc()
    sys.exit(1)


class DancingVisualization:
    """Pygame visualization for the dancing environment."""
    
    def __init__(self, width=1200, height=800):
        """Initialize pygame visualization."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🕺 Humanoid Dancing Environment - Live Performance")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (50, 50, 255)
        self.YELLOW = (255, 255, 50)
        self.PURPLE = (200, 50, 200)
        self.CYAN = (50, 255, 255)
        self.ORANGE = (255, 150, 50)
        self.PINK = (255, 150, 200)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 36)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Dance floor visualization
        self.floor_center = (400, 400)
        self.floor_radius = 150
        
        # Beat visualization
        self.beat_circles = []
        self.max_beat_circles = 10
        
        # Performance graph
        self.performance_history = []
        self.max_history = 100
        
        # Crowd visualization
        self.crowd_particles = []
        self.init_crowd_particles()
        
        # Visual effects
        self.spotlight_angle = 0
        self.disco_ball_rotation = 0
        self.star_effects = []
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
    
    def init_crowd_particles(self):
        """Initialize crowd visualization particles."""
        colors = [self.YELLOW, self.ORANGE, self.PINK]
        for i in range(50):
            self.crowd_particles.append({
                'x': np.random.randint(50, 750),
                'y': np.random.randint(550, 750),
                'vx': 0,
                'vy': 0,
                'color': colors[np.random.randint(0, len(colors))],
                'size': np.random.randint(3, 6)
            })
    
    def update(self, env_info, reward, action_magnitudes):
        """Update visualization with new environment data."""
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen with gradient background
        self.draw_gradient_background()
        
        # Update visual effects
        self.update_visual_effects(env_info)
        
        # Draw components
        self.draw_dance_floor(env_info)
        self.draw_dancer_state(env_info)
        self.draw_beat_indicator(env_info)
        self.draw_performance_metrics(env_info, reward)
        self.draw_crowd_excitement(env_info)
        self.draw_move_info(env_info)
        self.draw_combo_meter(env_info)
        self.draw_action_heatmap(action_magnitudes)
        self.draw_performance_graph(reward)
        self.draw_episode_stats(env_info)
        
        # Draw title
        title = self.title_font.render("🕺 Humanoid Dancing Environment 💃", True, self.WHITE)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        return True
    
    def draw_gradient_background(self):
        """Draw a gradient background."""
        for y in range(self.height):
            color_value = int(20 + (y / self.height) * 30)
            color = (color_value, 0, color_value + 10)
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
    
    def update_visual_effects(self, env_info):
        """Update visual effects."""
        # Rotate spotlight
        self.spotlight_angle += 2
        if self.spotlight_angle >= 360:
            self.spotlight_angle = 0
        
        # Rotate disco ball
        self.disco_ball_rotation += 3
        if self.disco_ball_rotation >= 360:
            self.disco_ball_rotation = 0
        
        # Update beat circles
        beat_phase = env_info.get('beat_phase', 0)
        if beat_phase < 0.1 and len(self.beat_circles) < self.max_beat_circles:
            self.beat_circles.append({'radius': 0, 'alpha': 255})
        
        # Update existing beat circles
        for circle in self.beat_circles[:]:
            circle['radius'] += 3
            circle['alpha'] -= 5
            if circle['alpha'] <= 0:
                self.beat_circles.remove(circle)
        
        # Add star effects for high combo
        combo = env_info.get('combo_multiplier', 1.0)
        if combo > 3.0 and np.random.random() < 0.1:
            star_colors = [self.YELLOW, self.WHITE, self.CYAN]
            self.star_effects.append({
                'x': np.random.randint(100, 700),
                'y': np.random.randint(100, 500),
                'size': np.random.randint(5, 15),
                'lifetime': 30,
                'color': star_colors[np.random.randint(0, len(star_colors))]
            })
        
        # Update star effects
        for star in self.star_effects[:]:
            star['lifetime'] -= 1
            if star['lifetime'] <= 0:
                self.star_effects.remove(star)
    
    def draw_dance_floor(self, env_info):
        """Draw the dance floor visualization."""
        # Disco ball
        disco_x = self.floor_center[0]
        disco_y = self.floor_center[1] - 200
        pygame.draw.circle(self.screen, self.GRAY, (disco_x, disco_y), 20)
        
        # Disco ball sparkles
        for i in range(8):
            angle = self.disco_ball_rotation + i * 45
            x = disco_x + int(25 * np.cos(np.radians(angle)))
            y = disco_y + int(25 * np.sin(np.radians(angle)))
            pygame.draw.circle(self.screen, self.WHITE, (x, y), 3)
        
        # Spotlight beams
        for i in range(3):
            angle = self.spotlight_angle + i * 120
            end_x = self.floor_center[0] + int(200 * np.cos(np.radians(angle)))
            end_y = self.floor_center[1] + int(200 * np.sin(np.radians(angle)))
            
            # Draw translucent beam
            beam_color = (*self.YELLOW, 50)
            for j in range(5):
                offset = j * 2
                pygame.draw.line(self.screen, self.YELLOW,
                               (disco_x + offset, disco_y),
                               (end_x, end_y), 1)
        
        # Dance floor circle
        pygame.draw.circle(self.screen, self.WHITE, self.floor_center, self.floor_radius, 3)
        
        # Checkered pattern
        pattern_size = 30
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = self.floor_center[0] + i * pattern_size
                y = self.floor_center[1] + j * pattern_size
                
                # Check if inside circle
                dist = np.sqrt((x - self.floor_center[0])**2 + (y - self.floor_center[1])**2)
                if dist < self.floor_radius:
                    if (i + j) % 2 == 0:
                        color = self.DARK_GRAY
                    else:
                        color = self.GRAY
                    pygame.draw.rect(self.screen, color, 
                                   (x - pattern_size//2, y - pattern_size//2, 
                                    pattern_size, pattern_size))
        
        # Beat circles (ripple effect)
        for circle in self.beat_circles:
            color = (*self.CYAN, circle['alpha'])
            if circle['alpha'] > 0:
                pygame.draw.circle(self.screen, self.CYAN,
                                 self.floor_center, circle['radius'], 2)
        
        # Star effects
        for star in self.star_effects:
            points = []
            for i in range(10):
                angle = i * 36
                if i % 2 == 0:
                    radius = star['size']
                else:
                    radius = star['size'] // 2
                x = star['x'] + int(radius * np.cos(np.radians(angle)))
                y = star['y'] + int(radius * np.sin(np.radians(angle)))
                points.append((x, y))
            if len(points) > 2:
                pygame.draw.polygon(self.screen, star['color'], points)
    
    def draw_dancer_state(self, env_info):
        """Draw simplified dancer representation."""
        # Dancer position on floor (simplified)
        dancer_x = self.floor_center[0]
        dancer_y = self.floor_center[1]
        
        # Draw dancer as stylized figure
        # Body
        pygame.draw.circle(self.screen, self.BLUE, (dancer_x, dancer_y), 15, 0)
        
        # Arms (animated based on beat)
        beat_phase = env_info.get('beat_phase', 0)
        arm_angle = beat_phase * 360
        
        # Right arm
        right_arm_x = dancer_x + int(30 * np.cos(np.radians(arm_angle)))
        right_arm_y = dancer_y + int(30 * np.sin(np.radians(arm_angle)))
        pygame.draw.line(self.screen, self.BLUE, (dancer_x, dancer_y),
                        (right_arm_x, right_arm_y), 3)
        
        # Left arm
        left_arm_x = dancer_x - int(30 * np.cos(np.radians(arm_angle + 180)))
        left_arm_y = dancer_y - int(30 * np.sin(np.radians(arm_angle + 180)))
        pygame.draw.line(self.screen, self.BLUE, (dancer_x, dancer_y),
                        (left_arm_x, left_arm_y), 3)
        
        # Add glow effect if on beat
        if beat_phase < 0.1 or beat_phase > 0.9:
            pygame.draw.circle(self.screen, self.YELLOW, (dancer_x, dancer_y), 20, 2)
    
    def draw_beat_indicator(self, env_info):
        """Draw beat timing indicator."""
        # Beat bar position
        bar_x = 850
        bar_y = 100
        bar_width = 300
        bar_height = 30
        
        # Background
        pygame.draw.rect(self.screen, self.DARK_GRAY,
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.WHITE,
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Beat phase indicator
        beat_phase = env_info.get('beat_phase', 0)
        indicator_x = bar_x + int(beat_phase * bar_width)
        
        # Draw beat zones (perfect timing)
        perfect_zone_width = 20
        pygame.draw.rect(self.screen, self.GREEN,
                        (bar_x, bar_y, perfect_zone_width, bar_height))
        pygame.draw.rect(self.screen, self.GREEN,
                        (bar_x + bar_width - perfect_zone_width, bar_y, 
                         perfect_zone_width, bar_height))
        
        # Draw indicator
        pygame.draw.line(self.screen, self.YELLOW,
                        (indicator_x, bar_y - 5),
                        (indicator_x, bar_y + bar_height + 5), 3)
        
        # Label
        beat_label = self.font.render("Beat Timing", True, self.WHITE)
        self.screen.blit(beat_label, (bar_x, bar_y - 25))
        
        # Beat count
        beat_text = self.small_font.render(f"BPM: 120", True, self.WHITE)
        self.screen.blit(beat_text, (bar_x, bar_y + bar_height + 5))
    
    def draw_performance_metrics(self, env_info, reward):
        """Draw performance metrics."""
        metrics_x = 850
        metrics_y = 200
        
        # Performance score
        score = env_info.get('performance_score', 0)
        score_text = self.header_font.render(f"Score: {score:.0f}", True, self.YELLOW)
        self.screen.blit(score_text, (metrics_x, metrics_y))
        
        # Current reward
        reward_color = self.GREEN if reward > 0 else self.RED
        reward_text = self.font.render(f"Reward: {reward:+.1f}", True, reward_color)
        self.screen.blit(reward_text, (metrics_x, metrics_y + 40))
        
        # Stats
        stats = env_info.get('episode_stats', {})
        
        y_offset = metrics_y + 80
        stat_items = [
            ("Perfect Moves", stats.get('perfect_moves', 0), self.GREEN),
            ("Good Moves", stats.get('good_moves', 0), self.YELLOW),
            ("Missed Beats", stats.get('missed_beats', 0), self.RED),
            ("Energy Used", f"{stats.get('energy_used', 0):.1f}", self.CYAN),
            ("Time on Beat", f"{stats.get('time_on_beat', 0):.1f}s", self.PURPLE)
        ]
        
        for label, value, color in stat_items:
            text = self.small_font.render(f"{label}: {value}", True, color)
            self.screen.blit(text, (metrics_x, y_offset))
            y_offset += 25
    
    def draw_crowd_excitement(self, env_info):
        """Draw crowd excitement meter."""
        meter_x = 850
        meter_y = 450
        meter_width = 300
        meter_height = 30
        
        # Background
        pygame.draw.rect(self.screen, self.DARK_GRAY,
                        (meter_x, meter_y, meter_width, meter_height))
        pygame.draw.rect(self.screen, self.WHITE,
                        (meter_x, meter_y, meter_width, meter_height), 2)
        
        # Excitement level
        excitement = env_info.get('crowd_excitement', 0.5)
        fill_width = int(excitement * meter_width)
        
        # Color based on level
        if excitement < 0.3:
            color = self.RED
        elif excitement < 0.6:
            color = self.YELLOW
        else:
            color = self.GREEN
        
        pygame.draw.rect(self.screen, color,
                        (meter_x, meter_y, fill_width, meter_height))
        
        # Label
        crowd_label = self.font.render(f"Crowd Excitement: {excitement:.0%}", 
                                      True, self.WHITE)
        self.screen.blit(crowd_label, (meter_x, meter_y - 25))
        
        # Update crowd particles based on excitement
        for particle in self.crowd_particles:
            if excitement > 0.5:
                particle['vy'] = -np.random.random() * excitement * 5
                particle['vx'] = np.random.uniform(-2, 2)
            
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            
            # Bounce back
            if particle['y'] > 750:
                particle['y'] = 750
                particle['vy'] *= -0.5
            
            # Keep in bounds
            if particle['x'] < 50:
                particle['x'] = 50
            if particle['x'] > 750:
                particle['x'] = 750
            
            # Draw particle
            pygame.draw.circle(self.screen, particle['color'],
                             (int(particle['x']), int(particle['y'])),
                             particle['size'])
    
    def draw_move_info(self, env_info):
        """Draw current move information."""
        move_x = 850
        move_y = 530
        
        # Current move
        current_move = env_info.get('current_move', {})
        move_name = "Unknown"
        for name, details in [('basic_step', {'difficulty': 1}), 
                             ('spin', {'difficulty': 2}),
                             ('jump', {'difficulty': 2})]:
            if details == current_move:
                move_name = name.replace('_', ' ').title()
                break
        
        move_text = self.font.render(f"Current Move: {move_name}", True, self.CYAN)
        self.screen.blit(move_text, (move_x, move_y))
        
        # Move difficulty
        difficulty = current_move.get('difficulty', 1)
        diff_text = "★" * difficulty + "☆" * (4 - difficulty)
        diff_render = self.font.render(f"Difficulty: {diff_text}", True, self.YELLOW)
        self.screen.blit(diff_render, (move_x, move_y + 30))
        
        # Style points
        style_points = current_move.get('style_points', 0)
        style_text = self.small_font.render(f"Style Points: {style_points}", True, self.WHITE)
        self.screen.blit(style_text, (move_x, move_y + 60))
    
    def draw_combo_meter(self, env_info):
        """Draw combo multiplier."""
        combo_x = 850
        combo_y = 630
        
        combo = env_info.get('combo_multiplier', 1.0)
        
        # Combo text with effects
        if combo > 5.0:
            combo_color = self.PURPLE
            size = 48
        elif combo > 3.0:
            combo_color = self.YELLOW
            size = 40
        elif combo > 1.5:
            combo_color = self.GREEN
            size = 32
        else:
            combo_color = self.WHITE
            size = 28
        
        combo_font = pygame.font.Font(None, size)
        combo_text = combo_font.render(f"COMBO: {combo:.1f}x", True, combo_color)
        
        # Add glow effect for high combos
        if combo > 3.0:
            for i in range(3):
                glow_surf = combo_font.render(f"COMBO: {combo:.1f}x", True, combo_color)
                glow_surf.set_alpha(50)
                self.screen.blit(glow_surf, (combo_x - i, combo_y - i))
        
        self.screen.blit(combo_text, (combo_x, combo_y))
    
    def draw_action_heatmap(self, action_magnitudes):
        """Draw action magnitude heatmap."""
        heatmap_x = 50
        heatmap_y = 550
        box_size = 15
        
        # Title
        title = self.small_font.render("Joint Activity", True, self.WHITE)
        self.screen.blit(title, (heatmap_x, heatmap_y - 20))
        
        # Draw heatmap grid
        for i, magnitude in enumerate(action_magnitudes[:20]):  # First 20 joints
            row = i // 5
            col = i % 5
            
            x = heatmap_x + col * (box_size + 2)
            y = heatmap_y + row * (box_size + 2)
            
            # Color based on magnitude
            intensity = min(255, int(magnitude * 255))
            color = (intensity, 255 - intensity, 0)
            
            pygame.draw.rect(self.screen, color, (x, y, box_size, box_size))
            pygame.draw.rect(self.screen, self.WHITE, (x, y, box_size, box_size), 1)
    
    def draw_performance_graph(self, reward):
        """Draw performance history graph."""
        graph_x = 50
        graph_y = 650
        graph_width = 200
        graph_height = 80
        
        # Add to history
        self.performance_history.append(reward)
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
        
        # Draw background
        pygame.draw.rect(self.screen, self.DARK_GRAY,
                        (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, self.WHITE,
                        (graph_x, graph_y, graph_width, graph_height), 1)
        
        # Draw graph
        if len(self.performance_history) > 1:
            points = []
            for i, value in enumerate(self.performance_history):
                x = graph_x + int(i * graph_width / self.max_history)
                # Normalize value to graph height
                normalized_value = np.clip((value + 100) / 200, 0, 1)
                y = graph_y + graph_height - int(normalized_value * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
        
        # Label
        label = self.small_font.render("Reward History", True, self.WHITE)
        self.screen.blit(label, (graph_x, graph_y - 20))
    
    def draw_episode_stats(self, env_info):
        """Draw episode statistics."""
        stats_x = 300
        stats_y = 650
        
        stats = env_info.get('episode_stats', {})
        
        # Title
        title = self.small_font.render("Episode Stats", True, self.WHITE)
        self.screen.blit(title, (stats_x, stats_y - 20))
        
        # Stats
        longest_combo = stats.get('longest_combo', 0)
        creativity = stats.get('creativity_score', 0)
        crowd_rating = stats.get('crowd_rating', 0)
        
        stats_text = [
            f"Longest Combo: {longest_combo}",
            f"Creativity: {creativity:.1f}",
            f"Crowd Rating: {crowd_rating:.0%}"
        ]
        
        for i, text in enumerate(stats_text):
            render = self.small_font.render(text, True, self.CYAN)
            self.screen.blit(render, (stats_x, stats_y + i * 20))


def test_dancing_environment():
    """Test the humanoid dancing environment with visualization."""
    
    print("\n" + "="*60)
    print("🕺 HUMANOID DANCING ENVIRONMENT TEST 💃")
    print("="*60)
    
    # Initialize environment
    print("\n📌 Initializing environment...")
    try:
        register_env()
        env = HumanoidDancingEnv(render_mode='human')
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        traceback.print_exc()
        return
    
    # Initialize visualization
    viz = DancingVisualization()
    
    # Test parameters
    n_episodes = 3
    max_steps_per_episode = 600  # 10 seconds at 60 Hz
    
    # Episode results
    episode_results = []
    
    print(f"\n🎮 Running {n_episodes} test episodes...")
    print(f"   Each episode: max {max_steps_per_episode} steps ({max_steps_per_episode/60:.1f} seconds)")
    
    for episode in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*40}")
        
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0
        episode_data = {
            'episode': episode + 1,
            'steps': 0,
            'total_reward': 0,
            'perfect_moves': 0,
            'good_moves': 0,
            'missed_beats': 0,
            'max_combo': 0,
            'final_crowd_excitement': 0,
            'performance_score': 0,
            'termination_reason': None
        }
        
        # Run episode
        for step in range(max_steps_per_episode):
            # Random action (you can implement a policy here)
            action = env.action_space.sample()
            
            # Scale actions for more controlled movement
            action *= 0.5
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_data['steps'] = step + 1
            episode_data['total_reward'] = episode_reward
            
            # Calculate action magnitudes for visualization
            action_magnitudes = np.abs(action) / 200.0  # Normalize to [0, 1]
            
            # Update visualization
            if not viz.update(info, reward, action_magnitudes):
                print("\n🔴 Visualization window closed by user")
                env.close()
                pygame.quit()
                return
            
            # Print progress
            if step % 60 == 0:  # Every second
                print(f"   Step {step:4d}: Reward: {reward:+8.2f}, "
                      f"Combo: {info['combo_multiplier']:4.1f}x, "
                      f"Crowd: {info['crowd_excitement']:5.1%}")
            
            if terminated or truncated:
                if terminated:
                    episode_data['termination_reason'] = 'Fall or out of bounds'
                else:
                    episode_data['termination_reason'] = 'Time limit'
                break
        
        # Update episode data
        stats = info.get('episode_stats', {})
        episode_data['perfect_moves'] = stats.get('perfect_moves', 0)
        episode_data['good_moves'] = stats.get('good_moves', 0)
        episode_data['missed_beats'] = stats.get('missed_beats', 0)
        episode_data['max_combo'] = stats.get('longest_combo', 0)
        episode_data['final_crowd_excitement'] = info.get('crowd_excitement', 0)
        episode_data['performance_score'] = info.get('performance_score', 0)
        
        episode_results.append(episode_data)
        
        # Print episode summary
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Total Steps: {episode_data['steps']}")
        print(f"   Total Reward: {episode_data['total_reward']:.2f}")
        print(f"   Performance Score: {episode_data['performance_score']:.2f}")
        print(f"   Max Combo: {episode_data['max_combo']}")
        print(f"   Crowd Excitement: {episode_data['final_crowd_excitement']:.1%}")
        print(f"   Termination: {episode_data['termination_reason']}")
        
        # Wait a bit between episodes
        time.sleep(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📈 FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    avg_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_score = np.mean([r['performance_score'] for r in episode_results])
    avg_combo = np.mean([r['max_combo'] for r in episode_results])
    avg_crowd = np.mean([r['final_crowd_excitement'] for r in episode_results])
    
    print(f"Average Total Reward: {avg_reward:.2f}")
    print(f"Average Performance Score: {avg_score:.2f}")
    print(f"Average Max Combo: {avg_combo:.1f}")
    print(f"Average Crowd Excitement: {avg_crowd:.1%}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"dancing_test_results_{timestamp}.json"
    
    test_results = {
        'timestamp': timestamp,
        'environment': 'HumanoidDancing-v0',
        'test_parameters': {
            'n_episodes': n_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'render_mode': 'human'
        },
        'summary': {
            'average_reward': float(avg_reward),
            'average_performance_score': float(avg_score),
            'average_max_combo': float(avg_combo),
            'average_crowd_excitement': float(avg_crowd)
        },
        'episodes': episode_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Close environment
    env.close()
    pygame.quit()
    
    print("\n✅ Test completed successfully! [[memory:7080335]] [[memory:7080345]]")


if __name__ == "__main__":
    try:
        test_dancing_environment()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        pygame.quit()
