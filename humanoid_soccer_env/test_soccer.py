#!/usr/bin/env python3
"""Simple test script for Humanoid Soccer Environment.

Author: Hasnain Fareed
License: MIT (2025)
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid_soccer_env import HumanoidSoccerEnv

def create_environment(render_mode='human'):
    """Create soccer environment."""
    print("🤖⚽ Creating Improved Humanoid Soccer Environment...")
    print("   - Clean field without unnecessary objects")
    print("   - More realistic humanoid proportions")
    print("   - Better joint ranges and movement")
    print("   - Fixed gravity and physics")
    print("   - Ball always positioned in front of player (centered)")
    print("   - Football-like ball appearance (larger size)")
    print("   - Much longer episodes (1000 steps, ~10x longer)")
    
    try:
        env = HumanoidSoccerEnv(render_mode=render_mode)
        print("✅ Environment created successfully!")
        return env
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return None

def test_environment_basic(env, episodes=10, episode_steps=1000):
    """Test basic environment functionality with longer episodes."""
    print(f"\n🧪 Testing improved environment ({episodes} episodes, {episode_steps} steps each)...")
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset environment
        obs, info = env.reset()
        print(f"  ✅ Reset successful, obs shape: {obs.shape}")
        
        if env.render_mode == "human":
            print("  🎮 3D visualization window should be visible now!")
            print("  👀 Look for: Clean field, realistic humanoid, no extra objects")
            time.sleep(3)  # Give more time to see the improvements
        
        # Run episode
        total_reward = 0
        for step in range(episode_steps):
            # Simple walking actions
            action = env.action_space.sample() * 0.1  # Small random actions
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                ball_pos = info.get('ball_position', [0, 0, 0])
                robot_pos = info.get('robot_position', [0, 0, 0])
                print(f"    Step {step}: reward={reward:.2f}, ball=({ball_pos[0]:.1f},{ball_pos[1]:.1f})")
            
            if terminated or truncated:
                print(f"    Episode ended at step {step}")
                break
            
            time.sleep(0.01)  # Slightly faster visualization for long episodes
        
        print(f"  📊 Total reward: {total_reward:.2f}")
        
        # Check for ball contact
        if info.get('ball_contact', False):
            print("  ⚽ Ball contact detected!")
        
        # Check for goal
        if info.get('goal_scored', False):
            print("  🥅 GOAL SCORED!")

def main():
    """Main function."""
    print("🤖⚽ Improved Humanoid Soccer Environment Test")
    print("=" * 60)
    print("🎯 Changes made:")
    print("   • Removed unnecessary objects (wind zones, extra lighting)")
    print("   • Cleaned up field boundaries (just lines, not walls)")
    print("   • Improved humanoid proportions and joint ranges")
    print("   • Better foot design for soccer gameplay")
    print("   • Fixed gravity and physics issues")
    print("   • Ball always positioned in front of player (centered)")
    print("   • Football-like ball appearance (larger size)")
    print("   • Much longer episodes (1000 steps, ~10x longer than before)")
    print("=" * 60)
    
    # Create environment
    env = create_environment(render_mode='human')
    if env is None:
        return
    
    try:
        # Test basic functionality with more episodes and much longer episodes
        test_environment_basic(env, episodes=10, episode_steps=1000)
        
        print("\n🎉 All tests completed successfully!")
        
        if env.render_mode == "human":
            print("🎮 Visualization window will close in 5 seconds...")
            print("👀 Take a final look at the improved environment!")
            time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        env.close()
        print("✅ Environment closed")

if __name__ == "__main__":
    main()
